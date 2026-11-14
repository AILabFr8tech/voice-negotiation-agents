"""Telefonía de PoC1: Twilio Media Streams sobre WebSocket, servido por FastAPI.

Es el mismo LLM, el mismo contexto, los mismos manejadores y el mismo motor de
negociación que ya verifica el chequeo de humo. Lo único que cambia es el
extremo del pipeline: donde el humo pone `CapturadorDeSalida`, aquí va el
transporte de Twilio.

Dos endpoints y uno de cortesía:

    POST/GET /twiml  -> el TwiML que Twilio pide al contestar la llamada
    WS       /ws     -> el Media Stream bidireccional (audio en los dos sentidos)
    GET      /salud  -> para que el marcador confirme que el túnel está arriba
                        antes de gastar una llamada

Sobre la cuenta de Twilio (CLAUDE.md): este módulo **nunca** modifica la
configuración de un número ni de un trunk. Solo sirve TwiML por HTTP y habla
por el WebSocket. La `url` del TwiML viaja en la llamada saliente, así que el
webhook configurado en el número del agente despachador no se toca.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from poc1_pipecat.app.config import (
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc1_pipecat.app.latency import (
    AgentSpeechAnnotator,
    LatencyCollector,
    LatencyProbe,
)
from poc1_pipecat.app.pipeline import (
    RECORDINGS_PATH,
    build_turn_analyzer,
    build_context,
    build_llm,
    build_stt,
    build_tts,
    build_vad,
)
from poc1_pipecat.app.tools import register_functions
from shared.negotiation_engine import (
    TEST_TRIP,
    TripData,
    CloseReason,
    NegotiationResult,
    NegotiationSession,
)

logger = logging.getLogger(__name__)

RUTA_TWIML: Final[str] = "twiml"
RUTA_WS: Final[str] = "ws"
HEALTH_PATH: Final[str] = "health"

DEFAULT_PORT: Final[int] = 8000

# Etiqueta de la corrida. Viaja del marcador -> query string del TwiML ->
# `<Parameter>` del `<Stream>` -> `customParameters` del WebSocket -> JSON.
# Son los tres campos que `shared.metrics.llamadas.identificar()` busca primero
# en el JSON, antes de caer al archivo de referencia.
RUN_PARAMETERS: Final[tuple[str, ...]] = ("poc", "scenario", "repetition")

# Twilio manda y espera μ-law de 8 kHz. La **salida** se fija a esa tasa para no
# sintetizar en alta y tirar la mitad al serializar.
#
# La **entrada** se deja en el default de pipecat (16 kHz) a propósito. Fijarla en
# 8 kHz era lo natural —la fuente ya es telefónica y subirla no agrega
# información— pero deja al pipeline dependiendo de que cada consumidor
# remuestree bien por su cuenta, y el analizador de fin de turno de pipecat
# arrastró justo ese bug hasta hace poco (issues #3844 / #3857: Smart Turn v3
# espera 16 kHz y recibía 8 kHz "al doble de velocidad", clasificando frases
# incompletas como completas).
#
# En pipecat 1.6.0 el remuestreo interno ya está y **se verificó que funciona**:
# con el mismo audio, declarar 8000 o 16000 da la misma predicción al milésimo
# (0.5098 vs 0.5098; ver docs/comparativo_resultados.md). Aun así se deja que el
# serializer suba a 16 kHz una sola vez, río arriba, en vez de repartir la
# responsabilidad entre N consumidores.
TELEPHONY_SAMPLE_RATE: Final[int] = 8000


# --------------------------------------------------------------------------
# TwiML
# --------------------------------------------------------------------------


def build_twiml(
    url_websocket: str, parameters: Mapping[str, str] | None = None
) -> str:
    """TwiML que conecta la llamada al Media Stream bidireccional.

    `<Connect><Stream>` es bidireccional: el audio del transportista llega por
    el WebSocket y el del agente se devuelve por el mismo. `<Start><Stream>`
    sería de una sola vía y el transportista no oiría nada.

    Cuando el WebSocket se cierra, Twilio sigue con el verbo siguiente del
    TwiML; al no haber ninguno, cuelga.

    `parametros` se emite como `<Parameter>` dentro del `<Stream>`. Twilio los
    devuelve en `customParameters` del mensaje `start` del WebSocket, y es así
    como el escenario y la repetición llegan hasta el JSON de la corrida sin
    tener que adivinarlos después por Call SID.
    """
    labels = ""
    for name, valor in (parameters or {}).items():
        text = str(valor).strip()
        if not text:
            continue
        labels += f"<Parameter name={quoteattr(str(name))} value={quoteattr(text)} />"

    stream = f"<Stream url={quoteattr(url_websocket)}>{labels}</Stream>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Connect>{stream}</Connect>"
        "</Response>"
    )


# --------------------------------------------------------------------------
# Registro del diálogo
# --------------------------------------------------------------------------


@dataclass
class CallRecord:
    """Transcripción de la llamada, con el segundo relativo de cada intervención."""

    start: float = field(default_factory=time.monotonic)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def annotate(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.turns.append(
            {
                "t": round(time.monotonic() - self.start, 3),
                "role": role,
                "text": cleaned,
            }
        )
        logger.info("[%s] %s", role, cleaned)

    def as_text(self) -> str:
        return "\n".join(
            f"[{t['t']:>7.3f}s] {t['role']:>13}  {t['text']}" for t in self.turns
        )

    def roles(self) -> set[str]:
        """Qué partes quedaron registradas. Sirve para verificar el diálogo."""
        return {str(t.get("role", "")) for t in self.turns}


def read_identification(start: Mapping[str, Any]) -> dict[str, str]:
    """Saca `poc`/`escenario`/`repeticion` del mensaje `start` de Twilio.

    Twilio entrega los `<Parameter>` del TwiML en `customParameters`. Se lee de
    forma tolerante —claves en cualquier caja, valores vacíos fuera— porque un
    error aquí no debe tumbar una llamada que ya está sonando.
    """
    raw_values = start.get("customParameters") or {}
    if not isinstance(raw_values, Mapping):
        return {}

    by_lowercase = {
        str(clave).strip().lower(): valor for clave, valor in raw_values.items()
    }
    label: dict[str, str] = {}
    for clave in RUN_PARAMETERS:
        valor = by_lowercase.get(clave)
        if valor is None:
            continue
        text = str(valor).strip()
        if text:
            label[clave] = text
    return label


class DialogueAnnotator(FrameProcessor):
    """Copia al registro lo que transcribe el STT del transportista.

    Va justo después del STT. **La voz del agente ya no se anota aquí**: se anota
    después del TTS, con `AnotadorDeVozDelAgente`, para registrar lo que de
    verdad se pronunció y no lo que el LLM quiso decir. Ver el docstring de
    `app/latency.py`.

    Se conserva el parámetro `rol` para no romper el contrato de los tests y para
    dejar explícito de quién es la fila que escribe.
    """

    def __init__(self, record: CallRecord, role: str = "carrier") -> None:
        super().__init__()
        self._record = record
        self._role = role

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._record.annotate(self._role, frame.text)
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------
# Pipeline de voz
# --------------------------------------------------------------------------


@dataclass
class VoicePipeline:
    """El pipeline de una llamada real, listo para correr."""

    task: PipelineTask
    record: CallRecord
    session: NegotiationSession
    collector: LatencyCollector


@dataclass(frozen=True)
class HarnessHooks:
    """Lo que el arnés le inserta a este pipeline, sin duplicarlo.

    **Existe para que el arnés mida ESTE pipeline y no una copia suya.** El arnés
    necesita dos cosas que una llamada de campaña no: fechar el cero que produce
    el VAD, y devolverle por el cable lo que se transcribió. Copiar el armado del
    pipeline para agregárselas sería la forma segura de que las dos versiones se
    separen —una perilla que se cambia aquí y no allá— y entonces el arnés
    mediría un pipeline que nadie corre.

    Los tres campos son inertes por default: una llamada de campaña construye lo
    mismo que antes de que esto existiera.
    """

    # `False` en el arnés: no hay ninguna llamada de Twilio que colgar, el
    # `callSid` es inventado, y el serializer se niega a construirse con
    # `auto_hang_up` y sin credenciales.
    hang_up_via_rest: bool = True
    # Se llaman una vez por llamada y se insertan en tres puntos distintos,
    # porque **ningún punto del pipeline ve todo**:
    #
    # - `tras_el_stt`: el único lugar donde existe el `TranscriptionFrame` del
    #   transportista. El agregador de usuario se lo queda para armar el
    #   contexto, así que río abajo ya no pasa (medido el 17 de septiembre de
    #   2026: el arnés recibía la voz del agente y ni una sola transcripción).
    # - `tras_el_turno`: después del agregador, que es quien emite los frames
    #   del VAD y el fin de turno. Antes no existe ninguno de los dos.
    # - `antes_de_la_salida`: después del TTS, donde ya se sabe qué se pronunció.
    after_stt: Callable[[], Sequence[FrameProcessor]] | None = None
    after_turn: Callable[[], Sequence[FrameProcessor]] | None = None
    before_output: Callable[[], Sequence[FrameProcessor]] | None = None
    # Después de `guardar_corrida`, con el `callSid` y la ruta del JSON. Es
    # donde el arnés escribe lo suyo, que tiene que quedar **junto** al JSON del
    # pipeline para que el cruce sea un `join` por nombre y no una heurística.
    on_save: Callable[[str, Path], None] | None = None


def _insert(
    factory: Callable[[], Sequence[FrameProcessor]] | None,
) -> list[FrameProcessor]:
    return list(factory()) if factory is not None else []


def build_serializer(
    settings: Settings,
    *,
    stream_sid: str,
    call_sid: str,
    hooks: HarnessHooks,
) -> TwilioFrameSerializer:
    """El serializer de Twilio, con o sin colgado por REST.

    Es función aparte y no dos líneas dentro de `build_voice_pipeline`
    porque **la decisión de colgar es verificable y el pipeline entero no**:
    armarlo pide claves de OpenAI, Deepgram y ElevenLabs, así que una prueba que
    lo construyera no correría en CI. Aquí sí se puede preguntar lo único que
    importa — que con `hang_up_via_rest` apagado no haga falta credencial alguna,
    que es justo lo que el arnés necesita para correr sin cuenta de Twilio.
    """
    return TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        params=TwilioFrameSerializer.InputParams(
            auto_hang_up=hooks.hang_up_via_rest
        ),
    )


def build_voice_pipeline(
    settings: Settings,
    session: NegotiationSession,
    websocket: WebSocket,
    *,
    stream_sid: str,
    call_sid: str,
    hooks: HarnessHooks | None = None,
) -> VoicePipeline:
    """Arma el pipeline completo de una llamada telefónica.

    El VAD y el analizador de fin de turno van en los parámetros del agregador
    de usuario, que es donde pipecat 1.6.0 los consume — ya no son parámetros
    del transporte.

    El serializer recibe las credenciales de Twilio para poder colgar por REST
    cuando el pipeline emite un `EndFrame`: así la llamada termina cuando el
    agente cierra la negociación, no cuando se agota el límite de duración.
    """
    verify_for_voice(settings)
    hooks = hooks or HarnessHooks()

    serializer = build_serializer(
        settings, stream_sid=stream_sid, call_sid=call_sid, hooks=hooks
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Twilio ya trae su propio encuadre; un header WAV lo rompería.
            add_wav_header=False,
            serializer=serializer,
            # El tope duro de duración lo pone `time_limit` al marcar, que es
            # del lado de Twilio y no depende de que este proceso siga vivo.
            session_timeout=None,
        ),
    )

    llm = build_llm(settings)
    register_functions(llm)

    context = build_context(
        session.trip, compensations_path=session.compensations_path
    )
    aggregator_group = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=build_vad(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=build_turn_analyzer()
                    )
                ]
            ),
        ),
    )

    record = CallRecord()
    collector = LatencyCollector()

    # Tres sondas, porque ningún punto del pipeline ve todos los eventos: arriba
    # del STT no existe todavía el audio del TTS, y abajo del TTS ya no existe el
    # `LLMTextFrame` (el TTS lo convirtió en `TTSTextFrame`). Las marcas son
    # idempotentes, así que sobra con que cada evento cruce al menos una.
    pipeline = Pipeline(
        [
            transport.input(),
            build_stt(settings),
            DialogueAnnotator(record, "carrier"),
            LatencyProbe(collector, label="entry"),
            # Vacío salvo con el arnés, y **antes** del agregador: es el último
            # punto donde se ve el `TranscriptionFrame`.
            *_insert(hooks.after_stt),
            aggregator_group.user(),
            # Vacío salvo con el arnés. Va aquí y no más arriba porque los frames
            # del VAD y el fin de turno los emite el agregador: una sonda puesta
            # antes no vería ninguno de los dos.
            *_insert(hooks.after_turn),
            llm,
            LatencyProbe(collector, label="llm"),
            build_tts(settings),
            # Después del TTS: aquí se sabe qué se pronunció de verdad y cuándo
            # salió el primer byte de audio.
            AgentSpeechAnnotator(lambda text: record.annotate("agent", text)),
            LatencyProbe(collector, label="output"),
            *_insert(hooks.before_output),
            transport.output(),
            aggregator_group.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # Sin `audio_in_sample_rate`: el serializer sube el μ-law de 8 kHz al
            # default de pipecat (16 kHz) una sola vez, y de ahí para abajo todo
            # el pipeline —STT y analizador de fin de turno— ve la misma tasa.
            # Ver la nota de TASA_TELEFONIA.
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        app_resources=session,
        # No hay cliente RTVI escuchando.
        enable_rtvi=False,
    )

    # Cerrar la negociación cuelga la llamada: `stop_when_done` deja que el
    # agente termine de decir su despedida y recién entonces emite el EndFrame,
    # que el serializer convierte en un colgado por REST.
    async def _on_close(resultado: NegotiationResult) -> None:
        logger.info(
            "negociación cerrada como %r; se cuelga al terminar de hablar",
            resultado.resultado,
        )
        await task.stop_when_done()

    session.close_callback = _on_close

    @transport.event_handler("on_client_connected")
    async def _on_connect(_transport, _ws) -> None:
        logger.info("Media Stream conectado (call_sid=%s)", call_sid)
        # Cero de la llamada: el turno de apertura se mide desde aquí, porque el
        # saludo del agente no tiene un "fin de habla" antes.
        collector.begin()
        # El agente habla primero: corre el LLM sobre el contexto inicial.
        await task.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, _ws) -> None:
        logger.info("Media Stream desconectado (call_sid=%s)", call_sid)
        if not session.is_closed:
            # Se cortó sin que el agente cerrara: queda registrado como colgado
            # para que la métrica de resultado no se pierda.
            session.close_without_agreement(CloseReason.HUNG_UP)
        await task.cancel()

    return VoicePipeline(
        task=task, record=record, session=session, collector=collector
    )


# --------------------------------------------------------------------------
# Persistencia de la corrida
# --------------------------------------------------------------------------


def save_run(
    session: NegotiationSession,
    record: CallRecord,
    *,
    base_path: Path | None = None,
    collector: LatencyCollector | None = None,
    identification: Mapping[str, str] | None = None,
) -> Path:
    """Vuelca el resultado, la transcripción y la latencia a JSON.

    Cae en `grabaciones/`, que está gitignoreada: la transcripción puede
    contener voz y datos de un transportista real (LFPDPPP).

    El resumen de la sesión no incluye `max_price`, por diseño de
    `SesionNegociacion.resumen()`. Tampoco se agrega aquí: `shared/metrics/` lo
    recibe por parámetro desde la config de la corrida, nunca desde este archivo.

    `identificacion` son `poc`, `escenario` y `repeticion` — llegan de los
    `<Parameter>` del TwiML y se escriben al nivel de arriba, que es donde
    `shared.metrics.llamadas.identificar()` los busca primero.
    """
    folder = base_path or RECORDINGS_PATH
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"call_{session.call_id or 'no_sid'}.json"

    content: dict[str, Any] = {"call_sid": session.call_id}

    # La identificación va antes del resumen para que se vea al abrir el archivo.
    for clave in ("poc", "scenario", "repetition"):
        valor = (identification or {}).get(clave)
        if valor is None or str(valor).strip() == "":
            continue
        try:
            content[clave] = int(str(valor).strip())
        except ValueError:
            # Un valor no numérico se conserva tal cual en vez de tirarse: es
            # dato de trazabilidad y perderlo es peor que guardarlo raro.
            content[clave] = str(valor).strip()

    content["summary"] = session.summary()
    content["transcript"] = record.turns
    if collector is not None:
        collector.close_pending()
        content["latencies"] = collector.to_json()

    destination.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


# --------------------------------------------------------------------------
# Servidor
# --------------------------------------------------------------------------


def create_app(
    settings: Settings,
    *,
    trip: TripData = TEST_TRIP,
    hooks: HarnessHooks | None = None,
) -> FastAPI:
    """La app de FastAPI que atiende el TwiML y el Media Stream.

    `ganchos` es cómo el arnés corre **este** servidor en vez de uno parecido;
    ver `GanchosDelArnes`. En la campaña va en `None` y no cambia nada.
    """
    app = FastAPI(title="PoC1 — agente de voz negociador (Pipecat)")

    @app.get(f"/{HEALTH_PATH}")
    async def health() -> JSONResponse:
        """Para que el marcador confirme el túnel antes de gastar una llamada."""
        return JSONResponse(
            {
                "ok": True,
                "websocket": settings.url_websocket(RUTA_WS),
                "modelos": {
                    "llm": settings.openai_model,
                    "stt": f"{settings.deepgram_model}/{settings.deepgram_language}",
                    "tts": settings.eleven_model,
                },
            }
        )

    @app.api_route(f"/{RUTA_TWIML}", methods=["GET", "POST"])
    async def twiml(request: Request) -> Response:
        """El TwiML que Twilio pide al contestar.

        `poc`, `escenario` y `repeticion` llegan como query string en la URL que
        el marcador le pasó a Twilio, y salen como `<Parameter>` del `<Stream>`
        para que vuelvan por el WebSocket. Así la etiqueta de la corrida viaja
        con la llamada y no hay que reconstruirla después por Call SID.
        """
        # Twilio manda los datos de la llamada como formulario.
        try:
            form = await request.form()
            call_sid = str(form.get("CallSid", "")) or "?"
        except Exception:  # noqa: BLE001 - un GET de prueba no trae formulario
            call_sid = "?"

        label = {
            clave: request.query_params[clave]
            for clave in RUN_PARAMETERS
            if request.query_params.get(clave)
        }
        url_ws = settings.url_websocket(RUTA_WS)
        logger.info(
            "TwiML pedido (call_sid=%s, corrida=%s) -> %s", call_sid, label, url_ws
        )
        return Response(
            content=build_twiml(url_ws, label), media_type="application/xml"
        )

    @app.websocket(f"/{RUTA_WS}")
    async def media_stream(websocket: WebSocket) -> None:
        """El Media Stream: una llamada, una sesión de negociación."""
        await websocket.accept()

        # Los dos primeros mensajes de Twilio se leen a mano porque el
        # serializer necesita el streamSid y el callSid para construirse, y eso
        # tiene que pasar antes de que el transporte tome el WebSocket.
        await websocket.receive_text()  # {"event": "connected"}
        raw_start = await websocket.receive_text()
        start = json.loads(raw_start).get("start", {})
        stream_sid = start.get("streamSid", "")
        call_sid = start.get("callSid", "")
        identification = read_identification(start)

        if not stream_sid or not call_sid:
            logger.error("el mensaje `start` de Twilio no trae streamSid/callSid")
            await websocket.close()
            return

        logger.info(
            "llamada entrando al pipeline: call_sid=%s corrida=%s",
            call_sid,
            identification or "(sin etiquetar)",
        )
        if not identification:
            logger.warning(
                "esta llamada no trae scenario/repetition: pásalos al marcar "
                "(--scenario/--repetition/--poc) o habrá que etiquetarla a mano "
                "en el archivo de referencia"
            )
        session = NegotiationSession(trip=trip, call_id=call_sid)
        pipeline = build_voice_pipeline(
            settings,
            session,
            websocket,
            stream_sid=stream_sid,
            call_sid=call_sid,
            hooks=hooks,
        )

        runner = PipelineRunner(handle_sigint=False)
        try:
            await runner.run(pipeline.task)
        finally:
            destination = save_run(
                session,
                pipeline.record,
                collector=pipeline.collector,
                identification=identification,
            )
            print("\n" + "=" * 70)
            print(f"Llamada terminada: {call_sid}")
            print("=" * 70)
            print(pipeline.record.as_text() or "(sin transcripción)")
            print()
            samples = pipeline.collector.samples
            measured_items = [
                m.full_turn_ms
                for m in samples
                if m.full_turn_ms is not None and not m.is_opening
            ]
            print(f"  turnos con latencia  {len(samples)}")
            if measured_items:
                print(
                    f"  turno completo       régimen: "
                    f"{min(measured_items):,.0f}-{max(measured_items):,.0f} ms "
                    f"({len(measured_items)} turnos)"
                )
            opening = next((m for m in samples if m.is_opening), None)
            if opening and opening.full_turn_ms is not None:
                print(f"  apertura (frío)      {opening.full_turn_ms:,.0f} ms")
            print(f"  roles registrados    {sorted(pipeline.record.roles()) or '(ninguno)'}")

            resultado = session.resultado
            print(f"  historial validado   {session.offer_history or '(ninguno)'}")
            print(f"  compensaciones       {session.offered_compensations or '(ninguna)'}")
            print(f"  resultado            {resultado.resultado if resultado else '(sin cierre)'}")
            if resultado and resultado.monto_final is not None:
                print(f"  monto final          {resultado.monto_final:,.0f}")
                print(f"  monto validado       {resultado.validated_final_amount}")
            print(f"\n  corrida guardada en  {destination}")

            if hooks is not None and hooks.on_save is not None:
                # Después de `guardar_corrida` y dentro del mismo `finally`: lo
                # que el arnés escribe tiene que quedar aunque el pipeline haya
                # muerto por una excepción, igual que el JSON de la llamada.
                try:
                    hooks.on_save(call_sid, destination)
                except Exception as error:  # noqa: BLE001 - diagnóstico, no la llamada
                    logger.warning("el gancho del arnés falló: %r", error)

    return app


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.telephony",
        description=(
            "Servidor de telefonía de PoC1. Sirve el TwiML y el Media Stream; "
            "no marca. Para marcar, usa `poc1_pipecat.app.llamada`."
        ),
    )
    arg_parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="puerto local"
    )
    args = arg_parser.parse_args(argv)

    # Antes de cualquier print: el volcado de la transcripción ocurre después de
    # colgar una llamada ya pagada, y morir ahí por el encoding de la consola
    # perdería la evidencia.
    force_utf8_output()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = load_settings()
    verify_for_telephony(settings)
    verify_for_voice(settings)

    print("Servidor de telefonía de PoC1")
    print(f"  local          http://localhost:{args.port}")
    print(f"  TwiML          {settings.url_webhook(RUTA_TWIML)}")
    print(f"  Media Stream   {settings.url_websocket(RUTA_WS)}")
    print(f"  salud          {settings.url_webhook(HEALTH_PATH)}")
    print(
        f"\nEl túnel debe apuntar a este puerto:\n"
        f"  ngrok http {args.port} --domain={settings.public_host()}\n"
    )

    import uvicorn

    # 127.0.0.1 y no 0.0.0.0: ngrok corre en esta misma máquina, así que no hay
    # razón para exponer el puerto a la red local.
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
