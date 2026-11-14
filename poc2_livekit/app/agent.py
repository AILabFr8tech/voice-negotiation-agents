"""El agente de voz de PoC2: `AgentSession` de LiveKit sobre el trunk SIP.

Es el mismo prompt, las mismas 4 funciones y el mismo motor de negociación que
PoC1. Lo que cambia es el ensamblado: donde PoC1 arma una lista de
`FrameProcessor` en un `Pipeline`, aquí se declaran los servicios en una
`AgentSession` y LiveKit se encarga del orden.

## Lo que NO tiene PoC2 y sí tenía PoC1

No hay servidor de TwiML, ni WebSocket de Media Stream, ni túnel de ngrok, ni
serializador de μ-law. El participante SIP entra a la sala y el worker ya está
dentro. Es la diferencia de esfuerzo de integración más grande entre las dos
arquitecturas, y va al reporte como tal.

## Aislamiento del despachador

El worker se registra con `agent_name` propio, así que **no recibe dispatch
automático**: solo entra a las salas donde se le pide por nombre. Ver el README.

## Grabación

**PoC2 no graba audio.** El WER se calcula contra el guion actuado: las líneas
`stt:` del JSON son la hipótesis y las `yo:` la referencia, que se corrige a
mano. El método es idéntico en las 3 PoC, así que no depende del framework.

`record=False` es aparte, y también a propósito: el `record=True` de LiveKit
sube audio, transcripción y trazas a la observabilidad del proyecto en la nube
— y el proyecto es el del despachador real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from poc2_livekit.app.config import (
    CARRIER_IDENTITY,
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_livekit,
    verify_for_voice,
)
from poc2_livekit.app.end_of_turn import install as install_end_of_turn_capture
from poc2_livekit.app.latency import LatencyCollector
from poc2_livekit.app.marks import MarksRecord
from poc2_livekit.app.tools import build_tools
from shared import llm_transport
from shared.negotiation_engine import (
    TEST_TRIP,
    TripData,
    CloseReason,
    NegotiationResult,
    NegotiationSession,
)
from shared.system_prompt import render_system_prompt

logger = logging.getLogger(__name__)

RECORDINGS_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "recordings"

AGENT_ROLE: Final[str] = "agent"
CARRIER_ROLE: Final[str] = "carrier"

# Etiqueta de la corrida. En PoC1 viaja por el TwiML; aquí, por el `metadata`
# del dispatch. Son las tres llaves que `shared.metrics.llamadas.identificar()`
# busca primero en el JSON.
RUN_PARAMETERS: Final[tuple[str, ...]] = ("poc", "scenario", "repetition")

# Marca de una corrida de diagnóstico del arnés (por ejemplo el contraste de
# banda). No es una llamada de la campaña: sus números no describen el
# comportamiento telefónico de PoC2 y **nunca** deben entrar a una poza de
# percentiles ni al comparativo. Se escribe en `excluir_de_pooled`, que es el
# mecanismo que `shared/metrics/` ya respeta al consolidar.
DIAGNOSTICS_KEY: Final[str] = "diagnostics"

# Atributo con el que LiveKit publica el estado de la llamada SIP en el
# participante. Pasa por `dialing` mientras timbra y solo llega a `active`
# cuando el transportista descuelga.
ATTRIBUTE_SIP_STATE: Final[str] = "sip.callStatus"
SIP_ACTIVE: Final[str] = "active"

# Tope de espera a que contesten. El marcador tiene su propio tope de llamada
# (`--limite`, 600 s por default); este solo evita que el job se quede colgado
# para siempre si nadie descuelga y el marcador muere sin avisar.
ANSWER_WAIT_S: Final[float] = 90.0


# --------------------------------------------------------------------------
# Servicios
# --------------------------------------------------------------------------


def preload_plugins() -> None:
    """Importa los plugins de LiveKit **en el hilo principal**, antes del worker.

    En Windows el worker corre cada job en un HILO, no en un proceso
    (`livekit/agents/worker.py`: `_default_job_executor_type = THREAD` cuando
    `sys.platform` empieza con "win", por un BrokenPipeError al crear procesos).
    Y registrar un plugin exige el hilo principal
    (`livekit/agents/plugin.py::register_plugin`):

        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Plugins must be registered on the main thread")

    El registro ocurre en el **primer import** de cada plugin. Con los imports
    perezosos de los constructores de abajo, ese primer import cae dentro de
    `prewarm` o del `entrypoint`, que ya corren en el hilo del job: los cuatro
    job runners mueren al arrancar, el dispatch falla con "failed to launch job
    after 3 attempts" y el transportista contesta a una sala **sin agente**.
    Nada en el log dice "no hay audio", así que este comentario se queda.

    Importar aquí los deja en `sys.modules`; los imports de los constructores
    siguen siendo perezosos —así el humo nivel 0 no paga la carga— pero ya solo
    son una búsqueda en caché.
    """
    from livekit.plugins import (  # noqa: F401  (se importan por su efecto: registrarse)
        deepgram,
        elevenlabs,
        openai,
        silero,
    )


def build_vad() -> Any:
    """Silero con los defaults de LiveKit (`min_silence_duration = 0.55 s`).

    **No se baja a los 0.2 s de PoC1.** Cada framework corre su detección como
    su fabricante la recomienda; además el audio turn detector exige un mínimo
    de 0.25 s, así que forzar el valor de pipecat lo dejaría fuera de contrato.
    La diferencia es parte de lo que se compara y está documentada.
    """
    from livekit.plugins import silero

    return silero.VAD.load()


def build_turn_detector(settings: Settings) -> Any:
    """El audio turn detector, con la versión fijada desde el `.env`.

    Se pasa `version=` explícito a propósito: sin él, LiveKit elige según cómo
    se lanzó el worker (ver la nota en `config.py`).
    """
    from livekit.agents import inference

    return inference.TurnDetector(version=settings.turn_detector_version)  # type: ignore[arg-type]


def build_stt(settings: Settings) -> Any:
    """Deepgram con el mismo modelo e idioma que PoC1.

    `endpointing_ms` y `utterance_end_ms` se dejan en los defaults del plugin, y
    PoC1 también los deja en los suyos. No coinciden entre frameworks, y eso se
    anota como asimetría en vez de emparejarse a mano: tocarlos cambiaría el
    turn-taking de una sola PoC.
    """
    from livekit.plugins import deepgram

    return deepgram.STT(
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        api_key=settings.deepgram_api_key,
    )


def build_llm(settings: Settings) -> Any:
    """gpt-4.1-mini. Congelado para las 3 PoC — no cambiar a media campaña.

    Los timeouts salen de `shared/llm_transport/`, no de los defaults del
    plugin: son parte del contrato de la comparativa. El default de LiveKit
    (`connect=15 s`) convierte una red caída en 15 s de silencio por intento.
    """
    import httpx
    from livekit.plugins import openai

    return openai.LLM(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        timeout=httpx.Timeout(
            connect=llm_transport.CONNECTION_TIMEOUT_S,
            read=llm_transport.FIRST_BYTE_TIMEOUT_S,
            write=llm_transport.FIRST_BYTE_TIMEOUT_S,
            pool=llm_transport.CONNECTION_TIMEOUT_S,
        ),
    )


def build_tts(settings: Settings) -> Any:
    """ElevenLabs Flash, misma voz y mismo modelo que PoC1."""
    from livekit.plugins import elevenlabs

    return elevenlabs.TTS(
        voice_id=settings.eleven_voice_id,
        model=settings.eleven_model,
        api_key=settings.eleven_api_key,
    )


def build_agent(
    session: NegotiationSession,
    *,
    ruta_prompt: Path | str | None = None,
) -> Any:
    """El `Agent` con el prompt v0 interpolado y las 4 funciones registradas.

    `render_system_prompt` verifica en cada render que el precio máximo no
    se haya filtrado, así que ese guardarraíl corre también aquí y no solo en
    los tests.
    """
    from livekit.agents import Agent

    instructions = render_system_prompt(
        session.trip,
        ruta_prompt=ruta_prompt,
        compensations_path=session.compensations_path,
    )
    return Agent(
        instructions=instructions,
        tools=build_tools(
            session, compensations_path=session.compensations_path
        ),
    )


def build_voice_session(
    settings: Settings,
    *,
    vad: Any = None,
    with_voice: bool = True,
) -> tuple[Any, Any]:
    """La `AgentSession` completa. Devuelve `(session, detector)`.

    El detector se devuelve aparte porque hay que leerle la versión resuelta
    antes y después de la llamada: puede caer sola del modelo remoto al local si
    el servicio de inferencia falla, y eso cambia la arquitectura de lo que se
    midió.

    Con `with_voice=False` se arma sin STT ni TTS, que es el modo texto que usa el
    chequeo de humo: ejercita LLM y tool calling de verdad, sin gastar audio.
    """
    from livekit.agents import AgentSession

    detector = build_turn_detector(settings)
    connections = build_conn_options()

    if not with_voice:
        # Sin audio no hay turn-taking que detectar: LiveKit corre en modo texto.
        return (
            AgentSession(llm=build_llm(settings), conn_options=connections),
            detector,
        )

    verify_for_voice(settings)
    voice_session = AgentSession(
        stt=build_stt(settings),
        llm=build_llm(settings),
        tts=build_tts(settings),
        vad=vad if vad is not None else build_vad(),
        conn_options=connections,
        turn_handling={
            "turn_detection": detector,
            # `endpointing` se deja en los defaults del detector por streaming
            # (min_delay 0.3 s, max_delay 2.5 s). Fijarlos a mano sería calibrar
            # una PoC y no la otra.
        },
    )
    return voice_session, detector


def build_conn_options() -> Any:
    """Reintentos del LLM, con los números de `shared/llm_transport/`.

    Solo se toca el camino del **LLM**. STT y TTS son WebSockets de larga vida,
    con otro modo de falla y otra política de reconexión: se quedan en los
    defaults de LiveKit, igual que en PoC1 se quedan en los de pipecat.

    `retry_interval=0` es deliberado: LiveKit ya hace el primer reintento
    inmediato (0.1 s fijos en `_interval_for_retry`) y los siguientes con este
    valor. Cada segundo de espera aquí es un segundo de silencio con alguien
    escuchando al otro lado del teléfono.
    """
    from livekit.agents import APIConnectOptions
    from livekit.agents.voice.agent_session import SessionConnectOptions

    return SessionConnectOptions(
        llm_conn_options=APIConnectOptions(
            max_retry=llm_transport.retries_after_first(),
            retry_interval=llm_transport.WAIT_BETWEEN_ATTEMPTS_S,
            timeout=llm_transport.CONNECTION_TIMEOUT_S,
        )
    )


# --------------------------------------------------------------------------
# Registro del diálogo
# --------------------------------------------------------------------------


@dataclass
class CallRecord:
    """Transcripción de la llamada, con el segundo relativo de cada intervención."""

    start: float = field(default_factory=time.time)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def annotate(self, role: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        self.turns.append(
            {
                "t": round(time.time() - self.start, 3),
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
        return {str(t.get("role", "")) for t in self.turns}


def connect_observers(
    voice_session: Any, record: CallRecord, collector: LatencyCollector
) -> None:
    """Engancha la transcripción y la captura de latencia a la sesión.

    Todo cuelga de `conversation_item_added`, que es la fuente de verdad de
    LiveKit para las métricas por turno (`metrics_collected` está deprecado).

    Sobre el texto del agente: LiveKit sustituye el contenido por el transcript
    **sincronizado** cuando hubo interrupción y el TTS provee alineación
    (ElevenLabs la provee, `sync_alignment=True` por default). O sea, registra lo
    que se **pronunció**, que es el mismo criterio por el que PoC1 anota después
    del TTS. Queda en la lista de cosas a confirmar en la llamada de diagnóstico.
    """

    @voice_session.on("conversation_item_added")
    def _on_add(event: Any) -> None:  # pragma: no cover - requiere sesión viva
        item = getattr(event, "item", None)
        if item is None:
            return
        role = str(getattr(item, "role", ""))
        text = getattr(item, "text_content", None) or ""
        metrics = getattr(item, "metrics", None) or {}
        interrupted = bool(getattr(item, "interrupted", False))

        if role == "user":
            record.annotate(CARRIER_ROLE, text)
        elif role == "assistant":
            record.annotate(AGENT_ROLE, text)

        collector.record_message(role, metrics, interrupted=interrupted)

    @voice_session.on("error")
    def _on_failure(event: Any) -> None:  # pragma: no cover - requiere sesión viva
        """Errores de proveedor (LLM/STT/TTS), incluidos los reintentos.

        Sin esto, un atorón del LLM solo existe en la consola: el JSON —que es
        la evidencia de la comparativa— no traía ni una señal de que hubiera
        pasado, y el turno afectado entraba a los percentiles como cualquier
        otro. Ver `latencia.ColectorDeLatencia.record_failure`.
        """
        error = getattr(event, "error", None)
        source = getattr(event, "source", None)
        collector.record_failure(
            str(getattr(error, "type", "unknown_error")),
            recuperable=bool(getattr(error, "recoverable", False)),
            provider=getattr(source, "provider", None),
            model_name=getattr(source, "model", None),
            detail=str(getattr(error, "error", None) or error or ""),
        )


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------


def read_identification(metadata: str | None) -> dict[str, str]:
    """Saca `poc`/`escenario`/`repeticion` del `metadata` del dispatch.

    Tolerante a propósito: un metadata mal formado no debe tumbar una llamada
    que ya está sonando. Si no viene nada, la corrida queda sin etiquetar y hay
    que ponerla a mano en el archivo de referencia.
    """
    if not metadata:
        return {}
    try:
        data = json.loads(metadata)
    except (TypeError, ValueError):
        logger.warning("el metadata del dispatch no es JSON válido: %r", metadata)
        return {}
    if not isinstance(data, dict):
        return {}

    label: dict[str, str] = {}
    for clave in (*RUN_PARAMETERS, DIAGNOSTICS_KEY):
        valor = data.get(clave)
        if valor is None:
            continue
        text = str(valor).strip()
        if text:
            label[clave] = text
    return label


def save_run(
    session: NegotiationSession,
    record: CallRecord,
    *,
    collector: LatencyCollector | None = None,
    identification: dict[str, str] | None = None,
    base_path: Path | None = None,
) -> Path:
    """Vuelca resultado, transcripción y latencia a JSON.

    Misma forma que el JSON de PoC1, para que `shared/metrics/` lo lea sin una
    sola rama por PoC. Cae en `grabaciones/`, que está gitignoreada: la
    transcripción puede contener voz y datos de un transportista real (LFPDPPP).
    """
    folder = base_path or RECORDINGS_PATH
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"call_{session.call_id or 'no_id'}.json"

    content: dict[str, Any] = {"call_sid": session.call_id}

    for clave in RUN_PARAMETERS:
        valor = (identification or {}).get(clave)
        if valor is None or str(valor).strip() == "":
            continue
        try:
            content[clave] = int(str(valor).strip())
        except ValueError:
            content[clave] = str(valor).strip()

    content["summary"] = session.summary()
    content["transcript"] = record.turns
    if collector is not None:
        content["latencies"] = collector.to_json()
        content["livekit_extra"] = collector.extra_block()
        # Familia "confiabilidad" del contrato de métricas. Va al nivel de la
        # llamada y no dentro de `latencias` porque un fallo puede caer entre
        # turnos, sin muestra a la que colgarse.
        content["technical_failures"] = collector.failures
        # Si el detector se degradó, la llamada no entra a la poza de
        # percentiles: mezclaría dos arquitecturas de turn-taking. `shared/
        # metrics/` respeta esta llave al consolidar; la llamada sigue
        # apareciendo en el detalle, con su aviso.
        # Una corrida de diagnóstico se excluye ANTES de mirar nada más: no es
        # una llamada de la campaña y sus números no describen el
        # comportamiento telefónico de PoC2.
        mark = (identification or {}).get(DIAGNOSTICS_KEY)
        if mark:
            content["exclude_from_pooled"] = {
                "motivo": "harness_diagnostics",
                "detail": (
                    f"corrida de diagnóstico del arnés ({mark}): no es una "
                    "llamada de la campaña y no entra a percentiles, comparativo "
                    "ni WER"
                ),
            }
            logger.warning("corrida de diagnóstico %r: excluida de la poza", mark)

        degradation = collector.detector_degradation()
        if degradation is not None:
            content["exclude_from_pooled"] = degradation
            logger.warning(
                "llamada excluida de la poza de percentiles: %s", degradation["detail"]
            )

    destination.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def print_summary(
    session: NegotiationSession,
    record: CallRecord,
    collector: LatencyCollector,
    destination: Path,
) -> None:
    """Lo mismo que imprime PoC1 al colgar, más el veredicto del cero."""
    print("\n" + "=" * 70)
    print(f"Llamada terminada: {session.call_id}")
    print("=" * 70)
    print(record.as_text() or "(sin transcripción)")
    print()

    samples = collector.samples
    measured_items = [
        m.full_turn_ms
        for m in samples
        if m.full_turn_ms is not None and not m.is_opening
    ]
    print(f"  turnos con latencia  {len(samples)}")
    if measured_items:
        print(
            f"  turno completo       régimen: "
            f"{min(measured_items):,.0f}-{max(measured_items):,.0f} ms ({len(measured_items)} turnos)"
        )
    opening = next((m for m in samples if m.is_opening), None)
    if opening and opening.full_turn_ms is not None:
        print(f"  apertura (frío)      {opening.full_turn_ms:,.0f} ms")
    print(f"  roles registrados    {sorted(record.roles()) or '(ninguno)'}")

    extra = collector.extra_block()
    detector = extra["turn_detector"]
    print(
        f"  turn detector        {detector['resolved_at_start']} -> "
        f"{detector['resolved_at_end']}"
    )
    verdict = extra["zero_verdict"]
    print(f"  veredicto del cero   {verdict['verdict']}")
    print(f"                       {verdict['detail']}")

    failures = collector.failures
    if not failures:
        print("  fallos de proveedor  ninguno")
    else:
        recovered_items = sum(1 for f in failures if f.get("recuperable"))
        print(
            f"  fallos de proveedor  {len(failures)} "
            f"({recovered_items} recuperados, {len(failures) - recovered_items} no)"
        )
        for failure in failures:
            mark = f"{failure['t']:.1f}s" if failure.get("t") is not None else "?"
            print(f"                       [{mark}] {failure['tipo']}: {failure['detail']}")

    resultado = session.resultado
    print(f"  historial validado   {session.offer_history or '(ninguno)'}")
    print(f"  compensaciones       {session.offered_compensations or '(ninguna)'}")
    print(f"  resultado            {resultado.resultado if resultado else '(sin cierre)'}")
    print(f"\n  corrida guardada en  {destination}")


# --------------------------------------------------------------------------
# Ciclo de la llamada
# --------------------------------------------------------------------------


async def wait_for_answer(
    ctx: Any, *, tope_seg: float = ANSWER_WAIT_S
) -> Any | None:
    """Espera a que el transportista **descuelgue**, no solo a que entre a la sala.

    Son dos instantes distintos, y confundirlos deja la llamada muda. LiveKit
    mete al participante SIP a la sala en cuanto empieza a marcar, con
    `sip.callStatus = "dialing"`, y solo lo pasa a `"active"` cuando contestan.
    Si el agente saluda al ver entrar al participante, suelta el saludo contra el
    tono de llamada y después se calla a esperar turno: el transportista
    descuelga y no oye nada.

    Es también el cero honesto de la latencia: el contrato define la apertura
    desde que la llamada fue contestada (`latencia.ColectorDeLatencia.iniciar`).
    Arrancar el reloj al conectar le sumaría los segundos de timbrado.

    Devuelve el participante, o `None` si nadie contestó dentro de `tope_seg` o
    si colgaron antes de descolgar.
    """
    from livekit import rtc

    participant = await ctx.wait_for_participant(identity=CARRIER_IDENTITY)

    def _estado(p: Any) -> str:
        return str((getattr(p, "attributes", None) or {}).get(ATTRIBUTE_SIP_STATE, ""))

    if getattr(participant, "kind", None) != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # No es una llamada: es un cliente de prueba entrando a la sala. No hay
        # estado de telefonía que esperar.
        logger.info("participante %r no es SIP: se arranca sin esperar", CARRIER_IDENTITY)
        return participant

    if _estado(participant) == SIP_ACTIVE:
        return participant

    logger.info("timbrando (estado %r); se espera a que contesten", _estado(participant))
    answered: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    def _resolver(valor: bool) -> None:
        if not answered.done():
            answered.set_result(valor)

    def _on_attributes_changed(changes: dict[str, str], p: Any) -> None:
        if getattr(p, "identity", "") != CARRIER_IDENTITY:
            return
        if ATTRIBUTE_SIP_STATE in changes:
            logger.info("estado de la llamada: %s", changes[ATTRIBUTE_SIP_STATE])
            if changes[ATTRIBUTE_SIP_STATE] == SIP_ACTIVE:
                _resolver(True)

    def _on_disconnect(p: Any) -> None:
        if getattr(p, "identity", "") == CARRIER_IDENTITY:
            _resolver(False)

    ctx.room.on("participant_attributes_changed", _on_attributes_changed)
    ctx.room.on("participant_disconnected", _on_disconnect)
    try:
        responded = await asyncio.wait_for(answered, tope_seg)
    except asyncio.TimeoutError:
        logger.warning("nadie contestó en %.0f s", tope_seg)
        return None
    finally:
        ctx.room.off("participant_attributes_changed", _on_attributes_changed)
        ctx.room.off("participant_disconnected", _on_disconnect)

    return participant if responded else None


# --------------------------------------------------------------------------
# Entrypoint del worker
# --------------------------------------------------------------------------


async def entrypoint(ctx: Any) -> None:  # pragma: no cover - requiere LiveKit vivo
    """Lo que corre cuando a este agente lo despachan a una sala.

    El worker está registrado con `agent_name`, así que esto **solo** se ejecuta
    tras un `CreateAgentDispatch` explícito con ese nombre: nunca por una sala
    del despachador.
    """
    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_voice(settings)

    identification = read_identification(getattr(ctx.job, "metadata", None))
    room = getattr(ctx.room, "name", "") or getattr(ctx.job, "room", "") or "no_room"

    logger.info("PoC2 despachado a %s (corrida=%s)", room, identification or "(sin etiquetar)")
    if not identification:
        logger.warning(
            "esta llamada no trae scenario/repetition: pásalos al marcar "
            "(--scenario/--repetition/--poc) o habrá que etiquetarla a mano"
        )

    negotiation = NegotiationSession(trip=TEST_TRIP, call_id=room)
    record = CallRecord()
    # La captura se instala una sola vez por proceso del worker: el filtro vive
    # en el logger de LiveKit, que es global. `instalar` devuelve `None` si está
    # apagada por entorno, y entonces el colector simplemente no publica el
    # bloque — nunca cambia lo que se mide.
    # El cero real de fin de voz que publica el arnés por el canal de datos. En
    # una llamada telefónica nadie publica en ese tópico, el registro se queda
    # vacío y el colector ni siquiera emite la llave. Ver `app/marcas.py`.
    marks = MarksRecord()
    collector = LatencyCollector(
        end_of_turn_capture=install_end_of_turn_capture(), marks=marks
    )
    collector.requested_detector = settings.turn_detector_version

    vad = ctx.proc.userdata.get("vad") if hasattr(ctx, "proc") else None
    voice_session, detector = build_voice_session(settings, vad=vad)
    collector.initial_detector = getattr(detector, "model", None)

    connect_observers(voice_session, record, collector)

    # Referencia fuerte a la tarea de cuelgue: una tarea sin dueño puede ser
    # recolectada a media ejecución y dejar la llamada abierta.
    closing_tasks: set[asyncio.Task[None]] = set()

    async def _hang_up_after_speaking() -> None:
        """`drain()` = "cuelga cuando acabes de hablar", el `stop_when_done()` de pipecat.

        LiveKit agenda la respuesta a una función con `force=True` justamente para
        que salga aunque la sesión esté drenando, y le fuerza `tool_choice="none"`
        para que sea texto y no otra llamada a función. O sea: la despedida del
        paso 6 del prompt se pronuncia completa, y hasta entonces no se cuelga.
        """
        await voice_session.drain()
        await voice_session.aclose()
        ctx.delete_room()

    async def _on_close(resultado: NegotiationResult) -> None:
        """Cerrar la negociación cuelga la llamada — pero **no desde aquí**.

        Este callback lo espera el manejador del tool (`tools.py`), que corre
        dentro del turno de habla del agente. Y `drain()` espera a que **todas**
        las tareas de habla terminen por completo, function tools incluidas — lo
        dice el propio SDK en `AgentActivity._pause_scheduling_task`:

            # When pausing/draining, we ensure that all speech_tasks complete
            # fully. This means that even if the SpeechHandle themselves have
            # finished, we still wait for the entire execution (e.g function_tools)

        Awaitarlo aquí es un abrazo mortal: el tool espera a `drain` y `drain`
        espera al tool. El síntoma: tras `registrar_acuerdo` el agente queda
        mudo, LiveKit tira la voz del transportista con "skipping user input,
        speech scheduling is paused" —eso es `_scheduling_paused`, que `drain`
        levanta antes de esperar— y la despedida nunca se pronuncia.

        Se agenda aparte para que el tool pueda devolver.
        """
        logger.info(
            "negociación cerrada como %r; se cuelga al terminar de hablar",
            resultado.resultado,
        )
        tarea = asyncio.create_task(_hang_up_after_speaking())
        closing_tasks.add(tarea)
        tarea.add_done_callback(closing_tasks.discard)

    negotiation.close_callback = _on_close

    async def _on_finish() -> None:
        collector.final_detector = getattr(detector, "model", None)
        # Si la llamada se cortó con el transportista a media intervención, sus
        # fragmentos siguen pendientes: se vuelcan aquí o se pierden.
        collector.close_pending_ones()
        if not negotiation.is_closed:
            # Se cortó sin que el agente cerrara.
            negotiation.close_without_agreement(CloseReason.HUNG_UP)
        destination = save_run(
            negotiation,
            record,
            collector=collector,
            identification=identification,
        )
        print_summary(negotiation, record, collector, destination)

    ctx.add_shutdown_callback(_on_finish)

    await ctx.connect()

    # Que el transportista cuelgue tiene que terminar el job, no dejarlo vivo
    # hasta el `empty_timeout` de la sala: el JSON de la corrida se escribe en el
    # shutdown, y esperar cinco minutos por él invita a matar el worker y
    # perderlo. La `AgentSession` se cierra sola por `close_on_disconnect`, pero
    # NO borra la sala (`delete_room_on_close` es False por default).
    def _on_participant_disconnected(p: Any) -> None:
        if getattr(p, "identity", "") == CARRIER_IDENTITY:
            logger.info("el transportista colgó: se cierra la sala")
            ctx.delete_room()

    ctx.room.on("participant_disconnected", _on_participant_disconnected)

    def _on_data_received(packet: Any) -> None:
        """Marcas del arnés. Cualquier otro tópico se ignora en silencio.

        Es puro diagnóstico: no toca la sesión de voz, no responde nada, y un
        paquete que no se entienda se anota y se descarta. Nada de lo que entre
        por aquí puede cambiar lo que el agente dice o lo que se mide.
        """
        participant = getattr(packet, "participant", None)
        marks.receive(
            getattr(packet, "data", b"") or b"",
            topic=str(getattr(packet, "topic", "") or ""),
            identity=str(getattr(participant, "identity", "") or ""),
        )

    ctx.room.on("data_received", _on_data_received)

    # Nada de audio antes de que descuelguen: el saludo iría contra el tono de
    # llamada y el transportista contestaría a un agente ya callado.
    if await wait_for_answer(ctx) is None:
        logger.warning("la llamada no fue contestada: no hay nada que negociar")
        negotiation.close_without_agreement(CloseReason.NO_RESPONSE)
        ctx.delete_room()
        return

    # El cero de la llamada: contestaron. A partir de aquí se mide la apertura,
    # que por definición del contrato paga el arranque en frío — incluido lo que
    # cueste levantar la sesión y abrir los WebSockets de TTS y STT.
    collector.begin()

    agent = build_agent(negotiation)
    await voice_session.start(agent, room=ctx.room, record=False)

    # El agente habla primero: es una llamada saliente.
    voice_session.generate_reply()


def prewarm(proc: Any) -> None:  # pragma: no cover - lo llama el worker
    """Carga Silero antes de que entre la llamada, no durante el saludo."""
    proc.userdata["vad"] = build_vad()


def main(argv: list[str] | None = None) -> int:
    """Levanta el worker. **No marca**: para marcar, usa `app.llamada`."""
    force_utf8_output()

    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.agent",
        description=(
            "Worker del agente de PoC2. Se registra con su propio agent_name y "
            "espera dispatch explícito; no marca ni contesta nada por su cuenta."
        ),
        add_help=False,
    )
    arg_parser.add_argument("--ayuda", action="help", help="esta ayuda")
    _, remaining_items = arg_parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_voice(settings)

    # Antes de que exista cualquier hilo de job: ver `precargar_plugins`.
    preload_plugins()

    print("Worker de PoC2 — LiveKit Agents")
    print(f"  proyecto        {settings.livekit_url}")
    print(f"  agent_name      {settings.agent_name}")
    print(f"  turn detector   {settings.turn_detector_version}")
    print(f"  LLM / STT / TTS {settings.openai_model} / {settings.deepgram_model} / {settings.eleven_model}")
    print("\nEsperando dispatch explícito. Para marcar, en otra terminal:")
    print("  uv run --project poc2_livekit python -m poc2_livekit.app.llamada --accept-cost\n")

    import sys

    from livekit.agents import WorkerOptions, cli

    # `cli.run_app` lee sus propios argumentos de sys.argv.
    sys.argv = [sys.argv[0], *(remaining_items or ["start"])]
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Esto es el aislamiento: con agent_name, el worker NO recibe
            # dispatch automático de ninguna sala.
            agent_name=settings.agent_name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
