"""Arnés de PoC1: actúa el guion contra el pipeline completo, sin teléfono.

Es el análogo del arnés de PoC2 (`poc2_livekit/app/harness.py`), y comparte con él
todo lo que decide qué es un turno medible: el reproductor, la regla de avance,
el cruce del cero, la bitácora y el reporte viven en
`shared/carrier_simulator/`. Lo que PoC1 pone es su transporte —el WebSocket de
Media Streams— y su ruta de la marca, que es un archivo en disco.

## Qué corre y qué no

Corre el pipeline **de producción**: `telefonia.create_app`, con STT, LLM, TTS,
tool calling y el motor de negociación. No una copia suya — ver
`telefonia.GanchosDelArnes` para por qué eso importa y cómo se le insertan las
dos cosas que solo el arnés necesita.

No corre Twilio. No se marca a nadie, no suena ningún teléfono y no se toca
ningún número: el `callSid` lo inventa `ClienteDeMediaStreams` con un prefijo
que ningún SID real usa, y el serializer va con `auto_hang_up=False`.

## Dos procesos, y no es un detalle

El servidor va en un subproceso. Compartir bucle de eventos con la inferencia de
Silero y del analizador de turnos le mete al arnés pausas de cientos de
milisegundos, y esas pausas entran enteras en el sesgo que se quiere medir. Con
procesos separados el ritmo vuelve a un desvío de ~1 ms.

## El cruce del cero

El pipeline escribe `recordings/call_{call_sid}.json` como en cualquier
llamada. El arnés escribe `diagnostics/harness/zeros_{call_sid}.jsonl` con los dos
ceros reales de cada turno, y el servidor deja `cero_{call_sid}.json` con el
cruce ya hecho. Las tres llaves son el mismo `call_sid`, así que juntarlas es un
`join` por nombre de archivo y no una heurística.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from poc1_pipecat.app.config import (
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_voice,
)
from poc1_pipecat.app.marks import (
    DIAGNOSTICS_PATH,
    DiskMarkEmitter,
    AudioOutputProbe,
    ZeroProbe,
    lines_block,
    harness_block,
    marks_de,
    latency_samples,
    zeros_path,
)
from poc1_pipecat.app.media_streams import (
    MediaStreamsClient,
    MediaStreamsPlayer,
)
from shared.audio import TELEPHONY_SAMPLE_RATE, ulaw_a_pcm16
from shared.carrier_simulator import (
    EstimatedZero,
    MAX_AGENT_WAIT_S,
    WAIT_AFTER_CLIP_S,
    MIN_VALID_TURNS,
    AUDIT_PREFIX,
    RUTA_CLIPS_DEFAULT,
    CostContext,
    ReportContext,
    HarnessFailure,
    TextFragment,
    AuditRecorder,
    Observed,
    ScriptStep,
    Tariffs,
    UnauditableSampleRate,
    event_log_de_la_corrida,
    load_clip,
    load_scenario,
    negotiation_state,
    write_summary,
    measure_return_path,
    measure_overlap,
    evaluate_run,
    print_cost_warning,
    print_report as print_simulator_report,
    audio_frame_line,
    published_frame_line,
    play_script,
    quote_summary,
    call_summary,
    output_sample_rate,
)

logger = logging.getLogger(__name__)

# El agente contesta por el mismo cable y con la misma tasa: Media Streams es
# μ-law de 8 kHz en las dos direcciones. En PoC2 no es así —se publica a 8 k y se
# escucha a 48 k— y por eso allá la auditoría remuestrea y aquí no.
LISTEN_SAMPLE_RATE: Final[int] = TELEPHONY_SAMPLE_RATE

# El evento con el que el servidor le devuelve al arnés lo que se transcribió.
# No es de Twilio: es un `OutputTransportMessageUrgentFrame` que el serializer
# reenvía como JSON crudo. Ver `EcoDeTexto`.
TEXT_EVENT: Final[str] = "harness.text"

FROM_CARRIER: Final[str] = "carrier"
FROM_AGENT: Final[str] = "agent"

# Marca del fragmento cuyo texto YA trae los espacios que lo separan del
# anterior. Viaja en `FragmentoDeTexto.segmento`, que es el campo libre que cada
# PoC llena con lo que sepa leer de su protocolo. Ver `ensamblar_agente`.
WITH_SPACES: Final[str] = "has_spaces"

# Tarifas de `docs/costos_poc1.md`. Sin renglón de telefonía: aquí no hay
# ninguna. El TTS y el LLM se cobran completos, que es cota alta y no estimación
# fina.
STT_COST_PER_MINUTE: Final[float] = 0.0077
TTS_COST_PER_CALL: Final[float] = 0.03
LLM_COST_PER_CALL: Final[float] = 0.02

HARNESS_TARIFFS: Final[Tariffs] = Tariffs(
    per_minute=STT_COST_PER_MINUTE,
    per_call=TTS_COST_PER_CALL + LLM_COST_PER_CALL,
)


def estimate_cost(seconds: float) -> float:
    """Costo aproximado de una corrida. Sin telefonía: aquí no hay trunk."""
    return HARNESS_TARIFFS.estimate(seconds)


# --------------------------------------------------------------------------
# Lo que el arnés le inserta al pipeline
# --------------------------------------------------------------------------


def _build_echo(what_to_forward: str) -> Any:
    """Un procesador que le devuelve al arnés el texto que pasa por él.

    El arnés de PoC2 lee el texto de la sala; aquí no hay sala, solo un
    WebSocket de audio. Sin este eco la bitácora saldría sin `stt` ni `agente` y
    los descartes por avance no podrían decir **qué** se descartó — que es justo
    lo que hizo legible el descarte de PoC2 ("Gracias,").

    Va por `OutputTransportMessageUrgentFrame`, que el `TwilioFrameSerializer`
    reenvía como JSON crudo por el mismo WebSocket. No toca el audio y no entra
    en ninguna latencia: es texto que ya existía, reenviado.

    **Son dos procesadores y no uno** porque los dos textos no coexisten en
    ningún punto del pipeline: el del transportista se lo queda el agregador de
    usuario, y el del agente no existe hasta después del TTS. Ver
    `telefonia.GanchosDelArnes`.
    """
    from pipecat.frames.frames import (
        Frame,
        OutputTransportMessageUrgentFrame,
        TranscriptionFrame,
        TTSTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class TextEcho(FrameProcessor):
        """Reenvía al arnés el texto de una dirección, sin tocar nada más."""

        async def process_frame(
            self, frame: Frame, direction: FrameDirection
        ) -> None:
            await super().process_frame(frame, direction)

            message: dict[str, Any] | None = None
            # `TranscriptionFrame` y no `InterimTranscriptionFrame`: los
            # interinos se reemplazan entre sí y concatenarlos daría la frase
            # repetida en pedazos. Los finales de Deepgram son segmentos que no
            # se solapan.
            if what_to_forward == FROM_CARRIER and isinstance(
                frame, TranscriptionFrame
            ):
                message = {
                    "event": TEXT_EVENT,
                    "de": FROM_CARRIER,
                    "text": frame.text,
                    "final": True,
                }
            elif what_to_forward == FROM_AGENT and isinstance(frame, TTSTextFrame):
                message = {
                    "event": TEXT_EVENT,
                    "de": FROM_AGENT,
                    "text": frame.text,
                    # ElevenLabs manda un frame por palabra y **sin** los
                    # espacios que las separan. Sin esta bandera el arnés
                    # reensambla "Carlos,quierohablarsobreunenvío".
                    "has_spaces": bool(
                        getattr(frame, "includes_inter_frame_spaces", True)
                    ),
                    "final": False,
                }

            await self.push_frame(frame, direction)
            if message is not None:
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message=message),
                    FrameDirection.DOWNSTREAM,
                )

    return TextEcho()


def build_carrier_echo() -> Any:
    """El eco del STT. Va **antes** del agregador, que se queda el frame."""
    return _build_echo(FROM_CARRIER)


def build_agent_echo() -> Any:
    """El eco del TTS. Va después de él, que es donde el texto existe."""
    return _build_echo(FROM_AGENT)


def harness_hooks(
    probes: list[ZeroProbe],
    *,
    outputs: list[AudioOutputProbe] | None = None,
    on_save: Any = None,
) -> Any:
    """Los tres puntos que el arnés le pide al pipeline de producción.

    Está aquí y no dentro de `_servir` para poder preguntarle, sin levantar un
    servidor, **qué** monta en cada punto: que el eco del transportista no vaya
    donde no lo ve, y que el del agente no duplique lo que ya mandó el otro.

    `sondas` se llena a medida que el pipeline pide una por llamada; quien la
    pasó se queda con la lista para leerle los ceros al colgar.
    """
    from poc1_pipecat.app.telephony import HarnessHooks

    def _after_stt() -> Sequence[Any]:
        return (build_carrier_echo(),)

    def _after_turn() -> Sequence[Any]:
        probe = ZeroProbe()
        probes.append(probe)
        return (probe,)

    def _before_output() -> Sequence[Any]:
        # La sonda va **después** del eco de texto y antes del transporte: ahí
        # el audio ya es el que va a salir por el cable, y su instante es la
        # punta izquierda del camino de vuelta.
        probe = AudioOutputProbe()
        if outputs is not None:
            outputs.append(probe)
        return (build_agent_echo(), probe)

    return HarnessHooks(
        # No hay ninguna llamada de Twilio que colgar, y el serializer se niega
        # a construirse con `auto_hang_up` y sin credenciales.
        hang_up_via_rest=False,
        after_stt=_after_stt,
        after_turn=_after_turn,
        before_output=_before_output,
        on_save=on_save,
    )


# --------------------------------------------------------------------------
# El servidor: el pipeline de producción, en su propio proceso
# --------------------------------------------------------------------------


async def _serve(port: int, *, folder: Path, stop_secs: float) -> None:
    """Levanta `telefonia.create_app` con los ganchos del arnés y espera una llamada.

    Cuando el arnés cuelga, el pipeline guarda su JSON como en cualquier llamada
    y el gancho escribe, junto a él, el cruce del cero. Después el servidor se
    apaga solo: es de una sola llamada por diseño, y dejarlo vivo obligaría al
    arnés a matarlo — que es como se pierde el volcado.
    """
    import uvicorn

    from poc1_pipecat.app.telephony import HarnessHooks, create_app

    probes: list[ZeroProbe] = []
    outputs: list[AudioOutputProbe] = []
    servidor: Any = None

    def _on_save(call_sid: str, destination: Path) -> None:
        nonlocal servidor
        for probe in probes:
            # El último turno no lo cierra nadie: el arnés deja de hablar y el
            # analizador no vuelve a disparar. Perderlo sería perder una muestra
            # que sí se midió.
            probe.close_pending_turn()
        zeros = [c for probe in probes for c in probe.zeros]
        marks_path = zeros_path(call_sid, folder=folder)

        block = harness_block(
            marks_path,
            zeros,
            stop_secs=stop_secs,
            pipeline_host=socket.gethostname(),
            audio_outputs=sorted(t for s in outputs for t in s.outputs),
            audible_outputs=sorted(
                t for s in outputs for t in s.audible_outputs
            ),
        )
        block["how_it_was_run"] = {
            "pipeline": (
                "el de producción (`telefonia.create_app`) entero: transporte, "
                "serializer, VAD, analizador de fin de turno, STT, LLM, TTS y "
                "tool calling. No es una copia: ver `telefonia.GanchosDelArnes`"
            ),
            "telephony": "ninguna — WebSocket local, sin Twilio y sin marcar",
            "pipeline_json": str(destination),
            "vad_speech_ends": sum(
                len(probe.speech_ends) for probe in probes
            ),
            "turn_boundary": (
                "la declara el analizador de pipecat, no el arnés: con STT sí hay "
                "transcripción, así que `TurnAnalyzerUserTurnStopStrategy` cierra "
                "los turnos. Es la diferencia con `zero_diagnostics`, donde la "
                "ventana la ponía el arnés"
            ),
        }
        output = folder / f"cero_{call_sid}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  cruce del cero en    {output}")

        if servidor is not None:
            servidor.should_exit = True

    settings = load_settings()
    app = create_app(
        settings,
        hooks=harness_hooks(probes, outputs=outputs, on_save=_on_save),
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    servidor = uvicorn.Server(config)
    await servidor.serve()


# --------------------------------------------------------------------------
# Lo observado
# --------------------------------------------------------------------------


@dataclass
class ObservedFromPoC1(Observed):
    """Lo que el arnés vio por el WebSocket de Media Streams.

    Dos cosas que solo pasan aquí:

    - **El audio llega en μ-law** y hay que decodificarlo antes de medirle
      energía. Es el único punto del arnés que lo hace.
    - **El texto llega por el mismo cable**, como eventos `arnes.texto` que el
      servidor reenvía. Ver `EcoDeTexto`.
    """

    own_identity: str = FROM_CARRIER

    # Cómo salió el audio de este lado. Diagnóstico del arnés: no toca latencias.
    publishing_pacing: dict[str, Any] = field(default_factory=dict)
    measured_sends: list[float] = field(default_factory=list)
    # Tramas que se quisieron publicar después de que el agente colgó.
    frames_without_destination: int = 0

    def annotate_audio_in_ulaw(self, ulaw: bytes) -> None:
        """Decodifica y mide. El arnés escucha en μ-law; `Observado` mide PCM16."""
        self.annotate_audio(ulaw_a_pcm16(ulaw).tobytes())

    def assemble_stt(self, fragments: list[TextFragment]) -> str:
        """Los finales de Deepgram son segmentos, no deltas: se unen con espacio.

        Concatenarlos sin separador pegaría la última palabra de una frase con
        la primera de la siguiente.

        **La regla es de aquí y no se copia a PoC3.** Es correcta aquí porque
        Deepgram entrega segmentos finalizados; los fragmentos de PoC3 son
        deltas, no segmentos, y con esta regla el espacio repuesto parte palabras
        (`"die ciocho mil"`) y despega la puntuación. Ver
        `docs/hallazgo_ensamblado_stt_poc3.md`.
        """
        chunks = [f.text.strip() for f in fragments if f.text.strip()]
        return " ".join(chunks)

    def assemble_agent(self, fragments: list[TextFragment]) -> str:
        """Con la misma regla que usa el pipeline, y no con `"".join`.

        ElevenLabs emite un `TTSTextFrame` por palabra **sin** los espacios que
        las separan, así que concatenar da "Carlos,quierohablar…". El pipeline ya
        resuelve esto en `join_tts_fragments`; se reusa esa función para que el
        texto del arnés y el del JSON de la llamada no puedan discrepar.
        """
        from poc1_pipecat.app.pipeline import join_tts_fragments

        return join_tts_fragments(
            [(f.text, f.segment == WITH_SPACES) for f in fragments]
        )


def annotate_hang_up(
    observed: ObservedFromPoC1, closed: asyncio.Event
) -> Any:
    """El aviso de "el agente colgó", listo para pasarle al cliente.

    Está fuera de `run_session` para poder probarlo: es la señal de la que
    depende que el guion se corte, y sin ella se pierde la corrida entera.
    """

    def _hung_up(motivo: str) -> None:
        if closed.is_set():
            return
        closed.set()
        observed.was_closed_by_agent = True
        observed.close_reason = motivo
        observed.tally(f"el agente colgó: {motivo}")

    return _hung_up


def close_observed(
    observed: ObservedFromPoC1, *, player: Any, client: Any
) -> None:
    """Vuelca a `observado` lo que solo sabe el transporte.

    Aparte por lo mismo que `annotate_hang_up`: corre en el `finally` de una
    corrida ya pagada, que es donde nadie mira hasta que falta un número.
    """
    observed.publishing_pacing = player.pacing.summary()
    observed.measured_sends = list(player.pacing.reales)
    observed.frames_without_destination = client.frames_without_destination


def first_speech_per_turn(observed: ObservedFromPoC1) -> dict[str, float]:
    """Cuándo se oyó al agente después de cada intervención, en reloj de pared.

    Lo mide el arnés sobre la pista que le llega, así que es el mismo número en
    las 3 PoC y no depende de que el worker lo publique. Es la punta derecha de
    la latencia por intervención; la izquierda es el fin de voz de la marca.
    """
    timings: dict[str, float] = {}
    for turn in observed.turns:
        if turn.t_agent_first_speech is None:
            continue
        timings[turn.step.turn] = observed.start + turn.t_agent_first_speech
    return timings


def add_lines(
    crossing_path: Path,
    *,
    path_of_marks: Path,
    observed: ObservedFromPoC1,
    pipeline_json: Path | None,
) -> dict[str, Any] | None:
    """Le agrega al cruce la cuenta por intervención. Corre tras colgar.

    Va en el proceso del arnés y no en el del servidor porque hace falta lo que
    **solo el arnés sabe**: cuándo se oyó al agente. El servidor ya dejó los
    ceros crudos en el archivo; aquí se agrupan contra las marcas.
    """
    if not crossing_path.is_file():
        return None
    try:
        cruce = json.loads(crossing_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # noqa: BLE001 - diagnóstico
        print(f"  [!] no se pudo leer el cruce: {error!r}")
        return None

    zeros = [
        EstimatedZero(indice=int(c["indice"]), start=c.get("start"), end=c.get("end"))
        for c in cruce.get("zeros_of_pipeline", [])
    ]
    block = lines_block(
        marks_de(path_of_marks),
        zeros,
        first_speech=first_speech_per_turn(observed),
        pipeline_samples=(
            latency_samples(pipeline_json) if pipeline_json else None
        ),
    )
    cruce["lines"] = block
    crossing_path.write_text(
        json.dumps(cruce, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return block


def _on_text_received(observed: ObservedFromPoC1, message: dict[str, Any]) -> None:
    """Traduce un evento `arnes.texto` a un fragmento de `Observado`."""
    text = str(message.get("text", ""))
    if not text:
        return
    observed.annotate_text(
        str(message.get("de", FROM_AGENT)),
        text,
        # El campo libre lleva si el fragmento ya trae su separador. Sin esto,
        # reensamblar la voz del agente pega las palabras.
        segment=WITH_SPACES if message.get("has_spaces") else "",
        final=bool(message.get("final", False)),
    )


# --------------------------------------------------------------------------
# La corrida
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait_for_server(
    port: int, process: Any, tope_seg: float = 180.0
) -> None:
    """Espera a que el `/ws` acepte conexiones. Arrancar carga dos modelos ONNX."""
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if process.returncode is not None:
            raise HarnessFailure(
                f"el servidor del arnés se murió al arrancar "
                f"(código {process.returncode})"
            )
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.25)
    raise HarnessFailure(f"el servidor no abrió el puerto {port} en {tope_seg:.0f} s")


async def run_session(
    *,
    script: list[ScriptStep],
    folder: Path,
    label: str,
    wait_after_clip: float,
    recorder: Any = None,
    stop_secs: float,
) -> tuple[ObservedFromPoC1, str]:
    """Levanta el servidor, actúa el guion y cuelga. Devuelve lo observado."""
    from poc1_pipecat.app.telephony import RUTA_WS

    folder.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    root = Path(__file__).resolve().parents[2]

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "poc1_pipecat.app.harness",
        "--run-server",
        "--port",
        str(port),
        "--folder",
        str(folder),
        "--stop-secs",
        str(stop_secs),
        cwd=str(root),
    )

    observed = ObservedFromPoC1(recorder=recorder)
    client: MediaStreamsClient | None = None
    # El agente cerró y colgó. Es una de las dos condiciones de fin, no un
    # error: en el escenario 1 el agente cierra en T7 y quedan turnos de guion
    # por actuar. Sin esta señal el arnés se los sigue reproduciendo a un
    # pipeline que ya no está — y en PoC1 además se cae, porque el WebSocket ya
    # no acepta tramas. Es la misma señal que usa PoC2.
    closed = asyncio.Event()
    _agent_hung_up = annotate_hang_up(observed, closed)

    try:
        print("  levantando el pipeline (carga Silero y Smart Turn)...")
        await _wait_for_server(port, process)

        client = MediaStreamsClient(
            f"ws://127.0.0.1:{port}/{RUTA_WS}",
            label=label,
            # Sin `escenario`/`repeticion` a propósito: etiquetar esta corrida
            # la volvería indistinguible de una de las 15 de la campaña.
            identification={"poc": "1"},
            on_receive_audio=observed.annotate_audio_in_ulaw,
            on_receive_message=lambda m: _on_text_received(observed, m),
            on_other_side_closed=_agent_hung_up,
        )

        player = MediaStreamsPlayer(
            client, observed=observed, recorder=recorder
        )
        emitter = DiskMarkEmitter(
            zeros_path(client.call_sid, folder=folder),
            publish_sample_rate=player.sample_rate,
        )

        halt = asyncio.Event()
        async with client:
            pumping = asyncio.create_task(player.launch(halt))
            try:
                # El agente habla primero: su saludo arranca cuando el
                # transporte ve al cliente conectado. Se le deja terminar antes
                # de soltar el primer clip.
                from shared.carrier_simulator import (
                    AGENT_SILENCE_S,
                    wait_until_agent_is_silent,
                )

                motivo = await wait_until_agent_is_silent(
                    observed,
                    silence_s=AGENT_SILENCE_S,
                    tope_seg=MAX_AGENT_WAIT_S,
                )
                print(f"  [{observed.now():>7.3f}s] {motivo}\n")

                await play_script(
                    script,
                    player=player,
                    observed=observed,
                    wait_for_agent_s=wait_after_clip,
                    emitter=emitter,
                    closed=closed,
                    estado=negotiation_state(script),
                )
            except KeyboardInterrupt:  # pragma: no cover - depende de la terminal
                print("\n  cortado a mano.")
            finally:
                halt.set()
                # Ancho a propósito: lo que venga de aquí es de una corrida que
                # **ya se pagó**, y dejar que una falla del bombeo se lleve el
                # reporte, la bitácora y el WAV sería tirar la evidencia entera.
                # Se anota y se sigue.
                try:
                    await pumping
                except asyncio.CancelledError:
                    pass
                except Exception as error:  # noqa: BLE001 - reporta y sigue
                    observed.annotate_incident(
                        f"el bucle de publicación terminó mal: {error!r}"
                    )

                # El resumen va a disco, no al agente, así que se escribe aunque
                # ya haya colgado: es la segunda oportunidad de las marcas que
                # no hayan quedado.
                try:
                    observed.marks_summary_sent = await emitter.shut(
                        turns_without_audio=[
                            t.step.turn for t in observed.turns_without_audio()
                        ]
                    )
                except Exception as error:  # noqa: BLE001 - reporta y sigue
                    observed.annotate_incident(
                        f"el resumen de ceros no se pudo cerrar: {error!r}"
                    )
                observed.sent_marks = emitter.sent
                observed.failed_marks = list(emitter.faults)
                close_observed(
                    observed, player=player, client=client
                )

        # Colgar el WebSocket es lo que le dispara al pipeline su volcado: el
        # JSON de la llamada y, con él, el cruce del cero.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=60)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)

    return observed, (client.call_sid if client is not None else "")


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


def print_report(observed: ObservedFromPoC1, call_sid: str) -> int:
    """El reporte compartido, con lo que solo se puede decir de Media Streams."""
    header = [
        f"  publicado a           {TELEPHONY_SAMPLE_RATE} Hz, μ-law por Media Streams",
        published_frame_line(observed),
        audio_frame_line(observed),
    ]
    if observed.sent_marks:
        summary = "sí" if observed.marks_summary_sent else "NO"
        header.append(
            f"  ceros escritos        {observed.sent_marks} "
            f"(resumen al cerrar: {summary}"
            + (
                f", {len(observed.failed_marks)} fallaron"
                if observed.failed_marks
                else ""
            )
            + ")"
        )
    if observed.was_closed_by_agent:
        header.append(
            f"  el agente colgó       sí — {observed.close_reason}"
            + (
                f" · {observed.frames_without_destination} trama(s) sin destino"
                if observed.frames_without_destination
                else ""
            )
        )

    return print_simulator_report(
        observed,
        ReportContext(
            title=f"Arnés de PoC1 — {call_sid}",
            header=header,
            stt_name="Deepgram",
            where_the_mutes_are_marked="el cruce del cero",
            footer=(
                f"\n  El pipeline deja su JSON en poc1_pipecat/grabaciones/llamada_{call_sid}.json"
                "\n  Es una corrida de arnés, NO de campaña: no la dejes en esa carpeta si"
                "\n  vas a correr `shared.metrics.cli calcular` sobre ella.",
            ),
        ),
    )


def print_lines(block: dict[str, Any]) -> None:
    """La tabla por intervención. Es la unidad de la comparativa (§2.2.5)."""
    print("\n  por intervención (la unidad de análisis, no el turno del VAD)\n")
    print(
        f"    {'turn':<6}{'clip':<8}{'events':>9}{'latency':>11}"
        f"{'sesgo del cero':>16}"
    )
    print("    " + "-" * 50)
    for row in block.get("per_line", []):
        lat = row["line_latency_ms"]
        bias = row["speech_bias_ms"]
        print(
            f"    {str(row['turn']):<6}{str(row['clip_id']):<8}"
            f"{row['pipeline_events']:>9}"
            f"{('—' if lat is None else f'{lat:,.0f} ms'):>11}"
            f"{('—' if bias is None else f'{bias:,.1f} ms'):>16}"
        )

    frag = block.get("fragmentation")
    print(
        f"\n    intervenciones {block['lines']} · eventos del pipeline "
        f"{block['pipeline_events']} · fragmentación "
        f"{'—' if frag is None else f'{frag:.2f}'}"
    )
    if block.get("sample_fragmentation") is not None:
        print(
            f"    muestras de latencia {block['pipeline_latency_samples']}"
            f" · fragmentación de muestras {block['sample_fragmentation']:.2f}"
        )
    lat = block.get("line_latency_ms")
    if lat:
        print(
            f"    latencia por intervención  n {lat['n']} · min {lat['min']:,.0f}"
            f" · p50 {lat['p50']:,.0f} · max {lat['max']:,.0f} ms"
        )
    if block.get("events_without_line"):
        print(
            f"    [!] {block['events_without_line']} evento(s) del pipeline "
            "fuera de toda intervención"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_script(args: argparse.Namespace) -> list[ScriptStep]:
    """Traduce las banderas a la lista de turnos que se va a actuar."""
    folder = Path(args.clips)
    if args.scenario is not None:
        script = load_scenario(
            args.scenario, folder, publish_sample_rate=TELEPHONY_SAMPLE_RATE
        )
    else:
        step = load_clip(Path(args.clip), publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        script = [step] * max(1, int(args.repeat))

    if args.turns:
        # Para el humo: media docena de turnos menos es media docena de turnos
        # que no se pagan. El recorte se anota en la bitácora por el guion que
        # queda, así que una corrida corta no se puede confundir con una entera.
        script = script[: max(1, int(args.turns))]
    return script


def _print_cost_warning(
    script: list[ScriptStep], seconds: float, *, call_sid: str
) -> None:
    print_cost_warning(
        CostContext(
            title="Corrida del arnés de PoC1, SIN telefonía.",
            tariffs=HARNESS_TARIFFS,
            is_spent=(
                ("STT", "Deepgram, sobre los clips que publica el arnés"),
                ("TTS", "ElevenLabs, el saludo del agente y sus respuestas"),
                ("LLM", "gpt-4.1-mini, la apertura del prompt v0 y cada turno"),
            ),
            is_not_spent=(
                ("Twilio", "no se marca: el WebSocket es local y el callSid inventado"),
                ("ngrok", "no hace falta túnel: el arnés y el servidor son locales"),
            ),
            fields=(
                ("callSid", call_sid),
                ("transport", "ws://127.0.0.1 — Media Streams local"),
            ),
        ),
        script,
        seconds,
    )


def _dump_audit(recorder: Any, call_sid: str, *, folder: Path) -> Path | None:
    """Escribe el WAV de auditoría y su JSON. Devuelve la ruta del WAV."""
    ruta_wav = folder / f"{AUDIT_PREFIX}_{call_sid}.wav"
    try:
        descriptive = recorder.dump(ruta_wav, room=call_sid)
    except Exception as error:  # noqa: BLE001 - diagnóstico: reporta y sigue
        print(f"\n  [!] no se pudo escribir la grabación de auditoría: {error!r}")
        return None

    ruta_json = ruta_wav.with_suffix(".json")
    ruta_json.write_text(
        json.dumps(descriptive, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  Auditoría (NO es evidencia de campaña) en {ruta_wav}\n"
        f"    {descriptive['duration_s']:.1f} s a {descriptive['sample_rate_hz']} Hz · "
        f"canal 0 el arnés, canal 1 el agente · qué es, en {ruta_json.name}"
    )
    return ruta_wav


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.harness",
        description=(
            "Arnés de PoC1: actúa el guion contra el pipeline completo por un "
            "WebSocket local. No usa Twilio y no marca a nadie."
        ),
    )
    arg_parser.add_argument(
        "--run-server",
        action="store_true",
        help="modo interno: levanta el pipeline y espera. Lo usa el propio arnés",
    )
    arg_parser.add_argument("--port", type=int, help="con --run-server, el puerto")
    arg_parser.add_argument(
        "--folder",
        default=str(DIAGNOSTICS_PATH),
        help="dónde van las marcas, el cruce, la bitácora y la auditoría",
    )
    arg_parser.add_argument(
        "--stop-secs",
        type=float,
        default=None,
        help="internal: el `stop_secs` del VAD, para el veredicto del cero",
    )
    arg_parser.add_argument(
        "--scenario", type=int, help="número de escenario del guion (1-5)"
    )
    arg_parser.add_argument(
        "--clip", help="un solo artefacto de shared/audio/ en vez de un escenario"
    )
    arg_parser.add_argument(
        "--clips",
        default=str(RUTA_CLIPS_DEFAULT),
        help="carpeta de los clips preparados",
    )
    arg_parser.add_argument("--repeat", type=int, default=1, metavar="N")
    arg_parser.add_argument(
        "--turns",
        type=int,
        default=0,
        metavar="N",
        help="corta el guion en los primeros N turnos. Para el humo: cuesta menos",
    )
    arg_parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_AFTER_CLIP_S,
        help=f"segundos de observación tras el clip (default {WAIT_AFTER_CLIP_S:.0f})",
    )
    arg_parser.add_argument(
        "--record-audit",
        action="store_true",
        help=(
            "escribe un WAV de dos canales de la corrida (canal 0 el arnés, "
            "canal 1 el agente). Es para OÍR la corrida: no entra a latencias, "
            "ni al WER, ni a ningún percentil"
        ),
    )
    arg_parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        metavar="N",
        help="qué repetición del escenario es. Solo etiqueta el resumen",
    )
    arg_parser.add_argument(
        "--summary",
        default=None,
        metavar="RUTA",
        help=(
            "escribe un JSON chico con cómo terminó la llamada, las "
            "intervenciones, la latencia y el WER. Sin --accept-cost escribe "
            "en su lugar el estimado y no gasta. Lo usa el modo lote"
        ),
    )
    arg_parser.add_argument(
        "--accept-cost", action="store_true", help="obligatorio para ejecutar"
    )
    arg_parser.add_argument(
        "--interruptions",
        action="store_true",
        help=(
            "escenario de interrupciones: en algunos turnos el arnés le habla "
            "encima al agente (interrupción, apoyo verbal o control). Necesita "
            "--scenario y --repetition y los clips de "
            "`python -m shared.audio.interrupciones preparar`"
        ),
    )
    arg_parser.add_argument(
        "--interruption-clips",
        default=None,
        help="carpeta de los clips de interrupción preparados",
    )
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    folder = Path(args.folder)

    if args.run_server:
        if args.port is None:
            print("[FALLA] --run-server necesita --port")
            return 1
        # El `stop_secs` lo manda el padre para no cargar Silero dos veces solo
        # para leer un número.
        stop_secs = args.stop_secs
        if stop_secs is None:
            from poc1_pipecat.app.pipeline import build_vad

            stop_secs = build_vad().params.stop_secs
        asyncio.run(_serve(args.port, folder=folder, stop_secs=float(stop_secs)))
        return 0

    if (args.scenario is None) == (args.clip is None):
        print("[FALLA] hay que dar --scenario o --clip, y solo uno")
        return 1

    settings = load_settings()
    verify_for_voice(settings)
    # `verificar_para_telefonia` NO se llama: este camino no necesita Twilio.

    try:
        script = build_script(args)
        if args.interruptions:
            from shared.carrier_simulator.interruptions import (
                INTERRUPTION_CLIPS_PATH,
                activate,
            )

            activate(
                script,
                scenario=args.scenario,
                repetition=args.repetition,
                folder=args.interruption_clips or INTERRUPTION_CLIPS_PATH,
            )
        if args.record_audit:
            # Antes del aviso de costo y de gastar nada: enterarse de que el WAV
            # no se puede armar después de la llamada sería el peor momento.
            output_sample_rate(TELEPHONY_SAMPLE_RATE, LISTEN_SAMPLE_RATE)
    except (HarnessFailure, UnauditableSampleRate) as error:
        print(f"[FALLA] {error}")
        return 1

    label = time.strftime("%Y%m%d-%H%M%S")
    call_sid = f"arnes-call-{label}"
    total = MAX_AGENT_WAIT_S + sum(
        step.max_duration_s() + args.wait for step in script
    )

    if not args.accept_cost:
        print()
        _print_cost_warning(script, total, call_sid=call_sid)
        if args.summary:
            write_summary(
                Path(args.summary),
                quote_summary(
                    poc=1,
                    scenario=args.scenario,
                    script=script,
                    seconds=total,
                    cost_usd=estimate_cost(total),
                ),
            )
        return 0

    from poc1_pipecat.app.pipeline import build_vad

    stop_secs = build_vad().params.stop_secs

    recorder = (
        AuditRecorder(harness_sample_rate=TELEPHONY_SAMPLE_RATE, agent_sample_rate=LISTEN_SAMPLE_RATE)
        if args.record_audit
        else None
    )

    onset = time.monotonic()
    observed, call_sid = asyncio.run(
        run_session(
            script=script,
            folder=folder,
            label=label,
            wait_after_clip=args.wait,
            recorder=recorder,
            stop_secs=stop_secs,
        )
    )
    duration = time.monotonic() - onset
    faults = print_report(observed, call_sid)

    # La cuenta por intervención: aquí y no en el servidor, porque hace falta
    # cuándo se oyó al agente, que solo este lado mide. Ver §2.2.5 del contrato.
    from poc1_pipecat.app.pipeline import RECORDINGS_PATH

    crossing_path = folder / f"cero_{call_sid}.json"
    pipeline_json = RECORDINGS_PATH / f"call_{call_sid}.json"
    lines = add_lines(
        crossing_path,
        path_of_marks=zeros_path(call_sid, folder=folder),
        observed=observed,
        pipeline_json=pipeline_json,
    )
    if lines:
        print_lines(lines)

    log_path = folder / f"event_log_{call_sid}.json"
    event_log = event_log_de_la_corrida(observed, call_sid, scenario=args.scenario)
    event_log["audio_delivery"] = {
        "pacing": observed.publishing_pacing,
        "source_tail": {
            "n": 0,
            "note": (
                "PoC1 no tiene cola: el WebSocket no expone ocupación y no hay "
                "`AudioSource` entre el envío y el cable. Es una de las piezas "
                "por las que su sesgo no se compara de frente con el de PoC2 — "
                "ver docs/METRICS_CONTRACT.md §2.2.2"
            ),
        },
        "series": {
            "what_they_are": (
                "una entrada por trama publicada: `instante` es el envío medido "
                "(reloj de pared). PoC1 no tiene `cola_ms` que acompañarlo"
            ),
            "moment": [round(t, 6) for t in observed.measured_sends],
        },
    }
    log_path.write_text(
        json.dumps(event_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Lo que se reprodujo quedó en {log_path}")

    ruta_auditoria = (
        _dump_audit(recorder, call_sid, folder=folder)
        if recorder is not None
        else None
    )

    if observed.decisions:
        print(
            "  La corrida se ramificó: corrige la referencia del WER con esas\n"
            "  líneas, no con las del guion, o el WER mide contra otra frase."
        )

    resultado = evaluate_run(observed)

    if args.summary:
        # Antes de cualquier `return`: el lote necesita el resumen **también**
        # de las llamadas que salieron en rojo. Una corrida no utilizable es un
        # dato de la campaña —y de su costo—, no un hueco.
        write_summary(
            Path(args.summary),
            call_summary(
                observed,
                resultado,
                poc=1,
                scenario=args.scenario,
                repetition=args.repetition,
                label=call_sid,
                duration_s=duration,
                cost_usd=estimate_cost(duration),
                failed_criteria=faults,
                lines=lines,
                overlap=measure_overlap(ruta_auditoria),
                return_path=measure_return_path(crossing_path),
                artifacts={
                    "event_log": str(log_path),
                    "zero_crossing": str(crossing_path),
                    "pipeline_json": str(pipeline_json),
                    **({"auditoria_wav": str(ruta_auditoria)} if ruta_auditoria else {}),
                },
            ),
        )

    print()
    if resultado.retry:
        print(
            f"Corrida no utilizable ({resultado.motivo}): {resultado.reached} de "
            f"{resultado.of_script} turnos, mínimo {MIN_VALID_TURNS}."
        )
        return 1
    if faults:
        print(f"{faults} criterio(s) fallaron.")
        return 1
    print(
        f"Verde: {resultado.reached} turno(s) actuados, oídos y contestados "
        f"({resultado.motivo})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
