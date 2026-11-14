"""Cuánto se corre el cero de PoC1, medido contra el instante real del arnés.

Levanta el `/ws` de PoC1 con **el transporte, el serializer y el VAD de verdad**
—lo mismo que corre en una llamada, y en su propio proceso— y le habla desde el
arnés. Después cruza lo que el VAD estimó contra lo que el arnés publicó.

## Qué lleva el pipeline y qué no

Lleva el transporte, el serializer, el VAD y el analizador de fin de turno,
montados donde la campaña los monta. Los cuatro corren local y no cuestan nada.

No lleva STT, LLM ni TTS, porque cuelgan de ahí para abajo: el cero nace en
`VADUserStoppedSpeakingFrame`, y quitarlos no cambia el tramo medido pero baja el
costo de la corrida a **cero**. Lo que sí cambia es la carga de CPU —una llamada
completa tiene además tres clientes de red— así que el número que sale de aquí es
el **piso** del sesgo de PoC1, y una corrida con el pipeline completo lo confirma
o lo sube.

**Quitar el STT tiene una consecuencia que no es de costo**, y hay que decirla:
`TurnAnalyzerUserTurnStopStrategy` espera la transcripción antes de declarar el
fin de turno, así que sin STT el turno casi no cierra (4 clips dieron 7 fines de
habla y un solo fin de turno). La frontera del turno la declara entonces el
arnés, que la sabe de cierto — ver `zeros_per_window`. **El cero no se toca**:
sigue saliendo del VAD.

## Dos procesos, como en la campaña

El pipeline corre en un subproceso y el arnés en el principal. No es cosmético:
compartir bucle de eventos con la inferencia de Silero y del analizador le mete
al arnés pausas de hasta 244 ms, que entrarían enteras en el sesgo que se quiere
medir.

## Qué NO hace

No toca `latencias` ni ninguna muestra de la campaña: `SondaDelCero` solo lee.
No escribe en `grabaciones/`. Lo que produce vive en `diagnostics/harness/` y está
marcado como diagnóstico.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# A nivel de módulo y no dentro de la función: con `from __future__ import
# annotations` FastAPI resuelve las anotaciones del handler contra los globales
# del módulo. Con `WebSocket` importado adentro no lo encuentra, lo toma por un
# parámetro de query y rechaza el handshake con un 403 que no explica nada.
from fastapi import FastAPI, WebSocket

from poc1_pipecat.app.marks import (
    DIAGNOSTICS_PATH,
    DiskMarkEmitter,
    ZeroProbe,
    harness_block,
    marks_de,
    zeros_path,
)
from poc1_pipecat.app.media_streams import (
    MediaStreamsClient,
    MediaStreamsPlayer,
)
from poc1_pipecat.app.pipeline import build_vad
from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.carrier_simulator.script import (
    RUTA_CLIPS_DEFAULT,
    HarnessFailure,
    ScriptStep,
    load_scenario,
)
from shared.carrier_simulator.cero import ZeroMark
from shared.carrier_simulator.cruce import EstimatedZero
from shared.carrier_simulator.observed import Observed
from shared.carrier_simulator.player import high_clock_resolution

logger = logging.getLogger(__name__)

# Hueco entre clips. No hay agente que conteste, así que el avance de turno no
# puede colgar de su voz: es un tiempo fijo, y tiene que ser cómodamente mayor
# que los `stop_secs` del VAD (0.2 s) para que cada clip produzca su propio fin
# de habla en vez de fundirse con el siguiente.
CLIP_GAP_S: float = 3.0

# Ruido de confort antes del primer clip. Silero necesita oír algo de piso antes
# de que llegue la voz: si el primer sonido de la sesión es el clip, el arranque
# de habla cae dentro del clip y el primer turno sale distinto de los demás.
INITIAL_NOISE_S: float = 2.0

# Los `stop_secs` con los que corre la campaña. Se leen del VAD que construye
# `pipeline.py` para que no puedan discrepar, pero una sola vez: cargar el modelo
# de Silero otra vez solo para leer un número tarda segundos.
STOP_SECS: float = build_vad().params.stop_secs


class _ObservedSilent(Observed):
    """El agente no habla en esta corrida: no hay transcripción que leer.

    `Observado` existe para juntar lo que dijo el agente, y aquí no dice nada.
    Se instancia igual porque el reproductor lo usa para fechar el clip.
    """


async def _serve_ws_with_vad(port: int, output: Path) -> None:
    """Levanta el `/ws` de PoC1 con el transporte, el VAD y el analizador.

    Corre en **su propio proceso**. En la campaña el arnés y el pipeline son dos
    procesos, y aquí tiene que ser igual por una razón medible: compartir bucle
    de eventos con la inferencia de Silero y del analizador le mete al arnés
    pausas de hasta 244 ms, que entran enteras en el sesgo que se quiere medir.
    Con procesos separados el ritmo del arnés vuelve a su desvío de ~1 ms."""
    import uvicorn
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
        TurnAnalyzerUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    from poc1_pipecat.app.pipeline import build_turn_analyzer, build_context
    from poc1_pipecat.app.telephony import RUTA_WS
    from shared.negotiation_engine import TEST_TRIP

    probes: list[ZeroProbe] = []
    app = FastAPI()
    servidor: Any = None

    def _dump() -> None:
        """Lo que vio el VAD, a disco. Es lo único que el proceso devuelve."""
        output.write_text(
            json.dumps(
                {
                    "speech_ends": [
                        {"start": i, "end": f}
                        for probe in probes
                        for i, f in probe.speech_ends
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @app.websocket(f"/{RUTA_WS}")
    async def media_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        # Igual que en `telefonia.py`: los dos primeros mensajes se leen a mano
        # porque el serializer necesita los SID antes de tomar el WebSocket.
        await websocket.receive_text()
        start = json.loads(await websocket.receive_text()).get("start", {})

        serializer = TwilioFrameSerializer(
            stream_sid=start.get("streamSid", ""),
            call_sid=start.get("callSid", ""),
            # Sin credenciales y sin colgar por REST: aquí no hay ninguna
            # llamada de Twilio que colgar, y el serializer se niega a
            # construirse con `auto_hang_up` y sin credenciales. Lo que sí es
            # igual que en la campaña es el (des)empaquetado de μ-law, que es lo
            # único suyo que está en el camino que se mide.
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
                session_timeout=None,
            ),
        )

        probe = ZeroProbe()
        probes.append(probe)
        # El VAD y el analizador de fin de turno son los mismos que construye el
        # pipeline de la campaña, montados donde la campaña los monta. Los dos
        # corren local y no cuestan nada, y el analizador hace falta de verdad:
        # sin él no hay frontera de turno, y con `stop_secs=0.2` el VAD dispara
        # varias veces dentro de un mismo clip.
        aggregator_group = LLMContextAggregatorPair(
            build_context(TEST_TRIP),
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
        tarea = PipelineTask(
            Pipeline(
                [
                    transport.input(),
                    aggregator_group.user(),
                    probe,
                    transport.output(),
                ]
            ),
            params=PipelineParams(enable_metrics=False),
            enable_rtvi=False,
        )
        @transport.event_handler("on_client_disconnected")
        async def _on_disconnect(_transport: Any, _ws: Any) -> None:
            # Sin esto el pipeline se queda esperando audio que ya no va a
            # llegar, el proceso no termina y el arnés acaba matándolo — con lo
            # que el volcado de abajo nunca corre y el diagnóstico sale vacío.
            await tarea.cancel()

        try:
            await PipelineRunner(handle_sigint=False).run(tarea)
        finally:
            # En el `finally` y no después: el arnés cuelga el WebSocket, y si
            # el pipeline muriera por una excepción igual queremos lo que el VAD
            # alcanzó a ver. Un diagnóstico que se va en blanco no se distingue
            # de uno que midió cero.
            _dump()
            if servidor is not None:
                servidor.should_exit = True

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    servidor = uvicorn.Server(config)
    await servidor.serve()


async def _play_without_agent(
    player: MediaStreamsPlayer,
    observed: Observed,
    script: list[ScriptStep],
    emitter: DiskMarkEmitter,
) -> None:
    """Toca los clips uno tras otro, con hueco fijo. Sin esperar respuesta.

    Las marcas salen con exactamente la misma llamada que hace
    `play_script`, para que el paquete sea el de la campaña y no uno
    parecido escrito aquí.
    """
    await asyncio.sleep(INITIAL_NOISE_S)

    for step in script:
        if step.is_silence:
            await asyncio.sleep(step.silence_s)
            continue
        end = await player.play_step(step)
        start = end - step.duration_s
        await emitter.send(
            turn=step.turn,
            clip_id=step.clip_id,
            start_of_clip=observed.start + start,
            clip_end=observed.start + end,
            end_of_speech_s=step.end_of_speech_s,
            drop_db=step.speech_drop_db,
            speech_threshold_dbfs=step.speech_threshold_dbfs,
        )
        await asyncio.sleep(CLIP_GAP_S)


def zeros_per_window(
    ends: Sequence[Mapping[str, Any]], marks: Sequence[ZeroMark]
) -> list[EstimatedZero]:
    """El cero de cada turno: el ÚLTIMO fin de habla dentro de la ventana del clip.

    En la campaña la frontera del turno la pone el analizador de fin de turno, y
    `ColectorDeLatencia` se queda con el último fin de habla antes de esa
    frontera. Aquí no hay STT, y `TurnAnalyzerUserTurnStopStrategy` **espera la
    transcripción** antes de declarar el fin de turno: sin ella el turno
    prácticamente no cierra (4 clips produjeron 7 fines de habla y un solo fin de
    turno).

    Así que la frontera la declara el arnés, que la sabe de cierto: la ventana va
    del inicio del clip hasta el siguiente. **Eso no toca el cero** — el cero
    sigue siendo el instante que produjo el VAD, sin corregir. Lo único que
    decide esta función es *cuál* de los fines de habla del clip es el del turno,
    y con 3 s de silencio entre clips no hay ambigüedad: es el mismo que elegiría
    el analizador.
    """
    ordered = sorted(marks, key=lambda m: m.start_of_clip)
    zeros: list[EstimatedZero] = []
    for i, mark in enumerate(ordered):
        limit = (
            ordered[i + 1].start_of_clip
            if i + 1 < len(ordered)
            else float("inf")
        )
        inside = [
            f
            for f in ends
            if mark.start_of_clip <= float(f["end"]) < limit
        ]
        if not inside:
            continue
        ultimo = inside[-1]
        zeros.append(
            EstimatedZero(
                indice=i,
                start=inside[0].get("start"),
                end=float(ultimo["end"]),
            )
        )
    return zeros


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait_for_server(port: int, process: Any, tope_seg: float = 120.0) -> None:
    """Espera a que el `/ws` acepte conexiones. Arrancar carga dos modelos ONNX."""
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if process.returncode is not None:
            raise HarnessFailure(
                f"el servidor de diagnóstico se murió al arrancar "
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


async def launch(
    *, scenario: int, clips: int | None, folder: Path
) -> dict[str, Any]:
    from poc1_pipecat.app.telephony import RUTA_WS

    script = load_scenario(
        scenario, RUTA_CLIPS_DEFAULT, publish_sample_rate=TELEPHONY_SAMPLE_RATE
    )
    script = [p for p in script if not p.is_silence]
    if clips:
        script = script[:clips]
    if not script:
        raise HarnessFailure(f"el escenario {scenario} no trajo clips que tocar")

    folder.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    label = time.strftime("%Y%m%d-%H%M%S")
    vad_output = folder / f"vad_{label}.json"

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "poc1_pipecat.app.zero_diagnostics",
        "--run-server",
        "--port",
        str(port),
        "--vad-output",
        str(vad_output),
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    try:
        await _wait_for_server(port, process)

        client = MediaStreamsClient(
            f"ws://127.0.0.1:{port}/{RUTA_WS}", label=label
        )
        observed = _ObservedSilent()
        player = MediaStreamsPlayer(client, observed=observed)
        ruta = zeros_path(client.call_sid, folder=folder)
        emitter = DiskMarkEmitter(ruta, publish_sample_rate=player.sample_rate)

        halt = asyncio.Event()
        with high_clock_resolution():
            async with client:
                pumping = asyncio.create_task(player.launch(halt))
                try:
                    await _play_without_agent(
                        player, observed, script, emitter
                    )
                    # Un respiro al final: el último fin de habla llega
                    # `stop_secs` después del clip, y cerrar antes lo perdería.
                    await asyncio.sleep(2.0)
                finally:
                    halt.set()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pumping
                    await emitter.shut()

        # El servidor escribe lo que vio el VAD cuando se le cae el WebSocket.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=30)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)

    if not vad_output.is_file():
        raise HarnessFailure(
            f"el servidor no dejó sus fines de habla en {vad_output}: sin "
            "ellos no hay contra qué cruzar las marcas"
        )
    ends = json.loads(vad_output.read_text(encoding="utf-8"))["speech_ends"]
    marks = marks_de(ruta)
    zeros = zeros_per_window(ends, marks)

    block = harness_block(
        ruta, zeros, stop_secs=STOP_SECS, pipeline_host=socket.gethostname()
    )
    block["how_it_was_run"] = {
        "pipeline": (
            "transporte + serializer + VAD + analizador de fin de turno, en su "
            "propio proceso; sin STT, sin LLM y sin TTS"
        ),
        "cost_usd": 0.0,
        "caution": (
            "es el PISO del sesgo: una llamada completa corre además tres "
            "clientes de red (Deepgram, OpenAI, ElevenLabs) y esa contención "
            "solo puede subirlo"
        ),
        "harness_pacing_ms": player.pacing.summary(),
        "published_frames": client.sent_frames,
        "vad_speech_ends": len(ends),
        "turn_boundary": (
            "la declara el arnés (ventana de cada clip), no el analizador: sin "
            "STT no hay transcripción y `TurnAnalyzerUserTurnStopStrategy` la "
            "espera antes de cerrar. El cero sigue saliendo del VAD sin tocar; "
            "ver `zeros_per_window`"
        ),
    }
    destination = folder / f"cero_{client.call_sid}.json"
    destination.write_text(
        json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    block["file"] = str(destination)
    return block


def _display(block: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("El cero de PoC1 contra el instante real del arnés")
    print("=" * 70)
    cob = block["cobertura"]
    print(
        f"  marcas del arnés     {cob['marks']}"
        f"  ·  fines de habla del VAD {cob['vad_turns']}"
        f"  ·  emparejados {cob['paired_turns']}"
    )
    print(f"  ritmo del arnés      {block['how_it_was_run']['harness_pacing_ms']}")
    print()
    print(f"  {'turn':7s}{'clip':9s}{'sesgo voz':>11s}{'sesgo clip':>12s}{'cola':>9s}")
    for row in block["per_turn"]:
        if not row.get("paired"):
            print(f"  {row['indice']:<7}{'(sin marca)':9s}")
            continue
        print(
            f"  {row['harness_turn']:7s}{row['clip_id']:9s}"
            f"{row['speech_bias_ms']:11.1f}{row['clip_bias_ms']:12.1f}"
            f"{row['cola_ms']:9.1f}"
        )
    print()
    lost = block["turns_with_lost_end"]
    if lost["n"]:
        print(
            f"  {lost['n']} turno(s) con sesgo negativo "
            f"({', '.join(lost['clips'])}): fuera del resumen.\n"
            f"  Revisa los fines de habla crudos antes de leerlo como que el "
            f"VAD perdió el final del clip — la primera vez fue un bug del arnés."
        )
    print(f"  sesgo de voz   {block['speech_bias_ms']}")
    print(f"  sesgo de clip  {block['clip_bias_ms']}")
    verdict = block["zero_verdict"]
    print(f"\n  VEREDICTO      {verdict['verdict']}")
    print(f"  {verdict['detail']}")
    print(f"\n  {block['how_it_was_run']['caution']}")
    print(f"  escrito en     {block['file']}")


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.zero_diagnostics",
        description=(
            "Mide el sesgo del cero de PoC1 contra el instante real del arnés. "
            "Sin STT, LLM ni TTS: costo cero."
        ),
    )
    arg_parser.add_argument("--scenario", type=int, default=1)
    arg_parser.add_argument(
        "--clips",
        type=int,
        default=None,
        help="cuántos clips del escenario tocar (por defecto, todos)",
    )
    arg_parser.add_argument(
        "--folder", type=Path, default=DIAGNOSTICS_PATH, help="dónde escribir"
    )
    arg_parser.add_argument(
        "--run-server",
        action="store_true",
        help=(
            "modo interno: corre solo el pipeline y escribe lo que vio el VAD. "
            "Lo lanza la corrida principal como proceso aparte"
        ),
    )
    arg_parser.add_argument("--port", type=int, default=0)
    arg_parser.add_argument("--vad-output", type=Path, default=None)
    args = arg_parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    if args.run_server:
        if not args.port or args.vad_output is None:
            arg_parser.error("--run-server necesita --port y --vad-output")
        asyncio.run(_serve_ws_with_vad(args.port, args.vad_output))
        return 0

    block = asyncio.run(
        launch(scenario=args.scenario, clips=args.clips, folder=args.folder)
    )
    _display(block)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
