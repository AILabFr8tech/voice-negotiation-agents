"""Arnés de PoC2: corre la sala del agente con clips pregrabados, sin telefonía.

Proceso aparte del worker. Entra a la sala como `transportista`, le publica al
agente audio ya degradado a banda telefónica (`shared/audio/`) y observa qué
returns: voz del agente y fragmentos de `lk.transcription`. Publica además, por el
canal de datos, el fin del clip y el fin de voz medido dentro de él, para que el
worker los contraste contra su `stopped_speaking_at` — ver `app/marks.py`.

## Qué es de PoC2 y qué está en `shared/carrier_simulator/`

Aquí vive solo lo que no existiría sin LiveKit: el token y la conexión a la
sala, la publicación por la `AudioSource`, la lectura de `lk.transcription` con
sus atributos, el reensamblado del stream **no-delta** del STT, y el emisor de
ceros por el canal de datos (`app/marks.py`).

Lo demás —el reproductor por turno, el ruido de confort continuo, la regla de
los turnos medibles, la bitácora, el reporte y el aviso de costo— está en
`shared/carrier_simulator/`, porque entra en cómo se leen las métricas y tiene
que ser idéntico en las 3 PoC. Los nombres se reexportan aquí para que el arnés
se siga leyendo como un solo módulo desde la CLI y desde las pruebas.

## Por qué publica a 8 kHz y no sube a 48 él mismo

En una llamada real el audio entra por el trunk SIP a 8 kHz y es **LiveKit** quien
lo sube a 48 y lo mete en Opus. Subirlo aquí metería un remuestreo que la llamada
real no tiene, justo en el camino cuyo WER se mide. `--sample-rate 48000` existe por si
la pila rechaza publicar a 8 kHz, y el reporte lo dice, porque deja de ser el
mismo camino.

## Lo que este módulo NO hace, a propósito

- **No crea participantes SIP**: ningún teléfono puede sonar por accidente.
- **No toca el contrato de métricas.** No escribe en `latencias`; lo único que
  deja en el JSON del worker va en `livekit_extra.arnes`, que es diagnóstico.
- **No cambia ninguna constante congelada** (`stop_secs`,
  `min_silence_duration`, versión del turn detector).
- **No degrada audio en vivo**: el clip llega ya degradado, porque hacer DSP
  durante la llamada le metería su propio retraso a lo que se está midiendo.
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
from poc2_livekit.app.call import (
    ESTIMATED_LIVEKIT_COST_PER_MINUTE,
    LLM_COST_PER_CALL,
    STT_COST_PER_MINUTE,
    TTS_COST_PER_CALL,
)
from poc2_livekit.app.marks import MarkEmitter
from shared.audio import TELEPHONY_SAMPLE_RATE

# La mitad del arnés que no depende de LiveKit. Se reexporta con `as` explícito
# —y no con `import *`— para que se lea de un vistazo qué es compartido: todo lo
# que aparezca aquí abajo tiene que comportarse igual en PoC1 y PoC3, y cambiarlo
# es cambiarlo para la comparativa entera.
from shared.carrier_simulator import (
    MAX_AGENT_WAIT_S,
    WAIT_AFTER_CLIP_S,
    AUDIT_PREFIX,
    END_AGENT_MUTE,
    END_CLOSED_BY_AGENT,
    END_COMPLETE,
    END_INCOMPLETE,
    END_TIMEOUT,
    MIN_VALID_TURNS,
    MS_PER_FRAME,
    COMFORT_NOISE_PEAK,
    RMS_WITH_SPEECH,
    RUTA_CLIPS_DEFAULT,
    AGENT_SILENCE_S,
    CostContext,
    ReportContext,
    HarnessFailure,
    TextFragment,
    AgentWait,
    AuditRecorder,
    MIN_SUBSTANTIVE_SPEECH_S,
    SpeechStreak,
    BACKCHANNEL_CAP_S,
    UnauditableSampleRate,
    ScriptStep,
    RunResult,
    Tariffs,
    ObservedTurn,
    event_log_de_la_corrida,
    load_clip,
    load_scenario,
    is_backchannel,
    write_summary,
    measure_return_path,
    measure_overlap,
    wait_until_agent_is_silent,
    wait_for_agent_turn,
    negotiation_state,
    evaluate_run,
    print_cost_warning,
    audio_frame_line,
    published_frame_line,
    split_into_frames,
    play_script,
    quote_summary,
    call_summary,
    rms_pcm16,
    output_sample_rate,
    text_from_deltas,
    comfort_noise_frame,
)
from shared.carrier_simulator import Observed as ObservedFromSimulator
from shared.carrier_simulator import Player as SimulatorPlayer
from shared.carrier_simulator import print_report as print_simulator_report

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_AGENT_WAIT_S",
    "WAIT_AFTER_CLIP_S",
    "END_AGENT_MUTE",
    "END_CLOSED_BY_AGENT",
    "END_COMPLETE",
    "END_INCOMPLETE",
    "END_TIMEOUT",
    "MIN_VALID_TURNS",
    "MIN_SUBSTANTIVE_SPEECH_S",
    "MS_PER_FRAME",
    "COMFORT_NOISE_PEAK",
    "RMS_WITH_SPEECH",
    "RUTA_CLIPS_DEFAULT",
    "AGENT_SILENCE_S",
    "BACKCHANNEL_CAP_S",
    "AgentWait",
    "HarnessFailure",
    "TextFragment",
    "Observed",
    "ScriptStep",
    "SpeechStreak",
    "Player",
    "RunResult",
    "ObservedTurn",
    "build_script",
    "event_log_de_la_corrida",
    "load_clip",
    "load_scenario",
    "run_session",
    "wait_until_agent_is_silent",
    "wait_for_agent_turn",
    "is_backchannel",
    "estimate_cost",
    "evaluate_run",
    "print_report",
    "main",
    "split_into_frames",
    "play_script",
    "rms_pcm16",
    "text_from_deltas",
    "replacements_text",
    "comfort_noise_frame",
    "verify_constants",
]

# --------------------------------------------------------------------------
# Constantes del protocolo de transcripción de LiveKit
# --------------------------------------------------------------------------
#
# Se declaran aquí y se **verifican** contra el SDK en `verificar_constantes`,
# en vez de importarse: el arnés debe depender solo de `livekit.rtc` (el SDK del
# cliente), no de `livekit.agents` (el del worker). Y si LiveKit les cambia el
# nombre, un import roto se ve; una constante desactualizada **no**: el handler
# nunca se dispararía y el arnés reportaría "el agente no transcribe", que es la
# conclusión equivocada.

TRANSCRIPT_TOPIC: Final[str] = "lk.transcription"
ATTRIBUTE_SEGMENT: Final[str] = "lk.segment_id"
ATTRIBUTE_FINAL: Final[str] = "lk.transcription_final"

# Tope de la cola de la `AudioSource`. Es un TOPE y no una ocupación —solo se
# llena empujando más rápido que el tiempo real— pero **en llamada real se
# alcanza**, aunque en banco aislado se quede en una trama: en una llamada el
# bucle se atora, con el agente entrando por el otro lado, y después recupera
# empujando en ráfaga, que es justo lo que llena la cola.
#
# Por eso `resumen_de_la_cola` se publica por corrida y no se supone. Ver
# `docs/CONTRATO_METRICAS.md` §2.2.2.
SOURCE_TAIL_MS: Final[int] = 200

# Tasa a la que se publica. Ver el docstring del módulo.
DEFAULT_PUBLISH_SAMPLE_RATE: Final[int] = TELEPHONY_SAMPLE_RATE

# Tasa a la que se lee el audio del agente: la que sale a la sala
# (`RoomOutputOptions.audio_sample_rate`), así que pedir otra metería un
# remuestreo más de este lado.
#
# **No es la tasa de origen.** El TTS entrega `mp3_22050_32` —22.05 kHz, el
# default del plugin de ElevenLabs, que `construir_tts` no sobreescribe—, así que
# para cuando el audio llega aquí ya lo remuestrearon una vez: en el WAV de
# auditoría, por encima de 11.025 kHz (el Nyquist del origen) hay tanta potencia
# como justo por debajo, y eso no puede ser voz. Ver la asimetría declarada en `docs/CONTRATO_METRICAS.md`.
LISTEN_SAMPLE_RATE: Final[int] = 24_000

# Nombre de la pista del arnés, distinto de `roomio_audio` (la del agente) para
# poder distinguirlas de un vistazo en el dashboard.
TRACK_NAME: Final[str] = "harness_carrier"

# Dónde vive lo que produce el arnés. **No** es `grabaciones/`, a propósito: ahí
# viven las llamadas de la campaña y `shared.metrics.cli calcular` las lee todas.
DIAGNOSTICS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "harness"
)


def verify_constants() -> str:
    """Compara las constantes de arriba con las del SDK del worker.

    Corre antes de conectarse, así que un rename del SDK se ve como una falla de
    configuración y no como "el agente no manda transcripción".
    """
    from livekit.agents.types import (
        ATTRIBUTE_TRANSCRIPTION_FINAL,
        ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
        TOPIC_TRANSCRIPTION,
    )

    expected_items = (
        ("tópico", TRANSCRIPT_TOPIC, TOPIC_TRANSCRIPTION),
        ("atributo de segmento", ATTRIBUTE_SEGMENT, ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID),
        ("atributo final", ATTRIBUTE_FINAL, ATTRIBUTE_TRANSCRIPTION_FINAL),
    )
    for label, our, of_sdk in expected_items:
        if our != of_sdk:
            raise HarnessFailure(
                f"el {label} de la transcripción cambió en el SDK: "
                f"nosotros usamos {our!r} y LiveKit publica en {of_sdk!r}"
            )
    return f"constantes de transcripción al día ({TRANSCRIPT_TOPIC})"


# --------------------------------------------------------------------------
# Lo observado, en una sala
# --------------------------------------------------------------------------


def replacements_text(fragments: list[TextFragment]) -> str:
    """Reensambla un stream **no-delta**: el STT de lo que nosotros publicamos.

    Los dos viajan por `lk.transcription`, pero **no se reensamblan igual**, y
    confundirlos rompe el WER en las dos direcciones posibles.
    `room_io.py` construye las dos salidas con banderas opuestas:

        _user_tr_output  = _ParticipantTranscriptionOutput(is_delta_stream=False, …)
        _agent_tr_output = _ParticipantTranscriptionOutput(is_delta_stream=True,  …)

    Con `is_delta_stream=False`, `capture_text` abre un **writer nuevo por cada
    actualización** y escribe la transcripción *completa* hasta ese momento
    (`_output.py`: "always create a new writer"), y el `flush` manda una vez más
    el `_latest_text` ya con `lk.transcription_final`. O sea que los fragmentos
    no se suman: **el último manda**.

    Un clip que dice "Sí, soy Carlos. Dígame." puede llegar en 4 fragmentos, y
    `"".join` daría `'Sí, soy Carlos.Sí, soy Carlos.Sí, soy Carlos. Dígame.Sí,
    soy Carlos. Dígame.'`, con el WER arruinado.

    Se agrupa por `lk.segment_id` (estable dentro de una intervención,
    `_reset_state` lo regenera en la siguiente) y, por si LiveKit cambiara a
    emitir un id por actualización, una transcripción que **prolonga** a la
    anterior también la reemplaza. Las intervenciones distintas sí se suman.
    """
    useful = [f for f in fragments if not f.closure and f.text]
    if not useful:
        return ""

    # 1) Por segmento: la actualización más completa gana.
    per_segment: list[tuple[str, str]] = []
    for fragment in useful:
        if per_segment and per_segment[-1][0] == fragment.segment:
            previous = per_segment[-1][1]
            if len(fragment.text) >= len(previous):
                per_segment[-1] = (fragment.segment, fragment.text)
        else:
            per_segment.append((fragment.segment, fragment.text))

    # 2) Entre segmentos: si uno prolonga al anterior, es la misma intervención.
    parts: list[str] = []
    for _, text in per_segment:
        if parts and (text.startswith(parts[-1]) or parts[-1].startswith(text)):
            if len(text) >= len(parts[-1]):
                parts[-1] = text
        else:
            parts.append(text)
    return " ".join(parts)


@dataclass
class Observed(ObservedFromSimulator):
    """Lo del simulador, más lo que solo existe dentro de una sala de LiveKit."""

    # Nuestra identidad la fija el agente: espera exactamente a este nombre.
    own_identity: str = CARRIER_IDENTITY

    published_track: bool = False
    participants: dict[str, float] = field(default_factory=dict)
    subscribed_tracks: list[str] = field(default_factory=list)

    # Ceros reales publicados al worker por el canal de datos (`app/marcas.py`).
    sent_marks: int = 0
    failed_marks: list[str] = field(default_factory=list)
    marks_summary_sent: bool = False

    # Cómo salió el audio de este lado. Se llenan al terminar la sesión y viajan
    # en la bitácora, no en `latencias`: sirven para leer el sesgo del cero sin
    # tener que suponer ni el ritmo del arnés ni lo que retuvo la `AudioSource`.
    publishing_pacing: dict[str, Any] = field(default_factory=dict)
    source_tail: dict[str, Any] = field(default_factory=dict)
    # Las dos series crudas, una entrada por trama y en el mismo orden.
    measured_sends: list[float] = field(default_factory=list)
    tail_per_frame: list[float] = field(default_factory=list)

    def annotate_fragment(
        self,
        sender: str,
        text: str,
        attribute_map: dict[str, str],
        *,
        closure: bool = False,
    ) -> None:
        """Un trozo de `lk.transcription`, con los atributos del stream."""
        self.annotate_text(
            sender,
            text,
            segment=str(attribute_map.get(ATTRIBUTE_SEGMENT, "")),
            final=str(attribute_map.get(ATTRIBUTE_FINAL, "")).lower() == "true",
            closure=closure,
        )

    def assemble_stt(self, fragments: list[TextFragment]) -> str:
        """El STT del usuario llega como reemplazos. Ver `replacements_text`."""
        return replacements_text(fragments)

    @property
    def agent_identity(self) -> str | None:
        for identity in self.participants:
            if identity != self.own_identity:
                return identity
        return None


# --------------------------------------------------------------------------
# Sesión
# --------------------------------------------------------------------------


def build_token(settings: Settings, room: str) -> str:
    """Token del arnés: entrar a `sala` como "transportista" y poder publicar.

    La identidad tiene que ser **exactamente** `CARRIER_IDENTITY`: el
    agente espera a ese nombre (`ctx.wait_for_participant(identity=…)`) y cuelga
    cuando ese nombre se desconecta. Con cualquier otra el worker se queda
    esperando su tope de 90 s y no negocia nada.

    `can_publish`/`can_subscribe` ya son `True` por default; se pasan explícitos
    porque son el permiso del que depende todo el arnés y no debe quedar como un
    default heredado que alguien cambie sin darse cuenta.
    """
    from livekit.api import AccessToken, VideoGrants

    return (
        AccessToken(api_key=settings.livekit_api_key, api_secret=settings.livekit_api_secret)
        .with_identity(CARRIER_IDENTITY)
        .with_name("Transportista (arnés)")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def dispatch_agent(
    settings: Settings, room: str, identification: dict[str, Any]
) -> str:
    """Mete al agente a la sala. **Sin `create_sip_participant`.**

    Es el paso 1 de `app/call.py::marcar` y nada más. El worker está
    registrado con `agent_name`, así que sin este dispatch no entra a ninguna
    sala; y sin el paso 2 —que aquí no existe— no hay teléfono que suene.
    """
    from livekit import api

    async with api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    ) as lk:
        dispatch_result = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=settings.agent_name,
                room=room,
                metadata=json.dumps(identification, ensure_ascii=False),
            )
        )
    return str(getattr(dispatch_result, "id", "?"))


class Player(SimulatorPlayer):
    """El reproductor compartido, empujando por la `AudioSource` de la sala.

    Lo único que PoC2 pone es la trama: el ritmo, el ruido de confort y el
    instante en que termina de sonar el clip son del simulador, para que las 3
    PoC fechen el cero del turno igual.

    ## Por qué mide la ocupación de la cola

    La `AudioSource` se crea con `queue_size_ms=200`, y esos 200 ms se leían como
    si fueran retraso: audio esperando turno antes de salir al encoder, que
    caería entero dentro del sesgo del cero. **No lo es, y por eso se mide en vez
    de suponerse.** 200 ms es un *tope*, y un tope solo se alcanza empujando más
    rápido que el tiempo real; el reproductor empuja contra deadline absoluto, o
    sea exactamente a tiempo real, así que la cola se queda en una trama.

    Muestrear `queued_duration` cuesta una resta y una lectura de reloj — está
    fuera del `await`, y el ritmo publicado en el reporte deja ver si alguna vez
    costara más.
    """

    def __init__(
        self,
        source: Any,
        *,
        sample_rate: int,
        observed: ObservedFromSimulator,
        recorder: Any = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, observed=observed, recorder=recorder)
        self.source = source
        self._rtc: Any = None
        # Ocupación de la cola justo antes de empujar cada trama, en ms.
        self.cola_ms: list[float] = []

    async def prepare(self) -> None:
        # Se importa aquí y no arriba para que las pruebas offline puedan correr
        # el bucle con un `livekit.rtc` falso en `sys.modules`.
        from livekit import rtc

        self._rtc = rtc

    async def publish(self, raw_value: bytes) -> None:
        occupancy = getattr(self.source, "queued_duration", None)
        if isinstance(occupancy, (int, float)):
            self.cola_ms.append(float(occupancy) * 1000.0)
        await self.source.capture_frame(
            self._rtc.AudioFrame(
                data=raw_value,
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=self.per_frame,
            )
        )

    def queue_summary(self) -> dict[str, Any]:
        """Cuánto audio había esperando en la `AudioSource` al empujar.

        Lo que importa no es el promedio sino el **máximo**: si alguna vez se
        acercara a `queue_size_ms`, el audio estaría saliendo tarde y ese retraso
        caería dentro del sesgo del cero sin que nada lo avisara.
        """
        if not self.cola_ms:
            return {"n": 0, "note": "la fuente no expone `queued_duration`"}
        ordered = sorted(self.cola_ms)
        n = len(ordered)
        maximum = ordered[-1]
        return {
            "n": n,
            "declared_ms": SOURCE_TAIL_MS,
            "median_ms": round(ordered[n // 2], 2),
            "p95_ms": round(ordered[min(n - 1, int(0.95 * n))], 2),
            "max_ms": round(maximum, 2),
            "cap_usage_pct": round(100.0 * maximum / SOURCE_TAIL_MS, 1),
            "note": (
                "la cola declarada es un TOPE, no una ocupación: solo se llena "
                "empujando más rápido que el tiempo real, y el reproductor "
                "empuja contra deadline absoluto. Si `uso_del_tope_pct` se "
                "acercara a 100, el audio saldría tarde y ese retraso entraría "
                "en el sesgo del cero"
            ),
        }


async def read_agent_audio(audio_track: Any, observed: Observed) -> None:
    """Mide la energía de la voz del agente trama por trama.

    Por **energía** y nunca por llegada de tramas: el humo de sala midió 9,935
    tramas del agente de las que solo 230 (2%) traían voz. La pista es continua;
    lo que no es continuo es que el agente hable.
    """
    from livekit import rtc

    stream = rtc.AudioStream(audio_track, sample_rate=LISTEN_SAMPLE_RATE, num_channels=1)
    try:
        async for event in stream:
            trama = getattr(event, "frame", None)
            if trama is None:
                continue
            observed.annotate_audio(bytes(trama.data))
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - el arnés reporta, no propaga
        observed.annotate_incident(f"la lectura del audio del agente falló: {error!r}")
    finally:
        await stream.aclose()


async def run_session(
    settings: Settings,
    *,
    room: str,
    script: list[ScriptStep],
    publish_sample_rate: int,
    wait_after_clip: float,
    identification: dict[str, Any],
    silent: bool = False,
    estado: Any = None,
    recorder: Any = None,
) -> Observed:
    """La corrida completa: despacha, conecta, deja abrir al agente y actúa el guion.

    Con `grabadora` se acumulan las dos direcciones del audio para poder oír la
    corrida después (`--record-audit`). No cambia nada de lo que se mide: el
    volcado a disco pasa al colgar, no aquí. Ver
    `shared/carrier_simulator/auditoria.py`.
    """
    from livekit import rtc

    # Referencias fuertes a las tareas de fondo: una tarea sin dueño puede ser
    # recolectada a media ejecución.
    tasks: set[asyncio.Task[Any]] = set()

    def _adopt(tarea: asyncio.Task[Any]) -> asyncio.Task[Any]:
        tasks.add(tarea)
        tarea.add_done_callback(tasks.discard)
        return tarea

    print(f"  despachando al agente {settings.agent_name!r} a la sala {room} …")
    dispatch_result = await dispatch_agent(settings, room, identification)
    print(f"  dispatch creado: {dispatch_result}")

    # El cero del reloj se pone aquí, justo antes de conectar, y no se vuelve a
    # tocar: el dispatch ya ocurrió y todas las marcas que siguen son
    # comparables entre sí.
    observed = Observed(silent=silent, recorder=recorder)
    if recorder is not None:
        # Que la grabadora tenga el mismo cero que todo lo demás: se construye
        # antes, en `main`, porque tiene que sobrevivir a la corrida para poder
        # volcarse.
        recorder.start = observed.start

    rtc_room = rtc.Room()

    # --- Handlers, ANTES de conectar ---
    # Registrarlos después dejaría pasar los eventos de quien ya estuviera
    # dentro, y el agente puede entrar antes que nosotros.

    @rtc_room.on("participant_connected")
    def _on_enter(participant: Any) -> None:
        identity = str(getattr(participant, "identity", ""))
        observed.participants[identity] = observed.now()
        print(f"  [{observed.now():>7.3f}s] entró {identity!r}")

    @rtc_room.on("track_subscribed")
    def _on_subscribe(audio_track: Any, publishing: Any, participant: Any) -> None:
        if getattr(audio_track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        identity = str(getattr(participant, "identity", ""))
        name = str(getattr(publishing, "name", ""))
        observed.subscribed_tracks.append(f"{identity}/{name}")
        print(f"  [{observed.now():>7.3f}s] suscrito al audio de {identity!r}")
        _adopt(asyncio.create_task(read_agent_audio(audio_track, observed)))

    # Segunda condición de fin. Se declara antes de los handlers porque
    # uno de ellos lo levanta.
    closed = asyncio.Event()

    @rtc_room.on("disconnected")
    def _on_disconnect(motivo: Any) -> None:
        """El agente cerró la negociación y borró la sala.

        **No es un incidente**: es una de las dos formas normales de terminar.
        Lo que sí sería un problema es no enterarse: el arnés seguiría
        reproduciendo el resto del guion contra una sala borrada, y los turnos
        que "no contestaron" parecerían fallas del agente.
        """
        observed.was_closed_by_agent = True
        observed.close_reason = str(motivo)
        closed.set()
        observed.tally(f"la sala se cerró: {motivo!r}")

    async def _leer_stream(lector: Any, sender: str) -> None:
        info = getattr(lector, "info", None)
        try:
            async for piece in lector:
                attribute_map = dict(getattr(info, "attributes", None) or {})
                observed.annotate_fragment(sender, piece, attribute_map)
            # El trailer ya se fundió en `info.attributes`: aquí, y solo aquí, se
            # puede leer `lk.transcription_final` de verdad.
            observed.annotate_fragment(
                sender, "", dict(getattr(info, "attributes", None) or {}), closure=True
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            observed.annotate_incident(f"la lectura de un stream de texto falló: {error!r}")

    def _on_text_received(lector: Any, sender: str) -> None:
        # El handler que registra LiveKit es **sync**, así que la lectura tiene
        # que irse a una tarea o el stream se queda sin consumir.
        _adopt(asyncio.create_task(_leer_stream(lector, sender)))

    rtc_room.register_text_stream_handler(TRANSCRIPT_TOPIC, _on_text_received)

    # --- Conectar ---
    token = build_token(settings, room)
    print(f"  conectando como {CARRIER_IDENTITY!r} …")
    await rtc_room.connect(settings.livekit_url, token)
    observed.connected = True
    print(f"  conectado. identidad local: {rtc_room.local_participant.identity!r}")

    for identity, participant in rtc_room.remote_participants.items():
        observed.participants.setdefault(str(identity), observed.now())
        for publishing in participant.track_publications.values():
            if publishing.track is not None:
                _on_subscribe(publishing.track, publishing, participant)

    halt = asyncio.Event()
    # Fuera del `try` para que el `finally` pueda consultarlo aunque la
    # publicación de la pista falle antes de crearlo.
    emitter: MarkEmitter | None = None
    try:
        source = rtc.AudioSource(
            publish_sample_rate, 1, queue_size_ms=SOURCE_TAIL_MS
        )
        audio_track = rtc.LocalAudioTrack.create_audio_track(TRACK_NAME, source)
        publishing = await rtc_room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE,
                # Ver `trama_de_ruido_de_confort`: con DTX activo las pausas
                # dejan de mandar paquetes y el endpointing del agente se rompe.
                dtx=False,
            ),
        )
        observed.published_track = True
        print(
            f"  [{observed.now():>7.3f}s] pista publicada: {publishing.sid} "
            f"({publish_sample_rate} Hz mono, dtx=False)"
        )

        player = Player(
            source, sample_rate=publish_sample_rate, observed=observed, recorder=recorder
        )
        _adopt(asyncio.create_task(player.launch(halt)))
        emitter = MarkEmitter(
            rtc_room.local_participant, publish_sample_rate=publish_sample_rate
        )

        print("\n  esperando a que el agente abra …")
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
            estado=estado,
        )
    except KeyboardInterrupt:  # pragma: no cover - depende de la terminal
        print("\n  cortado a mano.")
    finally:
        halt.set()
        for tarea in list(tasks):
            tarea.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # El resumen de ceros va aquí: con la pista ya quieta y **antes** de
        # desconectarse. Desconectarse es lo que le dispara al worker su
        # shutdown y la escritura del JSON, así que un paquete publicado después
        # no llega a ningún lado. Ver `marcas.EmisorDeMarcas.cerrar`.
        if emitter is not None:
            if closed.is_set():
                # Publicar contra una sala borrada solo produce
                # `engine is closed`. Las marcas en caliente ya viajaron; el
                # resumen era la red de seguridad y aquí ya no hay red.
                observed.tally(
                    "no se manda el resumen de ceros: la sala ya estaba cerrada"
                )
            else:
                # La lista de turnos mudos viaja aquí porque aquí es donde ya
                # se sabe: al publicar cada marca el agente ni había empezado a
                # contestar. Es lo único de esto que el worker no puede deducir
                # solo — él ve su TTS pedir audio, no si ese audio sonó.
                observed.marks_summary_sent = await emitter.shut(
                    turns_without_audio=[t.step.turn for t in observed.turns_without_audio()]
                )
            observed.sent_marks = emitter.sent
            observed.failed_marks = list(emitter.faults)
            if emitter.faults:
                observed.annotate_incident(
                    f"{len(emitter.faults)} marca(s) de cero de turno no salieron: "
                    "el worker publicará cobertura incompleta"
                )

        # Cómo salió el audio de este lado, para poder leer el sesgo del cero
        # con algo más que suposiciones: el ritmo dice si el arnés entregó
        # parejo, y la cola si la `AudioSource` retuvo audio. Los dos son
        # diagnóstico del arnés y no tocan `latencias`.
        observed.publishing_pacing = player.pacing.summary()
        observed.source_tail = player.queue_summary()
        observed.measured_sends = list(player.pacing.reales)
        observed.tail_per_frame = list(player.cola_ms)

        # Desconectarnos es lo que cierra la sesión del agente
        # (`close_on_disconnect`) y dispara su `delete_room` — o sea, lo que hace
        # que el worker escriba su JSON en vez de quedarse vivo hasta el
        # `empty_timeout` de la sala.
        await rtc_room.disconnect()
        print(f"  [{observed.now():>7.3f}s] desconectados.")

    return observed


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


def print_report(observed: Observed, room: str, *, publish_sample_rate: int) -> int:
    """El reporte compartido, con lo que solo se puede decir de una sala."""
    agent = observed.agent_identity
    header = [
        f"  publicado a           {publish_sample_rate} Hz, dtx=False",
        published_frame_line(observed),
        f"  agente en la sala     {agent!r}"
        if agent
        else "  agente en la sala     NO ENTRÓ",
        audio_frame_line(observed),
    ]
    if observed.sent_marks:
        summary = "sí" if observed.marks_summary_sent else "NO"
        header.append(
            f"  ceros publicados      {observed.sent_marks} "
            f"(resumen antes de colgar: {summary}"
            + (
                f", {len(observed.failed_marks)} fallaron"
                if observed.failed_marks
                else ""
            )
            + ")"
        )

    return print_simulator_report(
        observed,
        ReportContext(
            title=f"Arnés de PoC2 — {room}",
            header=header,
            stt_name="Deepgram",
            where_the_mutes_are_marked="livekit_extra.arnes",
            footer=(
                f"\n  El worker deja su JSON en poc2_livekit/grabaciones/llamada_{room}.json"
                "\n  Es una corrida de arnés, NO de campaña: no la dejes en esa carpeta si"
                "\n  vas a correr `shared.metrics.cli calcular` sobre ella.",
            ),
        ),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_script(args: argparse.Namespace) -> list[ScriptStep]:
    """Traduce las banderas a la lista de turnos que se va a actuar."""
    if args.scenario is not None:
        return load_scenario(
            args.scenario, args.clips, publish_sample_rate=args.sample_rate
        )

    samples, origin = load_clip(args.clip, publish_sample_rate=args.sample_rate)
    times = max(1, int(args.repeat))
    return [
        ScriptStep(
            turn=f"T{numero}",
            text="(clip suelto)",
            samples=samples,
            clip_id=Path(args.clip).stem,
            origin=origin,
            sample_rate=args.sample_rate,
        )
        for numero in range(1, times + 1)
    ]


# Alias histórico: las pruebas y el diagnóstico de banda lo llaman por este
# nombre. La función vive en el simulador porque el catálogo de compensaciones
# es compartido.
_negotiation_state = negotiation_state


# Sin trunk: aquí no hay telefonía. Las tarifas salen de `app/llamada.py`,
# verificadas contra `docs/costos_poc2.md`.
HARNESS_TARIFFS: Final[Tariffs] = Tariffs(
    per_minute=STT_COST_PER_MINUTE + ESTIMATED_LIVEKIT_COST_PER_MINUTE,
    per_call=TTS_COST_PER_CALL + LLM_COST_PER_CALL,
)


def estimate_cost(seconds: float) -> float:
    """Costo aproximado de una corrida. Sin trunk: aquí no hay telefonía."""
    return HARNESS_TARIFFS.estimate(seconds)


def _print_cost_warning(
    settings: Settings,
    room: str,
    script: list[ScriptStep],
    seconds: float,
    *,
    retries: int = 0,
) -> None:
    print_cost_warning(
        CostContext(
            title="Corrida del arnés de PoC2, SIN telefonía.",
            tariffs=HARNESS_TARIFFS,
            is_spent=(
                ("STT", "Deepgram, sobre los clips que publica el arnés"),
                ("TTS", "ElevenLabs, el saludo del agente y sus respuestas"),
                ("LLM", "gpt-4.1-mini, la apertura del prompt v0 y cada turno"),
                ("LiveKit", "minutos de participante (estimación sin verificar)"),
            ),
            is_not_spent=(
                ("Twilio", "no se crea participante SIP: ningún teléfono suena"),
            ),
            fields=(
                ("room", room),
                ("proyecto", settings.livekit_url),
                ("agent", settings.agent_name),
            ),
        ),
        script,
        seconds,
        retries=retries,
    )


def wait_and_add_lines(
    observed: Observed, room: str, *, tope_seg: float = 20.0
) -> dict[str, Any] | None:
    """Espera el JSON del worker y le agrega la cuenta por intervención.

    Hay que esperar porque el worker lo escribe en su `shutdown`, disparado por
    nuestra desconexión: llegar antes que él encontraría el archivo a medias o
    no encontrarlo. En PoC1 no hace falta porque el arnés mata al servidor y lo
    espera; aquí el worker es de otra terminal y su ritmo no es nuestro.
    """
    from poc2_livekit.app.agent import RECORDINGS_PATH
    from poc2_livekit.app.marks import add_lines

    ruta = RECORDINGS_PATH / f"call_{room}.json"
    first_speech = {
        t.step.turn: observed.start + t.t_agent_first_speech
        for t in observed.turns
        if t.t_agent_first_speech is not None
    }

    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if ruta.is_file():
            block = add_lines(ruta, first_speech=first_speech)
            if block is not None:
                return block
        time.sleep(0.5)

    print(
        f"\n  [!] el worker no dejó {ruta.name} en {tope_seg:.0f} s: la cuenta por\n"
        "      intervención se puede agregar después con el JSON ya escrito."
    )
    return None


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


def _dump_audit(recorder: Any, room: str) -> Path | None:
    """Escribe el WAV de auditoría y su JSON. Devuelve la ruta del WAV.

    Va envuelto en un `except` amplio a propósito: la grabación es para oír la
    corrida, y una corrida buena no se tira porque el archivo de diagnóstico no
    se pudo escribir. Si falla, lo dice y sigue.
    """
    ruta_wav = DIAGNOSTICS_PATH / f"{AUDIT_PREFIX}_{room}.wav"
    try:
        descriptive = recorder.dump(ruta_wav, room=room)
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
        prog="poc2_livekit.app.harness",
        description=(
            "Arnés de PoC2: publica un clip pregrabado a la sala del "
            "agente y verifica que lo oiga, lo transcriba y conteste. Requiere "
            "el worker arriba en otra terminal."
        ),
    )
    source = arg_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scenario",
        type=int,
        help="número de escenario del guion (1-5): corre sus turnos en orden",
    )
    source.add_argument(
        "--clip",
        help="un solo artefacto de shared/audio/ (.ulaw preferido, o .wav de 8 kHz)",
    )
    arg_parser.add_argument(
        "--clips",
        default=str(RUTA_CLIPS_DEFAULT),
        help="carpeta de los clips preparados (default: shared/test_scenarios/clips)",
    )
    arg_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "con --clip, reproduce el mismo clip N veces como turnos separados. "
            "Sirve para juntar muestras de fin de turno sin tener grabado el "
            "escenario completo; la conversación no tiene sentido y el WER de "
            "esa corrida no significa nada"
        ),
    )
    arg_parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_PUBLISH_SAMPLE_RATE,
        help=(
            f"tasa de publicación (default {DEFAULT_PUBLISH_SAMPLE_RATE}, la del "
            "trunk SIP real). Súbela solo si la pila rechaza 8 kHz"
        ),
    )
    arg_parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_AFTER_CLIP_S,
        help=f"segundos de observación tras el clip (default {WAIT_AFTER_CLIP_S:.0f})",
    )
    arg_parser.add_argument(
        "--retries",
        type=int,
        default=0,
        metavar="N",
        help=(
            "repetir hasta N veces si la corrida no alcanza "
            f"{MIN_VALID_TURNS} turnos utilizables. Default 0: el arnés "
            "avisa que hay que repetir y no gasta por su cuenta. Cada reintento "
            "es una llamada más y entra en el estimado de costo"
        ),
    )
    arg_parser.add_argument(
        "--record-audit",
        action="store_true",
        help=(
            "escribe un WAV de dos canales de la corrida (canal 0 lo que publicó "
            "el arnés, canal 1 el agente) en diagnostics/harness/. Es para OÍR la "
            "run: no entra a latencias, ni al WER, ni a ningún percentil, y "
            "no activa la grabación de LiveKit. Apagada por default"
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

    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_voice(settings)
    # `verificar_para_telefonia` NO se llama: este camino no necesita trunk.

    try:
        print(verify_constants())
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
            # Se valida aquí, antes del aviso de costo y de conectarse a nada:
            # enterarse de que el WAV no se puede armar **después** de haber
            # gastado la llamada sería el peor momento posible.
            output_sample_rate(args.sample_rate, LISTEN_SAMPLE_RATE)
    except (HarnessFailure, UnauditableSampleRate) as error:
        print(f"[FALLA] {error}")
        return 1

    # El nombre de la sala lleva "arnes" para que sea imposible confundir esta
    # corrida con una de las 15 de la campaña, en el dashboard y en el JSON.
    room = settings.room_name(suffix=f"arnes-{time.strftime('%Y%m%d-%H%M%S')}")
    # Cota alta: cada turno paga su clip más el tope de espera al agente.
    # La rama más larga de cada turno: el árbol decide en vivo, y presupuestar
    # la base subestimaría la llamada justo en los escenarios ramificados.
    total = MAX_AGENT_WAIT_S + sum(
        step.max_duration_s() + args.wait for step in script
    )

    if not args.accept_cost:
        print()
        _print_cost_warning(
            settings, room, script, total, retries=max(0, int(args.retries))
        )
        if args.summary:
            write_summary(
                Path(args.summary),
                quote_summary(
                    poc=2,
                    scenario=args.scenario,
                    script=script,
                    seconds=total,
                    cost_usd=estimate_cost(total),
                ),
            )
        return 0

    if args.sample_rate != TELEPHONY_SAMPLE_RATE:
        print(
            f"  [!] publicando a {args.sample_rate} Hz y no a {TELEPHONY_SAMPLE_RATE}: el arnés "
            "sube el clip él mismo,\n      así que deja de ser el mismo camino "
            "que el del trunk SIP."
        )

    # Los reintentos gastan otra llamada cada uno, así que el default es 0: el
    # arnés dice que hay que repetir y quien paga decide. Ver `--reintentos`.
    attempts = 1 + max(0, int(args.retries))
    faults = 0
    # Cada intento es otra llamada y cada uno se paga; el resumen reporta las
    # métricas del último, que es el que vale, pero el costo de todos.
    run_cost = 0.0
    duration = 0.0
    ruta_auditoria: Path | None = None
    for attempt in range(1, attempts + 1):
        onset = time.monotonic()
        if attempt > 1:
            room = settings.room_name(
                suffix=f"arnes-{time.strftime('%Y%m%d-%H%M%S')}-r{attempt - 1}"
            )
            print(f"\n{'=' * 70}\nReintento {attempt - 1} de {attempts - 1} — sala {room}\n{'=' * 70}")

        # Una grabadora por intento: cada reintento es otra llamada y mezclar su
        # audio con el de la anterior daría un WAV que no corresponde a ninguna.
        recorder = (
            AuditRecorder(harness_sample_rate=args.sample_rate, agent_sample_rate=LISTEN_SAMPLE_RATE)
            if args.record_audit
            else None
        )

        observed = asyncio.run(
            run_session(
                settings,
                room=room,
                script=script,
                publish_sample_rate=args.sample_rate,
                wait_after_clip=args.wait,
                recorder=recorder,
                # `poc` sin `escenario`/`repeticion` **a propósito**: el agente
                # avisa que la corrida queda sin etiquetar, que es exactamente lo
                # correcto: etiquetarla la volvería indistinguible de una de las 15.
                identification={"poc": 2},
                # Estado nuevo por intento: un reintento es otra llamada, y
                # arrastrar los montos de la anterior ramificaría con lo que
                # dijo un agente que ya colgó.
                estado=negotiation_state(script),
            )
        )
        duration = time.monotonic() - onset
        run_cost += estimate_cost(duration)
        faults = print_report(observed, room, publish_sample_rate=args.sample_rate)

        # La cuenta por intervención, en este proceso y no en el worker: hace
        # falta cuándo se oyó al agente, que solo el arnés mide. El worker ya
        # dejó las marcas y los ceros crudos en su JSON. Ver §2.2.5 del contrato.
        lines = wait_and_add_lines(observed, room)
        if lines:
            print_lines(lines)

        log_path = DIAGNOSTICS_PATH / f"event_log_{room}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log = event_log_de_la_corrida(observed, room, scenario=args.scenario)
        # Cómo se entregó el audio. Va en la bitácora del arnés y no en el JSON
        # del worker porque es de este lado del cable: el worker no puede saber
        # ni con qué ritmo salió la trama ni cuánto esperó en la cola.
        event_log["audio_delivery"] = {
            "pacing": observed.publishing_pacing,
            "source_tail": observed.source_tail,
            # Las series crudas, no solo el resumen: con percentiles de toda la
            # corrida no se puede atar el llenado de la cola al crecimiento del
            # sesgo del cero dentro de la llamada. Con las dos series y el
            # instante de cada clip, el cruce se hace después y sin gastar otra
            # llamada.
            "series": {
                "what_they_are": (
                    "una entrada por trama publicada, en el mismo orden: "
                    "`instante` es el envío medido (reloj de pared) y `cola_ms` "
                    "la ocupación de la AudioSource justo antes de empujar"
                ),
                "moment": [round(t, 6) for t in observed.measured_sends],
                "cola_ms": [round(c, 2) for c in observed.tail_per_frame],
            },
        }
        log_path.write_text(
            json.dumps(event_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  Lo que se reprodujo quedó en {log_path}")

        if recorder is not None:
            ruta_auditoria = _dump_audit(recorder, room)

        if observed.decisions:
            print(
                "  La corrida se ramificó: corrige la referencia del WER con esas\n"
                "  líneas, no con las del guion, o el WER mide contra otra frase."
            )

        resultado = evaluate_run(observed)
        if not resultado.retry or attempt == attempts:
            break

    if args.summary:
        # Antes de cualquier `return`: el lote necesita el resumen **también**
        # de las llamadas que salieron en rojo. Una corrida no utilizable es un
        # dato de la campaña —y de su costo—, no un hueco.
        from poc2_livekit.app.agent import RECORDINGS_PATH

        write_summary(
            Path(args.summary),
            call_summary(
                observed,
                resultado,
                poc=2,
                scenario=args.scenario,
                repetition=args.repetition,
                label=room,
                duration_s=duration,
                cost_usd=run_cost,
                failed_criteria=faults,
                lines=lines,
                overlap=measure_overlap(ruta_auditoria),
                return_path=measure_return_path(RECORDINGS_PATH / f"call_{room}.json"),
                artifacts={
                    "event_log": str(log_path),
                    "worker_json": str(RECORDINGS_PATH / f"call_{room}.json"),
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
        f"({resultado.motivo}).\n"
        "Por qué cerró cada turno (confianza del detector o vencimiento de la\n"
        "espera larga) está en `livekit_extra.end_of_turn` del JSON del worker."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
