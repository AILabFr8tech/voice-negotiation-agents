"""Humo de sala de PoC2: una sesión real del agente **sin telefonía**.

Contesta una sola pregunta, la que `app/smoke.py` deja abierta en sus dos niveles
(el 0 no toca la red, el 1 corre el LLM en texto sin sala):

    ¿puede un participante que NO es SIP entrar a la sala del agente,
    publicarle audio, y recibir de vuelta su voz y su transcripción?

**No es el arnés**: no reproduce clips, no mide latencia y no escribe métricas.
Solo empuja ruido de confort y reporta qué llegó de vuelta.

Va primero porque el plan del arnés cuelga de cuatro supuestos verificados
leyendo el SDK, no corriéndolo: que `wait_for_answer` deja pasar a un
no-SIP (`app/agent.py`), que `RoomIO` engancha `PARTICIPANT_KIND_STANDARD` por
default, que el agente publica su voz como pista de la sala y que transmite su
texto a `lk.transcription` en fragmentos (`room_io/_output.py`). Si los dos
últimos no se cumplen en vivo **el plan cambia**: sin la pista no hay forma de
fechar el primer audio del agente desde fuera, y sin la transcripción no hay cómo
distinguir un acuse de una respuesta sustantiva, porque PoC2 no toca el audio y
no hay señal de energía de dónde sacarla.

## Lo que este módulo NO hace, a propósito

- **No crea un participante SIP**: no necesita trunk, así que queda demostrado
  que el arnés no puede marcarle a nadie por accidente.
- **No toca el contrato de métricas**: no escribe en `latencias` ni en ninguna
  otra llave del JSON.
- **No cambia ninguna constante congelada**: la sesión se arma con
  `build_voice_session` tal cual.

## Ruido de confort, no silencio digital

El audio se publica con `dtx=False` **y** con ruido de confort a ~-59 dBFS. Las
dos cosas, por la misma razón y en dos capas distintas: con DTX de Opus activo,
tramas de ceros hacen que el encoder deje de mandar paquetes, y entonces el
`AudioStream` del agente no recibe tramas durante las pausas — Silero nunca
acumula los 0.55 s de silencio *observado* que pide su `min_silence_duration` y
el fin de turno sale tarde o no sale. `dtx=False` lo apaga en el publisher; el
ruido evita que el encoder clasifique la trama como silencio de todos modos.

El nivel del ruido es **de este humo**, no una constante de la comparativa: no
entra en ningún número reportado.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import time
from array import array
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes del protocolo de transcripción de LiveKit
# --------------------------------------------------------------------------
#
# Se declaran aquí y se **verifican** contra el SDK en `_verificar_constantes`,
# en vez de importarse. Dos razones:
#
# - el arnés definitivo debe depender solo de `livekit.rtc`, no de
#   `livekit.agents` (que es el SDK del worker, no del cliente);
# - si LiveKit les cambia el nombre, un import roto se ve; una constante
#   hardcodeada que ya no coincide **no se ve**: el handler simplemente nunca
#   se dispara y el humo reportaría "el agente no manda transcripción", que es
#   la conclusión equivocada.
TRANSCRIPT_TOPIC: Final[str] = "lk.transcription"
ATTRIBUTE_SEGMENT: Final[str] = "lk.segment_id"
ATTRIBUTE_FINAL: Final[str] = "lk.transcription_final"

# --------------------------------------------------------------------------
# Formato del audio que publica el arnés
# --------------------------------------------------------------------------
#
# 48 kHz mono es la tasa nativa de WebRTC. No hay que remuestrear a mano: el
# lado del agente pide 24 kHz mono en tramas de 50 ms
# (`RoomInputOptions.audio_sample_rate`) y la pila remuestrea sola.
#
# **Esto no es la tasa del arnés.** El arnés publica los clips en banda
# telefónica (8 kHz + μ-law) para que sean comparables con PoC1 y PoC3
# (`app/arnes.py`); este humo solo prueba el transporte.
PUBLISH_SAMPLE_RATE: Final[int] = 48_000
MS_PER_FRAME: Final[int] = 20
SAMPLES_PER_FRAME: Final[int] = PUBLISH_SAMPLE_RATE * MS_PER_FRAME // 1000

# Tasa a la que se lee el audio del agente. Se pide 24 kHz porque es la que el
# agente publica (`RoomOutputOptions.audio_sample_rate`): pedir otra cosa
# metería un remuestreo que no aporta nada al humo.
LISTEN_SAMPLE_RATE: Final[int] = 24_000

# Amplitud pico del ruido de confort, en cuentas de int16. Con distribución
# uniforme, RMS = pico / sqrt(3) ≈ 35, o sea ≈ -59.4 dBFS: por debajo de
# cualquier umbral de voz, y por encima de "esto es silencio, apaga el encoder".
COMFORT_NOISE_PEAK: Final[int] = 60

# Nombre de la pista del arnés. Se elige distinto de `roomio_audio` (la del
# agente) para poder distinguirlas de un vistazo en el dashboard.
TRACK_NAME: Final[str] = "harness_carrier"

# Ventana de observación por default. Se pasa de 90 s **a propósito**: el tope
# de `ESPERA_PARA_CONTESTAR_SEG` del agente es justo 90 s, y aunque leyendo el
# código se ve que ese temporizador nunca se arma para un participante no-SIP
# (la rama devuelve antes de crear el future), el criterio de éxito de esta
# fase pide verlo, no deducirlo.
DEFAULT_WAIT_S: Final[float] = 100.0

# Umbral de energía solo para el reporte de este humo: sirve para decir "de las
# tramas que llegaron del agente, cuántas traían voz". **No es una perilla de
# medición** y no se compara con `UMBRAL_DE_VOZ_RMS` de PoC3.
RMS_WITH_SPEECH: Final[float] = 200.0


class SmokeFailure(AssertionError):
    """Un supuesto del que depende el arnés no se cumple."""


# --------------------------------------------------------------------------
# Verificación de constantes contra el SDK
# --------------------------------------------------------------------------


def _verify_constants() -> str:
    """Compara las constantes de arriba con las del SDK del worker.

    Corre antes de conectarse, así que un rename del SDK se ve como una falla
    de configuración y no como "el agente no manda transcripción".
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
            raise SmokeFailure(
                f"el {label} de la transcripción cambió en el SDK: "
                f"nosotros usamos {our!r} y LiveKit publica en {of_sdk!r}"
            )
    return f"constantes de transcripción al día ({TRANSCRIPT_TOPIC})"


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def comfort_noise_frame(samples: int = SAMPLES_PER_FRAME) -> bytes:
    """PCM int16 mono de ruido uniforme a ~-59 dBFS.

    Deliberadamente **no** se usa `livekit.agents.utils.audio.silence_frame`:
    eso genera ceros, que es exactamente lo que hay que evitar (ver el
    docstring del módulo).
    """
    pico = COMFORT_NOISE_PEAK
    pcm_samples = array("h", (random.randint(-pico, pico) for _ in range(samples)))
    return pcm_samples.tobytes()


def rms_pcm16(raw_value: bytes) -> float:
    """RMS de un buffer PCM int16. Solo para el reporte de este humo."""
    samples = array("h")
    samples.frombytes(raw_value[: len(raw_value) - len(raw_value) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(m * m for m in samples) / len(samples))


# --------------------------------------------------------------------------
# Lo observado
# --------------------------------------------------------------------------


@dataclass
class TextFragment:
    """Un trozo de `lk.transcription` con su marca de llegada local.

    `cierre` distingue el trozo de texto del **cierre del segmento**. Hacen
    falta los dos: el atributo `lk.transcription_final` viaja en el trailer del
    stream, no en el header, y `TextStreamReader` solo lo funde en
    `info.attributes` **después** de que la iteración termina
    (`data_stream.py::_on_stream_close`). O sea que leído por trozo siempre
    dice "false"; el valor de verdad solo existe al cerrar.
    """

    t: float
    sender: str
    text: str
    segment: str
    final: bool
    closure: bool = False


@dataclass
class Observed:
    """Todo lo que el arnés vio, con el cero en el instante de la conexión."""

    start: float = field(default_factory=time.time)

    connected: bool = False
    published_track: bool = False
    published_frames: int = 0

    # Participantes que entraron y no somos nosotros. El agente es el que
    # importa; su identidad la asigna LiveKit y no se conoce de antemano.
    participants: dict[str, float] = field(default_factory=dict)

    # Audio del agente
    subscribed_tracks: list[str] = field(default_factory=list)
    audio_frames: int = 0
    t_first_audio_frame: float | None = None
    t_first_frame_with_speech: float | None = None
    rms: list[float] = field(default_factory=list)

    # Texto
    fragments: list[TextFragment] = field(default_factory=list)

    # Fallas del transporte
    incidents: list[str] = field(default_factory=list)

    def now(self) -> float:
        return round(time.time() - self.start, 3)

    def annotate_incident(self, detail: str) -> None:
        self.incidents.append(f"[{self.now():>7.3f}s] {detail}")
        logger.warning("%s", detail)

    def annotate_audio(self, raw_value: bytes) -> None:
        self.audio_frames += 1
        if self.t_first_audio_frame is None:
            self.t_first_audio_frame = self.now()
        energy = rms_pcm16(raw_value)
        self.rms.append(energy)
        if energy >= RMS_WITH_SPEECH and self.t_first_frame_with_speech is None:
            self.t_first_frame_with_speech = self.now()

    def annotate_fragment(
        self,
        sender: str,
        text: str,
        attribute_map: dict[str, str],
        *,
        closure: bool = False,
    ) -> None:
        fragment = TextFragment(
            t=self.now(),
            sender=sender,
            text=text,
            segment=str(attribute_map.get(ATTRIBUTE_SEGMENT, "")),
            final=str(attribute_map.get(ATTRIBUTE_FINAL, "")).lower() == "true",
            closure=closure,
        )
        self.fragments.append(fragment)
        if not closure:
            logger.info("[%s] %r", sender, text)

    # -- Lecturas derivadas --

    @property
    def agent_identity(self) -> str | None:
        """La primera identidad que entró y no es la del arnés."""
        for identity in self.participants:
            if identity != CARRIER_IDENTITY:
                return identity
        return None

    def agent_fragments(self) -> list[TextFragment]:
        """Los que manda el agente sobre sí mismo (no el STT de nuestro audio).

        LiveKit publica los dos por el mismo tópico y los distingue con el
        `sender_identity` del stream: el transcript del agente viaja con la
        identidad del agente y el STT de lo que dijo el transportista viaja con
        la del transportista (`room_io/room_io.py::_init_task`). Es una
        distinción que el arnés definitivo va a necesitar tal cual.
        """
        return [
            f
            for f in self.fragments
            if f.sender != CARRIER_IDENTITY and not f.closure
        ]

    def our_stt_fragments(self) -> list[TextFragment]:
        return [
            f
            for f in self.fragments
            if f.sender == CARRIER_IDENTITY and not f.closure
        ]

    def closed_segments(self) -> list[TextFragment]:
        return [f for f in self.fragments if f.closure]


# --------------------------------------------------------------------------
# Sesión
# --------------------------------------------------------------------------


def build_token(settings: Settings, room: str) -> str:
    """Token del arnés: entrar a `sala` como "transportista" y poder publicar.

    La identidad tiene que ser **exactamente** `CARRIER_IDENTITY`: el
    agente espera a ese nombre (`ctx.wait_for_participant(identity=…)`) y cuelga
    cuando ese nombre se desconecta. Con cualquier otra identidad el worker se
    queda esperando su tope de 90 s y no negocia nada.

    `can_publish` y `can_subscribe` ya son `True` por default en `VideoGrants`;
    se pasan explícitos porque son el permiso del que depende todo el arnés y
    no debe quedar como un default heredado que alguien cambie sin darse cuenta.
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
            )
        )
        .to_jwt()
    )


async def dispatch_agent(settings: Settings, room: str, identification: dict[str, Any]) -> str:
    """Mete al agente a la sala. **Sin `create_sip_participant`.**

    Es el paso 1 de `app/call.py::marcar` y nada más. El worker está
    registrado con `agent_name`, así que sin este dispatch explícito no entra a
    ninguna sala; y sin el paso 2 no hay teléfono que suene.
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


async def publish_noise(source: Any, observed: Observed, halt: asyncio.Event) -> None:
    """Empuja ruido de confort a ritmo real hasta que se pida parar.

    El ritmo se lleva con un reloj monótono propio en vez de confiar en la
    contrapresión de `capture_frame`. Dos razones:

    - `AudioSource.capture_frame` dice devolver cuando la trama ya se empujó,
      pero el ritmo efectivo depende del lado Rust del FFI; el arnés
      definitivo necesita saber **exactamente** en qué instante de reloj de
      pared terminó la voz de un clip, y eso solo es cierto si el ritmo lo
      pone quien publica;
    - un `sleep` contra deadline absoluto no acumula deriva, mientras que
      `sleep(0.02)` en un bucle sí.
    """
    from livekit import rtc

    trama_num = 0
    onset = time.monotonic()
    while not halt.is_set():
        raw_value = comfort_noise_frame()
        await source.capture_frame(
            rtc.AudioFrame(
                data=raw_value,
                sample_rate=PUBLISH_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
        )
        observed.published_frames += 1
        trama_num += 1

        target = onset + trama_num * (MS_PER_FRAME / 1000.0)
        lag = target - time.monotonic()
        if lag > 0:
            await asyncio.sleep(lag)


async def read_agent_audio(audio_track: Any, observed: Observed) -> None:
    """Cuenta y mide las tramas de la voz del agente que llegan a la sala."""
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
    except Exception as error:  # noqa: BLE001 - el humo reporta, no propaga
        observed.annotate_incident(f"la lectura del audio del agente falló: {error!r}")
    finally:
        await stream.aclose()


async def run_session(
    settings: Settings, *, room: str, wait_s: float, identification: dict[str, Any]
) -> Observed:
    """El humo completo: despacha, conecta, publica, escucha y se va."""
    from livekit import rtc

    # Referencias fuertes a las tareas de fondo. Una tarea sin dueño puede ser
    # recolectada a media ejecución; el agente de PoC2 ya tiene esta misma nota
    # sobre su tarea de cuelgue.
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
    # comparables entre sí. Reajustarlo después de `connect` invalidaría las de
    # los participantes que ya estaban dentro.
    observed = Observed()

    rtc_room = rtc.Room()

    # --- Handlers, ANTES de conectar ---
    # Registrarlos después dejaría pasar los eventos de los participantes que
    # ya estaban dentro, y el agente puede entrar antes que nosotros.

    @rtc_room.on("participant_connected")
    def _on_enter(participant: Any) -> None:
        identity = str(getattr(participant, "identity", ""))
        kind = getattr(participant, "kind", None)
        observed.participants[identity] = observed.now()
        class_name = (
            rtc.ParticipantKind.Name(kind) if isinstance(kind, int) else str(kind)
        )
        print(f"  [{observed.now():>7.3f}s] entró {identity!r} ({class_name})")

    @rtc_room.on("participant_disconnected")
    def _on_exit(participant: Any) -> None:
        identity = str(getattr(participant, "identity", ""))
        print(f"  [{observed.now():>7.3f}s] salió {identity!r}")

    @rtc_room.on("track_subscribed")
    def _on_subscribe(audio_track: Any, publishing: Any, participant: Any) -> None:
        identity = str(getattr(participant, "identity", ""))
        name = str(getattr(publishing, "name", ""))
        if getattr(audio_track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        observed.subscribed_tracks.append(f"{identity}/{name}")
        print(f"  [{observed.now():>7.3f}s] suscrito a audio de {identity!r} ({name!r})")
        _adopt(asyncio.create_task(read_agent_audio(audio_track, observed)))

    @rtc_room.on("disconnected")
    def _on_disconnect(motivo: Any) -> None:
        observed.annotate_incident(f"la sala nos desconectó: {motivo!r}")

    async def _leer_stream(lector: Any, sender: str) -> None:
        info = getattr(lector, "info", None)
        try:
            async for piece in lector:
                attribute_map = dict(getattr(info, "attributes", None) or {})
                observed.annotate_fragment(sender, piece, attribute_map)
            # El trailer ya se fundió en `info.attributes`: aquí, y solo aquí,
            # se puede leer `lk.transcription_final` de verdad.
            observed.annotate_fragment(
                sender, "", dict(getattr(info, "attributes", None) or {}), closure=True
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            observed.annotate_incident(f"la lectura de un stream de texto falló: {error!r}")

    def _on_text_received(lector: Any, sender: str) -> None:
        # El handler que registra LiveKit es **sync** (`TextStreamHandler =
        # Callable[[TextStreamReader, str], None]`), así que la lectura tiene
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
    try:
        # --- Publicar la pista del arnés ---
        source = rtc.AudioSource(PUBLISH_SAMPLE_RATE, 1, queue_size_ms=200)
        audio_track = rtc.LocalAudioTrack.create_audio_track(TRACK_NAME, source)
        publishing = await rtc_room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE,
                # Ver el docstring del módulo: con DTX activo, las pausas dejan
                # de mandar paquetes y el endpointing del agente se rompe.
                dtx=False,
            ),
        )
        observed.published_track = True
        print(
            f"  [{observed.now():>7.3f}s] pista publicada: {publishing.sid} "
            f"({PUBLISH_SAMPLE_RATE} Hz mono, dtx=False)"
        )

        _adopt(asyncio.create_task(publish_noise(source, observed, halt)))

        print(f"\n  observando {wait_s:.0f} s (Ctrl+C para cortar antes) …\n")
        try:
            await asyncio.wait_for(halt.wait(), timeout=wait_s)
        except asyncio.TimeoutError:
            pass
    except KeyboardInterrupt:  # pragma: no cover - depende de la terminal
        print("\n  cortado a mano.")
    finally:
        halt.set()
        for tarea in list(tasks):
            tarea.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Desconectarnos es lo que cierra la sesión del agente
        # (`close_on_disconnect`) y dispara su `delete_room` — o sea, lo que
        # hace que el worker escriba su JSON y no se quede vivo hasta el
        # `empty_timeout` de la sala.
        await rtc_room.disconnect()
        print(f"  [{observed.now():>7.3f}s] desconectados.")

    return observed


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


def print_report(observed: Observed, room: str) -> int:
    """Imprime lo observado y devuelve el número de criterios que fallaron."""
    print("\n" + "=" * 70)
    print(f"Humo de sala — {room}")
    print("=" * 70)

    print(f"  conectado a la sala   {'sí' if observed.connected else 'NO'}")
    print(f"  pista publicada       {'sí' if observed.published_track else 'NO'}")
    print(f"  tramas publicadas     {observed.published_frames} "
          f"(~{observed.published_frames * MS_PER_FRAME / 1000:.1f} s de ruido)")

    agent = observed.agent_identity
    print(f"  agente en la sala     {agent!r}" if agent else "  agente en la sala     NO ENTRÓ")
    if observed.participants:
        for identity, t in observed.participants.items():
            print(f"                        [{t:>7.3f}s] {identity}")

    print(f"  pistas suscritas      {observed.subscribed_tracks or '(ninguna)'}")
    print(f"  tramas de audio       {observed.audio_frames}")
    if observed.t_first_audio_frame is not None:
        print(f"  primera trama         {observed.t_first_audio_frame:.3f}s")
    if observed.t_first_frame_with_speech is not None:
        print(
            f"  primera con voz       {observed.t_first_frame_with_speech:.3f}s "
            f"(RMS >= {RMS_WITH_SPEECH:.0f})"
        )
    if observed.rms:
        with_voice = [r for r in observed.rms if r >= RMS_WITH_SPEECH]
        print(
            f"  energía del agente    RMS máx {max(observed.rms):,.0f} | "
            f"{len(with_voice)}/{len(observed.rms)} tramas con voz "
            f"({100 * len(with_voice) / len(observed.rms):.0f}%)"
        )

    of_agent = observed.agent_fragments()
    of_our_stt = observed.our_stt_fragments()
    closures = observed.closed_segments()
    print(f"\n  fragmentos de {TRANSCRIPT_TOPIC}")
    print(f"    del agente          {len(of_agent)}")
    print(f"    STT de lo nuestro   {len(of_our_stt)}")
    print(
        f"    segmentos cerrados  {len(closures)} "
        f"({sum(1 for c in closures if c.final)} marcados final)"
    )

    if observed.fragments:
        print("\n  transcripción observada (marca local, remitente, texto)")
        for fragment in observed.fragments:
            if fragment.closure:
                mark = "FIN  " if fragment.final else "cutoff"
                print(
                    f"    [{fragment.t:>7.3f}s] {mark} {fragment.sender:<24} "
                    f"— cierra segmento {fragment.segment or '?'}"
                )
                continue
            print(
                f"    [{fragment.t:>7.3f}s]   ··· {fragment.sender:<24} "
                f"{fragment.text!r}"
            )

    if observed.incidents:
        print("\n  incidentes")
        for incident in observed.incidents:
            print(f"    {incident}")

    # --- Criterios de éxito de la Fase 0 ---
    print("\n  criterios de la Fase 0")
    criteria = (
        (
            "el participante no-SIP entró y el agente arrancó sin esperar telefonía",
            agent is not None,
            "el agente nunca entró a la sala: revisa que el worker esté arriba y "
            "que LIVEKIT_AGENT_NAME coincida",
        ),
        (
            "el agente no murió esperando SIP",
            agent is not None and agent in observed.participants,
            "no hay evidencia de que el agente siguiera en la sala",
        ),
        (
            "llegó audio del agente a la sala",
            observed.audio_frames > 0,
            "NO llegó una sola trama de audio del agente: sin esto no se puede "
            "fechar su primer audio por observación externa",
        ),
        (
            f"llegó al menos un fragmento de {TRANSCRIPT_TOPIC} del agente",
            len(of_agent) > 0,
            "NO llegó transcripción del agente: sin esto no hay forma de "
            "distinguir un acuse de una respuesta sustantiva en PoC2",
        ),
    )

    faults = 0
    for label, ok, detail in criteria:
        if ok:
            print(f"    [ok]    {label}")
        else:
            faults += 1
            print(f"    [FALLA] {label}")
            print(f"            {detail}")

    print(
        f"\n  El worker deja su JSON en poc2_livekit/grabaciones/llamada_{room}.json"
        "\n  Es una corrida de humo, NO de campaña: no la dejes en esa carpeta si"
        "\n  vas a correr `shared.metrics.cli calcular` sobre ella."
    )
    return faults


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def estimate_cost(wait_s: float) -> float:
    """Costo aproximado de una corrida. Sin trunk: aquí no hay telefonía.

    Las tarifas salen de `app/call.py`, que las tiene verificadas contra
    `docs/costos_poc2.md`. El TTS se cobra completo a propósito: el agente solo
    dice su saludo, así que es una cota alta y no una estimación fina.
    """
    minutes = wait_s / 60.0
    per_minute = STT_COST_PER_MINUTE + ESTIMATED_LIVEKIT_COST_PER_MINUTE
    return minutes * per_minute + TTS_COST_PER_CALL + LLM_COST_PER_CALL


def _print_cost_warning(settings: Settings, room: str, wait_s: float) -> None:
    print("Sesión real del agente de PoC2, SIN telefonía.\n")
    print("  lo que SÍ se gasta")
    print("    STT       Deepgram, sobre el ruido de confort que publica el arnés")
    print("    TTS       ElevenLabs, el saludo del agente")
    print("    LLM       gpt-4.1-mini, la apertura del prompt v0")
    print("    LiveKit   minutos de participante (estimación sin verificar)")
    print("\n  lo que NO se gasta")
    print("    Twilio    no se crea participante SIP: ningún teléfono suena")
    print(f"\n  sala             {room}")
    print(f"  proyecto         {settings.livekit_url}")
    print(f"  agente           {settings.agent_name}")
    print(f"  ventana          {wait_s:.0f} s")
    print(f"\n  estimado         ~${estimate_cost(wait_s):.2f} USD (cota alta)")
    print("\nVuelve a correrlo con --accept-cost para ejecutarlo.")


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.smoke_room",
        description=(
            "Humo de sala de PoC2: mete un participante no-SIP a la sala del "
            "agente, le publica ruido de confort y reporta qué audio y qué "
            "transcripción vuelven. Requiere el worker arriba en otra terminal."
        ),
    )
    arg_parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_S,
        help=(
            f"ventana de observación en segundos (default {DEFAULT_WAIT_S:.0f}, "
            "por encima del tope de 90 s del agente a propósito)"
        ),
    )
    arg_parser.add_argument(
        "--accept-cost", action="store_true", help="obligatorio para ejecutar"
    )
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_voice(settings)
    # `verificar_para_telefonia` NO se llama: este camino no necesita trunk.

    try:
        print(_verify_constants())
    except SmokeFailure as error:
        print(f"[FALLA] {error}")
        return 1

    # Sin `escenario`/`repeticion`: el nombre de la sala lleva "humo" para que
    # sea imposible confundir esta corrida con una de las 15 de la campaña,
    # tanto en el dashboard de LiveKit como en el nombre del JSON.
    room = settings.room_name(suffix=f"humo-{time.strftime('%Y%m%d-%H%M%S')}")

    if not args.accept_cost:
        print()
        _print_cost_warning(settings, room, args.wait)
        return 0

    observed = asyncio.run(
        run_session(
            settings,
            room=room,
            wait_s=args.wait,
            # `poc` sin `escenario`/`repeticion`: el agente va a avisar que la
            # corrida queda sin etiquetar, que es exactamente lo correcto.
            identification={"poc": 2},
        )
    )
    faults = print_report(observed, room)

    print()
    if faults:
        print(f"{faults} criterio(s) de la Fase 0 fallaron.")
        return 1
    print("Fase 0 verde: el arnés de clips es construible sobre este camino.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
