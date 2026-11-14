"""Cliente de Twilio Media Streams: le habla al `/ws` de PoC1 como lo haría Twilio.

Es la mitad de transporte del arnés de PoC1, el análogo de lo que en PoC2 hace
`AudioSource` sobre la sala de LiveKit. Abre un WebSocket contra el servidor de
`app/telephony.py`, manda la secuencia `connected` → `start` → N × `media` →
`stop`, y escucha lo que devuelve el `TwilioFrameSerializer` de pipecat.

**Sin credenciales y sin Twilio.** No se origina ninguna llamada: el `streamSid`
y el `callSid` los inventa este módulo con un prefijo que ningún SID real usa
(ver `HARNESS_SID`), así que una corrida de arnés no puede confundirse con una
llamada de la campaña ni en los logs ni en el JSON.

## El estímulo se manda tal cual, y eso es la regla del módulo

Los clips salen de `shared/audio/` ya degradados a banda telefónica, y lo que
viaja por el WebSocket son **exactamente** los bytes del `.ulaw` que registró el
manifiesto. No se decodifica ni se recodifica en ningún punto: μ-law → PCM16 →
μ-law daría un audio casi idéntico, pero rompería la única evidencia que hay de
que las 3 PoC reprodujeron el mismo estímulo, que es el hash. Si un clip llega
aquí sin sus bytes μ-law, el módulo **falla** en vez de convertirlo — ver
`ReproductorDeMediaStreams.play_step` y `verify_stimulus`.

Lo único que este módulo sintetiza es el ruido de confort de las pausas, que no
es estímulo: es el piso de la línea, y va en μ-law porque el medio es μ-law.

## Lo que NO hace, a propósito

- **No marca a nadie.** No importa el SDK de Twilio ni toca `app/call.py`.
- **No mide latencia.** Fecha sus propias tramas para poder demostrar que salen
  a 20 ms, y nada más: el cero del turno y las latencias que se reportan los
  escribe el pipeline, no el arnés.
- **No decide el turno.** Eso es `shared/carrier_simulator/player.py`.

## De dónde sale `websockets`

Entra con el extra `websocket` de pipecat y ya está fijado en `uv.lock` (16.1.1).
**No se declara en `pyproject.toml` a propósito**, aunque este módulo lo importe
directo y la convención del repo sea declararlo: tocar el `pyproject` obliga a
re-generar el lock, y eso cambiaría el ambiente documentado de PoC1 por una
necesidad del arnés. Es el mismo criterio con el que `shared/audio/resampling.py`
se quedó sin `soxr` en PoC2. Si algún día PoC1 deja de traer el extra, este
import se rompe de forma ruidosa, que es la falla correcta.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from poc1_pipecat.app.config import force_utf8_output
from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw, ulaw_a_pcm16
from shared.carrier_simulator import (
    MS_PER_FRAME,
    FramePacing,
    HarnessFailure,
    Observed,
    ScriptStep,
    Player,
    comfort_noise_frame,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# El protocolo
# --------------------------------------------------------------------------

# Los eventos que Twilio manda al servidor.
EVENT_CONNECTED: Final[str] = "connected"
EVENT_START: Final[str] = "start"
EVENT_MEDIA: Final[str] = "mean"
EVENT_END: Final[str] = "stop"
EVENT_MARK: Final[str] = "mark"

# Los que el servidor le devuelve a Twilio. `clear` es el que importa: el
# serializer de pipecat lo emite en cada barge-in para que Twilio tire el audio
# que todavía no reprodujo. Un cliente que no lo espere se cae justo en el
# momento más interesante de la llamada — cuando el agente se interrumpe.
EVENT_CLEAR: Final[str] = "clear"

# μ-law de 8 kHz, un byte por muestra. La trama de 20 ms son 160 bytes, que es
# el tamaño exacto de lo que manda Twilio.
SAMPLES_PER_FRAME: Final[int] = TELEPHONY_SAMPLE_RATE * MS_PER_FRAME // 1000
BYTES_PER_FRAME: Final[int] = SAMPLES_PER_FRAME

MEDIA_ENCODING: Final[str] = "audio/x-mulaw"

# Prefijos de los identificadores que este módulo inventa. Los de Twilio son
# `MZ`/`CA`/`AC` seguidos de 32 hexadecimales; estos dicen en el propio texto de
# dónde salieron, para que un `grep` en los logs los separe sin ambigüedad.
HARNESS_SID: Final[str] = "HARNESS"


def connected_message() -> dict[str, Any]:
    """El saludo. Twilio lo manda antes que nada y no lleva `streamSid`."""
    return {"event": EVENT_CONNECTED, "protocol": "Call", "version": "1.0.0"}


def start_message(
    *,
    stream_sid: str,
    call_sid: str,
    account_sid: str,
    sequence: int,
    parameters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """El `start`, que es donde viajan los `<Parameter>` del TwiML.

    `app/telephony.py` lee `streamSid` y `callSid` de aquí a mano —los necesita
    para construir el serializer antes de entregarle el socket al transporte— y
    saca la etiqueta de la corrida de `customParameters`. Si falta cualquiera de
    los dos, cierra el WebSocket sin decir por qué, así que los tres campos van
    siempre.
    """
    start: dict[str, Any] = {
        "accountSid": account_sid,
        "streamSid": stream_sid,
        "callSid": call_sid,
        "tracks": ["inbound"],
        "mediaFormat": {
            "encoding": MEDIA_ENCODING,
            "sampleRate": TELEPHONY_SAMPLE_RATE,
            "channels": 1,
        },
    }
    if parameters:
        start["customParameters"] = {str(k): str(v) for k, v in parameters.items()}
    return {
        "event": EVENT_START,
        "sequenceNumber": str(sequence),
        "start": start,
        "streamSid": stream_sid,
    }


def media_message(
    *, stream_sid: str, sequence: int, piece: int, ulaw: bytes
) -> dict[str, Any]:
    """Una trama de audio. `timestamp` es el ms de stream en que empieza.

    El `payload` es base64 de los bytes μ-law **sin tocar**: el único cambio
    entre el archivo y el cable es el base64, que es reversible por definición.
    """
    return {
        "event": EVENT_MEDIA,
        "sequenceNumber": str(sequence),
        "mean": {
            "track": "inbound",
            "chunk": str(piece),
            "timestamp": str((piece - 1) * MS_PER_FRAME),
            "payload": base64.b64encode(ulaw).decode("ascii"),
        },
        "streamSid": stream_sid,
    }


def end_message(
    *, stream_sid: str, call_sid: str, account_sid: str, sequence: int
) -> dict[str, Any]:
    """El `stop`. Twilio lo manda cuando la llamada termina de su lado."""
    return {
        "event": EVENT_END,
        "sequenceNumber": str(sequence),
        "stop": {"accountSid": account_sid, "callSid": call_sid},
        "streamSid": stream_sid,
    }


def message_audio(message: Mapping[str, Any]) -> bytes:
    """Los bytes μ-law de un `media`, vengan de donde vengan."""
    payload = (message.get("mean") or {}).get("payload") or ""
    return base64.b64decode(payload) if payload else b""


# --------------------------------------------------------------------------
# Tramas
# --------------------------------------------------------------------------


def noise_frame_in_ulaw() -> bytes:
    """El mismo ruido de confort de PoC2, en el dominio del medio.

    Se genera en PCM16 con `shared/carrier_simulator/audio.py` —para que el
    nivel salga de la misma calibración en las 3 PoC— y se codifica aquí. Esto
    **sí** se codifica y no rompe nada: el ruido no es el estímulo, no tiene
    hash en el manifiesto y no entra en ningún WER.

    Nunca ceros: μ-law no tiene un cero exacto, y una pausa de bytes constantes
    es una señal que ningún teléfono produce.
    """
    return pcm16_a_ulaw(
        np.frombuffer(comfort_noise_frame(SAMPLES_PER_FRAME), dtype=np.int16)
    )


def split_ulaw_into_frames(ulaw: bytes) -> list[bytes]:
    """Parte los bytes μ-law en tramas de 20 ms, rellenando la última.

    La cola se completa con ruido de confort y no con silencio, por la misma
    razón que el relleno entre clips. Los bytes del clip **no se tocan**: la
    concatenación de las tramas empieza por `ulaw` byte a byte.
    """
    if not ulaw:
        return []
    frames = [
        ulaw[i : i + BYTES_PER_FRAME] for i in range(0, len(ulaw), BYTES_PER_FRAME)
    ]
    missing_items = BYTES_PER_FRAME - len(frames[-1])
    if missing_items:
        frames[-1] = frames[-1] + noise_frame_in_ulaw()[:missing_items]
    return frames


def verify_stimulus(
    steps: Iterable[ScriptStep], folder: Path | str
) -> list[str]:
    """Compara los bytes que se van a mandar contra el hash del manifiesto.

    Es la verificación que hace comparable la corrida: si el `.ulaw` de disco no
    es el que el manifiesto registró, PoC1 estaría reproduciendo otro audio que
    PoC2, y el WER de los dos mediría cosas distintas sin que nada se vea roto.

    Corre **antes** de conectarse y devuelve la lista de líneas del informe.
    Levanta `FallaDelArnes` al primer desacuerdo: seguir sería gastar la llamada
    para producir un número que no se puede citar.
    """
    from shared.audio.manifest import MANIFEST_NAME, load_manifest

    ruta = Path(folder) / MANIFEST_NAME
    if not ruta.is_file():
        raise HarnessFailure(
            f"no hay manifiesto en {ruta}: sin él no hay contra qué comparar los "
            "bytes del estímulo. Corre `python -m shared.audio.cli preparar`"
        )
    of_manifest = load_manifest(ruta).by_id()

    text_lines: list[str] = []
    for step in steps:
        if step.is_silence:
            continue
        if not step.ulaw:
            raise HarnessFailure(
                f"{step.turn} ({step.clip_id or 'sin id'}) no trae bytes μ-law. "
                "PoC1 manda μ-law por el cable y este módulo no recodifica desde "
                "PCM: el hash del manifiesto es la evidencia de que las 3 PoC "
                "reprodujeron el mismo audio, y recodificar la anularía"
            )
        entry = of_manifest.get(step.clip_id)
        if entry is None:
            raise HarnessFailure(
                f"{step.clip_id} no está en el manifiesto {ruta}: no se puede "
                "verificar que sea el mismo clip que reprodujo PoC2"
            )
        # Se verifica el **archivo fuente**, no lo que sale al cable. Desde
        # §2.2.9 el arnés le suma el piso de ruido al clip antes de publicarlo,
        # así que los bytes publicados no pueden coincidir con el manifiesto por
        # construcción. Lo que la evidencia tiene que sostener sigue siendo lo
        # mismo —que las 3 PoC parten del mismo estímulo—, y el piso es una
        # función determinista de ese estímulo.
        source = step.ulaw_source or step.ulaw
        measured = hashlib.sha256(source).hexdigest()
        if measured != entry.sha256_ulaw:
            raise HarnessFailure(
                f"{step.clip_id}: el .ulaw de disco no es el del manifiesto.\n"
                f"        manifiesto {entry.sha256_ulaw}\n"
                f"        disco      {measured}\n"
                "        Vuelve a preparar los clips; reproducir este dejaría a "
                "PoC1 midiendo sobre un audio distinto al de PoC2"
            )
        published = hashlib.sha256(step.ulaw).hexdigest()
        text_lines.append(
            f"  {step.turn:<4} {step.clip_id:<14} {len(step.ulaw):>6} B  "
            f"sha256 {measured[:12]}…  ok"
            + (f"  (publicado con piso {published[:12]}…)" if step.ulaw_source else "")
        )
    return text_lines


# --------------------------------------------------------------------------
# El ritmo
# --------------------------------------------------------------------------
#
# `RitmoDeTramas` vive en `shared/carrier_simulator/ritmo.py` y lo lleva el
# reproductor, no el cliente: el instante de envío tiene que ser el mismo
# concepto en las 3 PoC, y la única forma de garantizarlo es tomarlo en el bucle
# compartido. El cliente solo lo expone para su CLI.


# --------------------------------------------------------------------------
# El cliente
# --------------------------------------------------------------------------


class MediaStreamsClient:
    """Un WebSocket hablando el protocolo de Twilio, sin Twilio.

    El envío lo maneja quien tenga el ritmo (`ReproductorDeMediaStreams`); aquí
    solo se serializa y se lleva la cuenta. La recepción corre en una tarea
    aparte porque el servidor manda cuando quiere, y bloquear el envío para leer
    metería en el audio el jitter de la respuesta.
    """

    def __init__(
        self,
        url: str,
        *,
        identification: Mapping[str, str] | None = None,
        on_receive_audio: Callable[[bytes], None] | None = None,
        on_clear: Callable[[], None] | None = None,
        on_receive_message: Callable[[Mapping[str, Any]], None] | None = None,
        on_other_side_closed: Callable[[str], None] | None = None,
        label: str = "",
    ) -> None:
        self.url = url
        self.identification = dict(identification or {})
        self._on_receive_audio = on_receive_audio
        self._on_clear = on_clear
        # Los eventos que no son de Twilio. Pipecat serializa los
        # `OutputTransportMessage*Frame` como JSON crudo por este mismo
        # WebSocket, así que es por donde el arnés recibe la transcripción.
        self._on_receive_message = on_receive_message
        # Se avisa cuando el servidor cierra el WebSocket por su cuenta, que es
        # lo que pasa cuando el agente cierra la negociación y cuelga. Sin este
        # aviso el arnés le sigue hablando a un pipeline que ya no existe, y aquí
        # además revienta el bucle de publicación sobre un socket muerto.
        self._on_other_side_closed = on_other_side_closed
        # `True` en cuanto el otro lado se va. A partir de ahí `enviar_trama` no
        # levanta: empujar contra un socket cerrado no es un error del arnés.
        self.remote_closed = False

        suffix = label or time.strftime("%Y%m%d-%H%M%S")
        self.stream_sid = f"{HARNESS_SID}-stream-{suffix}"
        self.call_sid = f"{HARNESS_SID}-call-{suffix}"
        self.account_sid = f"{HARNESS_SID}-account"

        self.pacing: FramePacing | None = None
        self.sent_frames = 0
        # Las que se quisieron publicar después de que el agente colgó. Va al
        # reporte: un número grande aquí significa que el guion siguió corriendo
        # contra nadie, y esos turnos no miden nada.
        self.frames_without_destination = 0
        self.received_frames = 0
        self.bytes_received = 0
        # Cuántas veces el serializer pidió tirar lo que no se había reproducido.
        # En una llamada real es el conteo de barge-ins vistos desde el cable.
        self.cleanups = 0
        self.received_events: list[str] = []
        self.unknown_events: list[str] = []
        self.incidents: list[str] = []

        self._ws: Any = None
        self._sequence = 0
        self._chunk = 0
        self._lector: asyncio.Task[None] | None = None
        self._closed = False

    # -- Ciclo de vida --

    async def begin(self) -> None:
        """Conecta y manda `connected` + `start`. Deja el lector corriendo."""
        from websockets.asyncio.client import connect

        self._ws = await connect(self.url, max_size=None)
        await self._send(connected_message())
        await self._send(
            start_message(
                stream_sid=self.stream_sid,
                call_sid=self.call_sid,
                account_sid=self.account_sid,
                sequence=self._next_sequence(),
                parameters=self.identification,
            )
        )
        self._lector = asyncio.create_task(self._leer())

    async def shut(self, *, send_stop: bool = True) -> None:
        """Manda `stop` y cierra. Idempotente: colgar dos veces no es un error."""
        if self._closed:
            return
        self._closed = True
        if self._ws is not None and send_stop:
            try:
                await self._send(
                    end_message(
                        stream_sid=self.stream_sid,
                        call_sid=self.call_sid,
                        account_sid=self.account_sid,
                        sequence=self._next_sequence(),
                    )
                )
            except Exception as error:  # noqa: BLE001 - colgar no debe tumbar nada
                self.incidents.append(f"el `stop` no salió: {error!r}")
        if self._lector is not None:
            self._lector.cancel()
            await asyncio.gather(self._lector, return_exceptions=True)
        if self._ws is not None:
            await self._ws.close()

    async def __aenter__(self) -> "MediaStreamsClient":
        await self.begin()
        return self

    async def __aexit__(self, *_exception: Any) -> None:
        await self.shut()

    # -- Envío --

    async def send_frame(self, ulaw: bytes) -> None:
        """Manda una trama de 20 ms. El ritmo lo pone quien llama."""
        if len(ulaw) != BYTES_PER_FRAME:
            raise HarnessFailure(
                f"la trama trae {len(ulaw)} bytes y Media Streams espera "
                f"{BYTES_PER_FRAME} (20 ms de μ-law a {TELEPHONY_SAMPLE_RATE} Hz)"
            )
        if self.remote_closed or self._closed:
            # El agente colgó. Se cuenta y se sigue: reventar aquí tumbaría el
            # bucle de publicación, y con él el reporte, la bitácora y el WAV de
            # una corrida que ya se pagó entera.
            self.frames_without_destination += 1
            return
        self._chunk += 1
        await self._send(
            media_message(
                stream_sid=self.stream_sid,
                sequence=self._next_sequence(),
                piece=self._chunk,
                ulaw=ulaw,
            )
        )
        self.sent_frames += 1

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def _send(self, message: Mapping[str, Any]) -> None:
        await self._ws.send(json.dumps(message))

    # -- Recepción --

    async def _leer(self) -> None:
        """Consume lo que devuelve el servidor hasta que se corte."""
        motivo = "el servidor cerró el Media Stream"
        try:
            async for raw_value in self._ws:
                self._dispatch(raw_value)
        except asyncio.CancelledError:
            # Somos nosotros colgando: no hay nada que avisar.
            raise
        except Exception as error:  # noqa: BLE001 - el arnés reporta, no propaga
            # Un cierre del otro lado es normal cuando el agente cuelga; lo que
            # no puede pasar es que se pierda sin quedar anotado.
            motivo = f"la lectura del WebSocket terminó: {error!r}"
            self.incidents.append(motivo)
        self._warn_that_closed(motivo)

    def _warn_that_closed(self, motivo: str) -> None:
        """El otro lado se fue. Idempotente: avisar dos veces no es un aviso."""
        if self.remote_closed:
            return
        self.remote_closed = True
        if self._on_other_side_closed is not None:
            try:
                self._on_other_side_closed(motivo)
            except Exception as error:  # noqa: BLE001 - el arnés reporta, no propaga
                self.incidents.append(f"el aviso de cierre falló: {error!r}")

    def _dispatch(self, raw_value: str | bytes) -> None:
        try:
            message = json.loads(raw_value)
        except (ValueError, TypeError):
            self.incidents.append(f"llegó algo que no es JSON: {raw_value!r:.80}")
            return
        if not isinstance(message, Mapping):
            self.incidents.append(f"llegó un JSON que no es objeto: {message!r:.80}")
            return

        event = str(message.get("event", ""))
        self.received_events.append(event)

        if event == EVENT_MEDIA:
            data = message_audio(message)
            self.received_frames += 1
            self.bytes_received += len(data)
            if data and self._on_receive_audio is not None:
                self._on_receive_audio(data)
        elif event == EVENT_CLEAR:
            # Barge-in: el serializer le pide a Twilio que tire lo que no ha
            # reproducido. Aquí no hay buffer que tirar —el arnés escucha, no
            # reproduce— así que se cuenta y se sigue. Lo importante es **no
            # romperse**: esto llega justo cuando el agente se interrumpe, que
            # es el momento que más interesa observar.
            self.cleanups += 1
            if self._on_clear is not None:
                self._on_clear()
        elif event in (EVENT_MARK, EVENT_CONNECTED):
            pass
        elif self._on_receive_message is not None:
            # Pipecat serializa los `TransportMessageFrame` como JSON libre, así
            # que aquí puede aparecer cualquier cosa. Quien sepa qué hacer con
            # ella la recibe entera; el arnés manda por aquí su transcripción.
            try:
                self._on_receive_message(message)
            except Exception as error:  # noqa: BLE001 - el arnés reporta, no propaga
                self.incidents.append(f"el mensaje {event!r} no se pudo leer: {error!r}")
        else:
            # Se anota una vez por tipo y no se trata como error: un evento nuevo
            # del serializer no es razón para tumbar una llamada que ya corre.
            if event not in self.unknown_events:
                self.unknown_events.append(event)


# --------------------------------------------------------------------------
# El reproductor
# --------------------------------------------------------------------------


class MediaStreamsPlayer(Player):
    """El reproductor compartido, con μ-law como dominio de la trama.

    Hereda el ritmo, el relleno continuo y el fechado del fin del clip de
    `shared/carrier_simulator/`, para que PoC1 y PoC2 fechen el cero del turno
    igual. Lo que cambia es qué hay dentro de la trama: aquí son los bytes μ-law
    del manifiesto, sin convertir.
    """

    def __init__(
        self,
        client: MediaStreamsClient,
        *,
        observed: Observed,
        recorder: Any = None,
    ) -> None:
        super().__init__(
            sample_rate=TELEPHONY_SAMPLE_RATE, observed=observed, recorder=recorder
        )
        self.client = client
        # El cliente expone el mismo objeto para su CLI. El dueño es el
        # reproductor: el instante de envío se toma en el bucle compartido.
        client.pacing = self.pacing

    async def publish(self, raw_value: bytes) -> None:
        await self.client.send_frame(raw_value)

    def filler_frame(self) -> bytes:
        return noise_frame_in_ulaw()

    def as_pcm16(self, raw_value: bytes) -> bytes:
        """Para la auditoría. Es el **único** lugar que decodifica, y no manda
        nada al cable: lo que sale de aquí va a un WAV en disco."""
        return ulaw_a_pcm16(raw_value).tobytes()

    def clip_frames(self, samples: np.ndarray) -> list[bytes]:
        raise HarnessFailure(
            "PoC1 manda μ-law por el cable y este reproductor no recodifica "
            "desde PCM16: usa `play_step(paso)`, que toma los bytes del "
            "manifiesto tal cual. Recodificar daría un audio casi igual y un "
            "hash distinto, y el hash es la evidencia de que las 3 PoC "
            "reprodujeron el mismo estímulo"
        )

    async def play_step(self, step: ScriptStep) -> float:
        """Encola los bytes μ-law del paso. Devuelve el fin del clip local."""
        if not step.ulaw:
            raise HarnessFailure(
                f"{step.turn} ({step.clip_id or 'sin id'}) no trae bytes μ-law: "
                "cárgalo con `load_scenario`, que los lee del `.ulaw` de "
                "`shared/test_scenarios/clips`"
            )
        return await self._enqueue(split_ulaw_into_frames(step.ulaw))


# --------------------------------------------------------------------------
# Servidor de eco: con qué se verifica el cliente sin tocar PoC1
# --------------------------------------------------------------------------


@dataclass
class MediaStreamsEcho:
    """Un servidor que habla el mismo protocolo y devuelve lo que le llega.

    Existe para poder verificar el cliente **sin levantar el pipeline**: sin
    Deepgram, sin ElevenLabs, sin OpenAI y sin un centavo de gasto. Lo que se
    comprueba contra él es lo que puede fallar en silencio — que las tramas
    salgan a 20 ms y que los bytes lleguen sin tocarse.

    `clean_up_after` hace que mande un `clear` después de N tramas, que es lo que
    el serializer de pipecat emite en un barge-in: sirve para probar que el
    cliente lo tolera en vez de esperar a que pase en una llamada real.
    """

    return_audio: bool = True
    clean_up_after: int | None = None
    # Tramas basura que el eco manda antes que nada, para probar que el cliente
    # no se cae con eventos que no conoce.
    unusual_events: tuple[str, ...] = ()

    start: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    received: list[bytes] = field(default_factory=list)
    halted: bool = False
    _servidor: Any = None
    _port: int = 0
    _done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}/ws"

    @property
    def audio_received(self) -> bytes:
        """Todo el μ-law que llegó, en orden y concatenado."""
        return b"".join(self.received)

    async def __aenter__(self) -> "MediaStreamsEcho":
        from websockets.asyncio.server import serve

        self._servidor = await serve(self.handle, "127.0.0.1", 0)
        self._port = self._servidor.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exception: Any) -> None:
        self._servidor.close()
        await self._servidor.wait_closed()

    async def wait_for_end(self, tope_seg: float = 5.0) -> bool:
        """Espera al `stop`. Devuelve si llegó antes del tope."""
        try:
            await asyncio.wait_for(self._done.wait(), tope_seg)
        except TimeoutError:
            return False
        return True

    async def handle(self, ws: Any) -> None:
        """El handler del WebSocket. Público: `--run-server` lo monta por su cuenta."""
        stream_sid = ""
        for odd in self.unusual_events:
            await ws.send(json.dumps({"event": odd, "streamSid": stream_sid}))

        async for raw_value in ws:
            message = json.loads(raw_value)
            event = str(message.get("event", ""))
            self.events.append(event)

            if event == EVENT_START:
                self.start = dict(message.get("start") or {})
                stream_sid = str(message.get("streamSid", ""))
            elif event == EVENT_MEDIA:
                data = message_audio(message)
                self.received.append(data)
                if self.return_audio:
                    await ws.send(
                        json.dumps(
                            {
                                "event": EVENT_MEDIA,
                                "streamSid": stream_sid,
                                "mean": {
                                    "payload": base64.b64encode(data).decode("ascii")
                                },
                            }
                        )
                    )
                if (
                    self.clean_up_after is not None
                    and len(self.received) == self.clean_up_after
                ):
                    await ws.send(
                        json.dumps(
                            {"event": EVENT_CLEAR, "streamSid": stream_sid}
                        )
                    )
            elif event == EVENT_END:
                self.halted = True
                self._done.set()
                break


# --------------------------------------------------------------------------
# CLI: la comprobación de ritmo y de bytes, a mano y sin gastar
# --------------------------------------------------------------------------


async def _publish_against(
    url: str, seconds: float, clip: Path | None
) -> MediaStreamsClient:
    """Publica ruido (y un clip) contra `url` y devuelve el cliente ya cerrado."""
    observed = Observed(silent=True)
    client = MediaStreamsClient(url, label="comprobacion")
    player = MediaStreamsPlayer(client, observed=observed)
    await client.begin()

    halt = asyncio.Event()
    pump = asyncio.create_task(player.launch(halt))
    try:
        if clip is not None:
            step = ScriptStep(
                turn="T1",
                text="(comprobación)",
                ulaw=clip.read_bytes(),
                clip_id=clip.stem,
            )
            await asyncio.sleep(0.5)
            await player.play_step(step)
        await asyncio.sleep(max(0.0, seconds))
    finally:
        halt.set()
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
        await client.shut()
    return client


def _print_measurements(
    client: MediaStreamsClient,
    clip: Path | None,
    *,
    echo: MediaStreamsEcho | None,
    where: str,
) -> int:
    print("\n" + "=" * 70)
    print(f"Cliente de Media Streams — {where}")
    print("=" * 70)
    if echo is not None:
        print(f"  eventos vistos por el eco   {_count(echo.events)}")
    print(f"  tramas enviadas             {client.sent_frames}")
    print(f"  tramas devueltas            {client.received_frames}")
    print(f"  clear tolerados             {client.cleanups}")
    if client.unknown_events:
        print(f"  eventos no vistos antes     {client.unknown_events}")
    print("\n  ritmo de las tramas")
    print(client.pacing.as_text())

    faults = 0
    if clip is not None and echo is not None:
        ulaw = clip.read_bytes()
        position = echo.audio_received.find(ulaw)
        print("\n  bytes del estímulo")
        print(f"  clip                 {clip.name}, {len(ulaw)} B")
        print(f"  sha256 en disco      {hashlib.sha256(ulaw).hexdigest()}")
        print(
            f"  aparece en el cable  {'sí' if position >= 0 else 'NO'}"
            + (f", en el byte {position}" if position >= 0 else "")
        )
        if position < 0:
            print("  [FALLA] el clip no llegó byte a byte: algo lo recodificó")
            faults += 1
    if client.incidents:
        print("\n  incidentes")
        for incident in client.incidents:
            print(f"    {incident}")
    return faults


async def _run_check(seconds: float, clip: Path | None) -> int:
    """El eco **en este mismo proceso**: cómodo, pero no es la topología real.

    El eco contesta cada trama desde el mismo bucle de eventos que la publica,
    así que le agrega su propio trabajo al ritmo que se está midiendo. Sirve
    para verificar el protocolo y los bytes; para medir el ritmo, `--run-server` en
    otra terminal.
    """
    async with MediaStreamsEcho() as echo:
        client = await _publish_against(echo.url, seconds, clip)
        await echo.wait_for_end()
    return _print_measurements(
        client, clip, echo=echo, where="eco en el mismo proceso"
    )


async def _run_against_url(url: str, seconds: float, clip: Path | None) -> int:
    """Contra un servidor que ya está arriba: el eco de `--run-server`, o PoC1."""
    client = await _publish_against(url, seconds, clip)
    return _print_measurements(client, clip, echo=None, where=f"contra {url}")


async def _serve(port: int) -> int:
    """El eco, en su propio proceso. Es la topología de una llamada real."""
    from websockets.asyncio.server import serve

    echo = MediaStreamsEcho()
    async with serve(echo.handle, "127.0.0.1", port):
        print(f"Eco de Media Streams escuchando en ws://127.0.0.1:{port}/ws")
        print("Ctrl-C para parar. No gasta nada: no hay STT, ni TTS, ni LLM.")
        await asyncio.Future()
    return 0


def _count(events: list[str]) -> str:
    seen_items: dict[str, int] = {}
    for event in events:
        seen_items[event] = seen_items.get(event, 0) + 1
    return ", ".join(f"{k}×{v}" for k, v in seen_items.items()) or "(ninguno)"


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.media_streams",
        description=(
            "Comprueba el cliente de Media Streams contra un eco local. No "
            "toca Twilio, no levanta el pipeline y no gasta nada."
        ),
    )
    arg_parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="cuánto publicar (default 10). Súbelo para medir la deriva larga",
    )
    arg_parser.add_argument(
        "--clip",
        help="un .ulaw de shared/test_scenarios/clips, para verificar los bytes",
    )
    destination = arg_parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--run-server",
        type=int,
        metavar="PORT",
        help=(
            "en vez de publicar, levanta el eco en este puerto y se queda. Es la "
            "topología real —servidor en otro proceso— y es como hay que medir "
            "el ritmo: con el eco dentro del mismo bucle de eventos, su trabajo "
            "se suma al que se está midiendo"
        ),
    )
    destination.add_argument(
        "--url",
        help=(
            "publicar contra un servidor que ya está arriba (el eco de --run-server). "
            "Sin esto se levanta un eco en este mismo proceso"
        ),
    )
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if args.run_server is not None:
        try:
            return asyncio.run(_serve(args.run_server))
        except KeyboardInterrupt:  # pragma: no cover - depende de la terminal
            print("\nEco detenido.")
            return 0

    clip = Path(args.clip) if args.clip else None
    if clip is not None and not clip.is_file():
        print(f"[FALLA] no existe {clip}")
        return 1
    if args.url:
        return asyncio.run(_run_against_url(args.url, args.seconds, clip))
    return asyncio.run(_run_check(args.seconds, clip))


if __name__ == "__main__":
    raise SystemExit(main())
