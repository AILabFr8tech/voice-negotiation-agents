"""Cliente de Twilio Media Streams: le habla al `/ws` de PoC3 como lo haría Twilio.

Es la mitad de transporte del arnés de PoC3. Abre un WebSocket contra el
servidor de `app/telephony.py`, manda la secuencia `connected` → `start` →
N × `media` → `stop`, y escucha lo que el puente devuelve.

**Sin credenciales de Twilio y sin marcar.** El `streamSid` y el `callSid` los
inventa este módulo con un prefijo que ningún SID real usa (`HARNESS_SID`),
así que una corrida de arnés no se puede confundir con una llamada de la
campaña ni en los logs ni en el JSON. Ojo con lo que **sí** cuesta: el puente
que hay del otro lado abre una sesión real contra el alpha, y aunque el tráfico
a `gpt-live` no se factura durante el alpha, **el modelo delegado sí**. Una
corrida de arnés de PoC3 no es gratis; lo gratis es probar este módulo contra
`EcoDeMediaStreams`.

## El estímulo se manda tal cual, y eso es la regla del módulo

Los clips salen de `shared/audio/` ya degradados a banda telefónica, y lo que
viaja por el WebSocket son **exactamente** los bytes del `.ulaw` que registró el
manifiesto. No se decodifica ni se recodifica en ningún punto: μ-law → PCM16 →
μ-law daría un audio casi idéntico, pero rompería la única evidencia de que las
3 PoC reprodujeron el mismo estímulo, que es el hash.

Lo único que este módulo sintetiza es el ruido de confort de las pausas, que no
es estímulo: es el piso de la línea, y va en μ-law porque el medio es μ-law.

## Por qué es una copia de `poc1_pipecat/app/media_streams.py`

Porque el transporte es el mismo —Twilio Media Streams— y el arnés de PoC1 ya
lo resolvió. Lo que **no** se hace es importarlo: vive bajo `poc1_pipecat/`, y
jalarlo desde aquí acoplaría el entorno de una arquitectura al de otra, que es
justo lo que la comparativa no puede permitirse (el mismo criterio con el que
`app/marks.py` duplica sus lectores). Lo que sí está compartido, y es lo único
que tiene que coincidir, es el ritmo y el reproductor de
`shared/carrier_simulator/`, **sin modificar**: `Reproductor` se subclasea
sobreescribiendo cinco ganchos y `play_script` se usa tal cual.

## En qué se aparta del de PoC1, por el otro lado del cable

**1. El `clear` no va a llegar nunca.** Pipecat emite `clear` en cada barge-in
para que Twilio tire el audio que no reprodujo, así que en PoC1 contar
`limpiezas` es contar barge-ins vistos desde el cable. El puente de PoC3 manda
**un solo tipo de evento hacia Twilio**, `media`
(`telefonia.PuenteDeLlamada._handle_audio`, única llamada a `send_text`): ni
`clear`, ni `mark`. El cliente lo sigue tolerando y contando —cuesta nada y el
día que el puente crezca se entera— pero `limpiezas == 0` en PoC3 **no es
evidencia de que no hubo barge-in**, es que el canal no lo reporta. Es la misma
familia de asimetría que la de `end_of_turn_verdicts` en PoC2: visibilidad
del turn-taking que una arquitectura da y la otra no.

**2. El texto llega por el cable, encendido también en campaña, y NO en el
mismo momento que en PoC1.** El puente emite `arnes.texto` con las mismas
etiquetas que PoC1 (`telefonia.TEXT_EVENT`). Va encendido por default,
incluida la campaña, porque **la campaña de PoC3 corre con el arnés por
WebSocket local y no hay Twilio en ninguna llamada de campaña**; se apaga con
`POC3_SIN_ECO_DE_TEXTO=1` y el único caso que lo justifica es una llamada
telefónica real. Una corrida que aun así quede sin texto se marca **no apta** en
su JSON (`telefonia.text_echo_block`), porque sin texto no hay WER del
agente ni ramas del guion.

Lo que no es equivalente es el momento, y va en la dirección contraria a la que
uno esperaría:

- En PoC1 el `TTSTextFrame` lo emite ElevenLabs **junto con** el audio de esa
  palabra, y el eco va después de reenviar el frame
  (`poc1_pipecat/app/telephony.py:417`). Texto y audio viajan pegados.
- En PoC3 `output_transcript.added` es un evento aparte y llega **antes** que el
  audio que describe. Medido contra la llegada del `output_audio.delta` del
  mismo `start_ms`, y **separado por modelo**, que es como hay que leerlo: con
  `gpt-live-1-boulder-alpha` (15 censos de agosto) el p50 por llamada va de −78
  a −141 ms, mediana −94; con `gpt-live-1` GA (`CA221584…`, la única corrida GA
  con censo completo) el p50 es **−531 ms**, y su valor menos adelantado (−234)
  ya es más negativo que el p50 más adelantado del alpha. Son dos poblaciones
  sin traslape. El modelo de campaña es el GA, así que la cifra que aplica es
  la de −531 ms, con n=1.

**Ese adelanto abre un hueco en el clasificador de rachas, y se cierra aquí.**
`Observado.agent_text_during` recorta a
`[racha.inicio, racha.fin + cola_seg]`: la cola de 3 s existe para el texto que
llega **tarde**, y no hay nada del lado izquierdo. Con 531 ms de adelanto se pierde el **principio** del
texto de cada racha, y en un apoyo verbal —que suele viajar en un solo
fragmento, 5.2 caracteres de media— se pierde entero: `is_backchannel("")`
devuelve `False` por diseño, así que la racha se toma como **la respuesta** y el
guion avanza encima de un "ajá". Afecta sólo a rachas cerradas de 0.6 a 2.0 s, que es la
única banda donde el clasificador consulta el texto; en la campaña de 200
llamadas esa banda no decidió ni una vez (los 102 descartes de PoC1 y PoC2
fueron todos por duración, < 600 ms), pero con el texto adelantado deja de ser
hipotético.

`ObservadoDePoC3` sobreescribe ese método y le pone
`TEXT_LEFT_TAIL_S` por delante. **`shared/` no se toca**: el único
llamador despacha por instancia, PoC1 y PoC2 siguen ejecutando el objeto
función de la base, y lo que la cola hace queda auditado por corrida en
`left_tail_block()` — cuántos fragmentos entraron **solo** por ella
y cuántas veces eso cambió un veredicto.

Lo que sigue sin disimularse: `ClienteDeMediaStreams.text_messages` cuenta
los eventos recibidos y `ObservadoDePoC3.deaf_to_text` dice con esas palabras
que una corrida no recibió ni una línea, para que no se lea como un agente que
no habló.

**3. El `callSid` sintético es la llave del cruce, y cuadra por construcción.**
`telefonia` lee `callSid` del mensaje `start` y de ahí salen `censo_<sid>.jsonl`,
`llamada_<sid>.json` y la grabación de auditoría; `app/marks.py` nombra
`ceros_<sid>.jsonl` con el mismo `call_id`. En una corrida de arnés ese SID es
el que inventa este módulo, así que los cuatro archivos comparten llave sin que
nadie tenga que sincronizar nada. Hay una prueba que lo afirma.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw, ulaw_a_pcm16
from shared.carrier_simulator import (
    MIN_SUBSTANTIVE_SPEECH_S,
    MS_PER_FRAME,
    BACKCHANNEL_CAP_S,
    HarnessFailure,
    TextFragment,
    Observed,
    ScriptStep,
    SpeechStreak,
    Player,
    FramePacing,
    is_backchannel,
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

# El que el servidor podría devolver en un barge-in. **El puente de PoC3 no lo
# manda** (ver el encabezado): se conoce para tolerarlo, no porque se espere.
EVENT_CLEAR: Final[str] = "clear"

# El canal lateral de texto del arnés. Mismos nombres que en
# `poc1_pipecat/app/arnes.py`: si las dos PoC usaran etiquetas distintas para el
# mismo evento, el arnés de cada una tendría que traducirlas. Lo emite
# `telefonia.PuenteDeLlamada._mandar_texto_al_arnes`, encendido por default.
TEXT_EVENT: Final[str] = "harness.text"
FROM_CARRIER: Final[str] = "carrier"
FROM_AGENT: Final[str] = "agent"

# Marca del fragmento que YA trae el separador que lo une al anterior — el que
# necesite, incluido ninguno. Viaja en `FragmentoDeTexto.segmento`, el campo
# libre que cada PoC llena con lo que sepa leer de su protocolo. Mismo nombre y
# mismo uso que en PoC1 — allá casi nunca está puesta (ElevenLabs manda un frame
# por palabra sin espacios) y aquí siempre lo está. Medido en GA sobre 7,132
# fragmentos: el 74.5% trae el espacio al principio y el 24.6% no trae nada
# porque es continuación de la palabra o de la puntuación anterior. Ver
# `telefonia.EL_ALPHA_TRAE_LOS_ESPACIOS`.
WITH_SPACES: Final[str] = "has_spaces"

# Cuánto se abre la ventana de texto por la IZQUIERDA en PoC3. La base de
# `shared/` abre en `racha.inicio` porque en PoC1 el texto llega con su audio o
# después; aquí llega **antes** (medio segundo de mediana), y sin esta cola el
# clasificador de rachas recibe cadena vacía justo en la banda donde consulta el
# texto. Ver el encabezado.
#
# El valor se fijó antes de la campaña, no después de ver los datos:
#
# - **Cota inferior.** Tiene que cubrir el adelanto o la racha pierde su propio
#   texto. 1.0 s es casi el doble del adelanto medido, que es el margen que pide
#   una medición de una sola llamada.
# - **La degradación no es un acantilado.** Con 1.0 s la captura es total para
#   cualquier adelanto de hasta 1,000 ms; por encima se pierden los primeros
#   fragmentos de la racha y los posteriores siguen entrando. Al clasificador le
#   basta con tener texto, no con tenerlo completo.
# - **Cota superior, que también existe.** Una cola larga importa el texto de la
#   racha anterior, y entonces `es_apoyo_verbal` devuelve `False` y la racha se
#   toma como la respuesta: **la misma falla en el otro sentido**. Como las
#   rachas se fusionan por debajo de `HANGOVER_DE_RACHA_SEG` (0.3 s), la fuga solo
#   cabe en un hueco corto, que es el agente pausando a media frase: el texto
#   "fugado" es de la misma intervención e incluirlo acierta más, no menos.
# - **Por qué 1.0 y no 0.8:** los dos errores no cuestan lo mismo. Quedarse sin
#   texto corrompe **en silencio** —el guion avanza encima y solo se detecta al
#   oído— mientras que la sobre-captura en esa banda es benigna. Con costos
#   asimétricos se yerra hacia la cola larga.
TEXT_LEFT_TAIL_S: Final[float] = 1.0

# μ-law de 8 kHz, un byte por muestra. La trama de 20 ms son 160 bytes, que es
# el tamaño exacto de lo que manda Twilio.
SAMPLES_PER_FRAME: Final[int] = TELEPHONY_SAMPLE_RATE * MS_PER_FRAME // 1000
BYTES_PER_FRAME: Final[int] = SAMPLES_PER_FRAME

MEDIA_ENCODING: Final[str] = "audio/x-mulaw"

# Prefijo de los identificadores que este módulo inventa. Los de Twilio son
# `MZ`/`CA`/`AC` seguidos de 32 hexadecimales; éste dice en el propio texto de
# dónde salió, para que un `grep` en los logs los separe sin ambigüedad.
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

    `telefonia.create_app` lee `streamSid` y `callSid` de aquí a mano —los
    necesita para poder devolver audio y para nombrar el censo— y saca la
    etiqueta de la corrida de `customParameters`. Si falta cualquiera de los
    dos, cierra el WebSocket, así que los tres campos van siempre.
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
    """El mismo ruido de confort de PoC1 y PoC2, en el dominio del medio.

    Se genera en PCM16 con `shared/carrier_simulator/audio.py` —para que el
    nivel salga de la misma calibración en las 3 PoC— y se codifica aquí. Esto
    **sí** se codifica y no rompe nada: el ruido no es el estímulo, no tiene
    hash en el manifiesto y no entra en ningún WER.

    Nunca ceros: μ-law no tiene un cero exacto, y una pausa de bytes constantes
    es una señal que ningún teléfono produce. Importa más en PoC3 que en las
    otras dos, porque aquí el endpointing lo hace el servidor sobre este mismo
    audio y no hay perilla que compense un piso falso.
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
    es el que el manifiesto registró, PoC3 estaría reproduciendo otro audio que
    PoC1 y PoC2, y los WER de los tres medirían cosas distintas sin que nada se
    vea roto.

    Corre **antes** de conectarse y devuelve la lista de líneas del informe.
    Levanta `FallaDelArnes` al primer desacuerdo: seguir sería gastar la corrida
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
                "PoC3 manda μ-law por el cable y este módulo no recodifica desde "
                "PCM: el hash del manifiesto es la evidencia de que las 3 PoC "
                "reprodujeron el mismo audio, y recodificar la anularía"
            )
        entry = of_manifest.get(step.clip_id)
        if entry is None:
            raise HarnessFailure(
                f"{step.clip_id} no está en el manifiesto {ruta}: no se puede "
                "verificar que sea el mismo clip que reprodujeron PoC1 y PoC2"
            )
        # Se verifica el **archivo fuente**, no lo que sale al cable: desde
        # §2.2.9 el arnés le suma el piso de ruido al clip antes de publicarlo,
        # así que los bytes publicados no pueden coincidir con el manifiesto por
        # construcción. Lo que la evidencia tiene que sostener sigue siendo que
        # las 3 PoC parten del mismo estímulo, y el piso es una función
        # determinista de ese estímulo.
        source = step.ulaw_source or step.ulaw
        measured = hashlib.sha256(source).hexdigest()
        if measured != entry.sha256_ulaw:
            raise HarnessFailure(
                f"{step.clip_id}: el .ulaw de disco no es el del manifiesto.\n"
                f"        manifiesto {entry.sha256_ulaw}\n"
                f"        disco      {measured}\n"
                "        Vuelve a preparar los clips; reproducir éste dejaría a "
                "PoC3 midiendo sobre un audio distinto al de PoC1 y PoC2"
            )
        published = hashlib.sha256(step.ulaw).hexdigest()
        text_lines.append(
            f"  {step.turn:<4} {step.clip_id:<14} {len(step.ulaw):>6} B  "
            f"sha256 {measured[:12]}…  ok"
            + (f"  (publicado con piso {published[:12]}…)" if step.ulaw_source else "")
        )
    return text_lines


# --------------------------------------------------------------------------
# El cliente
# --------------------------------------------------------------------------


class MediaStreamsClient:
    """Un WebSocket hablando el protocolo de Twilio, sin Twilio.

    El envío lo maneja quien tenga el ritmo (`ReproductorDeMediaStreams`); aquí
    solo se serializa y se lleva la cuenta. La recepción corre en una tarea
    aparte porque el puente manda cuando quiere, y bloquear el envío para leer
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
        # Cualquier evento que no sea de Twilio. Es por donde llega el texto del
        # agente: `telefonia.PuenteDeLlamada._mandar_texto_al_arnes`. El arnés
        # le pasa `ObservadoDePoC3.anotar_texto_del_puente`.
        self._on_receive_message = on_receive_message
        # Se avisa cuando el puente cierra el WebSocket por su cuenta, que es lo
        # que pasa cuando el agente cierra la negociación y cuelga. Sin este
        # aviso el arnés le sigue hablando a un puente que ya no existe.
        self._on_other_side_closed = on_other_side_closed
        # `True` en cuanto el otro lado se va. A partir de ahí `enviar_trama` no
        # levanta: empujar contra un socket cerrado no es un error del arnés.
        self.remote_closed = False

        suffix = label or time.strftime("%Y%m%d-%H%M%S")
        self.stream_sid = f"{HARNESS_SID}-stream-{suffix}"
        # **Ésta es la llave del cruce.** El puente la lee del `start` y con ella
        # nombra el censo, el JSON de la llamada y la auditoría; `app/marcas.py`
        # nombra el `.jsonl` de ceros con el mismo valor.
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
        # Cuántas veces el otro lado pidió tirar lo que no se había reproducido.
        # **En PoC3 esto es 0 por construcción**: el puente no manda `clear`. Se
        # cuenta igual para que el día que lo mande se note, y para poder decir
        # en el reporte que el 0 es del canal y no de la conversación.
        self.cleanups = 0
        # Eventos `arnes.texto` recibidos. Es la medida de si el arnés corrió
        # sordo —0 con el puente sin `eco_de_texto`, o con el eco apagado por
        # una falla— y por eso se cuenta aparte de `eventos_desconocidos`.
        self.text_messages = 0
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
        """Consume lo que devuelve el puente hasta que se corte."""
        motivo = "el puente cerró el Media Stream"
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
            # El puente de PoC3 no lo manda, pero tolerarlo cuesta nada y el día
            # que crezca esto tiene que no romperse: llegaría justo cuando el
            # agente se interrumpe, que es el momento que más interesa observar.
            self.cleanups += 1
            if self._on_clear is not None:
                self._on_clear()
        elif event in (EVENT_MARK, EVENT_CONNECTED):
            pass
        else:
            if event == TEXT_EVENT:
                self.text_messages += 1
            elif event not in self.unknown_events:
                # Se anota una vez por tipo y no se trata como error: un evento
                # nuevo del puente no es razón para tumbar una corrida.
                self.unknown_events.append(event)
            if self._on_receive_message is not None:
                try:
                    self._on_receive_message(message)
                except Exception as error:  # noqa: BLE001 - el arnés reporta
                    self.incidents.append(
                        f"el mensaje {event!r} no se pudo leer: {error!r}"
                    )


# --------------------------------------------------------------------------
# El reproductor
# --------------------------------------------------------------------------


class MediaStreamsPlayer(Player):
    """El reproductor compartido, con μ-law como dominio de la trama.

    Hereda el ritmo, el relleno continuo y el fechado del fin del clip de
    `shared/carrier_simulator/`, **sin modificarlo**, para que las 3 PoC fechen
    el cero del turno igual. Lo que cambia es qué hay dentro de la trama: aquí
    son los bytes μ-law del manifiesto, sin convertir.
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
            "PoC3 manda μ-law por el cable y este reproductor no recodifica "
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
# Lo que el arnés de PoC3 observa
# --------------------------------------------------------------------------


def join_fragments_of_alpha(fragments: list[TextFragment]) -> str:
    """Une los fragmentos de PoC3 y normaliza. La ÚNICA regla de reensamblado.

    Dos pasos y en este orden:

    1. **Unir.** Se repone el separador solo si el fragmento declara que no lo
       trae, leyendo `segmento == WITH_SPACES`. Es la regla de
       `pipeline.join_tts_fragments` de PoC1; con los fragmentos del alpha no
       inserta nada, y esa es la respuesta correcta. Medido sobre los 7,042
       bordes de la campaña de 90 llamadas: el 74.9% ya trae el separador y el
       25.1% restante **no lo trae a propósito**, porque es continuación de la
       palabra (`" die"+"ciocho"`) o de la puntuación (`" Sí"+", soy"`)
       anterior. Reponerlo ahí es lo que parte el texto.
    2. **Normalizar.** Con `telefonia.normalize_text`, la MISMA función que
       `RegistroDeLlamada.normalizar` le aplica al JSON de la llamada. Se reusa
       a propósito: si cada lado normalizara por su cuenta, el WER dependería
       de qué archivo se mirara.

    Vive suelta, y no como método, para que los dos ensambladores de
    `ObservadoDePoC3` no puedan divergir: divergieron una vez y costó un WER de
    0.1654 sobre uno real de 0.0403 (`docs/hallazgo_ensamblado_stt_poc3.md`).
    """
    from poc3_gpt_live.app.telephony import normalize_text

    parts: list[str] = []
    for fragment in fragments:
        text = fragment.text
        if not text:
            continue
        if (
            parts
            and fragment.segment != WITH_SPACES
            and not parts[-1][-1:].isspace()
            and not text[:1].isspace()
        ):
            parts.append(" ")
        parts.append(text)
    return normalize_text("".join(parts))


@dataclass
class ObservedFromPoC3(Observed):
    """Lo que el arnés vio por el WebSocket de Media Streams.

    Igual que el de PoC1 en lo que importa —el audio llega en μ-law y hay que
    decodificarlo antes de medirle energía— y distinto en cómo se reensambla el
    texto, que es donde las dos arquitecturas no se parecen.

    ## El reensamblado, con el formato real a la vista

    ElevenLabs manda un `TTSTextFrame` por palabra **sin** los espacios que las
    separan, así que PoC1 tiene que reponerlos. El alpha manda el espacio
    **dentro** del fragmento (`" Hola,"`, `" buen"`, `" día,"`), así que aquí
    reponerlo sería el error: partiría las palabras que llegan en dos trozos.
    `" ¿H"` + `"ablo con"` es una sola palabra en dos fragmentos, y `"diecis"` +
    `"iete mil"` otra; con una regla de separación propia saldrían `"¿H ablo"` y
    `"diecis iete"`.

    Los dos casos se resuelven con la MISMA regla, que es la de
    `pipeline.join_tts_fragments` de PoC1: se repone el separador solo cuando el
    fragmento declara que no lo trae. La bandera viaja en `segmento` y el puente
    la pone siempre (`telefonia.THE_ALPHA_BRINGS_THE_SPACES`), así que en PoC3 la
    regla no inserta nada y el reensamblado es la concatenación cruda — que es lo
    correcto. Se escribe con la bandera y no como `"".join` a secas para que el
    día que llegue un fragmento sin separador no se peguen dos palabras en
    silencio, que es el defecto que da un WER falso.
    """

    own_identity: str = FROM_CARRIER

    # Cómo salió el audio de este lado. Diagnóstico del arnés: no toca latencias.
    publishing_pacing: dict[str, Any] = field(default_factory=dict)
    measured_sends: list[float] = field(default_factory=list)
    # Tramas que se quisieron publicar después de que el agente colgó.
    frames_without_destination: int = 0
    # Rastro de la cola izquierda, indexado por el inicio de cada racha. El
    # bucle de espera reevalúa cada 50 ms, así que la entrada se sobreescribe y
    # lo que queda es la última evaluación — la que decidió.
    tail_trace: dict[float, dict[str, Any]] = field(default_factory=dict)

    def annotate_audio_in_ulaw(self, ulaw: bytes) -> None:
        """Decodifica y mide en tramos de 20 ms. El arnés escucha en μ-law.

        El puente entrega bloques de ~100 ms; se escuchan como los reproduciría
        el teléfono, tramo por tramo (`Observado.annotate_audio_in_frames`), y no
        como una sola trama fechada al llegar.
        """
        self.annotate_audio_in_frames(ulaw_a_pcm16(ulaw).tobytes(), sample_rate=TELEPHONY_SAMPLE_RATE)

    def assemble_stt(self, fragments: list[TextFragment]) -> str:
        """Lo que el arnés creyó decir. Sin fragmentos, cadena vacía.

        **La misma regla que `assemble_agent`, y tiene que seguir siéndolo.**
        Los dos lados de PoC3 llegan por `input_transcript.added` y
        `output_transcript.added`, que son el mismo formato del mismo motor:
        deltas que siguen la cadencia del habla, no segmentos finalizados.

        La regla de PoC1, `" ".join(f.texto.strip())`, es correcta allá porque
        Deepgram manda segmentos; aquí volvería un espacio **cada frontera de
        fragmento**, que parte palabras (`" die"+"ciocho"` → `"die ciocho"`) y
        despega la puntuación (`" Sí"+", soy"` → `"Sí , soy"`): `"die ciocho
        mil"` no es un monto extraíble. Ver `docs/hallazgo_ensamblado_stt_poc3.md`.
        """
        return join_fragments_of_alpha(fragments)

    def assemble_agent(self, fragments: list[TextFragment]) -> str:
        """Los fragmentos del alpha, con la misma regla que usa el JSON."""
        return join_fragments_of_alpha(fragments)

    def agent_text_during(
        self, streak: SpeechStreak, *, cola_seg: float = 3.0
    ) -> str:
        """Igual que la base, más `TEXT_LEFT_TAIL_S` por delante.

        `Observado.agent_text_during` abre la ventana en `racha.inicio`
        porque en PoC1 el texto llega con su audio o un poco después, y la cola
        de 3 s existe justo para ese caso. En PoC3 llega **antes**. Con la
        ventana de la base, el texto de una racha corta cae a la izquierda de
        `racha.inicio` y queda fuera, así que `wait_for_agent_turn`
        recibe cadena vacía — y `is_backchannel("")` devuelve `False` por
        diseño, o sea que **la trata como la respuesta** y el guion avanza
        encima de un "ajá", una falla que solo se detecta al oído.

        Se sobreescribe aquí y **no se toca `shared/`**: el único llamador es
        `reproductor.wait_for_agent_turn`, que despacha por instancia, y
        `Observado` ya deja `assemble_stt` y `assemble_agent` como métodos
        por esta misma razón — no son iguales en las 3 PoC. PoC1 y PoC2 siguen
        ejecutando el objeto función de la base, sin una línea nueva.

        El costo de la copia es que la base puede evolucionar y ésta no
        enterarse; lo cubre `TestLaColaNoSeSepara` forzando la cola a 0 y
        exigiendo salida byte a byte idéntica a la de `Observado`.
        """
        since = streak.start - TEXT_LEFT_TAIL_S
        until = streak.end + cola_seg
        of_agent = self.agent_fragments()
        inside = [f for f in of_agent if since <= f.t <= until]
        text = self.assemble_agent(inside)
        self._annotate_queue_trail(streak, of_agent, inside, text, cola_seg)
        return text

    def _annotate_queue_trail(
        self,
        streak: SpeechStreak,
        of_agent: list[TextFragment],
        inside: list[TextFragment],
        text: str,
        cola_seg: float,
    ) -> None:
        """Deja constancia de qué entró **solo** por la cola, y si decidió.

        Sin esto la cola es una perilla que no se puede auditar: una corrida
        saldría igual con 1.0 s que con 0, y la pregunta "¿cuántas veces la cola
        decidió sobre las 84?" no tendría respuesta en ningún archivo.

        `la_cola_decidio` es exacto y no una aproximación. El texto solo puede
        cambiar el veredicto de `_streak_verdict` en **una** banda: racha
        ya cerrada, de `MIN_SUBSTANTIVE_SPEECH_S` a
        `BACKCHANNEL_CAP_S`. Por encima se acepta sin preguntarle al
        texto, por debajo se descarta sin preguntarle. Dentro de la banda el
        veredicto **es** `is_backchannel(texto)`, así que evaluarlo con cola y
        sin ella responde la pregunta entera. No se registra si la racha estaba
        abierta —ahí el veredicto es `NOT_KNOWN_YET` pase lo que pase— y
        como el bucle reevalúa cada 50 ms, cada racha se sobreescribe por su
        `inicio` y lo que queda es la última evaluación, la que decidió.
        """
        with_tail = [f for f in inside if f.t < streak.start]
        no_queue = [f for f in inside if f.t >= streak.start]
        in_the_band = (
            MIN_SUBSTANTIVE_SPEECH_S <= streak.duration < BACKCHANNEL_CAP_S
        )
        text_without_tail = self.assemble_agent(no_queue)
        decided = in_the_band and (
            is_backchannel(text) != is_backchannel(text_without_tail)
        )
        self.tail_trace[round(streak.start, 6)] = {
            "turn": (self.turns[-1].step.turn if self.turns else ""),
            "streak_start": round(streak.start, 3),
            "streak_duration_ms": round(streak.duration * 1000.0, 1),
            "fragments_in_window": len(inside),
            # Los que NO habrían entrado con la cola en 0.
            "fragments_only_via_queue": len(with_tail),
            "in_the_band_where_text_decides": in_the_band,
            "the_tail_decided": decided,
            "text_with_tail": text,
            "text_without_tail": text_without_tail,
        }

    def left_tail_block(self) -> dict[str, Any]:
        """Lo que la cola hizo en esta corrida. Va al artefacto de la llamada.

        Se publica **siempre**, también cuando no hizo nada: una llave que solo
        aparece cuando la cola actuó se lee como ausente cuando falta el código
        que la escribe, y entonces "la cola nunca decidió" y "nadie lo midió"
        salen iguales. Sobre las 84 de campaña, `veces_que_decidio` sumado es la
        respuesta a cuántas veces la cola cambió un veredicto.
        """
        trail = [self.tail_trace[k] for k in sorted(self.tail_trace)]
        return {
            "left_tail_s": TEXT_LEFT_TAIL_S,
            "evaluated_streaks": len(trail),
            "streaks_with_fragments_only_from_the_tail": sum(
                1 for r in trail if r["fragments_only_via_queue"]
            ),
            "fragments_only_via_queue": sum(
                r["fragments_only_via_queue"] for r in trail
            ),
            "times_decided": sum(1 for r in trail if r["the_tail_decided"]),
            "per_streak": trail,
            "what_it_is": (
                "`fragmentos_solo_por_la_cola` son los fragmentos de texto que "
                "entraron a la ventana por abrirla "
                f"{TEXT_LEFT_TAIL_S:.1f} s antes de `racha.inicio` y "
                "que con la ventana de `shared/` habrían quedado fuera. "
                "`la_cola_decidio` es cuántas veces eso CAMBIÓ el veredicto de "
                "la racha, que solo puede ocurrir en rachas cerradas de "
                f"{MIN_SUBSTANTIVE_SPEECH_S:.1f} a "
                f"{BACKCHANNEL_CAP_S:.1f} s: fuera de esa banda el "
                "clasificador no consulta el texto"
            ),
        }

    def annotate_bridge_text(self, message: Mapping[str, Any]) -> None:
        """Traduce un evento `arnes.texto` a un fragmento de `Observado`.

        Gemelo de `poc1_pipecat/app/harness.py:_on_text_received`. El texto se
        guarda **crudo**: quien limpia es `assemble_agent`, al final.
        """
        text = str(message.get("text", ""))
        if not text:
            return
        self.annotate_text(
            str(message.get("de", FROM_AGENT)),
            text,
            segment=WITH_SPACES if message.get("has_spaces") else "",
            final=bool(message.get("final", False)),
        )

    def branches_block(self) -> dict[str, Any]:
        """Qué rama del guion se tomó en cada turno, y con qué evidencia.

        Va al artefacto de la corrida. Existe porque la ramificación depende del
        texto del agente (`play_script` se lo pasa a `decidir`), así que
        **antes del gancho de texto todos los turnos caían a su línea por
        defecto** y una corrida ramificada se veía igual que una que no lo
        estaba. Con `by_default` publicado por turno eso se distingue leyendo
        el archivo.

        No lo puede registrar el censo del puente: `censo_<sid>.jsonl` guarda
        los eventos del alpha y el puente no sabe que existe un guion — ni
        debería, o el arnés estaría midiendo un servidor que se comporta
        distinto cuando lo miran.
        """
        decisions = [d.as_json() for d in self.decisions]
        return {
            "decisions": decisions,
            "branched_turns": len(decisions),
            "by_default": sum(1 for d in decisions if d.get("by_default")),
            "deaf_to_text": self.deaf_to_text,
        }

    def text_block(self) -> dict[str, Any]:
        """El veredicto del arnés sobre el texto de esta corrida.

        Gemelo de `telefonia.text_echo_block`, visto desde el otro
        extremo del cable. Los dos tienen que coincidir: si el puente dice que
        emitió N fragmentos y aquí llegaron 0, lo que falló es el transporte.

        Va **siempre**, también cuando todo salió bien, por el mismo motivo que
        el otro: una llave que solo aparece al fallar se lee como ausente cuando
        falta el código que la escribe.
        """
        deaf = self.deaf_to_text
        return {
            "received_fragments": len(self.fragments),
            "of_agent": len(self.agent_fragments()),
            "deaf_to_text": deaf,
            "eligible": not deaf,
            "motivo": (
                "el arnés no recibió una sola línea de texto. Con el eco del "
                "puente encendido eso significa que el agente habló y el arnés "
                "no se enteró: sin texto no hay referencia del WER del agente, "
                "`turno.respuesta` sale en blanco y las ramas del guion caen "
                "todas a su línea por defecto. La corrida NO es apta"
                if deaf
                else ""
            ),
        }

    @property
    def deaf_to_text(self) -> bool:
        """¿Corrió esta corrida sin recibir una sola línea de texto?

        Existe para que el reporte pueda decirlo con esas palabras. Sin esto,
        una corrida sin el gancho del paso 3 se ve idéntica a una donde el
        agente no habló, y son cosas muy distintas: en la primera el agente
        habló y el arnés no se enteró.
        """
        return not self.fragments


# --------------------------------------------------------------------------
# Servidor de eco: con qué se verifica el cliente sin gastar un centavo
# --------------------------------------------------------------------------


@dataclass
class MediaStreamsEcho:
    """Un servidor que habla el mismo protocolo y devuelve lo que le llega.

    Existe para poder verificar el transporte **sin levantar el puente**: sin
    OpenAI, sin el alpha y sin un centavo de gasto. Lo que se comprueba contra
    él es lo que puede fallar en silencio — que las tramas salgan a 20 ms y que
    los bytes lleguen sin tocarse.

    `clean_up_after` y `text_after` mandan eventos que el puente de PoC3 **no**
    manda hoy: sirven para probar que el cliente los tolera y los cuenta cuando
    el paso 3 los agregue, en vez de descubrirlo con una corrida ya pagada.
    """

    return_audio: bool = True
    clean_up_after: int | None = None
    text_after: int | None = None
    # Eventos basura que el eco manda antes que nada, para probar que el cliente
    # no se cae con lo que no conoce.
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
        """El handler del WebSocket."""
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
                        json.dumps({"event": EVENT_CLEAR, "streamSid": stream_sid})
                    )
                if (
                    self.text_after is not None
                    and len(self.received) == self.text_after
                ):
                    await ws.send(
                        json.dumps(
                            {
                                "event": TEXT_EVENT,
                                "de": FROM_AGENT,
                                "text": " claro",
                                "final": False,
                                "streamSid": stream_sid,
                            }
                        )
                    )
            elif event == EVENT_END:
                self.halted = True
                self._done.set()
                break


__all__ = [
    "BYTES_PER_FRAME",
    "MEDIA_ENCODING",
    "FROM_AGENT",
    "FROM_CARRIER",
    "EVENT_CONNECTED",
    "TEXT_EVENT",
    "EVENT_END",
    "EVENT_START",
    "EVENT_CLEAR",
    "EVENT_MARK",
    "EVENT_MEDIA",
    "SAMPLES_PER_FRAME",
    "HARNESS_SID",
    "MediaStreamsClient",
    "MediaStreamsEcho",
    "ObservedFromPoC3",
    "MediaStreamsPlayer",
    "message_audio",
    "connected_message",
    "end_message",
    "start_message",
    "media_message",
    "split_ulaw_into_frames",
    "noise_frame_in_ulaw",
    "verify_stimulus",
]
