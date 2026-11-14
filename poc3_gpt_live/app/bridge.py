"""El puente: audio de Twilio <-> audio de GPT-Live, y el cliente de GPT-Live.

Dos cosas que no se mezclan:

1. **`PuenteDeAudio`** — el códec y el remuestreo. μ-law 8 kHz de Twilio a PCM16
   24 kHz mono para GPT-Live, y de vuelta.
2. **`ClienteGptLive`** — el WebSocket a `wss://api.openai.com/v1/live/sessions`,
   el `session.start` inicial, y el flujo de eventos JSON.

## Protocolo: el público

Las campañas de agosto y septiembre corrieron con el protocolo del alpha
(`docs_alpha/`): `?model=` en la URL, header `OpenAI-Alpha`, `session.update`
inicial, `input_audio.append`, `output_audio.delta` con `start_ms`/`end_ms`,
`turn.*`, los `response.*` sueltos al nivel de arriba y
`delegation.function_call_output.create`. Ese acceso se retiró
(`quicksilver_v2_access_denied`) y el mismo `gpt-live-1` quedó en la API
pública con otra forma. Lo de aquí sigue a la integración de referencia de
pipecat-ai 1.12.0 (`services/openai/live/`), que es la del propio fabricante del
framework de PoC1 contra la API pública:

| Antes (alpha) | Ahora (pública) |
|---|---|
| `session.update` inicial, modelo en la URL | `session.start` con `model` adentro |
| `input_audio.append` | `session.input_audio.append` |
| `output_audio.delta` (`audio`, `start_ms`, `end_ms`) | `session.output_audio.delta` (`delta`, **sin tiempos**) |
| `input/output_transcript.added` (`item.text`) | `session.input/output_transcript.delta` (`delta`, `start_ms`, `end_ms`) |
| `turn.created` / `turn.delta` / `turn.done` | **no existen** |
| `delegation.created` | `session.delegation.created` |
| `response.*` sueltos | envueltos en `response.event` (`delegation_id`, `event`) |
| `delegation.function_call_output.create` | `response.item.create` y luego `response.create` |
| `session.context.append` | `session.commentary.append` (y `instructions`/`thinking`) |

Lo que **no** cambia y es lo que la medición necesita: el audio sigue siendo
PCM16 24 kHz mono en los dos sentidos (se pide explícito en `session.start`), así
que el remuestreo, el umbral de voz y la ventana de 20 ms son los mismos que
validó la corrida de interrupciones. `escuchar()` desenvuelve `response.event`
para que el resto del puente vea los `response.*` igual que antes, con el
`delegation_id` agregado.

Quien los orquesta es `app/telephony.py`. Aquí no hay negociación, ni métricas,
ni FastAPI.

## Sobre el remuestreo: por qué `soxr` y no interpolación lineal

Subir de 8 kHz a 24 kHz repitiendo o interpolando muestras mete alias e imágenes
espectrales por encima de los 4 kHz — le entrega al modelo un espectro que la
fuente telefónica nunca tuvo. Bajar de 24 kHz a 8 kHz sin filtrar antes es peor:
todo lo que esté arriba de 4 kHz se **pliega** hacia abajo y aterriza dentro de
la banda de voz como distorsión audible.

`soxr` filtra con banda de transición antes de decidir, que es lo que hace que
la conversión sea reversible en el sentido que importa. Es además el mismo
remuestreador que usa PoC1 por dentro (entra con el extra `silero` de pipecat, y
`LocalSmartTurnAnalyzerV3._resample_to_model_rate` lo llama), así que las dos
PoC con conversión de tasa usan la misma aritmética y esa no es una variable de
la comparación.

**Se usa `ResampleStream` y no `resample()`.** Twilio manda tramas de 20 ms;
remuestrear cada trama por separado reinicia el filtro en cada una y deja un
chasquido en cada frontera —160 discontinuidades por segundo—. El stream
conserva el estado del filtro entre tramas.

### Lo que el remuestreo le cuesta a la latencia, medido

`ResampleStream` **no devuelve una trama por cada trama**: acumula y suelta en
ráfagas (dos de cada tres llamadas devuelven vacío). Eso mete un retraso que hay
que conocer, porque cae dentro de `full_turn_ms` y ninguna otra PoC lo
tiene en este punto del camino. Medido sobre 4 s de audio en tramas de 20 ms, en
las dos direcciones, con soxr 1.0.0 y 1.1.0 (los números salen idénticos):

| | 8 k -> 24 k | 24 k -> 8 k |
|---|---|---|
| Retraso en régimen | 24.2 ms | 24.1 ms |
| Peor ráfaga | 73.2 ms | 73.1 ms |
| Tramas que salen vacías | 135 / 200 | 135 / 200 |

O sea ~48 ms constantes de ida y vuelta, y hasta ~146 ms en el peor caso. Contra
una mediana de `full_turn_ms` de ~2,600 ms en PoC1, es ~2%: se documenta y
no se corrige. Bajar la calidad de `HQ` a `MQ` recortaría parte, pero cambiaría
el espectro que ve el modelo, que es justo lo que se está midiendo.

## Sobre el códec μ-law: por qué a mano y no con `audioop`

`audioop` haría esto en dos líneas, pero **se eliminó de la biblioteca estándar
en Python 3.13**. PoC3 corre 3.12 (igual que PoC1 y PoC2), así que hoy
funcionaría; dejar la PoC atada a un módulo ya borrado significa que actualizar
Python es un rediseño del puente. Son cuarenta líneas de tablas G.711, están
verificadas contra la implementación de referencia de Sun, y se prueban en
`tests/test_bridge.py` con un barrido de los 256 códigos.

## Sobre los timeouts del LLM: PoC3 no puede aplicar la política compartida

`shared/llm_transport/` fija 5 s al primer byte y 4 intentos, idénticos en PoC1
y PoC2. **En PoC3 no hay dónde ponerlos**: la llamada al modelo de texto ocurre
dentro del servicio de OpenAI, del otro lado del WebSocket, y el alpha no expone
ni timeout ni reintentos de la delegación. Lo único que este código controla es
su propio WebSocket. Es una asimetría de la arquitectura, no una omisión, y está
declarada en `docs/METRICS_CONTRACT.md` §2.5.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import soxr

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tasas y formato
# --------------------------------------------------------------------------

# Twilio Media Streams: μ-law de 8 kHz, mono.
TELEPHONY_SAMPLE_RATE: Final[int] = 8000
# GPT-Live: PCM16 little-endian, mono, 24 kHz
# (`docs_alpha/Alpha_guide.html`, "Quickstart: WebSocket").
LIVE_SAMPLE_RATE: Final[int] = 24000

# Cada muestra PCM16 son 2 bytes: el payload decodificado debe tener un número
# par de bytes o el alpha responde `invalid_audio`.
BYTES_PER_SAMPLE: Final[int] = 2

# Retraso que mete el remuestreador en régimen, por sentido, medido sobre tramas
# de 20 ms con soxr 1.0.0 y 1.1.0 (ver el docstring del módulo). Se guarda como dato
# y no como perilla: es lo que hay que restar mentalmente al leer la latencia de
# PoC3, y lo que hay que volver a medir si alguien cambia la calidad del filtro.
RESAMPLING_LAG_MS: Final[float] = 24.2
PEAK_RESAMPLING_LAG_MS: Final[float] = 73.2

# Formato de la sesión, en los dos sentidos (la API pública no permite uno
# distinto por sentido ni cambiarlo a media sesión):
# - `pcm24`: PCM16 24 kHz. Es el de la campaña de latencia de septiembre (alpha)
#   y el de la de interrupciones de octubre; el puente remuestrea 8 <-> 24 kHz.
# - `pcmu`: G.711 μ-law 8 kHz, el mismo de Twilio. El audio pasa sin convertir.
PCM24_FORMAT: Final[str] = "pcm24"
PCMU_FORMAT: Final[str] = "pcmu"
FORMATS: Final[tuple[str, ...]] = (PCM24_FORMAT, PCMU_FORMAT)


def session_format(format_kind: str) -> dict[str, Any]:
    """El `audio.format` de `session.start` para cada formato del puente."""
    if format_kind == PCMU_FORMAT:
        return {"type": "audio/pcmu", "rate": TELEPHONY_SAMPLE_RATE}
    if format_kind == PCM24_FORMAT:
        return {"type": "audio/pcm", "rate": LIVE_SAMPLE_RATE}
    raise ValueError(f"formato de audio desconocido: {format_kind!r} (válidos: {FORMATS})")


# --------------------------------------------------------------------------
# Códec G.711 μ-law
# --------------------------------------------------------------------------

_BIAS: Final[int] = 0x84
# Tope en el dominio de 14 bits, tras el desplazamiento de 2 bits.
_CLIP_14: Final[int] = 8159
# Fin de cada segmento, en el mismo dominio. De la implementación de referencia.
_SEGMENT_END: Final[np.ndarray] = np.array(
    [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32
)


def _decoding_table() -> np.ndarray:
    """LUT de los 256 códigos μ-law a PCM16. Se construye una vez al importar."""
    codes = np.arange(256, dtype=np.int32)
    inverted = ~codes & 0xFF
    mantissa = inverted & 0x0F
    exponent = (inverted & 0x70) >> 4
    magnitude = ((mantissa << 3) + _BIAS) << exponent
    negative = (inverted & 0x80) != 0
    samples = np.where(negative, _BIAS - magnitude, magnitude - _BIAS)
    return samples.astype(np.int16)


_DECODING: Final[np.ndarray] = _decoding_table()


def ulaw_a_pcm16(data: bytes) -> np.ndarray:
    """μ-law -> PCM16. Una búsqueda en tabla, exacta por construcción."""
    return _DECODING[np.frombuffer(data, dtype=np.uint8)]


def pcm16_a_ulaw(samples: np.ndarray) -> bytes:
    """PCM16 -> μ-law, vectorizado. Sigue la implementación de referencia G.711.

    El orden importa y no es intercambiable: se desplaza **antes** de tomar el
    valor absoluto, igual que el original. Hacerlo al revés mueve un LSB en las
    muestras negativas — inaudible, pero deja de ser bit-exacto contra cualquier
    otra implementación y vuelve inútil la prueba de ida y vuelta.
    """
    values = np.asarray(samples, dtype=np.int16).astype(np.int32) >> 2
    negative = values < 0
    magnitude = np.where(negative, -values, values)
    np.minimum(magnitude, _CLIP_14, out=magnitude)
    magnitude = magnitude + (_BIAS >> 2)

    # El segmento es el primer tramo de la tabla que alcanza a contener la
    # magnitud; 8 significa que se desbordó y va al código extremo.
    segment = np.searchsorted(_SEGMENT_END, magnitude, side="left")
    overflowed = segment >= 8
    safe_segment = np.where(overflowed, 0, segment)

    mantissa = (magnitude >> (safe_segment + 1)) & 0x0F
    code_value = np.where(overflowed, 0x7F, (safe_segment << 4) | mantissa)
    # La máscara invierte todo en las positivas y solo los 7 bits bajos en las
    # negativas: así el bit de signo queda en 1 para las positivas.
    mascara = np.where(negative, 0x7F, 0xFF)
    return (code_value ^ mascara).astype(np.uint8).tobytes()


class AudioBridge:
    """Convierte en los dos sentidos, conservando el estado del filtro.

    Una instancia por llamada: los dos remuestreadores llevan estado y mezclarlo
    entre llamadas metería la cola de la anterior en el arranque de la siguiente.

    No es seguro para uso concurrente sobre el mismo sentido; en la práctica cada
    sentido lo alimenta una sola tarea.
    """

    def __init__(self, format_kind: str = PCM24_FORMAT) -> None:
        session_format(format_kind)  # valida
        self.format_kind = format_kind
        self._upsampler = soxr.ResampleStream(
            TELEPHONY_SAMPLE_RATE, LIVE_SAMPLE_RATE, 1, dtype="int16", quality="HQ"
        )
        self._bajada = soxr.ResampleStream(
            LIVE_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE, 1, dtype="int16", quality="HQ"
        )

    @property
    def es_pcm(self) -> bool:
        """El audio de la sesión es PCM16 (2 bytes por muestra)."""
        return self.format_kind == PCM24_FORMAT

    @property
    def samples_per_window(self) -> int:
        """Muestras de una ventana de 20 ms en el audio que devuelve `to_measure`."""
        sample_rate = LIVE_SAMPLE_RATE if self.es_pcm else TELEPHONY_SAMPLE_RATE
        return int(sample_rate * SPEECH_WINDOW_S)

    def to_measure(self, audio: bytes) -> bytes:
        """El audio de la sesión como PCM16, para `has_speech`. En μ-law se decodifica."""
        if self.es_pcm or not audio:
            return audio
        return ulaw_a_pcm16(audio).tobytes()

    def entry(self, ulaw: bytes) -> bytes:
        """μ-law de Twilio -> lo que se manda a la sesión, en su formato."""
        return self.from_twilio(ulaw) if self.es_pcm else ulaw

    def output(self, audio: bytes) -> bytes:
        """Audio de la sesión -> μ-law para Twilio."""
        return self.a_twilio(audio) if self.es_pcm else audio

    def from_twilio(self, ulaw: bytes) -> bytes:
        """μ-law 8 kHz de Twilio -> PCM16 24 kHz para `input_audio.append`.

        Devuelve bytes crudos, **sin cabecera WAV ni contenedor**: el alpha los
        rechaza. Puede devolver vacío si el remuestreador todavía no soltó
        muestras, y quien llama debe saltarse los payloads vacíos (el alpha
        rechaza el audio vacío).
        """
        if not ulaw:
            return b""
        pcm = ulaw_a_pcm16(ulaw)
        upsampled = self._upsampler.resample_chunk(pcm)
        return np.asarray(upsampled, dtype=np.int16).tobytes()

    def a_twilio(self, pcm24: bytes) -> bytes:
        """PCM16 24 kHz de `output_audio.delta` -> μ-law 8 kHz para Twilio.

        Un payload con número impar de bytes está corrupto: se descarta el byte
        suelto en vez de desalinear todo lo que venga después (un desfase de un
        byte convierte el resto de la llamada en ruido blanco).
        """
        if not pcm24:
            return b""
        if len(pcm24) % BYTES_PER_SAMPLE:
            logger.warning(
                "payload de audio con %d bytes (impar): se descarta el último "
                "para no desalinear el resto de la llamada",
                len(pcm24),
            )
            pcm24 = pcm24[: len(pcm24) - (len(pcm24) % BYTES_PER_SAMPLE)]
        samples = np.frombuffer(pcm24, dtype="<i2")
        lowered = self._bajada.resample_chunk(samples)
        return pcm16_a_ulaw(np.asarray(lowered, dtype=np.int16))


# --------------------------------------------------------------------------
# Nombres de los eventos de GPT-Live
# --------------------------------------------------------------------------
#
# Los de la API pública (ver la tabla del docstring). Se declaran como
# constantes para que un cambio de protocolo se corrija en un solo lugar y para
# que un typo sea un `AttributeError` y no un evento que nunca entra por el `if`.

# Cliente -> servidor
EV_SESSION_START: Final[str] = "session.start"
EV_SESSION_UPDATE: Final[str] = "session.update"
EV_INPUT_AUDIO_APPEND: Final[str] = "session.input_audio.append"
# Los tres canales de contexto. `commentary` es el que el modelo dice en voz alta
# con sus palabras, y es como la API le pide que hable primero; `thinking` lo
# guarda sin decirlo; `instructions` se suma a las instrucciones de arranque.
EV_COMMENTARY_APPEND: Final[str] = "session.commentary.append"
EV_THINKING_APPEND: Final[str] = "session.thinking.append"
EV_INSTRUCTIONS_APPEND: Final[str] = "session.instructions.append"
EV_RESPONSE_ITEM_CREATE: Final[str] = "response.item.create"
EV_RESPONSE_CREATE: Final[str] = "response.create"
EV_SESSION_CLOSE: Final[str] = "session.close"

# Servidor -> cliente
EV_SESSION_STARTED: Final[str] = "session.started"
EV_SESSION_UPDATED: Final[str] = "session.updated"
EV_OUTPUT_AUDIO_DELTA: Final[str] = "session.output_audio.delta"
EV_INPUT_TRANSCRIPT: Final[str] = "session.input_transcript.delta"
EV_OUTPUT_TRANSCRIPT: Final[str] = "session.output_transcript.delta"
EV_DELEGATION_CREATED: Final[str] = "session.delegation.created"
# El sobre de la delegación Responses: trae `delegation_id` y el evento de la
# Responses API en `event`. `escuchar()` lo desenvuelve.
EV_RESPONSE_EVENT: Final[str] = "response.event"
# Ya desenvueltos, los eventos de la Responses API conservan su nombre. La
# llamada a función se ejecuta con `output_item.done` (item `function_call`,
# `status: completed`), que trae `call_id`, `name` y `arguments` juntos: es lo
# que usa la integración de referencia. `function_call_arguments.done` NO trae
# `call_id`, así que no sirve para ejecutar.
EV_FUNCTION_ARGS_DONE: Final[str] = "response.function_call_arguments.done"
EV_OUTPUT_ITEM_ADDED: Final[str] = "response.output_item.added"
EV_OUTPUT_ITEM_DONE: Final[str] = "response.output_item.done"
# El ciclo de vida de un turno del modelo delegado. Sirven para saber si hay una
# delegación **en vuelo**, que es lo que el cuelgue tiene que esperar: sin esto,
# el temporizador de silencio se dispara a media delegación y corta la despedida.
EV_RESPONSE_CREATED: Final[str] = "response.created"
EV_RESPONSE_COMPLETED: Final[str] = "response.completed"
EV_RESPONSE_INCOMPLETE: Final[str] = "response.incomplete"
EV_RESPONSE_FAILED: Final[str] = "response.failed"
RESPONSE_END_EVENTS: Final[frozenset[str]] = frozenset(
    {EV_RESPONSE_COMPLETED, EV_RESPONSE_INCOMPLETE, EV_RESPONSE_FAILED}
)

EV_SESSION_USAGE: Final[str] = "session.usage.updated"
EV_SESSION_CLOSED: Final[str] = "session.closed"
EV_ERROR: Final[str] = "error"

# Prefijo de los eventos de la delegación Responses, ya desenvueltos de su
# `response.event`. La API pide tolerar tipos nuevos: por eso se despacha por
# prefijo y no por lista cerrada.
RESPONSE_PREFIX: Final[str] = "response."

# Eventos cuyo payload es audio, y la llave donde viaja. No se loguean nunca
# completos.
AUDIO_KEY: Final[dict[str, str]] = {
    EV_INPUT_AUDIO_APPEND: "audio",
    EV_OUTPUT_AUDIO_DELTA: "delta",
}
EVENTS_WITH_AUDIO: Final[frozenset[str]] = frozenset(AUDIO_KEY)

# Los eventos del servidor sobre los que la app **despacha**. El censo los usa
# para separar "llegó algo que no manejo" de "llegó algo que ignoro a
# propósito": las dos cosas son hallazgos, pero solo la primera es sospechosa.
HANDLED_EVENTS: Final[frozenset[str]] = frozenset(
    {
        EV_SESSION_STARTED,
        EV_SESSION_UPDATED,
        EV_OUTPUT_AUDIO_DELTA,
        EV_INPUT_TRANSCRIPT,
        EV_OUTPUT_TRANSCRIPT,
        EV_DELEGATION_CREATED,
        EV_RESPONSE_EVENT,
        EV_SESSION_USAGE,
        EV_SESSION_CLOSED,
        EV_ERROR,
    }
)

# Documentados pero que la app no consume. Se distinguen de los desconocidos
# para que el censo no los reporte como sorpresa.
KNOWN_UNUSED_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "session.instructions.appended",
        "session.thinking.appended",
        "session.commentary.appended",
    }
)

# El servidor da 10 s como máximo para drenar tras `session.close`.
CLOSE_TIMEOUT_S: Final[float] = 10.0
# Tope para el handshake y para el `session.started`. No sale de los docs: es
# criterio propio, del mismo orden que el resto de la comparativa.
CONNECTION_TIMEOUT_S: Final[float] = 15.0
START_TIMEOUT_S: Final[float] = 15.0


class AlphaError(RuntimeError):
    """El alpha respondió con un `error` que impide continuar."""


# RMS por debajo del cual una trama de PCM16 es silencio y no voz. El habla
# normal anda en miles; el silencio digital es 0 y el ruido de línea, decenas.
#
# El valor está congelado y es el **mismo** en los tres sitios donde vive
# (`shared/carrier_simulator/observado.RMS_CON_VOZ`, el humo de sala de PoC2 y
# éste). No se unifican en un solo módulo porque eso tocaría la ruta de PoC1,
# que ya entregó campaña; en su lugar `tests/test_bridge.py::
# TestUmbralDeVozAnclado` afirma que los tres coinciden, para que una
# divergencia futura salte ahí y no dentro de una tabla de latencias.
#
# El relleno del modelo es silencio digital y no ruido de bajo nivel, así que el
# umbral no dispara sobre él. La calibración está en el anexo de
# docs/comparativo_resultados.md.
SPEECH_THRESHOLD_RMS: Final[float] = 200.0

# Ventana sobre la que se decide si hay voz. 20 ms es la trama de Media Streams
# y es lo que miden `shared/carrier_simulator` (y por lo tanto PoC1 y PoC2)
# sobre cada trama de Twilio.
#
# Promediar el delta entero (~100 ms) diluiría el arranque contra el silencio que
# lo precede y retrasaría el acuse solo en PoC3, justo en la única métrica de
# latencia que las tres comparten. Con la ventana igualada, la diferencia de
# método entre las tres no incluye ese sesgo (medición en el anexo de
# docs/comparativo_resultados.md).
SPEECH_WINDOW_S: Final[float] = 0.020
SAMPLES_PER_SPEECH_WINDOW: Final[int] = int(LIVE_SAMPLE_RATE * SPEECH_WINDOW_S)


def rms_pcm16(pcm: bytes) -> float:
    """Energía RMS de un bloque de PCM16 LE. 0.0 si viene vacío."""
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    pair = len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)
    samples = np.frombuffer(pcm[:pair], dtype="<i2").astype(np.float64)
    if not samples.size:
        return 0.0
    return float(np.sqrt((samples**2).mean()))


def has_speech(
    pcm: bytes,
    *,
    umbral: float = SPEECH_THRESHOLD_RMS,
    samples_per_window: int = SAMPLES_PER_SPEECH_WINDOW,
) -> bool:
    """¿Este bloque trae voz? Cruza si **alguna** ventana de 20 ms cruza.

    **Es un instrumento de medición, no un VAD.** Nada del comportamiento
    conversacional cuelga de esta señal: el turn-taking, la detección de fin de
    turno y las interrupciones las sigue decidiendo el servidor, que es lo que
    se está comparando. Esto solo pone marcas de tiempo para poder medir
    `full_turn_ms` desde fuera, sobre el audio que entra y sale del puente.

    Ver `docs/METRICS_CONTRACT.md` §2.7: es una asimetría **de método**, no de
    configuración. PoC1 y PoC2 componen la latencia desde eventos internos de su
    framework; PoC3 no tiene esos eventos utilizables y la mide por observación
    externa. Las tres reportan la misma magnitud.

    ## Por qué el máximo por ventana y no el promedio del bloque

    El bloque que llega aquí es un `output_audio.delta` entero, de ~100 ms.
    Promediarlo diluiría el arranque contra el silencio que lo precede dentro
    del mismo bloque y retrasaría la marca. Ver `SPEECH_WINDOW_S`.

    ## La cola incompleta

    Si el bloque tiene al menos una ventana completa, el resto de menos de 20 ms
    se descarta: su RMS no es comparable con el de una ventana entera y los
    deltas llegan contiguos, así que ese audio lo ve la llamada siguiente. Si el
    bloque **no** llega a una ventana completa se mide entero, para que un
    bloque corto no sea invisible por definición.
    """
    if len(pcm) < BYTES_PER_SAMPLE:
        return False
    pair = len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)
    samples = np.frombuffer(pcm[:pair], dtype="<i2")
    completed = len(samples) // samples_per_window
    if not completed:
        return rms_pcm16(pcm) >= umbral
    blocks = (
        samples[: completed * samples_per_window]
        .astype(np.float64)
        .reshape(completed, samples_per_window)
    )
    return bool(np.sqrt((blocks**2).mean(axis=1)).max() >= umbral)


def summarize_for_log(event: dict[str, Any]) -> str:
    """Un evento listo para loguear, con el audio reemplazado por su tamaño.

    Guardarraíl explícito: **nunca** sale a los logs el contenido de
    `input_audio.append` ni de `output_audio.delta`. Un solo volcado de esos
    escribe minutos de voz de un transportista en un archivo de texto (LFPDPPP),
    y además vuelve el log ilegible.

    Todo lo demás sí sale completo: los eventos JSON del alpha son justo lo que
    hay que poder auditar contra `docs_alpha/` cuando algo falle.
    """
    return json.dumps(no_audio(event), ensure_ascii=False)


def no_audio(event: dict[str, Any]) -> dict[str, Any]:
    """El evento con el payload de audio reemplazado por su tamaño.

    Mismo guardarraíl que `summarize_for_log`, pero devolviendo el diccionario:
    lo necesita el censo, que escribe JSON por línea en vez de texto.
    """
    tipo = str(event.get("type", "?"))
    key = AUDIO_KEY.get(tipo)
    if key is None:
        return dict(event)

    clone = {k: v for k, v in event.items() if k != key}
    raw_value = event.get(key)
    n = 0
    if isinstance(raw_value, str):
        # Tamaño decodificado aproximado: base64 crece 4/3.
        n = len(raw_value) * 3 // 4
    clone[key] = f"<audio {n} bytes>"
    return clone


class EventCensus:
    """Registra **todo** lo que llega del alpha, con la hora local de llegada.

    Instrumentación pura: no cambia el comportamiento del puente, corre en
    paralelo al despacho normal. Existe porque el alpha se desvía de su propia
    documentación en al menos un evento (`response.function_call_arguments.done`
    llega sin `call_id` ni `name`, contra lo que muestra `Alpha_guide.html`), y
    la única forma honesta de cerrar esa brecha es mirar lo que de verdad manda
    el servicio en vez de adivinar.

    Escribe un JSON por línea en `censo_<call_sid>.jsonl`, con `flush` en cada
    uno: si la corrida truena a la mitad, lo capturado hasta ahí sigue sirviendo
    y no hay que volver a gastar.

    El `t_local_seg` de cada evento es la otra mitad del trabajo: es lo que
    permite reconstruir fuera de línea la relación entre la línea de tiempo del
    servidor (`start_ms`/`end_ms`) y el reloj local, que hoy deriva y no está
    documentada.
    """

    def __init__(
        self,
        destination: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self._clock = clock
        self._t0 = clock()
        self._file = destination.open("w", encoding="utf-8")
        self._n = 0
        self.count: dict[str, int] = {}
        self.first_appearance: list[str] = []
        self.unknown_ones: dict[str, int] = {}

    def registrar(self, event: dict[str, Any]) -> None:
        tipo = str(event.get("type", "?"))
        elapsed = self._clock() - self._t0
        self._n += 1

        if tipo not in self.count:
            self.first_appearance.append(tipo)
        self.count[tipo] = self.count.get(tipo, 0) + 1
        if tipo not in HANDLED_EVENTS and tipo not in KNOWN_UNUSED_EVENTS:
            self.unknown_ones[tipo] = self.unknown_ones.get(tipo, 0) + 1

        row = {
            "n": self._n,
            "t_local_seg": round(elapsed, 6),
            "type": tipo,
            "keys": sorted(event),
            "manejado": tipo in HANDLED_EVENTS,
            "event": no_audio(event),
        }
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def summary(self) -> dict[str, Any]:
        """Qué llegó, cuánto de cada cosa y en qué orden apareció por primera vez."""
        return {
            "file": str(self.destination),
            "events": self._n,
            "count_by_kind": dict(
                sorted(self.count.items(), key=lambda pair: (-pair[1], pair[0]))
            ),
            "first_appearance": list(self.first_appearance),
            "unknown_ones": dict(self.unknown_ones),
            "response_unknown": {
                tipo: n
                for tipo, n in self.unknown_ones.items()
                if tipo.startswith(RESPONSE_PREFIX)
            },
        }

    def shut(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._file.close()


class GptLiveClient:
    """El WebSocket contra GPT-Live.

    Ciclo de vida:

        cliente = ClienteGptLive(url, headers, initial_session)
        await cliente.conectar()          # abre, configura y espera session.started
        async for evento in cliente.escuchar():
            ...
        await cliente.cerrar()            # session.close y drena hasta session.closed

    No interpreta los eventos: los entrega parseados. Quien decide qué significa
    cada uno es `app/telephony.py`, que es donde vive el estado de la llamada.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        initial_session: dict[str, Any],
        *,
        census: EventCensus | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers)
        self._initial_session = dict(initial_session)
        self._ws: Any = None
        self._event_counter = 0
        # Instrumentación opcional. Cuelga de `_recibir`, que es el único punto
        # por el que pasan **todos** los eventos —incluidos los del arranque,
        # antes de `session.started`, y los del drenado tras `session.close`.
        self.census = census
        # Lo que el servidor confirmó en `session.started`. Es la única fuente
        # de verdad de qué configuración corrió de verdad, y va al JSON.
        self.session_confirmed: dict[str, Any] = {}
        # Último `usage` reportado. El alpha lo manda ~1 vez por minuto y otra
        # vez completo en `session.closed`.
        self.usage: dict[str, Any] = {}
        self.closed: bool = False

    # -- Identificadores de evento --

    def _next_id(self, prefix: str) -> str:
        """`event_id` propio. Solo los errores lo devuelven, para correlacionar.

        Los acuses de recibo **no** hacen eco del `event_id`
        (`docs_alpha/Alpha_guide.html`): sirve para saber qué petición nuestra
        falló, no para emparejar respuestas.
        """
        self._event_counter += 1
        return f"{prefix}_{self._event_counter}"

    # -- Conexión --

    async def connect(self) -> dict[str, Any]:
        """Abre el WebSocket, manda `session.start` y espera `session.started`.

        `session.start` es el primer mensaje del socket y se manda una sola vez;
        lleva el modelo adentro. El esquema es estricto con los campos
        desconocidos, así que `initial_session` lleva solo los que existen.

        Devuelve el objeto `session` confirmado por el servidor.
        """
        import websockets

        logger.info("abriendo sesión de GPT-Live: %s", self._url)
        self._ws = await asyncio.wait_for(
            # `additional_headers` es el nombre en websockets >= 14.
            websockets.connect(self._url, additional_headers=self._headers),
            timeout=CONNECTION_TIMEOUT_S,
        )

        await self._send(
            {
                "type": EV_SESSION_START,
                "event_id": self._next_id("start"),
                "session": self._initial_session,
            }
        )

        # `session.started` es el primer evento de ciclo de vida, no
        # `session.updated`. Un `error` antes de él es un fallo de arranque y la
        # sesión no llega a existir.
        limit = asyncio.get_running_loop().time() + START_TIMEOUT_S
        while True:
            remaining = limit - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AlphaError(
                    f"GPT-Live no mandó {EV_SESSION_STARTED} en "
                    f"{START_TIMEOUT_S:.0f} s"
                )
            event = await asyncio.wait_for(self._receive(), timeout=remaining)
            tipo = str(event.get("type", ""))
            if tipo == EV_SESSION_STARTED:
                self.session_confirmed = dict(event.get("session") or {})
                logger.info(
                    "sesión iniciada: %s", summarize_for_log(event)
                )
                return self.session_confirmed
            if tipo == EV_ERROR:
                detail = event.get("error") or {}
                raise AlphaError(
                    f"error de arranque de GPT-Live: {json.dumps(detail, ensure_ascii=False)}"
                )
            logger.debug("evento antes de session.started: %s", summarize_for_log(event))

    # -- Envío --

    async def _send(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise AlphaError("la sesión de GPT-Live no está abierta")
        logger.debug("-> %s", summarize_for_log(event))
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    async def send_audio(self, pcm24: bytes) -> None:
        """`session.input_audio.append` con el audio de la sesión en base64.

        Los payloads vacíos y, en PCM16, los de tamaño impar no se mandan: el
        servidor los rechaza, y un rechazo por trama llenaría el log de
        errores mientras la llamada corre. En μ-law (el formato lo dice la
        sesión inicial) cada byte es una muestra y no hay alineación que
        cuidar. El audio crudo **no se acusa de recibo**, así que esto no
        espera nada.
        """
        format_kind = ((self._initial_session.get("audio") or {}).get("format") or {})
        pcm = format_kind.get("type", "audio/pcm") == "audio/pcm"
        if not pcm24 or (pcm and len(pcm24) % BYTES_PER_SAMPLE):
            if pcm24:
                logger.warning(
                    "no se manda un payload de %d bytes: PCM16 exige número par",
                    len(pcm24),
                )
            return
        await self._send(
            {
                "type": EV_INPUT_AUDIO_APPEND,
                "audio": base64.b64encode(pcm24).decode("ascii"),
            }
        )

    async def answer_function(self, event: dict[str, Any]) -> None:
        """Manda un `response.item.create` con la salida de una función.

        Lo construye `app/tools.build_output_event`: aquí solo se le pone
        el `event_id` si no lo trae y se manda. Encolar la salida **no** reanuda
        al modelo delegado: eso lo hace `proceed_delegation`, cuando ya se
        contestaron todas las llamadas de la respuesta (`SeguimientoDeRespuestas`).
        Un `call_id` desconocido o ya resuelto lo rechaza el servidor, así que
        **no se reintenta**.
        """
        output = dict(event)
        output.setdefault("event_id", self._next_id("function"))
        await self._send(output)

    async def proceed_delegation(self) -> None:
        """`response.create`: reanuda al modelo delegado con las salidas encoladas.

        El servidor lo rechaza (`function_call_outputs_required`) si falta
        alguna salida de la respuesta pendiente; por eso solo se manda cuando
        `SeguimientoDeRespuestas` dice que ya están todas.
        """
        await self._send(
            {"type": EV_RESPONSE_CREATE, "event_id": self._next_id("proceed")}
        )

    async def add_context(
        self, text: str, *, tipo: str = EV_COMMENTARY_APPEND
    ) -> None:
        """Agrega contexto a la sesión: un texto de 500 tokens como máximo.

        Por default va como `session.commentary.append`, que es como la API le
        pide al modelo que hable —en sus palabras, no leyéndolo—, y es lo que
        abre la llamada (`OPENING_CONTEXT`). `delegation_id` es obligatorio
        en estos eventos y `null` significa contexto general de la sesión.
        """
        await self._send(
            {
                "type": tipo,
                "event_id": self._next_id("context"),
                "delegation_id": None,
                "content": text,
            }
        )

    # -- Recepción --

    async def _receive(self) -> dict[str, Any]:
        if self._ws is None:
            raise AlphaError("la sesión de GPT-Live no está abierta")
        raw_value = await self._ws.recv()
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        data = json.loads(raw_value)
        event = data if isinstance(data, dict) else {"type": "?", "raw_value": data}
        if self.census is not None:
            self.census.registrar(event)
        return event

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        """Los eventos del servidor, ya parseados, hasta que la sesión cierre.

        Lleva por su cuenta dos cosas que interesan a todos los consumidores —el
        `usage` acumulado y la bandera de cerrada— y entrega todo lo demás sin
        interpretar, con una excepción: cada `response.event` sale
        **desenvuelto** (`unwrap_response`), para que el resto del puente
        vea los `response.*` como siempre. El censo guarda el sobre original.
        """
        import websockets

        try:
            while True:
                event = await self._receive()
                tipo = str(event.get("type", ""))

                if tipo == EV_RESPONSE_EVENT:
                    yield unwrap_response(event)
                    continue
                if tipo == EV_SESSION_USAGE:
                    self.usage = dict(event.get("usage") or {})
                elif tipo == EV_SESSION_CLOSED:
                    self.usage = dict(event.get("usage") or self.usage)
                    self.closed = True
                    yield event
                    return

                yield event
        except websockets.exceptions.ConnectionClosed as error:
            # Que el transporte se caiga no es una excepción del programa: es el
            # final del flujo, y quien escucha tiene que poder escribir el JSON
            # de la corrida. Se marca y se sale.
            logger.info("el WebSocket de GPT-Live cerró: %s", error)
            self.closed = True

    # -- Cierre --

    async def shut(self) -> None:
        """`session.close`, drena hasta `session.closed` y cierra el transporte.

        La API pide seguir leyendo hasta `session.closed`; el drenado tiene un
        máximo de 10 s del lado del servidor, así que aquí se espera lo mismo y
        ni un segundo más. Si ya cerró o el transporte murió, no hace nada.
        """
        if self._ws is None:
            return
        try:
            if not self.closed:
                await self._send(
                    {
                        "type": EV_SESSION_CLOSE,
                        "event_id": self._next_id("cierre"),
                    }
                )
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        self._drain_until_close(), timeout=CLOSE_TIMEOUT_S
                    )
        finally:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
            self.closed = True

    async def _drain_until_close(self) -> None:
        while True:
            event = await self._receive()
            if str(event.get("type", "")) == EV_SESSION_CLOSED:
                self.usage = dict(event.get("usage") or self.usage)
                logger.info("sesión cerrada: %s", summarize_for_log(event))
                return


# --------------------------------------------------------------------------
# Accesos a los eventos, con nombre
# --------------------------------------------------------------------------
#
# Existen para que `telefonia.py` no vaya sacando llaves anidadas a mano en
# medio de la orquestación, y para que la forma de cada evento esté escrita en
# un solo lugar.


def unwrap_response(on_top: dict[str, Any]) -> dict[str, Any]:
    """El evento de la Responses API que viene dentro de un `response.event`.

    Se le agrega el `delegation_id` del sobre, que es lo que correlaciona las
    llamadas a función de una misma respuesta (los eventos de adentro no todos
    nombran su respuesta). Un sobre sin `event` sale como `response.event`
    vacío, que nadie despacha.
    """
    internal = on_top.get("event")
    if not isinstance(internal, dict) or not isinstance(internal.get("type"), str):
        return {"type": EV_RESPONSE_EVENT, "delegation_id": on_top.get("delegation_id")}
    return {**internal, "delegation_id": on_top.get("delegation_id")}


def event_audio(event: dict[str, Any]) -> bytes:
    """Los bytes PCM16 de un `session.output_audio.delta`. Vacío si viene mal."""
    raw_value = event.get("delta")
    if not isinstance(raw_value, str) or not raw_value:
        return b""
    try:
        return base64.b64decode(raw_value)
    except (ValueError, TypeError):
        logger.warning("session.output_audio.delta con base64 ilegible: se descarta")
        return b""


def transcript_text(event: dict[str, Any]) -> str:
    """El texto de un `session.input_transcript.delta` / `output_transcript.delta`.

    **Se devuelve crudo, sin `strip()`.** El servidor manda el espacio que separa
    dos palabras al **principio** del fragmento (`" te"`, `" están"`), así que
    recortarlo y luego concatenar pega las palabras: la llamada del 12 de agosto
    de 2026 salió con `"teestán"` y el WER de un STT casi perfecto se reportó en
    84.5% (las 32 borraduras del CER eran justo los espacios comidos). Quien
    junta los fragmentos es `RegistroDeLlamada.anotar`, que limpia al final en
    `normalizar()`; aquí no se toca nada.
    """
    delta = event.get("delta")
    return delta if isinstance(delta, str) else ""


def function_call(event: dict[str, Any]) -> tuple[str, str, Any] | None:
    """`(call_id, nombre, argumentos)` de un `response.output_item.done` accionable.

    Accionable = item `function_call` con `status: completed`, `call_id` y
    `name`. `None` en cualquier otro caso (un item de texto, uno incompleto).
    `arguments` viene como cadena JSON y se deja cruda: quien la parsea es
    `app/tools.py`.
    """
    if str(event.get("type", "")) != EV_OUTPUT_ITEM_DONE:
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    if item.get("status") != "completed":
        logger.debug("function_call con status %r: no se ejecuta", item.get("status"))
        return None
    call_id = str(item.get("call_id") or "")
    name = str(item.get("name") or "")
    if not call_id or not name:
        logger.warning(
            "llamada a función sin call_id o sin name: %s", summarize_for_log(event)
        )
        return None
    return call_id, name, item.get("arguments")


@dataclass
class _PendingResponse:
    """Una respuesta del modelo delegado cuyas llamadas a función se van juntando."""

    unanswered: set[str] = field(default_factory=set)
    had_calls: bool = False
    termino: bool = False


class ResponseTracker:
    """Cuándo reanudar al modelo delegado, y qué llamadas ya se contestaron.

    ## Por qué hace falta

    Con el alpha, contestar una función (`delegation.function_call_output.create`)
    reanudaba solo al modelo delegado. En la API pública son dos pasos: la salida
    se **encola** con `response.item.create` y el modelo se reanuda con
    `response.create`, que el servidor rechaza si a la respuesta le falta alguna
    salida. Así que hay que saber, por respuesta, qué llamadas hizo, cuáles ya se
    contestaron y si ya terminó de emitir (`response.completed` /
    `.incomplete` / `.failed`). Es la misma lógica que `_PendingResponse` en la
    integración de referencia de pipecat-ai 1.12.0.

    La clave es el `delegation_id` del sobre (`unwrap_response`): los
    eventos de adentro no todos nombran su respuesta.

    ## Lo que se conserva del alpha

    El conjunto de `call_id` ya contestados. En la llamada del 12 de agosto de
    2026 el servidor rechazó una función contestada dos veces
    (`Unknown handoff function call id`); con este conjunto un item repetido no
    se vuelve a ejecutar.
    """

    _NO_DELEGATION: Final[str] = "no_delegation"

    def __init__(self) -> None:
        self._pending_items: dict[str, _PendingResponse] = {}
        self._call_key: dict[str, str] = {}
        self._answered: set[str] = set()

    def _clave(self, event: dict[str, Any]) -> str:
        return str(event.get("delegation_id") or self._NO_DELEGATION)

    def created(self, event: dict[str, Any]) -> None:
        """`response.created`: empieza a seguir esta respuesta."""
        self._pending_items.setdefault(self._clave(event), _PendingResponse())

    def new_call(self, event: dict[str, Any]) -> tuple[str, str, Any] | None:
        """La llamada accionable de un `output_item.done`, si no se contestó ya.

        Devuelve `(call_id, nombre, argumentos)` y la anota como pendiente de
        contestar en su respuesta. `None` si el evento no es una llamada
        accionable o si ese `call_id` ya se ejecutó.
        """
        data = function_call(event)
        if data is None:
            return None
        call_id = data[0]
        if call_id in self._answered or call_id in self._call_key:
            logger.warning(
                "%r ya se ejecutó: no se responde otra vez (el servidor rechaza "
                "los IDs ya resueltos)",
                call_id,
            )
            return None
        clave = self._clave(event)
        pending = self._pending_items.setdefault(clave, _PendingResponse())
        pending.unanswered.add(call_id)
        pending.had_calls = True
        self._call_key[call_id] = clave
        return data

    def answered(self, call_id: str) -> bool:
        """La salida de `call_id` ya se encoló. ¿Toca mandar `response.create`?"""
        self._answered.add(call_id)
        clave = self._call_key.pop(call_id, None)
        if clave is None:
            return False
        pending = self._pending_items.get(clave)
        if pending is not None:
            pending.unanswered.discard(call_id)
        return self._ready_to_continue(clave)

    def finished(self, event: dict[str, Any]) -> bool:
        """Fin de la respuesta. ¿Toca mandar `response.create`?"""
        clave = self._clave(event)
        pending = self._pending_items.get(clave)
        if pending is None:
            return False
        pending.termino = True
        return self._ready_to_continue(clave)

    def _ready_to_continue(self, clave: str) -> bool:
        pending = self._pending_items.get(clave)
        if (
            pending is None
            or not pending.termino
            or not pending.had_calls
            or pending.unanswered
        ):
            return False
        del self._pending_items[clave]
        return True

    @property
    def pending_items(self) -> int:
        """Llamadas ejecutadas cuya salida todavía no se encola."""
        return len(self._call_key)
