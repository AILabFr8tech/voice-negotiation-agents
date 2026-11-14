"""Códec G.711 μ-law, vectorizado. La degradación telefónica del arnés.

## Por qué a mano y no con `audioop`

`audioop` lo haría en dos líneas, pero **se eliminó de la biblioteca estándar en
Python 3.13**. Las tres PoC corren 3.12, así que hoy funcionaría; atar el arnés
a un módulo ya borrado significa que actualizar Python lo rompe. Son cuarenta
líneas de tablas G.711 y se prueban contra `audioop` mientras exista.

## Por qué está duplicado con `poc3_gpt_live/app/bridge.py`

Es el mismo códec, byte por byte, y la duplicación es deliberada: `shared/` no
puede depender de una PoC (CLAUDE.md), y PoC3 ya terminó sus 15 llamadas — su
`bridge.py` está congelado y no se toca para acomodar al arnés.

La copia no puede derivar en silencio: `tests/test_codec.py` compara las dos
implementaciones sobre los 256 códigos **cuando PoC3 es importable**, y contra
`audioop` siempre. Si alguna vez se unifican, el sentido correcto es que
`bridge.py` importe de aquí, nunca al revés.
"""

from __future__ import annotations

from typing import Final

import numpy as np

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


def round_trip_ulaw(samples: np.ndarray) -> np.ndarray:
    """PCM16 -> μ-law -> PCM16: la pérdida del códec, aplicada una vez.

    Es la mitad de la degradación telefónica (la otra es bajar a 8 kHz). Se
    aplica al preparar el clip, **no al reproducirlo**: el arnés de PoC1 y PoC3
    manda los bytes μ-law tal cual, y el de PoC2 publica el PCM ya degradado.

    No es idempotente por casualidad, lo es por construcción: el segundo viaje
    cuantiza valores que ya caen exactamente sobre los 256 niveles. Aun así, el
    arnés lo aplica una sola vez — aplicarlo dos veces no cambiaría el audio pero
    sí el hash del intermedio, y el hash es la evidencia.
    """
    return ulaw_a_pcm16(pcm16_a_ulaw(samples))
