"""Bajar a banda telefónica (8 kHz) y, si hace falta, volver a subir.

## Por qué hay dos motores y no uno

`soxr` es el remuestreador que ya usan PoC1 (por dentro de pipecat) y PoC3 (en
`bridge.py`), y es el que debería preparar los clips. Pero **no está instalado
en el entorno de PoC2**, y agregarlo cambiaría el `uv.lock` con el que corrieron
sus 15 llamadas — o sea, cambiaría el ambiente documentado de una arquitectura
por una necesidad del arnés. No se hace.

La salida es que el remuestreo caro ocurre **una sola vez, al preparar**, y el
manifiesto registra con qué motor se hizo. Los tres arneses reproducen después
los mismos bytes ya degradados, así que el motor no vuelve a aparecer en el
camino de ninguna llamada.

Cuando `soxr` no está, hay un respaldo en numpy puro que sirve para razones
enteras (48 k, 24 k, 16 k y 32 k hacia 8 kHz). No es bit-idéntico a soxr — no
puede serlo — y por eso el motor se guarda en el manifiesto y `verificar` avisa
si los clips se prepararon con uno distinto del disponible.

## Por qué no interpolación lineal

Bajar de 48 kHz a 8 kHz tomando una de cada seis muestras **pliega** todo lo que
esté arriba de 4 kHz hacia adentro de la banda de voz, como distorsión audible.
Un clip así no suena a teléfono: suena a clip roto, y el WER mediría eso. Hay
que filtrar antes de decidir, que es lo que hacen los dos motores.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Banda telefónica: la de Twilio, la del trunk SIP y la de las 15 llamadas de
# cada PoC. Mismo valor que `poc1_pipecat/app/telefonia.py` y `bridge.py`.
TELEPHONY_SAMPLE_RATE: Final[int] = 8000

# Corte del filtro, como fracción de la tasa de origen. 0.45 del Nyquist destino
# deja la banda de voz (300-3400 Hz) intacta y una transición cómoda por debajo
# de los 4 kHz. Subirlo a 0.5 dejaría alias en el borde; bajarlo empezaría a
# comerse las fricativas, que es justo lo que el escenario 5 quiere estresar.
_CUTOFF_FRACTION: Final[float] = 0.45

# Largo del FIR del respaldo, en múltiplos de la razón. 16 da ~97 taps para la
# razón 6 (48 k -> 8 k): suficiente para 60 dB de atenuación en la banda
# eliminada con ventana de Hamming.
_TAPS_PER_RATIO: Final[int] = 16


def _soxr():
    """`soxr` si está instalado, `None` si no. Import perezoso a propósito."""
    try:
        import soxr
    except ImportError:
        return None
    return soxr


def engine_description() -> str:
    """Qué remuestreador se usaría aquí y ahora. Va al manifiesto."""
    soxr = _soxr()
    if soxr is not None:
        return f"soxr {getattr(soxr, '__version__', '?')} (HQ)"
    return f"numpy-fir (hamming, {_TAPS_PER_RATIO} taps por razón)"


def _lowpass_filter(taps: int, cutoff: float) -> np.ndarray:
    """Sinc enventanado con Hamming. `corte` en ciclos por muestra (0 a 0.5)."""
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.hamming(taps)
    return h / h.sum()


def _downsample_int(samples: np.ndarray, reason: int) -> np.ndarray:
    """Decima por `razon` filtrando antes. Respaldo sin `soxr`."""
    taps = _TAPS_PER_RATIO * reason + 1  # impar: fase lineal
    h = _lowpass_filter(taps, _CUTOFF_FRACTION / reason)
    filtered = np.convolve(samples.astype(np.float64), h, mode="same")
    return _a_int16(filtered[::reason])


def _upsample_integer(samples: np.ndarray, reason: int) -> np.ndarray:
    """Inserta ceros y filtra. La ganancia del filtro se compensa con `razon`."""
    stretched = np.zeros(len(samples) * reason, dtype=np.float64)
    stretched[::reason] = samples.astype(np.float64)
    taps = _TAPS_PER_RATIO * reason + 1
    h = _lowpass_filter(taps, _CUTOFF_FRACTION / reason) * reason
    return _a_int16(np.convolve(stretched, h, mode="same"))


def _a_int16(values: np.ndarray) -> np.ndarray:
    """Redondea y satura. Sin el recorte, un pico se envuelve y suena a clic."""
    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


def downsample_to_telephony(samples: np.ndarray, origin_sample_rate: int) -> np.ndarray:
    """PCM16 a cualquier tasa -> PCM16 a 8 kHz.

    Con `soxr` acepta cualquier tasa. Sin él, solo razones enteras: 44.1 kHz y
    22.05 kHz no son múltiplos de 8 kHz y no hay respaldo honesto para ellas.
    """
    samples = np.asarray(samples, dtype=np.int16)
    if origin_sample_rate == TELEPHONY_SAMPLE_RATE:
        return samples
    if origin_sample_rate < TELEPHONY_SAMPLE_RATE:
        raise ValueError(
            f"el clip está a {origin_sample_rate} Hz, por debajo de los {TELEPHONY_SAMPLE_RATE} Hz "
            "de la banda telefónica: no se puede inventar el ancho de banda que le "
            "falta. Vuelve a grabarlo a 16 kHz o más"
        )

    soxr = _soxr()
    if soxr is not None:
        lowered = soxr.resample(samples, origin_sample_rate, TELEPHONY_SAMPLE_RATE, quality="HQ")
        return _a_int16(np.asarray(lowered))

    if origin_sample_rate % TELEPHONY_SAMPLE_RATE:
        raise ValueError(
            f"{origin_sample_rate} Hz no es múltiplo de {TELEPHONY_SAMPLE_RATE} Hz y este entorno "
            "no tiene `soxr`. Dos salidas: exporta los clips a 48 kHz (o 16/24/32 "
            "kHz), o corre la preparación en el entorno de PoC1 o PoC3, que sí lo "
            "traen. El manifiesto guarda con qué motor se preparó cada clip"
        )
    return _downsample_int(samples, origin_sample_rate // TELEPHONY_SAMPLE_RATE)


def upsample_from_telephony(samples: np.ndarray, target_sample_rate: int) -> np.ndarray:
    """8 kHz -> `target_sample_rate`. **Solo para el caso en que un arnés lo necesite.**

    Hoy ninguno lo usa en el camino normal: PoC1 y PoC3 mandan μ-law de 8 kHz
    tal cual, y el arnés de PoC2 publica su pista **a 8 kHz** para que el salto a
    48 kHz lo dé LiveKit — que es exactamente lo que hace con el audio del trunk
    SIP en una llamada real. Hacerlo nosotros metería un remuestreo que la
    arquitectura medida no tiene.

    Queda disponible por si publicar a 8 kHz resultara problemático, y en ese
    caso la sustitución se declara como asimetría.
    """
    samples = np.asarray(samples, dtype=np.int16)
    if target_sample_rate == TELEPHONY_SAMPLE_RATE:
        return samples
    if target_sample_rate < TELEPHONY_SAMPLE_RATE:
        raise ValueError(f"{target_sample_rate} Hz está por debajo de la banda telefónica")

    soxr = _soxr()
    if soxr is not None:
        upsampled = soxr.resample(samples, TELEPHONY_SAMPLE_RATE, target_sample_rate, quality="HQ")
        return _a_int16(np.asarray(upsampled))

    if target_sample_rate % TELEPHONY_SAMPLE_RATE:
        raise ValueError(
            f"{target_sample_rate} Hz no es múltiplo de {TELEPHONY_SAMPLE_RATE} Hz y este entorno "
            "no tiene `soxr`"
        )
    return _upsample_integer(samples, target_sample_rate // TELEPHONY_SAMPLE_RATE)
