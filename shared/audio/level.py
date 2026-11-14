"""Nivel de los clips: medirlo y ajustarlo al de una llamada real.

## Por qué el nivel es una variable de la medición

Un clip grabado en la laptop puede salir 15 dB más fuerte o más débil que lo que
entrega un teléfono. No cambia *lo que dice*, pero sí **todo lo que se mide con
él**: el VAD decide sobre energía (un clip fuerte cruza el umbral antes y el cero
de la latencia se mueve) y el STT degrada con la relación sig/noise (un WER
medido con audio más fuerte que el de las 15 llamadas no compara con ellas). Sin
calibrar, el arnés mediría su propia ganancia.

## Qué se normaliza y dónde se aplica

El **RMS de los segmentos con voz**, no el del archivo completo: el RMS completo
cuelga de cuánto silencio traiga la toma, así que la "calibración" estaría
midiendo el silencio. Es la misma métrica con la que se midió la referencia.

La ganancia se aplica a la tasa original, **antes** del remuestreo y del μ-law,
para que el códec cuantice la misma amplitud que cuantizaría en una llamada real
(μ-law es logarítmico: el error depende del nivel de entrada). El precio es que
el objetivo no se alcanza exacto —el antialias se lleva lo de arriba de 4 kHz—,
así que se **verifica después** y se avisa si la desviación pasa de
`TOLERANCE_DB`.

## El objetivo es un parámetro, no una constante del código

Vive en `calibracion_nivel.json`, junto a la medición de la que salió. Hoy es
**provisional**: se midió sobre una grabación de Twilio del agente despachador
—voz humana por la misma línea, pero de otro sistema—. Cuando haya acceso a las
grabaciones de 2 canales de PoC1 se cambia ese número y se regeneran los clips;
el manifiesto guarda con qué objetivo se preparó cada uno, así que un clip viejo
contra un objetivo nuevo se caza al verificar.

**Mientras `provisional` siga en `true`, cualquier WER medido con estos clips es
validación de plomería, no un dato citable.**
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

from shared.audio.noise import dbfs_pico, dbfs_rms

# La medición de referencia y el objetivo que sale de ella. Se commitea.
CALIBRATION_PATH: Final[Path] = Path(__file__).resolve().parent / "level_calibration.json"

# Valores de respaldo, solo por si el JSON no está. No son "el default": el
# default es el archivo, que es donde se cambia el objetivo sin tocar código.
FALLBACK_TARGET_DBFS: Final[float] = -18.0
FALLBACK_NOISE_DBFS: Final[float] = -60.0

# Ventana de análisis. 10 ms es más corto que cualquier fonema, así que no parte
# una sílaba a la mitad ni se come el arranque de una oclusiva.
WINDOW_MS: Final[int] = 10

# El umbral de voz cuelga del pico del propio clip: un piso absoluto dependería
# de qué tan fuerte grabó cada quien, que es justo lo que se está corrigiendo.
#
# Hay **dos** caídas y no una, porque las dos preguntas son distintas:
#
# - Para RECORTAR hace falta ser permisivo: lo que se busca es dónde deja de
#   haber voz, y morder una fricativa final ("...mande**s**") movería el cero del
#   fin de voz, que es justo lo que el recorte existe para proteger.
# - Para MEDIR EL NIVEL hace falta ser estricto: incluir los colgajos de baja
#   energía hunde el RMS y hace pedir ganancia de más.
#
# Conflacionarlas costó 3 dB de error medido (ver el default de más abajo).
DROP_FOR_TRIM_DB: Final[float] = 35.0
DROP_FOR_LEVEL_DB: Final[float] = 25.0
# Piso duro: por debajo de esto ya es ruido de micrófono en cualquier grabación.
MIN_THRESHOLD_DBFS: Final[float] = -55.0

# Tras aplicar la ganancia el pico no pasa de aquí. El objetivo de -18 dBFS RMS
# con el factor de cresta típico del habla (~15 dB) deja picos cerca de -3 dBFS;
# un clip con cresta mayor pediría ganancia que recortaría la señal, y recortar
# mete distorsión audible que el STT sí acusa. Antes de eso se limita la
# ganancia y se avisa: el clip queda por debajo del objetivo, pero limpio.
MAX_PEAK_AFTER_GAIN_DBFS: Final[float] = -1.0

# Cuánto puede alejarse del objetivo el RMS de voz ya degradado antes de avisar.
TOLERANCE_DB: Final[float] = 1.0

# Por debajo de este RMS de voz, la grabación original está tan baja que subirla
# al objetivo sube también su piso de ruido. Se avisa; se arregla regrabando.
SOURCE_RMS_TOO_LOW_DBFS: Final[float] = -40.0

_FULL_SCALE: Final[float] = 32768.0


class EmptyClip(ValueError):
    """El archivo no trae voz por encima del umbral: hay que volver a grabarlo."""


@dataclass(frozen=True)
class Calibration:
    """El objetivo de nivel y de dónde salió."""

    target_dbfs: float
    noise_dbfs: float
    provisional: bool
    source: str
    note: str = ""
    # El criterio viaja **con** el objetivo: el mismo audio mide hasta 9 dB
    # distinto según qué ventanas cuenten como voz, así que un objetivo sin su
    # caída no significa nada. Ver `CAIDA_PARA_NIVEL_DB`.
    drop_db: float = DROP_FOR_LEVEL_DB

    def with_target(self, target_dbfs: float | None) -> "Calibration":
        """Mismo origen, otro objetivo. Para la bandera `--target-dbfs`."""
        if target_dbfs is None:
            return self
        return replace(self, target_dbfs=float(target_dbfs))

    @property
    def caution(self) -> str:
        """Lo que hay que decir cada vez que se reporte un número con esto."""
        if not self.provisional:
            return ""
        return (
            "nivel PROVISIONAL: cualquier WER medido con estos clips es "
            "validación de plomería, no un dato citable"
        )


@lru_cache(maxsize=4)
def _read_calibration(ruta: str) -> Calibration:
    file = Path(ruta)
    if not file.is_file():
        return Calibration(
            target_dbfs=FALLBACK_TARGET_DBFS,
            noise_dbfs=FALLBACK_NOISE_DBFS,
            provisional=True,
            source=f"(respaldo: no se encontró {file.name})",
        )
    data = json.loads(file.read_text(encoding="utf-8"))
    target = data.get("calibration_target", {})
    return Calibration(
        target_dbfs=float(target.get("valor_provisional_dbfs", FALLBACK_TARGET_DBFS)),
        noise_dbfs=float(target.get("comfort_noise_dbfs", FALLBACK_NOISE_DBFS)),
        drop_db=float(target.get("threshold_drop_db", DROP_FOR_LEVEL_DB)),
        provisional=bool(target.get("provisional", True)),
        source=str(data.get("source", file.name)),
        note=str(data.get("note", "")),
    )


def load_calibration(ruta: Path | str | None = None) -> Calibration:
    """El objetivo vigente. Cambiar el número del JSON basta para moverlo."""
    return _read_calibration(str(Path(ruta or CALIBRATION_PATH)))


# --------------------------------------------------------------------------
# medición
# --------------------------------------------------------------------------


def levels_per_window(
    samples: np.ndarray, sample_rate: int, *, window_ms: int = WINDOW_MS
) -> tuple[np.ndarray, int]:
    """dBFS RMS de cada ventana. Devuelve `(niveles, muestras por ventana)`."""
    data = np.asarray(samples, dtype=np.int16)
    per_window = max(1, int(sample_rate) * window_ms // 1000)
    completed = len(data) // per_window
    if completed == 0:
        return np.zeros(0, dtype=np.float64), per_window

    blocks = data[: completed * per_window].reshape(completed, per_window)
    energy = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        levels = 20.0 * np.log10(np.maximum(energy, 1e-9) / _FULL_SCALE)
    return levels, per_window


def speech_threshold(
    samples: np.ndarray, *, drop_db: float = DROP_FOR_LEVEL_DB
) -> float:
    """Umbral por encima del cual una ventana cuenta como voz, en dBFS.

    `drop_db` es el criterio, y **no es el mismo para medir que para recortar**:
    ver el comentario de las dos constantes.
    """
    return max(dbfs_pico(samples) - drop_db, MIN_THRESHOLD_DBFS)


def windows_with_speech(
    samples: np.ndarray,
    sample_rate: int,
    *,
    umbral_dbfs: float | None = None,
    drop_db: float = DROP_FOR_LEVEL_DB,
) -> tuple[np.ndarray, int, float]:
    """`(índices de ventanas con voz, muestras por ventana, umbral usado)`."""
    levels, per_window = levels_per_window(samples, sample_rate)
    umbral = (
        speech_threshold(samples, drop_db=drop_db) if umbral_dbfs is None else umbral_dbfs
    )
    return np.flatnonzero(levels >= umbral), per_window, umbral


def rms_of_speech(
    samples: np.ndarray,
    sample_rate: int,
    *,
    umbral_dbfs: float | None = None,
    drop_db: float = DROP_FOR_LEVEL_DB,
) -> float:
    """RMS en dBFS de **solo** las ventanas con voz. `-inf` si no hay ninguna.

    Es la métrica que nombra el objetivo de calibración. El RMS del archivo
    completo no sirve: cuelga de cuánto silencio traiga la toma.
    """
    data = np.asarray(samples, dtype=np.int16)
    with_voice, per_window, _ = windows_with_speech(
        data, sample_rate, umbral_dbfs=umbral_dbfs, drop_db=drop_db
    )
    if with_voice.size == 0:
        return float("-inf")
    indices = (with_voice[:, None] * per_window + np.arange(per_window)).ravel()
    return dbfs_rms(data[indices])


@dataclass(frozen=True)
class EndOfSpeech:
    """Dónde deja de haber voz dentro de un clip, y con qué criterio.

    **El umbral viaja aquí adentro y no se puede separar del número.** El mismo
    audio mueve su medición hasta 9 dB según qué ventanas cuenten como voz (ver
    el comentario de las dos constantes de caída), así que un `end_s` suelto
    no es reproducible. Es el mismo trato que recibe el objetivo de nivel en
    `calibracion_nivel.json`.
    """

    end_s: float
    cola_seg: float
    drop_db: float
    umbral_dbfs: float
    duration_s: float


def end_of_speech(
    samples: np.ndarray,
    sample_rate: int,
    *,
    drop_db: float = DROP_FOR_LEVEL_DB,
) -> EndOfSpeech:
    """Hasta dónde llega la voz dentro del clip, medido desde su inicio.

    Existe porque **el último sample de un archivo no es la última sílaba**. Un
    clip puede arrastrar respiración, cola de reverberación o ruido de sala
    después de que la persona terminó de hablar, y el arnés que fecha su cero en
    el fin del archivo lo pone ahí de más.

    La cola varía mucho de un clip a otro, y es casi entera la explicación del
    sesgo del cero que publica el arnés de PoC2 (medición en el anexo de
    `docs/comparativo_results.md`).

    Se usa el umbral **estricto** (el mismo de la medición de nivel) y no el
    permisivo del recorte: aquí la pregunta es dónde dejó de haber voz de
    verdad, no dónde se puede cortar sin morder una fricativa.
    """
    data = np.asarray(samples, dtype=np.int16)
    duration = len(data) / float(sample_rate)
    with_voice, per_window, umbral = windows_with_speech(data, sample_rate, drop_db=drop_db)
    if with_voice.size == 0:
        # Sin voz no hay fin de voz. Devolver 0 o la duración serían dos
        # mentiras distintas; el fin cae en el arranque y la cola es todo.
        return EndOfSpeech(0.0, duration, drop_db, umbral, duration)

    end = min(len(data), (int(with_voice[-1]) + 1) * per_window) / float(sample_rate)
    return EndOfSpeech(
        end_s=round(end, 6),
        cola_seg=round(duration - end, 6),
        drop_db=float(drop_db),
        umbral_dbfs=round(float(umbral), 2),
        duration_s=round(duration, 6),
    )


# --------------------------------------------------------------------------
# normalización
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppliedLevel:
    """El resultado de normalizar, con todo lo que el manifiesto necesita."""

    samples: np.ndarray
    target_dbfs: float
    drop_db: float
    source_speech_rms_dbfs: float
    source_peak_dbfs: float
    gain_db: float
    clamped_gain: bool


def normalize_to_target(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_dbfs: float,
    drop_db: float = DROP_FOR_LEVEL_DB,
) -> AppliedLevel:
    """Ajusta la ganancia para que el RMS de voz caiga en el objetivo.

    Es ganancia pura —un solo factor para todo el clip—, no compresión: la
    dinámica del habla actuada no se toca, solo se mueve el nivel entero.
    """
    data = np.asarray(samples, dtype=np.int16)
    if data.size == 0:
        raise EmptyClip("el clip no tiene una sola muestra")

    umbral = speech_threshold(data, drop_db=drop_db)
    speech_rms = rms_of_speech(data, sample_rate, umbral_dbfs=umbral)
    if not math.isfinite(speech_rms):
        raise EmptyClip(
            f"no hay una sola ventana por encima de {umbral:.1f} dBFS: el clip es "
            "silencio. Revisa que la grabadora estuviera tomando el micrófono"
        )

    pico = dbfs_pico(data)
    gain = target_dbfs - speech_rms
    margin = MAX_PEAK_AFTER_GAIN_DBFS - pico
    limited = gain > margin
    if limited:
        gain = margin

    factor = 10.0 ** (gain / 20.0)
    escalado = np.clip(
        np.rint(data.astype(np.float64) * factor), -32768.0, 32767.0
    ).astype(np.int16)

    return AppliedLevel(
        samples=np.ascontiguousarray(escalado),
        target_dbfs=float(target_dbfs),
        drop_db=float(drop_db),
        source_speech_rms_dbfs=round(speech_rms, 2),
        source_peak_dbfs=round(pico, 2),
        gain_db=round(gain, 2),
        clamped_gain=bool(limited),
    )
