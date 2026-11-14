"""Degradación de un clip a banda telefónica, con sus hashes y sus medidas.

Un clip pasa por estos pasos, en este orden y una sola vez:

    WAV 16 bits -> mono -> nivel -> 8 kHz -> recorte de silencio -> μ-law

El recorte va **antes** del códec para que los dos artefactos que salen —el PCM
de 8 kHz y los bytes μ-law— describan exactamente el mismo audio. Se cumple
siempre que `pcm8k == ulaw_a_pcm16(ulaw)`, así que verificar un hash verifica
los dos.

El nivel va **antes** del remuestreo: la ganancia se aplica sobre el audio
limpio para que el códec cuantice la misma amplitud que cuantizaría en una
llamada real. El porqué, y por qué eso obliga a verificar el nivel *después*,
está en `nivel.py`.

## Por qué se recorta el silencio, y por qué no es cosmético

El arnés fecha el **fin de la voz del clip** para poner ahí el cero de la
latencia. Si el archivo trae un segundo de cuarto silencioso al final,
ese cero se corre un segundo y toda la latencia de ese turno sale inflada — en
las tres PoC, y de forma distinta en cada una según su endpointing. Recortar es
lo que hace que el cero signifique lo que dice.

Cuánto se recortó de cada lado queda en el manifiesto: es un dato de
preparación que hay que poder auditar, no un detalle que se pierde.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from shared.audio.codec import pcm16_a_ulaw, ulaw_a_pcm16
from shared.audio.level import (
    DROP_FOR_LEVEL_DB,
    DROP_FOR_TRIM_DB,
    SOURCE_RMS_TOO_LOW_DBFS,
    TOLERANCE_DB,
    Calibration,
    EmptyClip,
    load_calibration,
    end_of_speech,
    levels_per_window,
    normalize_to_target,
    rms_of_speech,
    speech_threshold,
)
from shared.audio.resampling import TELEPHONY_SAMPLE_RATE, downsample_to_telephony, engine_description
from shared.audio.noise import dbfs_pico, dbfs_rms
from shared.audio.wav import leer_wav

__all__ = [
    "WEAK_TAIL_CAP_MS",
    "DegradedClip",
    "EmptyClip",
    "degrade",
    "write_artifacts",
    "prepare_clip",
    "trim_silence",
]

# Margen que se deja a cada lado de la voz detectada. Sin él, el recorte muerde
# el ataque de la primera consonante.
#
# **No sirve como corrección del cero, y por eso no se publica.** El recorte usa
# el umbral permisivo, así que la cola real que queda después de la última voz
# es mucho mayor que estos 30 ms y varía por clip: son una cota inferior, no una
# corrección, y publicarlos invitaría a usarlos como si corrigieran. La cola se
# **mide** por clip (`nivel.fin_de_voz`) y viaja en el manifiesto.
_TRIM_MARGIN_MS: Final[int] = 30

# Cuánta señal débil se conserva DESPUÉS de la última voz fuerte. Es el tope que
# separa "fricativa final" de "respiración", y se decide por **duración**, no
# bajando el umbral permisivo: bajarlo mordería el ataque de las fricativas en
# todos los clips para arreglar la cola de unos pocos.
#
# El valor sale de medir los 49 clips crudos. La distancia entre el fin de la voz
# fuerte (25 dB) y el de la débil (35 dB) **no la explica el sonido final de la
# línea**: la misma fricativa final da 50 ms en un clip y 880 en otro. O sea que
# lo que hay ahí arriba de ~200 ms es respiración y cola de sala, no un fonema
# (mediciones en el anexo de `docs/comparativo_resultados.md`).
#
# 200 ms queda por encima de cualquier /s/ final del español (80-150 ms) y por
# debajo del grupo de las colas largas. Y el recorte **nunca corta antes** de la
# última ventana de voz fuerte, así que ningún contenido audible se puede
# perder: lo único que se puede ir es señal débil a más de 200 ms de la última
# sílaba.
WEAK_TAIL_CAP_MS: Final[float] = 200.0

# Cola por encima de la cual el clip desplaza el cero de la latencia lo
# suficiente como para avisar. Sobre los 8 clips del escenario 1, seis quedaron
# entre 50 y 170 ms y los dos que se pasaron desplazaron el cero 570 y 1,289 ms.
# No es una constante de la comparativa: es el punto donde vale la pena regrabar.
_OBSTRUCTING_TAIL_MS: Final[float] = 250.0

# Pico por encima del cual se considera que la grabación viene recortada
# (saturada). Se mide sobre el **original**, antes de la ganancia: la saturación
# es un defecto de la toma, y se arregla regrabando, no en software.
_SUSPICIOUS_PEAK_DBFS: Final[float] = -0.5


@dataclass(frozen=True)
class DegradedClip:
    """Un clip ya en banda telefónica, con todo lo que el manifiesto necesita."""

    pcm8k: np.ndarray
    ulaw: bytes
    origin_sample_rate: int
    origin_samples: int
    resampling_engine: str
    start_cut_ms: float
    end_cut_ms: float
    trim_threshold_dbfs: float
    # --- dónde deja de haber voz dentro del clip ya recortado ---
    # El arnés de cada PoC fecha su cero en el fin del **archivo**, que es lo
    # único que puede garantizar físicamente. Esto es lo otro: dónde dejó de
    # haber voz de verdad. Viaja con su umbral (`caida_voz_db`) porque sin él el
    # número no es reproducible. Ver `nivel.fin_de_voz`.
    end_of_speech_s: float
    cola_seg: float
    speech_drop_db: float
    speech_threshold_dbfs: float
    # --- calibración de nivel ---
    target_dbfs: float
    drop_db: float
    speech_rms_dbfs: float
    source_speech_rms_dbfs: float
    source_peak_dbfs: float
    gain_db: float
    clamped_gain: bool
    provisional_calibration: bool
    calibration_source: str

    @property
    def duration_s(self) -> float:
        return len(self.pcm8k) / TELEPHONY_SAMPLE_RATE

    @property
    def pico_dbfs(self) -> float:
        return dbfs_pico(self.pcm8k)

    @property
    def rms_dbfs(self) -> float:
        """RMS del clip completo. Para el nivel, `speech_rms_dbfs` es la medida."""
        return dbfs_rms(self.pcm8k)

    @property
    def deviation_db(self) -> float:
        """Qué tan lejos del objetivo quedó el nivel **ya degradado**."""
        return round(self.speech_rms_dbfs - self.target_dbfs, 2)

    @property
    def calibrated(self) -> bool:
        return abs(self.deviation_db) <= TOLERANCE_DB

    @property
    def saturated_source(self) -> bool:
        """La grabación original venía recortada. Se avisa; no se corrige."""
        return self.source_peak_dbfs >= _SUSPICIOUS_PEAK_DBFS

    @property
    def sha256_pcm8k(self) -> str:
        return hashlib.sha256(self.pcm8k.tobytes()).hexdigest()

    @property
    def sha256_ulaw(self) -> str:
        return hashlib.sha256(self.ulaw).hexdigest()

    def warning_list(self) -> list[str]:
        """Lo que vale la pena mirar antes de gastar 15 llamadas con este clip."""
        problems: list[str] = []
        if self.saturated_source:
            problems.append(
                f"pico a {self.source_peak_dbfs:.1f} dBFS en la grabación original: "
                "viene saturada y el códec va a distorsionarla — vuelve a grabar "
                "más bajo"
            )
        if self.duration_s < 0.3:
            problems.append(
                f"dura {self.duration_s:.2f} s: ¿se recortó de más o se grabó a medias?"
            )
        if self.duration_s > 20.0:
            problems.append(
                f"dura {self.duration_s:.1f} s: más largo que cualquier línea del guion"
            )
        if self.source_speech_rms_dbfs < SOURCE_RMS_TOO_LOW_DBFS:
            problems.append(
                f"la grabación original trae la voz a {self.source_speech_rms_dbfs:.1f} "
                f"dBFS: subirla {self.gain_db:+.1f} dB al objetivo sube también "
                "su piso de ruido. Vuelve a grabar más cerca del micrófono"
            )
        if self.cola_seg * 1000.0 > _OBSTRUCTING_TAIL_MS:
            problems.append(
                f"quedan {self.cola_seg * 1000:.0f} ms de cola después de la "
                f"última voz (umbral {self.speech_threshold_dbfs:.1f} dBFS, caída "
                f"{self.speech_drop_db:.0f} dB). El arnés fecha su cero en el fin "
                "del archivo, así que este clip lo corre esos milisegundos y "
                "toda la latencia de su turno sale inflada, en las 3 PoC. Es "
                "respiración o cola de sala: vuelve a grabar cortando antes"
            )
        if self.clamped_gain:
            problems.append(
                f"no se alcanzó el objetivo de {self.target_dbfs:.1f} dBFS: la "
                f"ganancia se limitó a {self.gain_db:+.1f} dB para no recortar "
                "los picos"
            )
        if not self.calibrated:
            problems.append(
                f"nivel de voz {self.speech_rms_dbfs:.1f} dBFS, {self.deviation_db:+.1f} "
                f"dB del objetivo ({self.target_dbfs:.1f}): fuera de la tolerancia "
                f"de ±{TOLERANCE_DB:.0f} dB"
            )
        return problems


def trim_silence(
    samples: np.ndarray, *, umbral_dbfs: float | None = None
) -> tuple[np.ndarray, float, float, float]:
    """Quita el silencio de los extremos.

    Devuelve `(recortado, ms quitados al inicio, ms al final, umbral usado)`.
    El silencio **interno** no se toca: una pausa a media frase es parte de la
    línea actuada, y quitarla cambiaría lo que el detector de turnos ve.

    Usa un umbral **más permisivo** que el de la medición de nivel, a propósito:
    aquí un falso negativo se come una fricativa final y mueve el cero del fin
    de voz, que es exactamente lo que el recorte existe para proteger.

    Y por eso mismo la cola se topa por **duración**, no bajando ese umbral.
    Medido sobre las 49 grabaciones: la señal débil que sigue a la última voz
    fuerte llega a 1,460 ms, y el sonido final de la línea no la predice — es
    respiración y cola de sala. Se conservan `WEAK_TAIL_CAP_MS` de esa
    señal, que alcanza para cualquier fricativa final, y se corta el resto.

    El corte **nunca** cae antes de la última ventana de voz fuerte, así que no
    puede morder contenido audible: el tope solo acorta lo que ya era débil.
    """
    samples = np.asarray(samples, dtype=np.int16)
    if samples.size == 0:
        raise EmptyClip("el clip no tiene una sola muestra")

    umbral = (
        speech_threshold(samples, drop_db=DROP_FOR_TRIM_DB)
        if umbral_dbfs is None
        else umbral_dbfs
    )
    levels, per_window = levels_per_window(samples, TELEPHONY_SAMPLE_RATE)
    if levels.size == 0:
        return samples, 0.0, 0.0, umbral

    with_voice = np.flatnonzero(levels >= umbral)
    if with_voice.size == 0:
        raise EmptyClip(
            f"no hay una sola ventana por encima de {umbral:.1f} dBFS: el clip es "
            "silencio. Revisa que la grabadora estuviera tomando el micrófono"
        )

    margin = _TRIM_MARGIN_MS * TELEPHONY_SAMPLE_RATE // 1000
    start = max(0, int(with_voice[0]) * per_window - margin)
    weak_end = (int(with_voice[-1]) + 1) * per_window

    # El tope de cola: de la señal débil que sigue a la última voz FUERTE, se
    # conserva solo lo que quepa en `TOPE_DE_COLA_DEBIL_MS`. Si no hay voz
    # fuerte —un clip entero al borde del umbral— no se topa nada: preferible
    # dejarlo largo y que el aviso de cola lo delate, a cortar a ciegas.
    strong = np.flatnonzero(
        levels >= speech_threshold(samples, drop_db=DROP_FOR_LEVEL_DB)
    )
    if strong.size:
        tope = WEAK_TAIL_CAP_MS * TELEPHONY_SAMPLE_RATE / 1000.0
        strong_end = (int(strong[-1]) + 1) * per_window
        weak_end = min(weak_end, int(strong_end + tope))

    end = min(len(samples), weak_end + margin)

    removed_start = start / TELEPHONY_SAMPLE_RATE * 1000.0
    removed_end = (len(samples) - end) / TELEPHONY_SAMPLE_RATE * 1000.0
    return samples[start:end], removed_start, removed_end, umbral


def degrade(
    samples: np.ndarray,
    origin_sample_rate: int,
    *,
    calibration: Calibration | None = None,
) -> DegradedClip:
    """PCM16 a cualquier tasa -> clip en banda telefónica, listo para reproducir.

    `calibracion` en `None` usa la de `calibracion_nivel.json`. La normalización
    de nivel **siempre** corre: un clip sin calibrar mediría la ganancia del
    micrófono con el que se grabó, no la latencia ni el WER de la arquitectura.
    """
    calibration = calibration or load_calibration()
    samples = np.asarray(samples, dtype=np.int16)

    leveled = normalize_to_target(
        samples,
        origin_sample_rate,
        target_dbfs=calibration.target_dbfs,
        drop_db=calibration.drop_db,
    )
    a_8k = downsample_to_telephony(leveled.samples, origin_sample_rate)
    trimmed, start_ms, final_ms, umbral = trim_silence(a_8k)

    # El μ-law se calcula del recorte, y el PCM que se guarda es el que sale de
    # decodificarlo: así los dos artefactos son literalmente el mismo audio.
    ulaw = pcm16_a_ulaw(trimmed)
    pcm8k = np.ascontiguousarray(ulaw_a_pcm16(ulaw), dtype=np.int16)

    # El fin de voz se mide sobre el PCM final, el mismo que se reproduce: si se
    # midiera antes del códec, el número no describiría el audio que oye el STT.
    voice = end_of_speech(pcm8k, TELEPHONY_SAMPLE_RATE, drop_db=leveled.drop_db)

    # El nivel se verifica sobre el audio **ya degradado**, que es el que se
    # reproduce: el filtro antialias y el códec mueven el RMS unas décimas.
    return DegradedClip(
        pcm8k=pcm8k,
        ulaw=ulaw,
        origin_sample_rate=int(origin_sample_rate),
        origin_samples=int(samples.size),
        resampling_engine=engine_description(),
        start_cut_ms=round(start_ms, 1),
        end_cut_ms=round(final_ms, 1),
        trim_threshold_dbfs=round(umbral, 1),
        end_of_speech_s=voice.end_s,
        cola_seg=voice.cola_seg,
        speech_drop_db=voice.drop_db,
        speech_threshold_dbfs=voice.umbral_dbfs,
        target_dbfs=leveled.target_dbfs,
        drop_db=leveled.drop_db,
        speech_rms_dbfs=round(rms_of_speech(pcm8k, TELEPHONY_SAMPLE_RATE, drop_db=leveled.drop_db), 2),
        source_speech_rms_dbfs=leveled.source_speech_rms_dbfs,
        source_peak_dbfs=leveled.source_peak_dbfs,
        gain_db=leveled.gain_db,
        clamped_gain=leveled.clamped_gain,
        provisional_calibration=calibration.provisional,
        calibration_source=calibration.source,
    )


def prepare_clip(
    ruta: Path | str, *, calibration: Calibration | None = None
) -> DegradedClip:
    """Lee un WAV y lo degrada. El paso que corre una sola vez por clip."""
    samples, sample_rate = leer_wav(ruta)
    return degrade(samples, sample_rate, calibration=calibration)


def write_artifacts(clip: DegradedClip, base: Path | str) -> tuple[Path, Path]:
    """Escribe `<base>.wav` (PCM16 8 kHz) y `<base>.ulaw` (bytes crudos).

    Los dos, y no uno: el `.wav` se puede oír para revisar que el clip diga lo
    que debe, y el `.ulaw` es exactamente el payload que PoC1 y PoC3 le meten al
    WebSocket de Media Streams, sin conversión en vivo.
    """
    from shared.audio.wav import write_wav

    base = Path(base)
    ruta_wav = write_wav(base.with_suffix(".wav"), clip.pcm8k, TELEPHONY_SAMPLE_RATE)
    ruta_ulaw = base.with_suffix(".ulaw")
    ruta_ulaw.write_bytes(clip.ulaw)
    return ruta_wav, ruta_ulaw
