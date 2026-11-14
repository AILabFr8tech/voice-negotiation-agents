"""Lectura y escritura de WAV PCM16 mono, con la biblioteca estándar.

Solo se acepta **PCM entero de 16 bits**, que es lo que exporta cualquier
grabadora cuando se le pide "WAV 16-bit". Los otros formatos se rechazan con un
mensaje que dice qué hacer, porque el caso típico no es un bug: es alguien que
grabó en el celular y trae un `.m4a`, o que exportó de Audacity en flotante de
32 bits sin darse cuenta.

`wave` (stdlib) no es `audioop`: sigue vivo y no está marcado para eliminarse.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class AudioNotSupported(ValueError):
    """El archivo existe pero no es WAV PCM16 mono/estéreo legible."""


def leer_wav(ruta: Path | str) -> tuple[np.ndarray, int]:
    """Devuelve `(muestras int16 mono, tasa)`.

    El estéreo se promedia a mono. No es una decisión de calidad: un clip de voz
    grabado en estéreo trae el mismo micrófono en los dos canales, y el
    transportista del teléfono es mono por definición del medio.
    """
    ruta = Path(ruta)
    try:
        with wave.open(str(ruta), "rb") as file:
            channel_list = file.getnchannels()
            width = file.getsampwidth()
            sample_rate = file.getframerate()
            raw_value = file.readframes(file.getnframes())
    except wave.Error as error:
        raise AudioNotSupported(
            f"{ruta.name}: no se pudo leer como WAV ({error}). Suele ser un WAV "
            "en flotante de 32 bits o un contenedor que no es WAV (.m4a, .mp3). "
            "Vuelve a exportarlo como WAV PCM de 16 bits"
        ) from error
    except FileNotFoundError:
        raise
    except OSError as error:  # pragma: no cover - depende del sistema de archivos
        raise AudioNotSupported(f"{ruta.name}: no se pudo abrir ({error})") from error

    if width != 2:
        raise AudioNotSupported(
            f"{ruta.name}: son muestras de {width * 8} bits y hacen falta 16. "
            "Vuelve a exportarlo como WAV PCM de 16 bits"
        )
    if channel_list not in (1, 2):
        raise AudioNotSupported(
            f"{ruta.name}: trae {channel_list} canales; solo se aceptan mono o estéreo"
        )
    if not raw_value:
        raise AudioNotSupported(f"{ruta.name}: el archivo no tiene una sola muestra")

    samples = np.frombuffer(raw_value, dtype="<i2")
    if channel_list == 2:
        # A int32 antes de promediar: la suma de dos int16 se desborda.
        pairs = samples.reshape(-1, 2).astype(np.int32)
        samples = (pairs.sum(axis=1) // 2).astype(np.int16)

    return np.ascontiguousarray(samples, dtype=np.int16), sample_rate


def read_wav_stereo(ruta: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    """Devuelve `(canal 0, canal 1, tasa)` **sin mezclar**.

    Es lo contrario de lo que hace `leer_wav`, y a propósito: aquel promedia a
    mono porque el medio es mono, y esto existe para la grabación de auditoría
    del arnés, donde cada lado de la conversación va en su canal. Promediarlos
    borraría justo lo que se quiere leer, que es si los dos sonaron a la vez.

    Un WAV mono se rechaza en vez de duplicarse: devolver el mismo canal dos
    veces daría 100% de solapamiento y parecería un hallazgo.
    """
    ruta = Path(ruta)
    try:
        with wave.open(str(ruta), "rb") as file:
            channel_list = file.getnchannels()
            width = file.getsampwidth()
            sample_rate = file.getframerate()
            raw_value = file.readframes(file.getnframes())
    except wave.Error as error:
        raise AudioNotSupported(
            f"{ruta.name}: no se pudo leer como WAV ({error})"
        ) from error

    if width != 2:
        raise AudioNotSupported(
            f"{ruta.name}: son muestras de {width * 8} bits y hacen falta 16"
        )
    if channel_list != 2:
        raise AudioNotSupported(
            f"{ruta.name}: trae {channel_list} canal(es) y hacen falta 2. Un WAV mono "
            "no sirve aquí: no se puede saber quién habló encima de quién"
        )
    if not raw_value:
        raise AudioNotSupported(f"{ruta.name}: el archivo no tiene una sola muestra")

    pairs = np.frombuffer(raw_value, dtype="<i2").reshape(-1, 2)
    return (
        np.ascontiguousarray(pairs[:, 0], dtype=np.int16),
        np.ascontiguousarray(pairs[:, 1], dtype=np.int16),
        sample_rate,
    )


def write_wav(ruta: Path | str, samples: np.ndarray, sample_rate: int) -> Path:
    """Escribe PCM16 mono. Crea la carpeta si hace falta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.int16)
    with wave.open(str(ruta), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(int(sample_rate))
        file.writeframes(data.tobytes())
    return ruta


def write_wav_stereo(
    ruta: Path | str, left_side: np.ndarray, derecho: np.ndarray, sample_rate: int
) -> Path:
    """Escribe PCM16 de dos canales **sin mezclar**, para poder oírlos aparte.

    No es una alternativa a `write_wav`: el audio que se mide es mono porque
    el medio es mono. Esto existe solo para la grabación de auditoría del arnés,
    donde cada lado de la conversación va en su canal — mezclarlos borraría
    justo lo que se quiere oír, que es si los dos hablaron encima y quién
    empezó.

    Los dos canales tienen que traer el mismo número de muestras: un desfase de
    longitud sería un desfase de tiempo, y este archivo se usa para juzgar
    quién habló primero.
    """
    ruta = Path(ruta)
    left = np.asarray(left_side, dtype=np.int16)
    right = np.asarray(derecho, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError(
            f"los canales miden distinto ({left.size} y {right.size} muestras): "
            "alinéalos antes de escribir o el WAV mentiría sobre los tiempos"
        )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.empty(left.size * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(str(ruta), "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(int(sample_rate))
        file.writeframes(interleaved.tobytes())
    return ruta
