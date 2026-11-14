"""El piso de ruido va SUMADO por debajo del clip, no alternado con él.

El primer diseño alternaba dos fuentes y eso dejaba un escalón de ~9 dB en los
dos bordes de cada intervención. Ver `docs/METRICS_CONTRACT.md` §2.2.9.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.audio.codec import pcm16_a_ulaw, ulaw_a_pcm16
from shared.audio.noise import COMFORT_NOISE_DBFS, dbfs_rms
from shared.carrier_simulator.audio import noise_floor_under_clip
from shared.carrier_simulator.script import load_clip, load_clip_to_publish

SAMPLE_RATE = 8000


def _clip(tmp_path, samples: np.ndarray, name: str = "clip"):
    ruta = tmp_path / f"{name}.ulaw"
    ruta.write_bytes(pcm16_a_ulaw(np.asarray(samples, dtype=np.int16)))
    return ruta


def _tone(seconds: float, *, amplitude: int = 8000) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 300.0 * t)).astype(np.int16)


# --------------------------------------------------------------------------
# La suma
# --------------------------------------------------------------------------


def test_the_clip_silence_stays_at_the_comfort_noise_level():
    """Antes quedaba en silencio digital y el borde del clip daba un escalón."""
    output = noise_floor_under_clip(np.zeros(SAMPLE_RATE, dtype=np.int16), seed=1)
    assert dbfs_rms(output) == pytest.approx(COMFORT_NOISE_DBFS, abs=1.0)


def test_the_voice_is_not_touched_appreciably():
    """El piso se suma por debajo; 40 dB más abajo no mueve el nivel de la voz."""
    voice = _tone(0.5)
    with_floor = noise_floor_under_clip(voice, seed=1)
    assert dbfs_rms(with_floor) == pytest.approx(dbfs_rms(voice), abs=0.1)


def test_the_floor_is_ALSO_under_the_speech_and_not_only_in_the_pauses():
    """Que sea una suma y no una alternancia es justo lo que se está probando.

    Se recupera el ruido restando el clip original: si el piso solo se pusiera
    donde no hay voz, el residuo bajo la voz sería cero.
    """
    voice = _tone(0.5)
    residue = noise_floor_under_clip(voice, seed=1).astype(np.int32) - voice
    assert dbfs_rms(residue) == pytest.approx(COMFORT_NOISE_DBFS, abs=1.0)


def test_the_same_seed_gives_the_same_bytes():
    """El arnés es reproducible por diseño y el ruido entra en lo que suena."""
    voice = _tone(0.2)
    uno = noise_floor_under_clip(voice, seed=7)
    otro = noise_floor_under_clip(voice, seed=7)
    assert (uno == otro).all()
    assert not (noise_floor_under_clip(voice, seed=8) == uno).all()


def test_a_clip_at_the_cap_does_not_wrap_around():
    """Sumar sin recortar convertiría un pico en su negativo: un chasquido."""
    tope = np.full(100, 32767, dtype=np.int16)
    assert noise_floor_under_clip(tope, seed=1).min() > 0


def test_an_empty_clip_does_not_blow_up():
    assert noise_floor_under_clip(np.zeros(0, dtype=np.int16), seed=1).size == 0


# --------------------------------------------------------------------------
# El cargador que publica
# --------------------------------------------------------------------------


def test_the_two_representations_are_THE_SAME_audio(tmp_path):
    """PoC2 publica el PCM y PoC1 los bytes μ-law: tienen que coincidir.

    Si el PCM saliera de la suma directa y el μ-law de codificarla, diferirían
    en la cuantización y las dos PoC estarían oyendo estímulos distintos.
    """
    ruta = _clip(tmp_path, _tone(0.3))
    samples, published, _, _ = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    assert (ulaw_a_pcm16(published) == samples).all()


def test_the_source_bytes_come_out_intact(tmp_path):
    """El hash del manifiesto se verifica contra estos, no contra lo publicado."""
    ruta = _clip(tmp_path, _tone(0.3))
    _, published, source, _ = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    assert source == ruta.read_bytes()
    assert published != source


def test_the_published_has_the_floor_and_the_clean_clip_does_not(tmp_path):
    """`load_clip` sigue dejando el estímulo limpio: es lo que se analiza."""
    ruta = _clip(tmp_path, np.zeros(SAMPLE_RATE, dtype=np.int16))
    cleaned, _ = load_clip(ruta, publish_sample_rate=SAMPLE_RATE)
    published, _, _, _ = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    assert dbfs_rms(cleaned) < COMFORT_NOISE_DBFS - 20.0
    assert dbfs_rms(published) == pytest.approx(COMFORT_NOISE_DBFS, abs=1.5)


def test_two_loads_of_the_same_clip_publish_the_same_bytes(tmp_path):
    """La semilla sale del contenido, así que no hace falta pasear un contador."""
    ruta = _clip(tmp_path, _tone(0.3))
    _, uno, _, _ = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    _, otro, _, _ = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    assert uno == otro


def test_two_different_clips_do_not_share_the_noise_realization(tmp_path):
    """Con una semilla fija para todos, el mismo ruido sonaría en cada turno.

    Los dos clips son silencio digital y difieren en **una sola muestra**, para
    que lo único que pueda separar las dos salidas sea la semilla y no la
    cuantización. Con la semilla derivada del contenido difieren en todas las
    muestras; con una semilla fija diferirían solo en esa una.
    """
    silent = np.zeros(SAMPLE_RATE, dtype=np.int16)
    different = silent.copy()
    different[0] = 20000
    uno = _clip(tmp_path, silent, "a")
    otro = _clip(tmp_path, different, "b")
    assert uno.read_bytes() != otro.read_bytes()
    a, _, _, _ = load_clip_to_publish(uno, publish_sample_rate=SAMPLE_RATE)
    b, _, _, _ = load_clip_to_publish(otro, publish_sample_rate=SAMPLE_RATE)
    # Saltando la muestra que los diferencia, todo lo demás tiene que diferir.
    equal_values = int((a[1:] == b[1:]).sum())
    assert equal_values < len(a) // 4, (
        f"{equal_values} de {len(a) - 1} muestras coinciden: parece la misma semilla"
    )


def test_the_fallback_path_raises_the_floor_along_with_the_speech(tmp_path):
    """A 48 kHz el piso tiene que seguir ahí, o el respaldo reintroduce el escalón."""
    ruta = _clip(tmp_path, np.zeros(SAMPLE_RATE, dtype=np.int16))
    samples, _, _, origin = load_clip_to_publish(ruta, publish_sample_rate=48_000)
    assert len(samples) > SAMPLE_RATE
    assert dbfs_rms(samples) == pytest.approx(COMFORT_NOISE_DBFS, abs=3.0)
    assert "piso de ruido" in origin


def test_the_source_says_it_carries_a_floor(tmp_path):
    """La bitácora tiene que distinguir una corrida con piso de una sin él."""
    ruta = _clip(tmp_path, _tone(0.2))
    _, _, _, origin = load_clip_to_publish(ruta, publish_sample_rate=SAMPLE_RATE)
    assert "piso de ruido" in origin
