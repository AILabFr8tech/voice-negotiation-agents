"""Pruebas de las piezas nuevas de la validación por los caminos de PoC2 y PoC3.

Las corridas completas no se prueban aquí: van en tiempo real, y la de PoC2
necesita una sala de LiveKit. Se prueba lo que decide la verdad: que el
reconocedor encuentre el clip y feche su fin de voz, y que la pista continua del
agente falso de LiveKit ponga la respuesta donde se programó.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from shared.audio import TELEPHONY_SAMPLE_RATE
from validation import recognizer as rc

SAMPLE_RATE = 24_000


def _clip(seed: int, seconds: float = 1.5) -> np.ndarray:
    chance = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    envelope = np.convolve(chance.uniform(0.2, 1.0, n // 2400 + 1).repeat(2400)[:n],
                             np.ones(480) / 480, mode="same")
    return (chance.normal(0, 1, n) * envelope * 4000).astype(np.int16)


def _template(clip_id: str, pcm: np.ndarray, end_s: float) -> rc.Template:
    return rc.Template(
        clip_id=clip_id, text=clip_id,
        start=pcm[: int(rc.TEMPLATE_LENGTH_S * SAMPLE_RATE)].astype(np.float64),
        end_of_speech=int(end_s * SAMPLE_RATE), largo=len(pcm),
    )


def test_the_normalized_correlation_finds_the_exact_shift():
    x = _clip(1)
    sig = np.concatenate([np.zeros(1234), x, np.zeros(500)]).astype(np.float64)
    corr = rc.normalized_correlation(sig, x[:4800].astype(np.float64))
    assert int(np.argmax(corr)) == 1234
    assert corr.max() == pytest.approx(1.0, abs=1e-6)


def test_the_recognizer_picks_the_clip_and_dates_the_end_of_speech_when_its_chunk_arrives():
    a, b = _clip(1), _clip(2)
    candidate_ones = [_template("a", a, 1.2), _template("b", b, 1.1)]
    ear = rc.Recognizer(candidate_ones, SAMPLE_RATE, min_correlation=0.9)
    silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.int16)
    stream = np.concatenate([silence, b, silence])
    piece = 480  # 20 ms
    resultado = None
    t0 = 100.0
    for i in range(0, len(stream), piece):
        # El trozo i llega cuando ya "sonó" completo.
        dice = ear.receive(stream[i:i + piece], t0 + (i + piece) / SAMPLE_RATE)
        if dice is not None:
            resultado = dice
            break
    assert isinstance(resultado, rc.EndOfSpeech)
    assert resultado.template.clip_id == "b"
    assert resultado.correlation == pytest.approx(1.0, abs=1e-6)
    # El fin de voz cae en la muestra 0.5 s + 1.1 s; llega con el trozo que la trae.
    sample = int(0.5 * SAMPLE_RATE) + int(1.1 * SAMPLE_RATE)
    expected = t0 + (sample // piece + 1) * piece / SAMPLE_RATE
    assert resultado.t_end_of_speech == pytest.approx(expected)


def test_the_recognizer_does_not_believe_a_peak_with_no_advantage_over_the_second():
    a = _clip(1)
    candidate_ones = [_template("a", a, 1.2), _template("again", a, 1.2)]
    ear = rc.Recognizer(candidate_ones, SAMPLE_RATE, min_correlation=0.5, min_advantage=0.1)
    stream = np.concatenate([np.zeros(SAMPLE_RATE // 2, dtype=np.int16), a])
    stated_items = [ear.receive(stream[i:i + 480], float(i)) for i in range(0, len(stream), 480)]
    assert any(isinstance(d, rc.NotRecognized) for d in stated_items)
    assert not any(isinstance(d, rc.EndOfSpeech) for d in stated_items)


def test_the_templates_upsample_the_published_clip_to_the_arrival_rate(tmp_path):
    from validation.synthetic_clips import generate

    generate(tmp_path)
    p = rc.templates(tmp_path, ["e1_t1"], SAMPLE_RATE)[0]
    assert len(p.start) == int(rc.TEMPLATE_LENGTH_S * SAMPLE_RATE)
    # El largo a 24 kHz es el triple del publicado a 8 kHz.
    assert p.largo == pytest.approx(3 * (tmp_path / "e1_t1.ulaw").stat().st_size, abs=3)


def test_the_continuous_track_yields_the_response_at_the_scheduled_sample():
    from validation.fake_agent_livekit import SAMPLES_PER_FRAME, ContinuousTrack

    class _Source:
        async def capture_frame(self, trama):
            return None

    class _Rtc:
        class AudioFrame:
            def __init__(self, **_):
                pass

    async def _launch():
        audio_track = ContinuousTrack(source=_Source(), rtc=_Rtc)
        halt = asyncio.Event()
        tarea = asyncio.create_task(audio_track.launch(halt))
        await asyncio.sleep(0.05)
        pcm = np.full(2400, 1000, dtype=np.int16)
        target = audio_track.sample_at(time.perf_counter() + 0.2)
        audio_track.schedule(pcm, onset=target, primero=300, phrase_start=300,
                        start_mm=None, row={})
        await asyncio.sleep(0.5)
        halt.set()
        await tarea
        return audio_track, target

    audio_track, target = asyncio.run(_launch())
    r = audio_track.replies[0]
    assert r.phrase_start == target
    trama = audio_track._trama(r.phrase_start // SAMPLES_PER_FRAME)
    assert trama[r.phrase_start % SAMPLES_PER_FRAME] == 1000
    assert audio_track.moment(r.phrase_start) is not None
