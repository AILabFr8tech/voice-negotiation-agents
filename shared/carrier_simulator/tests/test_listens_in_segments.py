"""La voz del agente que llega en bloques largos se escucha en tramos de 20 ms."""

from __future__ import annotations

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.carrier_simulator import observed as obs

SAMPLE_RATE = 8_000


def _block(ms_with_speech: int, silent_ms: int = 0) -> bytes:
    """Un bloque PCM16: voz al principio y silencio al final."""
    voice = (np.sin(np.arange(SAMPLE_RATE * ms_with_speech // 1000) * 0.3) * 6_000).astype(np.int16)
    return np.concatenate([voice, np.zeros(SAMPLE_RATE * silent_ms // 1000, dtype=np.int16)]).tobytes()


class _Clock:
    def __init__(self, start: float) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    r = _Clock(1_000.0)
    monkeypatch.setattr(obs.time, "time", r)
    return r


def test_the_last_speech_is_the_last_segment_that_sounds(clock):
    o = cs.Observed(silent=True, start=1_000.0)
    clock.t = 1_001.0
    o.annotate_audio_in_frames(_block(60, 40), sample_rate=SAMPLE_RATE)  # 100 ms: voz los primeros 60
    assert o.audio_frames == 5
    assert o.speech_streaks[0].start == pytest.approx(1.0)
    assert o.speech_streaks[0].end == pytest.approx(1.04)  # antes: 1.0, el arranque del bloque


def test_blocks_that_arrive_all_at_once_are_queued(clock):
    o = cs.Observed(silent=True, start=1_000.0)
    clock.t = 1_001.0
    o.annotate_audio_in_frames(_block(100), sample_rate=SAMPLE_RATE)
    clock.t = 1_001.01  # el segundo bloque llega 10 ms después: suena tras el primero
    o.annotate_audio_in_frames(_block(100), sample_rate=SAMPLE_RATE)
    assert o.speech_streaks[-1].end == pytest.approx(1.18)
    assert o.max_listening_offset_s == pytest.approx(0.09)


def test_a_pause_does_not_grow_a_block(clock):
    """Voz hasta 1.10, bloque callado, voz que llega a 1.70: la pausa es de 0.6 s."""
    o = cs.Observed(silent=True, start=1_000.0)
    clock.t = 1_001.0
    o.annotate_audio_in_frames(_block(100), sample_rate=SAMPLE_RATE)
    clock.t = 1_001.70
    o.annotate_audio_in_frames(_block(100), sample_rate=SAMPLE_RATE)
    a, b = o.speech_streaks
    assert b.start - (a.end + obs.LISTENING_SEGMENT_S) == pytest.approx(0.6)
