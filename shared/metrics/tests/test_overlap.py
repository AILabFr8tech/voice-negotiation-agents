"""Counting simultaneous speech on the two-channel WAV.

The test that holds up all the others is `test_finds_a_constructed_overlap`:
the result on the real runs is **zero**, and a zero is only worth something if
it is shown that the detector is not blind. Without that test, a bug that always
returned zero would read as "the agent never interrupts".
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.audio import write_wav_stereo
from shared.audio.wav import AudioNotSupported, write_wav, read_wav_stereo
from shared.metrics.overlap import (
    DROP_FOR_ACTIVITY_DB,
    STREAK_GAP_S,
    MIN_OVERLAP_S,
    Streak,
    measure,
    speech_streaks,
    summary,
    overlaps_between,
)

SAMPLE_RATE = 8000


def _track(*segments: tuple[str, float]) -> np.ndarray:
    """Builds a channel from `("voice"|"silence", seconds)` segments."""
    generator = np.random.default_rng(7)
    parts = []
    for kind, seconds in segments:
        n = int(round(seconds * SAMPLE_RATE))
        if kind == "voice":
            parts.append(
                generator.normal(0, 6000, n).clip(-30000, 30000).astype(np.int16)
            )
        else:
            # Not absolute silence: a comfort-noise floor, as on the wire.
            parts.append(
                generator.normal(0, 20, n).clip(-200, 200).astype(np.int16)
            )
    return np.concatenate(parts)


def _wav(tmp_path, harness_channel: np.ndarray, agent_channel: np.ndarray):
    n = max(harness_channel.size, agent_channel.size)
    a = np.pad(harness_channel, (0, n - harness_channel.size))
    b = np.pad(agent_channel, (0, n - agent_channel.size))
    ruta = tmp_path / "auditoria_prueba.wav"
    write_wav_stereo(ruta, a, b, SAMPLE_RATE)
    return ruta


# --------------------------------------------------------------------------
# Sensitivity: the zero of the real runs is only worth something if this passes
# --------------------------------------------------------------------------


def test_finds_a_constructed_overlap(tmp_path):
    """The agent starts at second 2 and the harness keeps talking until second 3."""
    harness = _track(("voice", 3.0), ("silence", 2.0))
    agent = _track(("silence", 2.0), ("voice", 3.0))
    data = measure(_wav(tmp_path, harness, agent))
    assert data["overlaps"] == 1
    assert data["the_agent_talked_over"] == 1
    assert data["the_harness_talked_over"] == 0
    assert data["detail"][0]["duration_ms"] == pytest.approx(1000, abs=60)


def test_distinguishes_who_came_in_over(tmp_path):
    """Same overlap, reversed: now the one who opens its mouth second is the harness."""
    agent = _track(("voice", 3.0), ("silence", 2.0))
    harness = _track(("silence", 2.0), ("voice", 3.0))
    data = measure(_wav(tmp_path, harness, agent))
    assert data["overlaps"] == 1
    assert data["the_agent_talked_over"] == 0
    assert data["the_harness_talked_over"] == 1


def test_a_turn_by_turn_conversation_does_not_overlap(tmp_path):
    """The real-runs case: they take turns and do not step on each other."""
    harness = _track(("voice", 2.0), ("silence", 3.0))
    agent = _track(("silence", 2.5), ("voice", 2.5))
    data = measure(_wav(tmp_path, harness, agent))
    assert data["overlaps"] == 0
    # And the coverage shows it was not due to blindness.
    assert data["detected_speech"]["harness_s"] > 1.5
    assert data["detected_speech"]["agent_s"] > 2.0


def test_the_comfort_noise_does_not_count_as_speech(tmp_path):
    """The harness channel carries a continuous floor: counting it would give 100% overlap."""
    harness = _track(("silence", 5.0))
    agent = _track(("voice", 5.0))
    data = measure(_wav(tmp_path, harness, agent))
    assert data["detected_speech"]["harness_s"] == 0.0
    assert data["overlaps"] == 0


def test_a_channel_without_speech_is_published_so_the_zero_does_not_deceive(tmp_path):
    harness = _track(("voice", 3.0))
    agent = _track(("silence", 3.0))
    data = measure(_wav(tmp_path, harness, agent))
    assert data["detected_speech"]["agent_s"] == 0.0
    assert data["overlaps"] == 0


# --------------------------------------------------------------------------
# The activity threshold
# --------------------------------------------------------------------------


def test_activity_is_measured_with_the_clip_drop_not_the_level_drop():
    """With 25 dB the harness channel gave 6.2 s of speech for eight clips."""
    assert DROP_FOR_ACTIVITY_DB == 35.0


def test_a_stricter_threshold_detects_less_speech():
    """This is the bug the first version had, in the form of a test."""
    audio_track = _track(("voice", 2.0), ("silence", 1.0), ("voice", 2.0))
    generous = sum(r.duration for r in speech_streaks(audio_track, SAMPLE_RATE, drop_db=35.0))
    strict = sum(r.duration for r in speech_streaks(audio_track, SAMPLE_RATE, drop_db=10.0))
    assert strict < generous


# --------------------------------------------------------------------------
# Streaks
# --------------------------------------------------------------------------


def test_two_segments_separated_by_a_long_gap_are_two_streaks():
    audio_track = _track(("voice", 1.0), ("silence", 1.0), ("voice", 1.0))
    assert len(speech_streaks(audio_track, SAMPLE_RATE)) == 2


def test_a_short_gap_does_not_split_the_streak():
    """Splitting one phrase too many does no harm; gluing two that do not belong together does."""
    audio_track = _track(("voice", 1.0), ("silence", 0.1), ("voice", 1.0))
    assert len(speech_streaks(audio_track, SAMPLE_RATE)) == 1


def test_a_mute_channel_has_no_streaks():
    assert speech_streaks(_track(("silence", 2.0)), SAMPLE_RATE) == []


def test_the_streak_covers_the_segment_with_speech():
    audio_track = _track(("silence", 1.0), ("voice", 2.0), ("silence", 1.0))
    streak = speech_streaks(audio_track, SAMPLE_RATE)[0]
    assert streak.start == pytest.approx(1.0, abs=0.05)
    assert streak.end == pytest.approx(3.0, abs=0.05)


def test_the_streak_that_is_NOT_the_last_also_closes_where_it_should():
    """The first one is closed in a different code branch, and an off-by-one fits there."""
    audio_track = _track(("voice", 1.0), ("silence", 1.0), ("voice", 1.0))
    first, second_one = speech_streaks(audio_track, SAMPLE_RATE)
    # Half-window tolerance: a one-window error shortens each streak by 10 ms
    # and the borderline overlaps are lost, which are the ones to detect.
    assert first.start == pytest.approx(0.0, abs=0.005)
    assert first.end == pytest.approx(1.0, abs=0.005)
    assert second_one.start == pytest.approx(2.0, abs=0.005)


# --------------------------------------------------------------------------
# Matching streaks
# --------------------------------------------------------------------------


def test_an_overlap_shorter_than_the_minimum_does_not_count():
    """It could be the tail of a frame that arrived late, not an interruption."""
    harness = [Streak(0.0, 1.0)]
    agent = [Streak(0.95, 2.0)]
    assert overlaps_between(harness, agent) == []


def test_a_long_overlap_does_count():
    harness = [Streak(0.0, 1.0)]
    agent = [Streak(0.5, 2.0)]
    overlaps = overlaps_between(harness, agent)
    assert len(overlaps) == 1
    assert overlaps[0].duration == pytest.approx(0.5)
    assert overlaps[0].agent_cut_it_off


def test_streaks_that_do_not_touch_do_not_overlap():
    assert overlaps_between([Streak(0.0, 1.0)], [Streak(1.5, 2.0)]) == []


def test_starting_at_the_same_time_is_not_interrupting():
    overlaps = overlaps_between([Streak(1.0, 3.0)], [Streak(1.0, 3.0)])
    assert len(overlaps) == 1
    assert overlaps[0].who_entered == "a la vez"
    assert not overlaps[0].agent_cut_it_off


def test_the_overlaps_come_out_ordered_in_time():
    harness = [Streak(5.0, 7.0), Streak(1.0, 3.0)]
    agent = [Streak(1.5, 4.0), Streak(5.5, 8.0)]
    overlaps = overlaps_between(harness, agent)
    assert [round(s.start, 1) for s in overlaps] == [1.5, 5.5]


def test_a_long_agent_streak_can_overlap_two_harness_ones():
    harness = [Streak(1.0, 2.0), Streak(3.0, 4.0)]
    agent = [Streak(0.5, 5.0)]
    assert len(overlaps_between(harness, agent)) == 2


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def test_the_summary_separates_who_interrupted():
    overlaps = overlaps_between(
        [Streak(0.0, 2.0), Streak(5.0, 8.0)], [Streak(1.0, 3.0), Streak(4.0, 6.0)]
    )
    data = summary(overlaps, duration_s=10.0)
    assert data["overlaps"] == 2
    assert data["the_agent_talked_over"] == 1
    assert data["the_harness_talked_over"] == 1


def test_the_summary_publishes_no_percentile():
    """The agent channel is timestamped on arrival: a p95 from here would be a lie."""
    data = summary(overlaps_between([Streak(0.0, 2.0)], [Streak(1.0, 3.0)]), duration_s=10.0)
    assert not any("p50" in k or "p95" in k or "p99" in k for k in data)


def test_without_overlaps_the_fraction_is_zero_and_not_null():
    data = summary([], duration_s=10.0)
    assert data["overlaps"] == 0
    assert data["run_fraction"] == 0.0


def test_a_run_of_zero_duration_does_not_divide_by_zero():
    assert summary([], duration_s=0.0)["run_fraction"] is None


# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------


def test_a_mono_wav_is_rejected(tmp_path):
    """Duplicating the channel would give 100% overlap and look like a finding."""
    ruta = tmp_path / "mono.wav"
    write_wav(ruta, _track(("voice", 1.0)), SAMPLE_RATE)
    with pytest.raises(AudioNotSupported, match="hacen falta 2"):
        read_wav_stereo(ruta)


def test_the_channels_are_not_mixed_when_reading(tmp_path):
    left_side = np.full(100, 1000, dtype=np.int16)
    derecho = np.full(100, -1000, dtype=np.int16)
    ruta = tmp_path / "dos.wav"
    write_wav_stereo(ruta, left_side, derecho, SAMPLE_RATE)
    a, b, sample_rate = read_wav_stereo(ruta)
    assert sample_rate == SAMPLE_RATE
    assert np.array_equal(a, left_side)
    assert np.array_equal(b, derecho)


def test_the_gap_between_streaks_is_short_on_purpose():
    assert STREAK_GAP_S <= 0.5


def test_the_overlap_minimum_is_the_same_poc3_uses():
    """Checks the duplicated value against PoC3's, wherever PoC3 can be imported.

    It is **skipped** in environments that do not have PoC3 instead of forcing
    the import: `shared/` cannot depend on any PoC, and this suite runs under
    all three interpreters. Where PoC3 is present, the check is done.
    """
    latency = pytest.importorskip(
        "poc3_gpt_live.app.latency", reason="PoC3 no está en este entorno"
    )
    assert MIN_OVERLAP_S == latency.OVERLAP_WINDOW_S
