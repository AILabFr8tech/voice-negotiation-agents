"""Matching VAD cuts against the pauses in the clips."""

from __future__ import annotations

import numpy as np
import pytest

from shared.carrier_simulator.cero import ZeroMark
from shared.carrier_simulator.cruce import EstimatedZero, group
from shared.metrics.fragmentation import (
    MIN_SILENCE_POC2,
    MIN_PAUSE_S,
    STOP_SECS_POC1,
    EFFECTIVE_THRESHOLD_POC1,
    EFFECTIVE_THRESHOLD_POC2,
    Pause,
    _attribute,
    _census,
    premature_cuts,
    inside_the_pause,
    chance_fraction,
    internal_pauses,
    verdict,
)

SAMPLE_RATE = 8000
UMBRAL = -40.0


def _sig(*segments: tuple[str, float]) -> np.ndarray:
    """Builds a signal from `("voice"|"silence", seconds)` segments."""
    parts = []
    for kind, seconds in segments:
        n = int(round(seconds * SAMPLE_RATE))
        if kind == "voice":
            # White noise comfortably above the -40 dBFS threshold.
            generator = np.random.default_rng(0)
            parts.append(
                (generator.normal(0, 3000, n)).clip(-32000, 32000).astype(np.int16)
            )
        else:
            parts.append(np.zeros(n, dtype=np.int16))
    return np.concatenate(parts)


# --------------------------------------------------------------------------
# The clip's pauses
# --------------------------------------------------------------------------


def test_finds_a_gap_between_two_speech_segments():
    sig = _sig(("voice", 0.5), ("silence", 0.4), ("voice", 0.5))
    pauses = internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=UMBRAL)
    assert len(pauses) == 1
    # Half-window tolerance: the pause starts at the first window WITHOUT
    # speech, not at the last one that had it. A one-window error would shift
    # every cut by 10 ms and the bias subtracted from them would stop adding up.
    assert pauses[0].duration == pytest.approx(0.4, abs=0.005)
    assert pauses[0].start == pytest.approx(0.5, abs=0.005)


def test_the_silence_at_the_start_is_not_a_pause():
    """The head of the clip is not a gap: it has no speech on its left."""
    sig = _sig(("silence", 0.6), ("voice", 0.5))
    assert internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=UMBRAL) == []


def test_the_clip_tail_is_not_a_pause():
    """Counting it would make every clip 'end in a pause'. The tail is the end of speech."""
    sig = _sig(("voice", 0.5), ("silence", 0.9))
    assert internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=UMBRAL) == []


def test_a_gap_shorter_than_the_minimum_does_not_count():
    sig = _sig(("voice", 0.5), ("silence", 0.1), ("voice", 0.5))
    assert internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=UMBRAL, min_s=0.2) == []


def test_finds_several_gaps_in_order():
    sig = _sig(
        ("voice", 0.3), ("silence", 0.3), ("voice", 0.3), ("silence", 0.5), ("voice", 0.3)
    )
    pauses = internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=UMBRAL)
    assert len(pauses) == 2
    assert pauses[0].start < pauses[1].start
    assert pauses[0].duration < pauses[1].duration


def test_a_higher_threshold_finds_more_pauses():
    """The threshold comes from the manifest on purpose: a different one yields different pauses."""
    sig = np.concatenate(
        [_sig(("voice", 0.4)), (_sig(("voice", 0.4)) // 40), _sig(("voice", 0.4))]
    )
    loose = internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=-60.0)
    strict = internal_pauses(sig, SAMPLE_RATE, umbral_dbfs=-25.0)
    assert len(strict) > len(loose)


def test_a_clip_without_speech_has_no_pauses():
    assert internal_pauses(_sig(("silence", 1.0)), SAMPLE_RATE, umbral_dbfs=UMBRAL) == []


# --------------------------------------------------------------------------
# The effective thresholds of each SDK
# --------------------------------------------------------------------------


def test_pipecat_rounds_down_and_waits_less_than_configured():
    """`round(0.2 / 0.032)` = 6 frames = 192 ms, not 200."""
    assert EFFECTIVE_THRESHOLD_POC1 == pytest.approx(0.192)
    assert EFFECTIVE_THRESHOLD_POC1 < STOP_SECS_POC1


def test_livekit_rounds_up_to_the_next_window_and_waits_longer():
    """It accumulates duration and fires on `>=`: 18 windows = 576 ms, not 550."""
    assert EFFECTIVE_THRESHOLD_POC2 == pytest.approx(0.576)
    assert EFFECTIVE_THRESHOLD_POC2 > MIN_SILENCE_POC2


def test_the_two_knobs_quantize_in_opposite_directions():
    """It widens the band that separates them: from [200, 550) to [192, 576)."""
    configured = MIN_SILENCE_POC2 - STOP_SECS_POC1
    effective = EFFECTIVE_THRESHOLD_POC2 - EFFECTIVE_THRESHOLD_POC1
    assert effective > configured


# --------------------------------------------------------------------------
# Attributing a cut to a pause
# --------------------------------------------------------------------------


def test_a_cut_inside_the_pause_is_attributed():
    pause = Pause(start=1.0, end=1.5)
    assert inside_the_pause(1.2, pause, bias_s=0.0)


def test_a_cut_before_the_pause_starts_does_not_count():
    """There is no slack on the left: that is cutting mid-word."""
    pause = Pause(start=1.0, end=1.5)
    assert not inside_the_pause(0.95, pause, bias_s=0.3)


def test_on_the_right_the_run_bias_is_added():
    """A premature zero arrives along the same path and pays the same delay."""
    pause = Pause(start=1.0, end=1.5)
    assert not inside_the_pause(1.7, pause, bias_s=0.0)
    assert inside_the_pause(1.7, pause, bias_s=0.25)


def test_the_pause_that_contains_it_is_preferred_over_the_nearest_start_one():
    pauses = [Pause(start=0.9, end=0.95), Pause(start=1.0, end=2.0)]
    pause, _, inside = _attribute(pauses, 1.5, bias_s=0.0)
    assert inside
    assert pause == pauses[1]


def test_if_none_contains_it_the_nearest_is_reported_and_marked_outside():
    pauses = [Pause(start=5.0, end=5.4)]
    pause, distance, inside = _attribute(pauses, 1.0, bias_s=0.1)
    assert not inside
    assert pause == pauses[0]
    assert distance == pytest.approx(-4.0)


def test_without_pauses_nothing_is_attributed():
    assert _attribute([], 1.0, bias_s=0.1) == (None, None, False)


# --------------------------------------------------------------------------
# The chance level
# --------------------------------------------------------------------------


def test_without_pauses_the_chance_is_zero():
    assert chance_fraction([], speech_onset=0.0, end_of_speech=2.0, bias_s=0.0) == 0.0


def test_chance_is_the_part_of_the_segment_covered_by_pauses():
    pauses = [Pause(start=0.5, end=1.5)]
    fraction = chance_fraction(
        pauses, speech_onset=0.0, end_of_speech=2.0, bias_s=0.0
    )
    # Exact, not approximate: the 10 ms grid over [0, 2] is 201 points and
    # 101 fall in [0.5, 1.5]. Requiring the exact number is what distinguishes
    # a reproducible sweep from one that starts at an arbitrary point.
    assert fraction == pytest.approx(101 / 201, abs=1e-9)


def test_the_bias_widens_the_chance():
    """If the criterion is more generous, chance goes up — and the comparison has to be against it."""
    pauses = [Pause(start=0.5, end=1.0)]
    sin = chance_fraction(pauses, speech_onset=0.0, end_of_speech=2.0, bias_s=0.0)
    having = chance_fraction(pauses, speech_onset=0.0, end_of_speech=2.0, bias_s=0.5)
    assert having > sin


def test_chance_does_not_depend_on_any_seed():
    """It is swept on a grid: two calls give exactly the same result."""
    pauses = [Pause(start=0.3, end=0.9)]
    a = chance_fraction(pauses, speech_onset=0.0, end_of_speech=2.0, bias_s=0.1)
    b = chance_fraction(pauses, speech_onset=0.0, end_of_speech=2.0, bias_s=0.1)
    assert a == b


def test_an_empty_segment_gives_no_chance():
    assert chance_fraction([], speech_onset=1.0, end_of_speech=1.0, bias_s=0.0) is None


# --------------------------------------------------------------------------
# Premature cuts
# --------------------------------------------------------------------------


def _mark(turn: str, clip_id: str, start: float) -> ZeroMark:
    return ZeroMark(
        sequence=1,
        turn=turn,
        clip_id=clip_id,
        start_of_clip=start,
        clip_end=start + 4.0,
        publish_sample_rate=SAMPLE_RATE,
        measured_end_of_speech=start + 3.8,
    )


def test_last_event_of_the_line_is_not_a_premature_cut():
    """There the carrier really did stop talking: it is the end, not a split."""
    marks = [_mark("T1", "c1", 100.0)]
    zeros = [
        EstimatedZero(indice=0, start=100.5, end=101.0),
        EstimatedZero(indice=1, start=101.5, end=103.9),
    ]
    lines, _ = group(marks, zeros)
    pauses = {"c1": [Pause(start=1.0, end=1.6)]}
    cortes = premature_cuts(lines, pauses)
    assert len(cortes) == 1
    assert cortes[0].offset_seg == pytest.approx(1.0)


def test_an_unsplit_line_contributes_no_cuts():
    marks = [_mark("T1", "c1", 100.0)]
    zeros = [EstimatedZero(indice=0, start=100.5, end=103.9)]
    lines, _ = group(marks, zeros)
    assert premature_cuts(lines, {"c1": []}) == []


def test_the_offset_is_relative_to_the_clip_start():
    marks = [_mark("T1", "c1", 500.0)]
    zeros = [
        EstimatedZero(indice=0, start=500.1, end=502.25),
        EstimatedZero(indice=1, start=502.5, end=503.9),
    ]
    lines, _ = group(marks, zeros)
    cortes = premature_cuts(lines, {"c1": []})
    assert cortes[0].offset_seg == pytest.approx(2.25)


def test_last_event_is_the_last_IN_TIME_not_in_the_list():
    """The pipeline does not guarantee order, and mixing them up would pick the wrong cut."""
    marks = [_mark("T1", "c1", 100.0)]
    # Reversed on purpose: the one that closes the line comes first in the list.
    zeros = [
        EstimatedZero(indice=1, start=101.5, end=103.9),
        EstimatedZero(indice=0, start=100.5, end=101.0),
    ]
    lines, _ = group(marks, zeros)
    cortes = premature_cuts(lines, {"c1": []})
    assert len(cortes) == 1
    assert cortes[0].offset_seg == pytest.approx(1.0)


def test_the_bias_reaches_the_attribution_of_the_cut():
    marks = [_mark("T1", "c1", 100.0)]
    zeros = [
        EstimatedZero(indice=0, start=100.2, end=101.7),
        EstimatedZero(indice=1, start=102.0, end=103.9),
    ]
    lines, _ = group(marks, zeros)
    pauses = {"c1": [Pause(start=1.0, end=1.5)]}
    assert not premature_cuts(lines, pauses, bias_s=0.0)[0].in_a_pause
    assert premature_cuts(lines, pauses, bias_s=0.3)[0].in_a_pause


# --------------------------------------------------------------------------
# The census and the verdict
# --------------------------------------------------------------------------


def test_the_census_separates_the_band_only_pipecat_cuts():
    pauses = {
        "c1": [Pause(0.0, 0.30), Pause(1.0, 1.40)],  # 300 and 400 ms: pipecat only
        "c2": [Pause(0.0, 0.80)],  # 800 ms: both
        "c3": [],
    }
    census = _census(pauses)
    assert census["pauses"] == 3
    assert census["clips_with_any"] == 2
    assert census["in_the_band_only_poc1"] == 2
    assert census["above_the_poc2_threshold"] == 1


def test_the_census_does_not_publish_a_trivial_poc1_count():
    """With the detection minimum at 200 ms it would be all of them, and it would say nothing."""
    census = _census({"c1": [Pause(0.0, 0.3)]})
    assert "pauses_that_would_cut_in_poc1" not in census


def _run(cortes: int, in_pause: int, chance: float, opportunities: int) -> dict:
    return {
        "premature_cuts": cortes,
        "cuts_in_a_clip_pause": in_pause,
        "chance_fraction": chance,
        "opportunities_in_split_clips": opportunities,
    }


def test_if_the_cuts_do_not_exceed_chance_the_clip_does_not_explain_them():
    final = verdict([_run(cortes=10, in_pause=6, chance=0.6, opportunities=30)])
    assert "no explica" in final["verdict"]


def test_if_almost_all_pauses_are_charged_the_fragmentation_is_from_the_stimulus():
    final = verdict([_run(cortes=9, in_pause=9, chance=0.2, opportunities=10)])
    assert "ESTÍMULO" in final["verdict"]


def test_cutting_only_at_pauses_but_at_few_is_the_knob_stimulus_pair():
    final = verdict([_run(cortes=9, in_pause=9, chance=0.57, opportunities=27)])
    assert "OPORTUNIDAD" in final["verdict"]
    assert final["opportunities_taken"] == pytest.approx(0.333, abs=0.001)


def test_without_cuts_there_is_no_verdict_to_give():
    final = verdict([_run(cortes=0, in_pause=0, chance=0.5, opportunities=0)])
    assert final["paused_fraction"] is None
    assert "no hubo" in final["verdict"]


def test_verdict_sums_several_runs():
    final = verdict(
        [
            _run(cortes=4, in_pause=4, chance=0.5, opportunities=16),
            _run(cortes=5, in_pause=5, chance=0.6, opportunities=11),
        ]
    )
    assert final["premature_cuts"] == 9
    assert final["opportunities"] == 27
    assert final["chance_fraction"] == pytest.approx(0.55)


def test_the_pause_minimum_is_the_stop_secs_of_pipecat():
    """Below that, no framework in the comparison can cut."""
    assert MIN_PAUSE_S == STOP_SECS_POC1
