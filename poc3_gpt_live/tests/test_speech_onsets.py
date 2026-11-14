"""The playhead offset at each speech onset, which corrects the acknowledgement.

It is the queueing term of the discriminant's error budget. A turn's
acknowledgement is timestamped when the delta **arrives** at the bridge; the
carrier hears it `offset_ms` later, because the monotonic playhead pushed that
block back so as not to overlap the previous one. Publishing it allows it to be
subtracted, and subtracting it is what leaves the dead band at 50 ms instead of
150.

**Why `line_onsets`, which already existed, is not enough.** That table groups
by silence between **arrivals**, and in the first block of each group the
playhead has been reset: it comes out 0 by construction. But a turn can start
speaking in the middle of a line that was already sounding, and there the
playhead is not at zero. On `CA221584…` there were **16 speech onsets and only
1 line onset**: the old table certified 0.0 ms for 6% of the cases and said
nothing about the rest.
"""

from __future__ import annotations

import numpy as np

from poc3_gpt_live.diagnostics.audit_recorder import (
    SILENCE_BEFORE_ONSET_S,
    TELEPHONY_SAMPLE_RATE,
    SPEECH_THRESHOLD_RMS,
    SPEECH_WINDOW_S,
    _Canal,
)


def _block(ms: int, amplitude: float) -> bytes:
    """A 300 Hz tone of `ms` milliseconds, in PCM16 at 8 kHz."""
    n = int(TELEPHONY_SAMPLE_RATE * ms / 1000)
    t = np.arange(n) / TELEPHONY_SAMPLE_RATE
    return (np.sin(2 * np.pi * 300 * t) * amplitude).astype("<i2").tobytes()


VOICE = 6_000.0  # RMS ~4,240: well above the threshold
RELLENO = 30.0  # RMS ~21: well below


class TestTheOnsetMidLine:
    """The case the old table did not see, and the one the acknowledgement corrects."""

    def test_publishes_the_offset_the_line_table_gives_at_zero(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        # Six filler blocks with 100 ms of audio arriving every 60 ms: the
        # playhead gets 40 ms further ahead per block. It is the alpha delivering
        # faster than real time, which is what really happens.
        t = 0.0
        for _ in range(6):
            canal.annotate(t, _block(100, RELLENO))
            t += 0.06
        # And now speech, within the same line.
        canal.annotate(t, _block(100, VOICE))

        accounting = canal.accounting()
        of_voice = accounting["speech_onsets"]
        of_line = accounting["line_onsets"]

        assert of_voice["n"] == 1
        assert of_voice["at_zero"] == 0
        assert of_voice["offset_ms"]["maximum"] == 240.0
        # The old table, over the same run, reports none of this.
        assert of_line["max_offset_ms"] == 0.0

    def test_the_onset_that_opens_a_line_still_comes_out_at_zero(self) -> None:
        """What already worked is not broken: after the silence, the playhead catches up."""
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        canal.annotate(0.0, _block(100, VOICE))
        canal.annotate(3.0, _block(100, VOICE))

        of_voice = canal.accounting()["speech_onsets"]

        assert of_voice["n"] == 2
        assert of_voice["at_zero"] == 2
        assert of_voice["offset_ms"]["maximum"] == 0.0


class TestTheCriterionIsTheSameAsTheBridgeOne:
    def test_the_filler_does_not_count_as_an_onset(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        for i in range(5):
            canal.annotate(i * 0.1, _block(100, RELLENO))

        assert canal.accounting()["speech_onsets"]["n"] == 0

    def test_the_continuous_speech_is_a_single_onset_and_not_one_per_block(self) -> None:
        """Without the hangover, each block of the same sentence would be an onset."""
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        for i in range(8):
            canal.annotate(i * 0.1, _block(100, VOICE))

        assert canal.accounting()["speech_onsets"]["n"] == 1

    def test_a_pause_shorter_than_the_hangover_does_not_open_an_onset(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        canal.annotate(0.0, _block(100, VOICE))
        # A 200 ms gap: below the hangover's 300.
        canal.annotate(0.3, _block(100, VOICE))

        assert canal.accounting()["speech_onsets"]["n"] == 1

    def test_a_longer_pause_does_open_an_onset(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        canal.annotate(0.0, _block(100, VOICE))
        canal.annotate(0.6, _block(100, VOICE))

        assert canal.accounting()["speech_onsets"]["n"] == 2

    def test_publishes_the_criterion_with_the_bridge_values(self) -> None:
        """If a different threshold were used here, the table would describe different onsets."""
        from poc3_gpt_live.app import bridge

        assert SPEECH_THRESHOLD_RMS == bridge.SPEECH_THRESHOLD_RMS
        assert SPEECH_WINDOW_S == bridge.SPEECH_WINDOW_S

        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        canal.annotate(0.0, _block(100, VOICE))
        criterion = canal.accounting()["speech_onsets"]["criterion"]

        assert "200" in criterion and "20 ms" in criterion
        assert f"{SILENCE_BEFORE_ONSET_S * 1000:.0f} ms" in criterion


class TestTheTableIsAlwaysIncluded:
    def test_a_mute_channel_neither_breaks_nor_invents(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)

        of_voice = canal.accounting()["speech_onsets"]

        assert of_voice["n"] == 0
        assert of_voice["tabla"] == []
        assert of_voice["offset_ms"]["maximum"] == 0.0

    def test_each_row_says_where_it_landed_and_with_how_much_energy(self) -> None:
        canal = _Canal(sample_rate=TELEPHONY_SAMPLE_RATE)
        canal.annotate(0.0, _block(100, VOICE))

        row = canal.accounting()["speech_onsets"]["tabla"][0]

        assert row["block"] == 0
        assert row["t_seg"] == 0.0
        assert row["rms"] > SPEECH_THRESHOLD_RMS
        assert row["offset_ms"] == 0.0

    def test_none_of_this_runs_inside_the_audio_loop(self) -> None:
        """`annotate` is still a subtraction, a comparison and two `append`s.

        The table is computed at dump time. If the computation crept into
        `annotate`, a call would pay for an energy analysis per frame.
        """
        import inspect

        source = inspect.getsource(_Canal.annotate)
        assert "speech_onsets" not in source
        assert "np." not in source
