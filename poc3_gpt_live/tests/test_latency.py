"""PoC3's reduced capture contract.

Two things that can only be tested here, without spending a call: that the 4
stages PoC3 does not measure **are omitted** from the JSON instead of coming out
as nulls, and that `interrupted` means a cut-off and not an overlap.
"""

from __future__ import annotations

import pytest

from poc3_gpt_live.app.latency import (
    INBOUND_FRAME_S,
    CUTOFF_WINDOW_S,
    LatencyCollector,
)
from shared.metrics.latency import CAMPO_PRINCIPAL, STAGE_FIELDS
from shared.metrics.calls import compute


class FakeClock:
    """Controlled clock: what is tested is the shape of the contract, not the ms."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _collector(clock: FakeClock) -> LatencyCollector:
    collector = LatencyCollector(clock=clock)
    collector.begin()
    collector.anchor_server({"model": "gpt-live-1-boulder-alpha"})
    return collector


def _incoming_streak(
    collector: LatencyCollector, clock: FakeClock, *, since: float, until: float
) -> None:
    """A carrier line, frame by frame.

    Twilio sends a frame every 20 ms, so in the real call a sentence arrives as
    dozens of `inbound_speech` marks. The tests simulate it the same way because
    the detector groups speech into **bursts**: a single loose mark is a click on
    the line, and it cannot be a turn's zero.
    """
    t = since
    while t <= until + 1e-9:
        clock.t = round(t, 6)
        collector.inbound_speech()
        t += INBOUND_FRAME_S


def _outgoing_streak(
    collector: LatencyCollector, clock: FakeClock, *, since: float, until: float
) -> None:
    """The agent sounding continuously. The alpha's deltas carry 100 ms of audio."""
    t = since
    while t <= until + 1e-9:
        clock.t = round(t, 6)
        collector.outbound_speech()
        t += 0.1


class TestEnergyMeasurement:
    """`latencies[]` comes from local energy, not from the alpha's events.

    The events do not work as a zero because `turn.done` arrives late
    (`docs/METRICS_CONTRACT.md` §2.7). The energy does not depend on the
    server's clock.
    """

    def test_the_opening_goes_from_the_call_zero_to_the_first_speech(self, clock):
        collector = _collector(clock)
        clock.t = 1.5
        collector.outbound_speech()

        sample = collector.samples[0]
        assert sample.is_opening is True
        assert sample.full_turn_ms == 1_500.0

    def test_turn_goes_from_end_of_incoming_streak_to_first_outgoing(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=4.0)
        clock.t = 6.2
        collector.outbound_speech()

        sample = collector.samples[-1]
        assert sample.is_opening is False
        assert sample.full_turn_ms == 2_200.0

    def test_a_stray_noise_frame_does_not_move_the_zero(self, clock):
        """Regression: line noise does not rewrite the zero.

        The carrier finishes at 4.0 s; at 5.9 s the line slips in a 20 ms click.
        The turn's zero has to remain 4.0, not 5.9.
        """
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=4.0)
        clock.t = 5.9
        collector.inbound_speech()
        clock.t = 7.0
        collector.outbound_speech()

        sample = collector.samples[-1]
        assert sample.full_turn_ms == 3_000.0
        turn_taking = collector.extra_block()["turn_taking"]
        assert turn_taking["discarded_streaks"] == 1

    def test_the_continued_outgoing_speech_does_not_open_another_sample(self, clock):
        """The agent speaking continuously is one turn, not twenty."""
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        for t in (1.02, 1.04, 1.5, 1.9):
            clock.t = t
            collector.outbound_speech()
        assert len(collector.samples) == 1

    def test_a_long_agent_pause_does_not_open_another_sample(self, clock):
        """Regression: an agent line is not split into several samples.

        The agent pauses mid-response while the delegation runs ("Un
        segundo.") and then continues. Without the carrier in between there is
        no new turn to measure: it is the same line.
        """
        from poc3_gpt_live.app.latency import SILENCE_BETWEEN_TURNS_S

        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        clock.t = 1.0 + SILENCE_BETWEEN_TURNS_S + 1.5
        collector.outbound_speech()

        assert len(collector.samples) == 1
        assert collector.extra_block()["turn_taking"]["agent_resumptions"] == 1

    def test_a_backchannel_mid_agent_sentence_does_not_build_a_zero(self, clock):
        """Regression: a backchannel does not produce a false latency of seconds.

        The carrier slips in an "ajá" while the agent is speaking, the agent
        keeps talking over it, and eight seconds later its next sentence would
        consume that zero as if it were a response. It is not: the agent was
        already speaking.
        """
        collector = _collector(clock)
        _outgoing_streak(collector, clock, since=1.0, until=10.0)
        # The backchannel lands in the middle of an agent line.
        _incoming_streak(collector, clock, since=3.0, until=3.3)
        # The agent kept talking over it until 10.0 and resumes at 18.0.
        clock.t = 18.0
        collector.outbound_speech()

        assert len(collector.samples) == 1  # only the opening
        turn_taking = collector.extra_block()["turn_taking"]
        assert turn_taking["zeros_vencidos_por_apoyo_verbal"] == 1
        assert turn_taking["discarded_streaks"] == 0

    def test_the_sample_reconciliation_is_auditable(self, clock):
        """samples = 1 opening + answered bursts. No leftovers."""
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        for start in (3.0, 8.0, 13.0):
            _incoming_streak(collector, clock, since=start, until=start + 0.5)
            clock.t = start + 2.0
            collector.outbound_speech()

        tt = collector.extra_block()["turn_taking"]
        answered_calls = (
            tt["carrier_streaks"]
            - tt["discarded_streaks"]
            - tt["streaks_without_response"]
            - tt["zeros_vencidos_por_apoyo_verbal"]
            - tt["zeros_absorbidos_por_la_apertura"]
        )
        assert len(collector.samples) == 1 + answered_calls == 4

    def test_a_mid_sentence_pause_no_longer_splits_the_streak(self, clock):
        """The hangover calibration.

        Carlos says half a sentence, pauses 500 ms and continues. With a 250 ms
        hangover that would close the burst and set a zero in the middle of his
        sentence; the agent answering 400 ms later would produce a false "375 ms
        of latency". The two halves are a single burst and the zero is the real
        end.
        """
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=3.6)
        _incoming_streak(collector, clock, since=4.1, until=4.7)  # after a 500 ms pause
        clock.t = 6.7
        collector.outbound_speech()

        tt = collector.extra_block()["turn_taking"]
        assert tt["carrier_streaks"] == 1
        assert len(collector.samples) == 2
        assert collector.samples[-1].full_turn_ms == 2_000.0
        # And the internal gap is published so the knob can be re-derived.
        assert 500 in tt["incoming_gaps_ms"]

    def test_after_the_carrier_speaks_it_does_open_another_sample(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=3.5)
        clock.t = 5.0
        collector.outbound_speech()

        assert len(collector.samples) == 2
        assert collector.samples[0].is_opening is True
        assert collector.samples[1].is_opening is False
        assert collector.samples[1].full_turn_ms == 1_500.0

    def test_the_opening_is_the_first_sample_even_if_they_answer_speaking(self, clock):
        """Regression: the opening comes out flagged even if the carrier answers.

        Deriving `is_opening` from "the carrier had not been heard" does not
        work: on the phone the carrier answers by saying something, and the cold
        start would slip entirely into the steady-state bucket.
        """
        collector = _collector(clock)
        _incoming_streak(collector, clock, since=0.5, until=1.0)
        clock.t = 3.0
        collector.outbound_speech()

        sample = collector.samples[0]
        assert sample.is_opening is True
        assert sample.full_turn_ms == 3_000.0
        assert any("contestó el teléfono" in note for note in sample.notes)

    def test_the_alpha_events_no_longer_produce_samples(self, clock):
        """They remain as diagnostics in `per_turn`, not as a metric."""
        collector = _collector(clock)
        clock.t = 2.0
        collector.closed_user_turn({"role": "user", "end_ms": 2_000})
        clock.t = 4.0
        collector.agent_first_audio(start_ms=4_000)

        assert collector.samples == []
        extras = collector.extra_block()["per_turn"]
        assert extras and extras[-1]["source"] == "alpha_events"
        assert extras[-1]["full_turn_server_ms"] == 2_000.0

    def test_the_zero_declares_that_it_is_energy(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        assert collector.zero_used() == "energy_local"
        assert collector.measurement_degradation() is None

    def test_without_outgoing_speech_the_call_leaves_the_pool(self, clock):
        collector = _collector(clock)
        clock.t = 3.0
        collector.inbound_speech()
        exclusion = collector.measurement_degradation()
        assert exclusion is not None
        assert "ninguna muestra" in exclusion["motivo"]


class TestBargeIn:
    """`interrupted` is a cut-off, not an overlap: in full-duplex, overlapping is normal."""

    def _launch(self, clock, *, go_silent_after: float) -> LatencyCollector:
        collector = _collector(clock)
        # The agent sounding continuously from 1.0 to 2.0.
        _outgoing_streak(collector, clock, since=1.0, until=2.0)
        # The carrier talks over it at 2.05 and speaks until 2.5.
        _incoming_streak(collector, clock, since=2.05, until=2.5)
        # The agent keeps sounding `go_silent_after` seconds after the overlap.
        _outgoing_streak(collector, clock, since=2.1, until=2.05 + go_silent_after)
        # It goes quiet, and answers well afterwards.
        clock.t = 2.05 + go_silent_after + 2.0
        collector.outbound_speech()
        return collector

    def test_the_overlap_that_silences_the_agent_marks_the_turn(self, clock):
        collector = self._launch(clock, go_silent_after=CUTOFF_WINDOW_S / 2)
        assert collector.samples[-1].interrupted is True

    def test_the_overlap_that_silences_no_one_marks_nothing(self, clock):
        collector = self._launch(clock, go_silent_after=CUTOFF_WINDOW_S * 4)
        assert collector.samples[-1].interrupted is False

    def test_speaking_after_the_agent_finished_is_not_a_cut(self, clock):
        """The sign of the gap: without a lower bound this would come out `interrupted`.

        The agent finishes at 2.0 and the carrier starts at 2.4, when no one was
        speaking any more.
        """
        collector = _collector(clock)
        _outgoing_streak(collector, clock, since=1.0, until=2.0)
        _incoming_streak(collector, clock, since=2.4, until=3.0)
        clock.t = 4.0
        collector.outbound_speech()

        sample = collector.samples[-1]
        assert sample.interrupted is False
        assert sample.full_turn_ms == 1_000.0
        assert collector.extra_block()["turn_taking"]["cortes"] == 0

    def test_the_two_counts_end_up_in_the_extra_block(self, clock):
        collector = self._launch(clock, go_silent_after=CUTOFF_WINDOW_S / 2)
        turn_taking = collector.extra_block()["turn_taking"]
        assert turn_taking["overlaps"] >= 1
        assert turn_taking["cortes"] >= 1


class TestJsonShape:
    def test_the_stages_poc3_does_not_measure_are_omitted(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()

        for row in collector.to_json():
            for campo in STAGE_FIELDS:
                assert campo not in row
            assert CAMPO_PRINCIPAL in row

    def test_shared_metrics_reads_it_without_per_poc_branches(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=2.5, until=3.0)
        clock.t = 5.0
        collector.outbound_speech()

        content = {
            "call_sid": "CA_test",
            "poc": 3,
            "scenario": 1,
            "repetition": 1,
            "latencies": collector.to_json(),
            "transcript": [
                {"t": 0.0, "role": "agent", "text": "Hola, ¿hablo con Carlos?"},
                {"t": 3.0, "role": "carrier", "text": "Sí, soy Carlos."},
            ],
        }
        metrics = compute(content)
        regimen = (metrics.latency.get("buckets") or {}).get("regimen") or {}
        assert (regimen.get("full_turn") or {}).get("n") == 1
        assert "stages" not in regimen

    def test_the_end_of_turn_verdicts_go_empty(self, clock):
        """The alpha does not expose the probability. It is a declared asymmetry."""
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        assert collector.samples[0].end_of_turn_verdicts == ()


class TestDeliveryDelay:
    """The drift of the server's clock is still measured, no longer as an error."""

    def test_the_drift_does_not_take_the_call_out_of_the_pool(self, clock):
        collector = _collector(clock)
        for i in range(1, 21):
            clock.t = i * 0.5 + i * 0.5
            collector.agent_audio(start_ms=int(i * 500))
        clock.t = 25.0
        collector.outbound_speech()

        delivery = collector.delivery_lag()
        assert delivery["final_delay_ms"] > delivery["initial_delay_ms"]
        assert delivery["delivery_pacing"] < 1.0
        assert collector.measurement_degradation() is None

    def test_without_deltas_it_does_not_rule(self, clock):
        collector = _collector(clock)
        assert collector.clock_verdict()["verdict"] == "no_data"


class TestFailures:
    def test_the_alpha_errors_stay_in_the_call_and_in_the_turn(self, clock):
        collector = _collector(clock)
        clock.t = 2.0
        collector.record_failure(
            "invalid_request_error",
            code_value="invalid_audio",
            message="PCM16 must contain an even number of bytes",
        )
        clock.t = 3.0
        collector.outbound_speech()

        assert len(collector.failures) == 1
        assert collector.failures[0]["t"] == 2.0
        # The sample is not voided: the measured number is still true, but it is
        # flagged so that no one cites it as the engine's latency.
        sample = collector.samples[-1]
        assert sample.full_turn_ms == 3_000.0
        assert any("error(es) del alpha" in note for note in sample.notes)


class TestTurnThatDidNotEnd:
    """Against a turn the carrier did not close there is no latency to measure."""

    def test_if_the_carrier_was_still_speaking_no_number_is_reported(self, clock):
        from poc3_gpt_live.app.latency import SILENCE_FOR_TURN_DONE_S

        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=3.4)
        # The agent starts before the burst completes the hangover: that is
        # full-duplex, not a response to a closed turn.
        clock.t = 3.4 + SILENCE_FOR_TURN_DONE_S / 2
        collector.outbound_speech()

        sample = collector.samples[-1]
        assert sample.full_turn_ms is None
        assert any("seguía hablando" in note for note in sample.notes)
        turn_taking = collector.extra_block()["turn_taking"]
        assert turn_taking["onsets_over_carrier"] == 1

    def test_with_sufficient_silence_it_is_measured(self, clock):
        from poc3_gpt_live.app.latency import SILENCE_FOR_TURN_DONE_S

        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=2.6, until=3.0)
        clock.t = 3.0 + SILENCE_FOR_TURN_DONE_S * 4
        collector.outbound_speech()

        assert collector.samples[-1].full_turn_ms == pytest.approx(
            SILENCE_FOR_TURN_DONE_S * 4000, abs=1
        )


class TestFullReconciliation:
    """The `turn_taking` reconciliation has to close exactly, with no uncounted paths.

    Without the `onsets_over_carrier` and the zero that stays set when the
    carrier speaks at the end and hangs up, the reconciliation comes out with a
    surplus even though the measured numbers are right.
    """

    def _cuadra(self, collector) -> bool:
        tt = collector.extra_block()["turn_taking"]
        answered_calls = (
            tt["carrier_streaks"]
            - tt["discarded_streaks"]
            - tt["streaks_without_response"]
            - tt["zeros_vencidos_por_apoyo_verbal"]
            - tt["zeros_absorbidos_por_la_apertura"]
            - tt["zeros_colgados_al_final"]
        )
        expected_variables = 1 + answered_calls + tt["onsets_over_carrier"]
        return len(collector.samples) == expected_variables

    def test_adds_up_with_an_onset_over_the_carrier(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        # Normal turn.
        _incoming_streak(collector, clock, since=3.0, until=3.6)
        clock.t = 6.0
        collector.outbound_speech()
        # And one where the agent starts over the carrier: it comes out without a number.
        _incoming_streak(collector, clock, since=8.0, until=8.6)
        clock.t = 8.9
        collector.outbound_speech()
        collector.close_pending_ones()

        assert self._cuadra(collector)

    def test_adds_up_when_the_carrier_speaks_at_the_end_and_hangs_up(self, clock):
        collector = _collector(clock)
        clock.t = 1.0
        collector.outbound_speech()
        _incoming_streak(collector, clock, since=3.0, until=3.6)
        clock.t = 6.0
        collector.outbound_speech()
        # The carrier says the last word and hangs up: the agent no longer answers.
        _incoming_streak(collector, clock, since=9.0, until=9.6)
        clock.t = 12.0
        collector.close_pending_ones()

        assert collector.extra_block()["turn_taking"]["zeros_colgados_al_final"] == 1
        assert self._cuadra(collector)

    def test_adds_up_when_the_phone_answers_speaking(self, clock):
        collector = _collector(clock)
        _incoming_streak(collector, clock, since=0.5, until=1.1)
        clock.t = 3.0
        collector.outbound_speech()
        collector.close_pending_ones()

        assert collector.extra_block()["turn_taking"]["zeros_absorbidos_por_la_apertura"] == 1
        assert self._cuadra(collector)


class WallClock:
    """The epoch axis, controlled. It runs offset from the internal one on purpose.

    The offset is what makes the test useful: if the collector mixed up the two
    axes, the numbers would come out the same and it would go unnoticed.
    """

    OFFERSET = 1_700_000_000.0

    def __init__(self, internal: FakeClock) -> None:
        self._internal = internal

    def __call__(self) -> float:
        return self.OFFERSET + self._internal.t


class TestTheWallAxis:
    """The acknowledgement also in `time.time()`, without moving the old axis.

    It is `marks.STEP_3_REQUIREMENT`. What is tested is that it is an **added**
    axis: `full_turn_ms` and the census's `t_local_seg` have to come out
    identical with and without the wall axis, because
    `diagnostics/step0_latencies.py` reads that field and all of PoC3's
    published numbers hang from it.
    """

    def _run(self, collector, clock):
        _incoming_streak(collector, clock, since=1.0, until=2.0)
        _outgoing_streak(collector, clock, since=3.5, until=4.0)
        _incoming_streak(collector, clock, since=6.0, until=7.0)
        _outgoing_streak(collector, clock, since=8.2, until=8.6)
        collector.close_pending_ones()

    def test_does_not_move_a_single_number_of_the_old_timeline(self, clock):
        no_wall = LatencyCollector(clock=clock)
        no_wall.begin()
        self._run(no_wall, clock)
        expected = no_wall.to_json()

        clock.t = 0.0
        with_wall = LatencyCollector(clock=clock, wall_clock=WallClock(clock))
        with_wall.begin()
        self._run(with_wall, clock)

        assert with_wall.to_json() == expected

    def test_publishes_zero_and_acknowledgement_in_epoch(self, clock):
        collector = LatencyCollector(clock=clock, wall_clock=WallClock(clock))
        collector.begin()
        self._run(collector, clock)

        samples = collector.extra_block()["wall_axis"]["samples"]
        regimen = [m for m in samples if not m["is_opening"]]
        assert regimen, "la corrida tiene un turno de régimen"
        turn = regimen[0]
        # Epoch, not uptime: the number has to be on the `time.time` scale.
        assert turn["t_zero_wall"] > WallClock.OFFERSET
        assert turn["t_ack_wall"] > turn["t_zero_wall"]

    def test_the_wall_difference_is_the_same_latency(self, clock):
        """The new axis has to measure the same thing, or it is useless for subtracting."""
        collector = LatencyCollector(clock=clock, wall_clock=WallClock(clock))
        collector.begin()
        self._run(collector, clock)

        for sample in collector.extra_block()["wall_axis"]["samples"]:
            if sample["t_zero_wall"] is None or sample["full_turn_ms"] is None:
                continue
            on_wall = (sample["t_ack_wall"] - sample["t_zero_wall"]) * 1000.0
            assert abs(on_wall - sample["full_turn_ms"]) < 1.0

    def test_without_a_timeline_the_block_says_so_instead_of_missing(self, clock):
        collector = LatencyCollector(clock=clock)
        collector.begin()
        self._run(collector, clock)

        block = collector.extra_block()["wall_axis"]
        assert block["presente"] is False
        assert block["samples"] == []
        assert block["fault"] is None

    def test_a_timeline_failure_turns_it_off_and_is_recorded(self, clock):
        """Same pattern as the hot-path guardrail: no retries."""
        calls = {"n": 0}

        def blows_up() -> float:
            calls["n"] += 1
            raise RuntimeError("el reloj se rompió")

        collector = LatencyCollector(clock=clock, wall_clock=blows_up)
        collector.begin()
        self._run(collector, clock)

        block = collector.extra_block()["wall_axis"]
        assert calls["n"] == 1, "se apaga a la primera, no se reintenta por trama"
        assert block["fault"]["excepcion"] == "RuntimeError"
        assert block["fault"]["sitio"] == "wall_clock"
        # And the latency kept coming out: what is lost is the cross-check, not the measurement.
        assert any(
            m.full_turn_ms is not None for m in collector.samples
        ), "una falla del eje de pared no puede dejar la llamada sin latencia"
