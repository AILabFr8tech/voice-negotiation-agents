"""Tests for the percentiles and the latency buckets.

`TestPercentil` compares against `statistics.quantiles(method="inclusive")`,
which is the same Hyndman-Fan method 7 that `numpy.percentile` uses. It serves
as an anchor: if someone changes the interpolation, the report would shift every
number and it shows up here immediately.
"""

from __future__ import annotations

import statistics

import pytest

from shared.metrics.latency import (
    BUCKET_OPENING,
    BUCKET_INTERRUPTED,
    BUCKET_REGIMEN,
    TurnSample,
    load_samples,
    classify,
    percentile_label,
    percentile,
    summarize_samples,
    summarize_values,
)


class TestPercentile:
    def test_matches_the_standard_library(self) -> None:
        values = [float(v) for v in range(1, 11)]
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        assert percentile(values, 25) == pytest.approx(quartiles[0])
        assert percentile(values, 50) == pytest.approx(quartiles[1])
        assert percentile(values, 75) == pytest.approx(quartiles[2])

    @pytest.mark.parametrize(
        ("p", "expected"),
        [(0, 1.0), (25, 3.25), (50, 5.5), (75, 7.75), (95, 9.55), (99, 9.91), (100, 10.0)],
    )
    def test_known_values(self, p: float, expected: float) -> None:
        values = [float(v) for v in range(1, 11)]
        assert percentile(values, p) == pytest.approx(expected)

    def test_a_single_sample(self) -> None:
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 99) == 42.0

    def test_empty_list(self) -> None:
        assert percentile([], 50) is None

    def test_does_not_care_about_the_order(self) -> None:
        assert percentile([9.0, 1.0, 5.0], 50) == 5.0

    def test_labels(self) -> None:
        assert percentile_label(50.0) == "p50"
        assert percentile_label(99.9) == "p99.9"


class TestValueSummary:
    def test_statistics(self) -> None:
        summary = summarize_values([100.0, 200.0, 300.0])
        assert summary["n"] == 3
        assert summary["min"] == 100.0
        assert summary["max"] == 300.0
        assert summary["mean"] == 200.0
        assert summary["p50"] == 200.0

    def test_flags_unreliable_percentiles(self) -> None:
        """With 3 samples, the p95 is the maximum in disguise and that has to be said."""
        summary = summarize_values([100.0, 200.0, 300.0])
        assert summary["reliable"]["p50"] is True
        assert summary["reliable"]["p95"] is False
        assert summary["reliable"]["p99"] is False

    def test_reliable_p95_with_enough_sample(self) -> None:
        summary = summarize_values([float(v) for v in range(30)])
        assert summary["reliable"]["p95"] is True

    def test_without_values(self) -> None:
        summary = summarize_values([])
        assert summary["n"] == 0
        assert summary["reliable"]["p50"] is False


class TestBuckets:
    def test_turn_zero_is_opening(self) -> None:
        assert classify(TurnSample(indice=0)) == BUCKET_OPENING

    def test_the_rest_are_regime(self) -> None:
        assert classify(TurnSample(indice=3)) == BUCKET_REGIMEN

    def test_interrupted_wins_over_opening(self) -> None:
        """Documented decision: the interruption dominates what is measured."""
        sample = TurnSample(indice=0, interrupted=True)
        assert classify(sample) == BUCKET_INTERRUPTED

    def test_explicit_opening(self) -> None:
        assert classify(TurnSample(indice=5, is_opening=True)) == BUCKET_OPENING

    def test_opening_is_derived_from_the_index(self) -> None:
        assert TurnSample(indice=0).is_opening is True
        assert TurnSample(indice=1).is_opening is False


class TestFromEvents:
    def test_computes_the_deltas_in_ms(self) -> None:
        """Pipecat's real order: the transcript arrives before the end of turn."""
        sample = TurnSample.from_events(
            0,
            t_speech_end=10.0,
            t_transcript=10.35,
            t_end_of_turn=10.4,
            t_primer_token=10.95,
            t_primer_audio=11.45,
        )
        assert sample.full_turn_ms == pytest.approx(1450.0)
        assert sample.stt_ms == pytest.approx(350.0)
        assert sample.end_of_turn_ms == pytest.approx(400.0)
        # Anchored on the end of turn, not on the transcript: the 50 ms of the
        # analyzer's decision are not LLM time.
        assert sample.llm_ttft_ms == pytest.approx(550.0)
        assert sample.tts_ttfb_ms == pytest.approx(500.0)
        assert sample.notes == ()

    def test_missing_mark_does_not_invent_zero(self) -> None:
        """A silent zero would ruin the p50; None is excluded and reported."""
        sample = TurnSample.from_events(
            1, t_speech_end=0.0, t_primer_audio=1.0
        )
        assert sample.full_turn_ms == pytest.approx(1000.0)
        assert sample.llm_ttft_ms is None
        assert sample.end_of_turn_ms is None

    def test_without_end_of_turn_the_llm_anchors_on_the_transcript(self) -> None:
        sample = TurnSample.from_events(
            1,
            t_speech_end=0.0,
            t_transcript=0.3,
            t_primer_token=0.9,
            t_primer_audio=1.2,
        )
        assert sample.llm_ttft_ms == pytest.approx(600.0)

    def test_a_negative_delta_comes_out_none_with_reason(self) -> None:
        """Regression: negative `stt_ms`.

        A negative latency is impossible; it means the marks arrived out of
        order. Writing it would silently poison the percentile, so the stage is
        discarded and the reason is noted.
        """
        sample = TurnSample.from_events(
            2,
            t_speech_end=1.0,
            t_transcript=0.9,  # before the end of speech: impossible
            t_end_of_turn=1.4,
            t_primer_token=2.0,
            t_primer_audio=2.3,
        )
        assert sample.stt_ms is None
        assert any("stt" in note for note in sample.notes)
        # The healthy stages of the same turn are kept.
        assert sample.end_of_turn_ms == pytest.approx(400.0)
        assert sample.full_turn_ms == pytest.approx(1300.0)

    def test_without_the_turn_zero_the_stages_that_depend_on_it_are_left_empty(self) -> None:
        sample = TurnSample.from_events(
            3, t_end_of_turn=1.0, t_primer_token=1.5, t_primer_audio=1.8
        )
        assert sample.full_turn_ms is None
        assert sample.stt_ms is None
        assert sample.llm_ttft_ms == pytest.approx(500.0)


class TestSampleSummary:
    def _samples(self) -> list[TurnSample]:
        return [
            TurnSample(indice=0, full_turn_ms=2000.0, llm_ttft_ms=900.0),
            TurnSample(indice=1, full_turn_ms=1000.0, llm_ttft_ms=400.0),
            TurnSample(indice=2, full_turn_ms=1200.0, llm_ttft_ms=500.0),
            TurnSample(indice=3, full_turn_ms=300.0, interrupted=True),
        ]

    def test_separates_the_three_buckets(self) -> None:
        summary = summarize_samples(self._samples())
        buckets = summary["buckets"]
        assert buckets[BUCKET_OPENING]["full_turn"]["n"] == 1
        assert buckets[BUCKET_REGIMEN]["full_turn"]["n"] == 2
        assert buckets[BUCKET_INTERRUPTED]["full_turn"]["n"] == 1

    def test_the_opening_does_not_contaminate_the_regime(self) -> None:
        """The point of separating buckets: the cold start does not raise the p50."""
        summary = summarize_samples(self._samples())
        regimen = summary["buckets"][BUCKET_REGIMEN]["full_turn"]
        assert regimen["p50"] == pytest.approx(1100.0)
        assert regimen["max"] == 1200.0

    def test_breakdown_by_stage(self) -> None:
        summary = summarize_samples(self._samples())
        stages = summary["buckets"][BUCKET_REGIMEN]["stages"]
        assert stages["llm_ttft"]["n"] == 2
        assert stages["llm_ttft"]["p50"] == pytest.approx(450.0)

    def test_without_stages_the_key_is_omitted(self) -> None:
        """PoC3 (Realtime) has no stages: it must not drag along empty columns."""
        samples = [
            TurnSample(indice=i, full_turn_ms=500.0 + i) for i in range(4)
        ]
        summary = summarize_samples(samples)
        assert "stages" not in summary["buckets"][BUCKET_REGIMEN]

    def test_counts_turns_without_measurement(self) -> None:
        samples = [
            TurnSample(indice=0, full_turn_ms=1000.0),
            TurnSample(indice=1),  # not measured
        ]
        summary = summarize_samples(samples)
        assert summary["buckets"][BUCKET_REGIMEN]["turns_without_measurement"] == 1
        assert any("cobertura" in a for a in summary["advertencias"])


class TestAdvisories:
    def test_without_samples_it_warns(self) -> None:
        summary = summarize_samples([])
        assert summary["n_turns"] == 0
        assert any("sin muestras" in a for a in summary["advertencias"])

    def test_warns_percentile_with_small_sample(self) -> None:
        samples = [
            TurnSample(indice=i, full_turn_ms=1000.0 + i) for i in range(5)
        ]
        summary = summarize_samples(samples)
        assert any("p95" in a for a in summary["advertencias"])

    def test_several_openings_in_one_call_warns(self) -> None:
        samples = [
            TurnSample(indice=0, full_turn_ms=1000.0, is_opening=True),
            TurnSample(indice=1, full_turn_ms=1000.0, is_opening=True),
        ]
        summary = summarize_samples(samples, calls=1)
        assert any("apertura" in a for a in summary["advertencias"])

    def test_consolidating_several_calls_does_not_warn(self) -> None:
        """15 calls have 15 openings: warning about that would be noise."""
        samples: list[TurnSample] = []
        for _ in range(15):
            samples.append(
                TurnSample(indice=0, full_turn_ms=2000.0, is_opening=True)
            )
            samples.extend(
                TurnSample(indice=i, full_turn_ms=1000.0 + i)
                for i in range(1, 8)
            )
        summary = summarize_samples(samples, calls=15)
        assert not any("marcados como apertura" in a for a in summary["advertencias"])


class TestSerialization:
    def test_round_trip(self) -> None:
        original = TurnSample(
            indice=2, full_turn_ms=1234.5, llm_ttft_ms=500.0, interrupted=True
        )
        clone = TurnSample.from_dict(original.to_dict())
        assert clone == original

    def test_ignores_unknown_keys(self) -> None:
        sample = TurnSample.from_dict(
            {"indice": 1, "full_turn_ms": 900.0, "something_odd": 5}
        )
        assert sample.full_turn_ms == 900.0

    def test_loading_samples_tolerates_garbage(self) -> None:
        assert load_samples(None) == []
        assert load_samples("no es lista") == []
        assert len(load_samples([{"full_turn_ms": 1.0}, "basura"])) == 1

    def test_loading_samples_fills_the_index(self) -> None:
        samples = load_samples(
            [{"full_turn_ms": 1.0}, {"full_turn_ms": 2.0}]
        )
        assert [m.indice for m in samples] == [0, 1]
        assert samples[0].is_opening is True
