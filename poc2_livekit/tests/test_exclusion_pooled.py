"""Exclusion from the percentile pool when the turn detector degrades.

Covers both halves of the mechanism: that PoC2 detects the degradation and
writes it, and that `shared/metrics/` honors it when consolidating.
"""

from __future__ import annotations

from typing import Any

from poc2_livekit.app.latency import LatencyCollector
from shared.metrics import calls as mod_calls
from shared.metrics import report as mod_report


def _collector(*, requested: str, initial: str | None, final: str | None):
    collector = LatencyCollector()
    collector.begin()
    collector.requested_detector = requested
    collector.initial_detector = initial
    collector.final_detector = final
    return collector


class TestDetection:
    def test_without_degradation_it_does_not_exclude(self):
        collector = _collector(
            requested="v1", initial="turn-detector-v1", final="turn-detector-v1"
        )
        assert collector.detector_degradation() is None

    def test_fallback_to_local_mid_call_excludes(self):
        collector = _collector(
            requested="v1", initial="turn-detector-v1", final="turn-detector-v1-mini"
        )
        degradation = collector.detector_degradation()
        assert degradation is not None
        assert degradation["final_detector"] == "turn-detector-v1-mini"
        assert "dos arquitecturas" in degradation["detail"]

    def test_starting_on_the_wrong_version_also_excludes(self):
        """If v1 was requested and mini started, the call did not measure what was intended."""
        collector = _collector(
            requested="v1",
            initial="turn-detector-v1-mini",
            final="turn-detector-v1-mini",
        )
        degradation = collector.detector_degradation()
        assert degradation is not None
        assert "distinta de la pedida" in degradation["motivo"]

    def test_asking_for_mini_and_running_mini_does_not_exclude(self):
        """Running the local detector on purpose is a valid run."""
        collector = _collector(
            requested="v1-mini",
            initial="turn-detector-v1-mini",
            final="turn-detector-v1-mini",
        )
        assert collector.detector_degradation() is None


def _call_json(*, call_sid: str, exclude: dict[str, Any] | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "call_sid": call_sid,
        "poc": 2,
        "scenario": 1,
        "repetition": 1,
        "transcript": [
            {"t": 0.0, "role": "agent", "text": "buenas tardes"},
            {"t": 3.0, "role": "carrier", "text": "sí dígame"},
        ],
        "latencies": [
            {"indice": 0, "full_turn_ms": 2000.0, "is_opening": True},
            {"indice": 1, "full_turn_ms": 1500.0, "is_opening": False},
            {"indice": 2, "full_turn_ms": 1600.0, "is_opening": False},
        ],
    }
    if exclude is not None:
        data["exclude_from_pooled"] = exclude
    return data


class TestPooled:
    def test_an_excluded_call_does_not_vote_in_the_percentiles(self):
        healthy = mod_calls.compute(_call_json(call_sid="healthy", exclude=None))
        degraded = mod_calls.compute(
            _call_json(
                call_sid="degraded",
                exclude={"motivo": "el detector cayó a local", "detail": "v1 -> v1-mini"},
            )
        )

        consolidated = mod_report.consolidate([healthy, degraded], poc="PoC2")
        regimen = consolidated.latency["buckets"]["regimen"]["full_turn"]

        # Only the 2 steady-state turns of the healthy call.
        assert regimen["n"] == 2
        assert consolidated.latency["excluded_from_pool"]
        assert len(consolidated.latency["excluded_from_pool"]) == 1

    def test_the_excluded_call_stays_in_the_detail(self):
        """It is not discarded: it loses its vote in the aggregates, not its existence."""
        degraded = mod_calls.compute(
            _call_json(
                call_sid="degraded", exclude={"motivo": "el detector cayó a local"}
            )
        )
        consolidated = mod_report.consolidate([degraded], poc="PoC2")

        assert len(consolidated.calls) == 1
        markdown = mod_report.generate_markdown(consolidated)
        assert "fuera de estos percentiles" in markdown
        assert "el detector cayó a local" in markdown

    def test_without_the_key_everything_works_as_before(self):
        """PoC1 and PoC3 never write the key: they must not be affected."""
        healthy = mod_calls.compute(_call_json(call_sid="healthy", exclude=None))
        assert healthy.exclude_from_pooled is None

        consolidated = mod_report.consolidate([healthy], poc="PoC1")
        assert "excluded_from_pool" not in consolidated.latency
        assert consolidated.latency["buckets"]["regimen"]["full_turn"]["n"] == 2

    def test_the_exclusion_appears_in_the_call_warnings(self):
        degraded = mod_calls.compute(
            _call_json(
                call_sid="degraded",
                exclude={"motivo": "el detector cayó a local", "detail": "v1 -> v1-mini"},
            )
        )
        assert any("fuera de la poza" in aviso for aviso in degraded.warning_list)
