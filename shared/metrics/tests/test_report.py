"""Tests for the per-PoC consolidation, and the end-to-end validation with synthetic data.

`TestAgregacionCorrecta` is the part most at risk of giving a believable but
wrong number: averaging percentiles or averaging per-call WER produces figures
that look reasonable and mean nothing. Here it is pinned down that samples are
pooled and counts are summed.

`TestEndToEnd` runs the full flow (15 synthetic calls, references, JSON
enrichment, Markdown and CSV) without spending a single real call.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from shared.metrics import latency as lat
from shared.metrics import calls as calls_mod
from shared.metrics import report
from shared.metrics.calls import CallMetrics, compute, enrich, list_calls
from shared.metrics.report import (
    CSV_COLUMNS,
    consolidate,
    write_csv,
    write_markdown,
    csv_row,
    generate_markdown,
)
from shared.metrics.synthetic import synthetic_corpus, write_corpus


def _metrics(
    *,
    scenario: int,
    repetition: int,
    latencies: list[float],
    wer_ref: str = "hola mundo",
    wer_hip: str = "hola mundo",
    resultado: str = "acuerdo",
    monto_final: float | None = 17_500.0,
    naturalness: int | None = 4,
) -> CallMetrics:
    """An already computed call, built by hand to control every number."""
    from shared.metrics import reference as ref

    data = {
        "call_sid": f"CA{scenario}{repetition}",
        "scenario": scenario,
        "repetition": repetition,
        "poc": 1,
        "summary": {
            "target_price": 18_000.0,
            "offer_history": [16_000.0, 17_500.0],
            "resultado": {
                "resultado": resultado,
                "monto_final": monto_final,
                "offer_history": [16_000.0, 17_500.0],
                "counteroffers": 1,
                "offered_compensations": [],
                "reason_category": None if resultado == "acuerdo" else "colgado",
                "validated_final_amount": True if monto_final else None,
            },
        },
        "transcript": [
            {"t": 1.0, "role": ref.AGENT_ROLE, "text": "Le ofrezco dieciséis mil."},
            {"t": 5.0, "role": ref.CARRIER_ROLE, "text": wer_hip},
        ],
        "latencies": [
            {
                "indice": i,
                "full_turn_ms": valor,
                "is_opening": i == 0,
            }
            for i, valor in enumerate(latencies)
        ],
    }
    notes = ""
    if naturalness is not None:
        notes = (
            f"{ref.NOTES_HEADER}\nnaturalness: {naturalness}\nturn_taking: 3\n"
        )
    file = ref.parse(f"## T1\nyo: {wer_ref}\n{notes}")
    return compute(data, reference_file=file)


def _with_stages(*, llm_ttft: list[float | None]) -> CallMetrics:
    """A call with a per-stage breakdown. `tts_ttfb` is always present, so the
    table exists even when there is no `llm_ttft` at all."""
    data = {
        "call_sid": "CAetapas",
        "scenario": 1,
        "repetition": 1,
        "poc": 1,
        "summary": {"resultado": {}},
        "transcript": [],
        "latencies": [
            {
                "indice": i,
                "full_turn_ms": 2_000.0,
                "llm_ttft_ms": valor,
                "tts_ttfb_ms": 300.0,
                "is_opening": i == 0,
            }
            for i, valor in enumerate(llm_ttft)
        ],
    }
    return compute(data)


class TestCorrectAggregation:
    def test_pools_samples_does_not_average_percentiles(self) -> None:
        """The p50 comes from the full pool, not from the mean of the p50s."""
        # Call A: regime 100, 200. Call B: regime 300, 400.
        a = _metrics(scenario=1, repetition=1, latencies=[9999.0, 100.0, 200.0])
        b = _metrics(scenario=1, repetition=2, latencies=[9999.0, 300.0, 400.0])
        consolidated = consolidate([a, b], poc="prueba")

        regimen = consolidated.latency["buckets"][lat.BUCKET_REGIMEN]["full_turn"]
        assert regimen["n"] == 4
        # p50 of [100,200,300,400] = 250. The mean of the p50s would give the same
        # here by symmetry, so the p25 is checked too, which would not.
        assert regimen["p50"] == pytest.approx(250.0)
        assert regimen["p25"] == pytest.approx(175.0)

    def test_opening_of_each_call_goes_to_its_bucket(self) -> None:
        a = _metrics(scenario=1, repetition=1, latencies=[9999.0, 100.0])
        b = _metrics(scenario=1, repetition=2, latencies=[9999.0, 200.0])
        consolidated = consolidate([a, b], poc="prueba")
        opening = consolidated.latency["buckets"][lat.BUCKET_OPENING]["full_turn"]
        assert opening["n"] == 2
        assert opening["p50"] == pytest.approx(9999.0)

    def test_wer_sums_counts(self) -> None:
        """A short call with an error cannot weigh the same as a long clean one."""
        corta = _metrics(
            scenario=1,
            repetition=1,
            latencies=[500.0],
            wer_ref="hola mundo",
            wer_hip="adios planeta",
        )
        largo = " ".join(["word"] * 98)
        long = _metrics(
            scenario=1, repetition=2, latencies=[500.0], wer_ref=largo, wer_hip=largo
        )
        consolidated = consolidate([corta, long], poc="prueba")
        assert consolidated.stt["wer"]["longitud_reference"] == 100
        assert consolidated.stt["wer"]["errors"] == 2
        assert consolidated.stt["wer"]["sample_rate"] == pytest.approx(0.02)

    def test_reliability_counts_turns_and_calls_without_latency(self) -> None:
        calls = [
            _metrics(scenario=1, repetition=1, latencies=[500.0, 600.0]),
            _metrics(scenario=1, repetition=2, latencies=[]),
        ]
        consolidated = consolidate(calls, poc="prueba")
        assert consolidated.reliability["measured_turns"] == 2
        assert consolidated.reliability["calls_without_latency"] == 1
        assert any("sin muestras de latencia" in a for a in consolidated.warning_list)

    def test_subjective_report_mean_and_minimum(self) -> None:
        """The mean alone hides a call rated 1."""
        calls = [
            _metrics(scenario=1, repetition=1, latencies=[500.0], naturalness=5),
            _metrics(scenario=1, repetition=2, latencies=[500.0], naturalness=5),
            _metrics(scenario=1, repetition=3, latencies=[500.0], naturalness=1),
        ]
        consolidated = consolidate(calls, poc="prueba")
        nat = consolidated.subjective["naturalness"]
        assert nat["n"] == 3
        assert nat["mean"] == pytest.approx(3.67, abs=0.01)
        assert nat["min"] == 1

    def test_sorts_by_scenario_and_repetition(self) -> None:
        calls = [
            _metrics(scenario=3, repetition=1, latencies=[500.0]),
            _metrics(scenario=1, repetition=2, latencies=[500.0]),
            _metrics(scenario=1, repetition=1, latencies=[500.0]),
        ]
        consolidated = consolidate(calls, poc="prueba")
        assert [m.label for m in consolidated.calls] == ["1.1", "1.2", "3.1"]


class TestWarnings:
    def test_warns_about_missing_scenarios(self) -> None:
        consolidated = consolidate(
            [_metrics(scenario=1, repetition=1, latencies=[500.0])], poc="prueba"
        )
        assert any("escenarios sin llamadas" in a for a in consolidated.warning_list)

    def test_warns_about_incomplete_calls(self) -> None:
        consolidated = consolidate(
            [_metrics(scenario=1, repetition=1, latencies=[500.0])], poc="prueba"
        )
        assert any("de las 15" in a for a in consolidated.warning_list)

    def test_warns_about_missing_notes(self) -> None:
        consolidated = consolidate(
            [
                _metrics(
                    scenario=1, repetition=1, latencies=[500.0], naturalness=None
                )
            ],
            poc="prueba",
        )
        assert any("puntuaciones" in a for a in consolidated.warning_list)

    def test_without_calls(self) -> None:
        consolidated = consolidate([], poc="prueba")
        assert consolidated.warning_list == ["no se encontró ninguna llamada"]


class TestCSV:
    def test_fixed_columns(self) -> None:
        """The columns are joined across the 3 PoCs: they have to match."""
        row = csv_row(_metrics(scenario=1, repetition=1, latencies=[500.0, 600.0]))
        assert set(row) == set(CSV_COLUMNS)

    def test_one_row_per_call(self, tmp_path: Path) -> None:
        calls = [
            _metrics(scenario=1, repetition=r, latencies=[500.0, 600.0])
            for r in (1, 2, 3)
        ]
        destination = write_csv(consolidate(calls, poc="p"), tmp_path / "m.csv")
        with destination.open(encoding="utf-8", newline="") as entry:
            rows = list(csv.DictReader(entry))
        assert len(rows) == 3
        assert rows[0]["scenario"] == "1"
        assert rows[0]["wer"] == "0.0"

    def test_only_brings_technology_metrics(self, tmp_path: Path) -> None:
        """The CSV is what gets joined across the 3 PoCs: nothing from the spoken content."""
        row = csv_row(_metrics(scenario=1, repetition=1, latencies=[500.0]))
        assert set(row) == set(CSV_COLUMNS)


class TestMarkdown:
    def test_has_all_the_sections(self) -> None:
        calls = [
            _metrics(scenario=i, repetition=1, latencies=[2000.0, 900.0, 1100.0])
            for i in range(1, 6)
        ]
        text = generate_markdown(consolidate(calls, poc="PoC1 — Pipecat"))
        for section in (
            "## Latencia",
            "## Calidad de STT",
            "## Naturalidad y turn-taking",
            "## Confiabilidad",
            "## Detalle por llamada",
        ):
            assert section in text

    def test_marks_the_unreliable_percentiles(self) -> None:
        calls = [_metrics(scenario=1, repetition=1, latencies=[2000.0, 900.0])]
        text = generate_markdown(consolidate(calls, poc="p"))
        assert "⚠" in text

    def test_warns_that_the_llm_ttft_is_not_comparable(self) -> None:
        """The row that has already been misquoted once.

        The table published 1,469 ms of TTFT in PoC1 versus 681 in PoC2 with the
        same model and the same prompt, and they were two different segments —
        not two different frameworks. The warning goes **in the generated
        file**, not only in the contract, because the mistake is made while
        reading the file.
        See `docs/METRICS_CONTRACT.md` §2.7.1.
        """
        text = generate_markdown(
            consolidate([_with_stages(llm_ttft=[None, 700.0, 1700.0])], poc="p")
        )
        assert "### Desglose por etapa" in text
        assert "no es comparable entre PoC1 y PoC2" in text
        assert "§2.7.1" in text

    def test_without_llm_ttft_it_does_not_put_the_warning(self) -> None:
        """Control for the previous test, and not a vacuous one: the stages table DOES appear.

        Without this, an unconditionally printed warning would pass the test
        above and nobody would know the condition is not evaluated. In PoC3
        there are no stages to isolate and the warning there would be noise.
        """
        text = generate_markdown(
            consolidate([_with_stages(llm_ttft=[None, None, None])], poc="p")
        )
        assert "### Desglose por etapa" in text, "la tabla tiene que existir"
        assert "tts_ttfb" in text
        assert "§2.7.1" not in text

    def test_without_latencies_it_says_so(self) -> None:
        data = {
            "call_sid": "CA1",
            "summary": {"resultado": {}},
            "transcript": [],
        }
        unmeasured = compute(data)
        text = generate_markdown(consolidate([unmeasured], poc="p"))
        assert "Sin muestras de latencia" in text

    def test_is_written_to_disk(self, tmp_path: Path) -> None:
        calls = [_metrics(scenario=1, repetition=1, latencies=[500.0])]
        destination = write_markdown(consolidate(calls, poc="p"), tmp_path / "r.md")
        assert destination.is_file()
        assert destination.read_text(encoding="utf-8").startswith("# Métricas")


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, list[CallMetrics]]:
    """The 15 synthetic calls, already computed and with the JSON enriched.

    Module scope: the corpus is built only once for all the end-to-end tests.
    """
    folder = tmp_path_factory.mktemp("corpus")
    calls = synthetic_corpus(poc=1)
    write_corpus(folder, calls)
    computed = [
        enrich(ruta) for ruta in list_calls(folder)
    ]
    return folder, computed


class TestEndToEnd:
    """The full flow with 15 synthetic calls. It costs nothing."""

    def test_fifteen_calls(self, run) -> None:
        _, computed = run
        assert len(computed) == 15

    def test_the_five_scenarios_with_three_repetitions(self, run) -> None:
        _, computed = run
        labels = sorted(m.label for m in computed)
        expected_variables = sorted(f"{e}.{r}" for e in range(1, 6) for r in range(1, 4))
        assert labels == expected_variables

    def test_all_have_wer(self, run) -> None:
        """The synthetic references come "corrected": there must be a WER."""
        _, computed = run
        assert all(m.stt is not None for m in computed)

    def test_wer_is_neither_zero_nor_absurd(self, run) -> None:
        """If it comes out 0 the STT was copied; if it comes out >50% the aligner is broken."""
        _, computed = run
        consolidated = consolidate(computed, poc="demo")
        sample_rate = consolidated.stt["wer"]["sample_rate"]
        assert 0.0 < sample_rate < 0.5

    def test_all_have_manual_notes(self, run) -> None:
        _, computed = run
        assert all((m.manual_notes or {}).get("completed") for m in computed)

    def test_the_block_stays_in_the_json(self, run) -> None:
        folder, _ = run
        for ruta in list_calls(folder):
            data = json.loads(ruta.read_text(encoding="utf-8"))
            assert "metrics" in data
            assert "latency" in data["metrics"]

    def test_consolidated_percentiles_have_a_sample(self, run) -> None:
        _, computed = run
        consolidated = consolidate(computed, poc="demo")
        regimen = consolidated.latency["buckets"][lat.BUCKET_REGIMEN]["full_turn"]
        # ~8 turns × 15 calls, minus openings and interrupted ones.
        assert regimen["n"] > 50
        assert regimen["reliable"]["p95"] is True

    def test_opening_slower_than_steady_state(self, run) -> None:
        """This is the reason for separating buckets: the cold start costs more."""
        _, computed = run
        consolidated = consolidate(computed, poc="demo")
        buckets = consolidated.latency["buckets"]
        assert (
            buckets[lat.BUCKET_OPENING]["full_turn"]["p50"]
            > buckets[lat.BUCKET_REGIMEN]["full_turn"]["p50"]
        )

    def test_does_not_warn_about_openings_when_consolidating(self, run) -> None:
        _, computed = run
        consolidated = consolidate(computed, poc="demo")
        assert not any("marcados como apertura" in a for a in consolidated.warning_list)

    def test_full_report_is_generated(self, run, tmp_path: Path) -> None:
        _, computed = run
        consolidated = consolidate(computed, poc="PoC1 — Pipecat (demo)")
        md = write_markdown(consolidated, tmp_path / "c.md")
        csv_ruta = write_csv(consolidated, tmp_path / "c.csv")
        assert md.is_file() and csv_ruta.is_file()
        with csv_ruta.open(encoding="utf-8", newline="") as entry:
            assert len(list(csv.DictReader(entry))) == 15

    def test_is_deterministic(self) -> None:
        """Same seed, same numbers: if they change, the computation changed."""
        first = synthetic_corpus(poc=1)
        second_one = synthetic_corpus(poc=1)
        assert [c.data for c in first] == [c.data for c in second_one]


class TestDelegateLlmRoundTrips:
    """The auxiliary LLM columns: CSV only, and never in the `llm_ttft` column.

    In PoC3 the delegate is a secondary brain that the voice model consults
    while it keeps talking, so its cost does NOT fall inside `full_turn_ms`.
    Putting it into `llm_ttft_p50_regimen_ms` would compare two different
    quantities.
    """

    def _call(self, round_trips):
        return calls_mod.compute(
            {
                "call_sid": "CA_rt",
                "poc": 3,
                "scenario": 1,
                "repetition": 1,
                "latencies": [
                    {"indice": 0, "full_turn_ms": 900.0, "is_opening": True},
                    {"indice": 1, "full_turn_ms": 1500.0},
                    {"indice": 2, "full_turn_ms": 2000.0},
                ],
                "transcript": [
                    {"t": 0.0, "role": "agent", "text": "Hola"},
                    {"t": 1.0, "role": "carrier", "text": "Sí"},
                ],
                "llm_round_trips": round_trips,
            }
        )

    def test_aggregates_p50_max_and_those_masked_by_speech(self):
        call = self._call(
            [
                {"ms": 900.0, "with_function": False},
                {"ms": 1000.0, "with_function": True},
                {"ms": 3200.0, "with_function": True},
            ]
        )
        row = report.csv_row(call)

        assert row["llm_delegate_n"] == 3
        assert row["llm_delegate_p50_ms"] == 1000.0
        assert row["llm_delegate_max_ms"] == 3200.0
        assert row["llm_delegate_with_function_n"] == 2
        # The worst regime turn measured 2,000 ms: only the 3,200 ms one exceeds it,
        # so only that one was covered by the agent's speech.
        assert row["llm_delegate_covered_by_speech_n"] == 1
        # And it does not contaminate the LLM TTFT column, which PoC3 does not measure.
        assert row["llm_ttft_p50_regime_ms"] is None

    def test_a_poc_that_does_not_publish_them_leaves_the_columns_empty(self):
        row = report.csv_row(self._call([]))
        for column in (
            "llm_delegate_n",
            "llm_delegate_p50_ms",
            "llm_delegate_max_ms",
            "llm_delegate_with_function_n",
            "llm_delegate_covered_by_speech_n",
        ):
            assert row[column] is None

    def test_a_malformed_entry_is_ignored_without_taking_down_the_call(self):
        call = self._call(
            [{"ms": 900.0}, {"ms": "mucho"}, "basura", {"with_function": True}]
        )
        assert call.round_trips_llm == [{"ms": 900.0, "with_function": False}]

    def test_the_columns_remain_in_the_csv_header(self):
        for column in ("llm_delegate_n", "llm_delegate_covered_by_speech_n"):
            assert column in report.CSV_COLUMNS
