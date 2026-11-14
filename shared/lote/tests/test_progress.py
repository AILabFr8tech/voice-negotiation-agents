"""`progress.jsonl`: it can be followed live and it supports resuming."""

from __future__ import annotations

import json

from shared.lote.plan import BatchItem
from shared.lote.progress import (
    FAILURE,
    OK,
    ROJA,
    append_row,
    finished_keys,
    cumulative_spend,
    read_progress,
    progress_row,
    summarize_batch,
)


def _summary(**changes):
    base = {
        "label": "arnes-call-1",
        "end": {
            "motivo": "cerrada_por_agente",
            "turns_reached": 7,
            "script_turns": 8,
            "retry": False,
            "detail": "el agente cerró",
        },
        "lines": {"heard_lines": 7, "fragmentation": 1.57},
        "line_latency_ms": {"n": 7, "p50": 2280.0},
        "stt": {"wer": 0.031},
    }
    base.update(changes)
    return base


# --- round trip ---


def test_what_is_appended_reads_the_same(tmp_path):
    ruta = tmp_path / "progress.jsonl"
    append_row(ruta, {"clave": "0001-poc1-e1-r1", "cost_usd": 0.08})
    append_row(ruta, {"clave": "0002-poc2-e1-r1", "cost_usd": 0.12})
    rows, discarded_ones = read_progress(ruta)
    assert [f["clave"] for f in rows] == ["0001-poc1-e1-r1", "0002-poc2-e1-r1"]
    assert discarded_ones == 0


def test_each_call_is_a_single_line(tmp_path):
    """For `Get-Content -Wait`: one line per call, ending in a newline."""
    ruta = tmp_path / "progress.jsonl"
    append_row(ruta, {"clave": "a", "detail": "con\nsalto adentro"})
    raw_value = ruta.read_text(encoding="utf-8")
    assert raw_value.count("\n") == 1
    assert raw_value.endswith("\n")


def test_without_a_file_nothing_is_done(tmp_path):
    assert read_progress(tmp_path / "does_not_exist.jsonl") == ([], 0)


# --- the half-written file ---


def test_a_line_split_at_the_end_does_not_drop_the_previous_ones(tmp_path):
    """This is what killing the process mid-write leaves behind."""
    ruta = tmp_path / "progress.jsonl"
    append_row(ruta, {"clave": "0001-poc1-e1-r1", "cost_usd": 0.08})
    with open(ruta, "a", encoding="utf-8") as file:
        file.write('{"clave": "0002-poc2-e1')  # unclosed and without newline
    rows, discarded_ones = read_progress(ruta)
    assert [f["clave"] for f in rows] == ["0001-poc1-e1-r1"]
    assert discarded_ones == 1


def test_a_line_without_key_is_discarded(tmp_path):
    ruta = tmp_path / "progress.jsonl"
    ruta.write_text('{"costo_usd": 1.0}\n', encoding="utf-8")
    rows, discarded_ones = read_progress(ruta)
    assert rows == []
    assert discarded_ones == 1


def test_blank_lines_do_not_count_as_discard(tmp_path):
    ruta = tmp_path / "progress.jsonl"
    ruta.write_text('\n{"clave": "a"}\n\n', encoding="utf-8")
    rows, discarded_ones = read_progress(ruta)
    assert len(rows) == 1
    assert discarded_ones == 0


# --- resuming ---


def test_the_finished_include_those_that_failed(tmp_path):
    """Retrying a failure just because the batch resumed would overspend."""
    rows = [
        {"clave": "a", "estado": OK},
        {"clave": "b", "estado": FAILURE},
    ]
    assert finished_keys(rows) == {"a", "b"}


def test_the_spend_adds_what_was_recorded():
    rows = [{"clave": "a", "cost_usd": 0.08}, {"clave": "b", "cost_usd": 0.12}]
    assert cumulative_spend(rows) == 0.2


def test_a_row_without_cost_counts_as_zero():
    assert cumulative_spend([{"clave": "a"}]) == 0.0


# --- the row ---


def test_the_row_carries_what_is_tracked_live():
    item = BatchItem(indice=3, poc=2, scenario=1, repetition=1)
    row = progress_row(
        item,
        estado=OK,
        summary=_summary(),
        motivo="",
        duration_s=118.42,
        cost_usd=0.12,
        spent_before=0.30,
        when="2026-09-19T10:00:00",
    )
    assert row["indice"] == 3
    assert row["poc"] == 2
    assert row["scenario"] == 1
    assert row["lines"] == 7
    assert row["fragmentation"] == 1.57
    assert row["latency_p50_ms"] == 2280.0
    # `indicative_wer`, not `wer`: the name is the warning that this number is
    # for monitoring the run live and is not cited. See §2.10.
    assert row["indicative_wer"] == 0.031
    assert "wer" not in row
    assert row["end"] == "cerrada_por_agente"
    assert row["turns_reached"] == 7
    assert row["duration_s"] == 118.4


def test_the_accumulated_cost_adds_what_came_before():
    item = BatchItem(indice=1, poc=1, scenario=1, repetition=1)
    row = progress_row(
        item,
        estado=OK,
        summary=_summary(),
        motivo="",
        duration_s=1.0,
        cost_usd=0.12,
        spent_before=0.30,
        when="x",
    )
    assert row["cost_usd"] == 0.12
    assert row["accumulated_cost_usd"] == 0.42


def test_without_a_summary_the_row_comes_out_with_gaps_and_does_not_blow_up():
    item = BatchItem(indice=1, poc=1, scenario=1, repetition=1)
    row = progress_row(
        item,
        estado=FAILURE,
        summary=None,
        motivo="exceeded the cap",
        duration_s=600.0,
        cost_usd=0.12,
        spent_before=0.0,
        when="x",
    )
    assert row["estado"] == FAILURE
    assert row["motivo"] == "exceeded the cap"
    assert row["latency_p50_ms"] is None
    assert row["fragmentation"] is None
    # It serializes: the row goes into a `.jsonl` and an odd object there would break it.
    json.dumps(row)


# --- batch summary ---


def test_the_summary_counts_by_poc():
    rows = [
        {"clave": "a", "poc": 1, "estado": OK, "cost_usd": 0.08},
        {"clave": "b", "poc": 1, "estado": ROJA, "cost_usd": 0.08},
        {"clave": "c", "poc": 2, "estado": FAILURE, "cost_usd": 0.12},
    ]
    summary = summarize_batch(rows)
    assert summary["calls"] == 3
    assert summary["cost_usd"] == 0.28
    assert summary["per_poc"]["1"]["calls"] == 2
    assert summary["per_poc"]["1"]["ok"] == 1
    assert summary["per_poc"]["1"]["failed_calls"] == 1
    assert summary["per_poc"]["1"]["failures"] == 0
    assert summary["per_poc"]["1"]["cost_usd"] == 0.16
    assert summary["per_poc"]["2"]["failures"] == 1


# --- tool round trips to the LLM ---


def test_the_row_carries_the_round_trips_of_the_call():
    row = progress_row(
        BatchItem(indice=1, poc=3, scenario=3, repetition=1),
        estado=OK,
        summary={
            "llm_round_trips": {
                "n": 3, "with_function_n": 2, "p50_ms": 980.0, "max_ms": 1_400.0
            }
        },
        motivo="",
        duration_s=150.0,
        cost_usd=0.13,
        spent_before=0.0,
        when="x",
    )
    assert row["round_trips_llm_n"] == 3
    assert row["round_trips_llm_p50_ms"] == 980.0
    json.dumps(row)


def test_without_the_block_the_column_is_none_and_not_zero():
    """PoC1 and PoC2 do not fill the key. A zero would say there were no tool calls."""
    row = progress_row(
        BatchItem(indice=1, poc=1, scenario=3, repetition=1),
        estado=OK,
        summary={"end": {"motivo": "guion_completo"}},
        motivo="",
        duration_s=150.0,
        cost_usd=0.07,
        spent_before=0.0,
        when="x",
    )
    assert row["round_trips_llm_n"] is None


def test_the_summary_breaks_down_the_round_trips_by_scenario():
    """They depend on the script, not the framework: the mean of the 5 hides it."""
    rows = [
        {"clave": "a", "poc": 3, "scenario": 1, "estado": OK,
         "cost_usd": 0.13, "round_trips_llm_n": 2},
        {"clave": "b", "poc": 3, "scenario": 1, "estado": OK,
         "cost_usd": 0.13, "round_trips_llm_n": 4},
        {"clave": "c", "poc": 3, "scenario": 3, "estado": ROJA,
         "cost_usd": 0.13, "round_trips_llm_n": 1},
    ]
    summary = summarize_batch(rows)

    assert summary["per_scenario"]["poc3-e1"]["round_trips_llm"]["total"] == 6
    assert summary["per_scenario"]["poc3-e1"]["round_trips_llm"]["per_call"] == 3.0
    assert summary["per_scenario"]["poc3-e3"]["round_trips_llm"]["per_call"] == 1.0
    # The red one counts all the same: it spent money and its tool calling happened.
    assert summary["per_scenario"]["poc3-e3"]["failed_calls"] == 1
    assert summary["per_poc"]["3"]["round_trips_llm"]["total"] == 7


def test_a_poc_that_does_not_fill_the_key_does_not_average_against_zero():
    """Mixing "no function was called" with "not measured" would give a false mean."""
    rows = [
        {"clave": "a", "poc": 1, "scenario": 1, "estado": OK,
         "cost_usd": 0.07, "round_trips_llm_n": None},
        {"clave": "b", "poc": 1, "scenario": 1, "estado": OK, "cost_usd": 0.07},
    ]
    rt = summarize_batch(rows)["per_poc"]["1"]["round_trips_llm"]

    assert rt["calls_with_data"] == 0
    assert rt["per_call"] is None
