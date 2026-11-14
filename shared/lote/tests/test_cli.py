"""The batch end to end, with faked calls.

What is tested here is what makes the batch usable overnight: that it resumes
where it was, that it stops at the cap, that one bad call does not take the
others down with it, and that the run order is the one that alternates the
PoCs. The calls are faked because they are the only thing that costs money;
everything else is real, including `progress.jsonl` on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from shared.lote import cli
from shared.lote.executor import PoCDescriptor, CallResult
from shared.lote.progress import FAILURE, OK, ROJA, read_progress


@pytest.fixture
def campana(tmp_path, monkeypatch):
    """Set up a batch with a fixed quote and faked calls.

    Returns an object with `runs` (what actually got launched, in order) and a
    `response` hook to decide how each call turns out.
    """

    class Banco:
        def __init__(self) -> None:
            self.runs: list[str] = []
            self.response = lambda item: ("ok", {})
            self.price = {1: 0.10, 2: 0.15}

        def launch(self, descriptor, item, **options):
            self.runs.append(item.clave)
            how, extra = self.response(item)
            if how == "muere":
                return CallResult(
                    code_value=7,
                    summary=None,
                    motivo="the harness exited with code 7",
                    duration_s=3.0,
                    log_file_path=options["log_file_path"],
                )
            summary = {
                "label": f"corrida-{item.clave}",
                "cost_usd": self.price[item.poc],
                "end": {
                    "motivo": "guion_completo",
                    "turns_reached": 8,
                    "script_turns": 8,
                    "retry": False,
                },
                "lines": {"heard_lines": 8, "fragmentation": 1.2},
                "line_latency_ms": {"n": 8, "p50": 2100.0},
                "stt": {"wer": 0.02},
            }
            summary.update(extra)
            Path(options["summary_path"]).write_text(
                json.dumps(summary), encoding="utf-8"
            )
            return CallResult(
                code_value=0 if how == "ok" else 1,
                summary=summary,
                motivo="",
                duration_s=100.0,
                log_file_path=options["log_file_path"],
            )

        def get_quote(self, descriptor, item, **_):
            return {
                "estimate_only": True,
                "poc": descriptor.poc,
                "scenario": item.scenario,
                "seconds_cap": 200.0,
                "estimated_usd": self.price[descriptor.poc],
            }

    banco = Banco()
    descriptors = {
        poc: PoCDescriptor(
            poc=poc,
            name=f"PoC{poc}",
            interpreter=Path(sys.executable),
            modulo=f"fake{poc}",
            needs_worker=poc == 2,
        )
        for poc in (1, 2)
    }
    monkeypatch.setattr(cli, "default_descriptors", lambda root: descriptors)
    monkeypatch.setattr(cli, "run_call", banco.launch)
    monkeypatch.setattr(cli, "get_quote", banco.get_quote)
    banco.base = tmp_path
    return banco


def _launch(campana, *extra, tope="10"):
    return cli.main(
        [
            "--label",
            "prueba",
            "--base",
            str(campana.base),
            "--pocs",
            "1,2",
            "--scenarios",
            "1,2",
            "--repetitions",
            "1",
            "--cap-usd",
            tope,
            "--no-worker",
            *extra,
        ]
    )


def _progress(campana):
    rows, _ = read_progress(campana.base / "prueba" / "progress.jsonl")
    return rows


# --- quote without spending ---


def test_without_accept_cost_no_call_is_run(campana, capsys):
    assert _launch(campana) == 0
    assert campana.runs == []
    output = capsys.readouterr().out
    assert "estimated" in output
    assert "--accept-cost" in output


def test_the_cost_table_sums_the_pending(campana, capsys):
    _launch(campana)
    # 2 PoC1 calls at 0.10 + 2 PoC2 calls at 0.15
    assert "0.50" in capsys.readouterr().out


# --- the order ---


def test_the_pocs_alternate_when_running(campana):
    _launch(campana, "--accept-cost")
    assert campana.runs == [
        "0001-poc1-e1-r1",
        "0002-poc2-e1-r1",
        "0003-poc1-e2-r1",
        "0004-poc2-e2-r1",
    ]


# --- resuming ---


def test_on_relaunch_what_is_already_recorded_is_not_repeated(campana):
    _launch(campana, "--accept-cost")
    first_ones = list(campana.runs)
    campana.runs.clear()
    assert _launch(campana, "--accept-cost") == 0
    assert first_ones  # the first pass did run
    assert campana.runs == []  # the second repeated nothing


def test_continues_from_where_it_left_off(campana):
    """The call in flight when the process was killed left no line: it is rerun."""
    def run_up_to_two(descriptor, item, **options):
        if item.indice > 2:
            raise KeyboardInterrupt
        return campana.launch(descriptor, item, **options)

    import shared.lote.cli as modulo

    original = modulo.run_call
    modulo.run_call = run_up_to_two
    try:
        assert _launch(campana, "--accept-cost") == 130
    finally:
        modulo.run_call = original

    assert [f["clave"] for f in _progress(campana)] == [
        "0001-poc1-e1-r1",
        "0002-poc2-e1-r1",
    ]

    campana.runs.clear()
    assert _launch(campana, "--accept-cost") == 0
    assert campana.runs == ["0003-poc1-e2-r1", "0004-poc2-e2-r1"]


def test_the_spend_of_what_is_already_done_carries_over_on_resume(campana):
    _launch(campana, "--accept-cost")
    spent = _progress(campana)[-1]["accumulated_cost_usd"]
    assert spent == pytest.approx(0.50)


def test_resuming_with_another_plan_under_the_same_label_is_refused(campana, capsys):
    _launch(campana, "--accept-cost")
    code_value = cli.main(
        [
            "--label",
            "prueba",
            "--base",
            str(campana.base),
            "--pocs",
            "1,2",
            "--scenarios",
            "1,2,3",  # one more: the indices would shift
            "--repetitions",
            "1",
            "--cap-usd",
            "10",
            "--no-worker",
            "--accept-cost",
        ]
    )
    assert code_value == 1
    assert "different plan" in capsys.readouterr().out


# --- spend cap ---


def test_the_batch_stops_on_its_own_on_reaching_the_cap(campana, capsys):
    # 0.10 + 0.15 = 0.25; the third (0.10) would exceed 0.30.
    _launch(campana, "--accept-cost", tope="0.30")
    assert campana.runs == ["0001-poc1-e1-r1", "0002-poc2-e1-r1"]
    assert "[cap]" in capsys.readouterr().out


def test_cap_is_checked_before_launching_not_after(campana):
    """Checking it afterwards would turn it into a warning, not a cap."""
    _launch(campana, "--accept-cost", tope="0.05")
    assert campana.runs == []


def test_what_did_not_fit_can_be_resumed_with_a_higher_cap(campana):
    _launch(campana, "--accept-cost", tope="0.30")
    campana.runs.clear()
    _launch(campana, "--accept-cost", tope="10")
    assert campana.runs == ["0003-poc1-e2-r1", "0004-poc2-e2-r1"]


# --- one bad call does not bring down the batch ---


def test_a_call_that_fails_is_recorded_and_the_batch_continues(campana):
    campana.response = lambda item: (
        ("muere", {}) if item.indice == 2 else ("ok", {})
    )
    _launch(campana, "--accept-cost")
    assert len(campana.runs) == 4
    rows = _progress(campana)
    assert [f["estado"] for f in rows] == [OK, FAILURE, OK, OK]
    assert "code 7" in rows[1]["motivo"]


def test_the_failed_call_is_charged_the_estimate(campana):
    """It left no summary to read the cost from, and may still have spent."""
    campana.response = lambda item: (
        ("muere", {}) if item.indice == 1 else ("ok", {})
    )
    _launch(campana, "--accept-cost")
    assert _progress(campana)[0]["cost_usd"] == pytest.approx(0.10)


def test_a_failed_run_is_distinguished_from_a_failure(campana):
    campana.response = lambda item: (
        ("roja", {}) if item.indice == 3 else ("ok", {})
    )
    _launch(campana, "--accept-cost")
    assert [f["estado"] for f in _progress(campana)] == [OK, OK, ROJA, OK]


# --- the file ---


def test_each_call_writes_its_line_when_finished(campana):
    """If the batch dies at call 40, the previous 39 are on disk."""
    seen_ones: list[int] = []

    def spy(descriptor, item, **options):
        # How many lines are already written when call N starts.
        seen_ones.append(len(_progress(campana)))
        return campana.launch(descriptor, item, **options)

    import shared.lote.cli as modulo

    original = modulo.run_call
    modulo.run_call = spy
    try:
        _launch(campana, "--accept-cost")
    finally:
        modulo.run_call = original
    assert seen_ones == [0, 1, 2, 3]


def test_the_plan_is_written_with_its_fingerprint(campana):
    _launch(campana)
    plan = json.loads((campana.base / "prueba" / "plan.json").read_text("utf-8"))
    assert len(plan["calls"]) == 4
    assert plan["fingerprint"]
    assert plan["parsed_args"]["scenarios"] == [1, 2]


def test_the_summary_of_each_call_stays_in_its_file(campana):
    _launch(campana, "--accept-cost")
    summaries = sorted((campana.base / "prueba" / "summaries").glob("*.json"))
    assert [r.stem for r in summaries] == [
        "0001-poc1-e1-r1",
        "0002-poc2-e1-r1",
        "0003-poc1-e2-r1",
        "0004-poc2-e2-r1",
    ]


# --- edge cases ---


def test_without_pending_it_neither_quotes_nor_runs(campana, capsys):
    _launch(campana, "--accept-cost")
    campana.runs.clear()
    assert _launch(campana, "--accept-cost") == 0
    assert "Nothing left to run" in capsys.readouterr().out


def test_a_poc_without_an_interpreter_is_rejected_before_spending(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "default_descriptors",
        lambda root: {
            1: PoCDescriptor(1, "PoC1", tmp_path / "no_existe.exe", "m"),
        },
    )
    code_value = cli.main(
        [
            "--label", "x", "--base", str(tmp_path), "--pocs", "1",
            "--scenarios", "1", "--repetitions", "1", "--cap-usd", "1",
            "--accept-cost",
        ]
    )
    assert code_value == 1
    assert "has no interpreter" in capsys.readouterr().out


def test_a_call_that_does_not_quote_is_not_run(campana, capsys, monkeypatch):
    original = campana.get_quote
    monkeypatch.setattr(
        cli,
        "get_quote",
        lambda descriptor, item, **_: (
            None if descriptor.poc == 2 else original(descriptor, item)
        ),
    )
    _launch(campana, "--accept-cost")
    assert campana.runs == ["0001-poc1-e1-r1", "0003-poc1-e2-r1"]
    assert "without a quote" in capsys.readouterr().out
