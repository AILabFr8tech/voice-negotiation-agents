"""Launch a call as a subprocess and collect what it left.

The tests that run a real process use a fake harness: a ten-line script that
writes a summary and exits with whatever code it is told to. This is not
ceremony: what is being tested is precisely the contract between the batch and
a foreign process, and an in-memory double does not fail in the three ways that
matter (exiting badly, hanging, leaving no file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from shared.lote.executor import (
    PoCDescriptor,
    call_command,
    run_call,
    default_descriptors,
    read_summary,
    time_cap,
)
from shared.lote.plan import BatchItem

FAKE_HARNESS = '''
import argparse, json, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--scenario", type=int)
p.add_argument("--repetition", type=int)
p.add_argument("--summary")
p.add_argument("--record-audit", action="store_true")
p.add_argument("--accept-cost", action="store_true")
a = p.parse_args()

config = json.loads(Path("config.json").read_text(encoding="utf-8"))
print("argv:", json.dumps(sys.argv[1:]))
if config.get("sleep"):
    time.sleep(config["sleep"])
    # Only reaches here if NOBODY killed it. It is the trace of an orphan process.
    Path("sobrevivi.txt").write_text("sigo vivo", encoding="utf-8")
if config.get("writes_summary", True):
    Path(a.summary).write_text(
        json.dumps({
            "scenario": a.scenario,
            "repetition": a.repetition,
            "accept_cost": a.accept_cost,
            "auditoria": a.record_audit,
            **config.get("summary", {}),
        }),
        encoding="utf-8",
    )
sys.exit(config.get("code_value", 0))
'''


@pytest.fixture
def fake_harness(tmp_path: Path):
    """Put the script in `tmp_path` and return a descriptor that launches it."""
    (tmp_path / "fake_harness.py").write_text(FAKE_HARNESS, encoding="utf-8")

    def configure(**config) -> PoCDescriptor:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return PoCDescriptor(
            poc=9,
            name="fake",
            interpreter=Path(sys.executable),
            modulo="fake_harness",
        )

    return configure


ITEM = BatchItem(indice=1, poc=9, scenario=3, repetition=2)


def _launch(descriptor, tmp_path, **changes):
    options = {
        "root": tmp_path,
        "summary_path": tmp_path / "summary.json",
        "log_file_path": tmp_path / "call.log",
        "tope_seg": 60.0,
        "accept_cost": True,
    }
    options.update(changes)
    return run_call(descriptor, ITEM, **options)


# --- the command ---


def test_the_command_carries_scenario_repetition_and_summary():
    descriptor = PoCDescriptor(9, "x", Path("py.exe"), "mod")
    command = call_command(
        descriptor, ITEM, summary_path=Path("r.json"), accept_cost=True
    )
    assert command[:3] == ["py.exe", "-m", "mod"]
    assert "--scenario" in command and command[command.index("--scenario") + 1] == "3"
    assert "--repetition" in command and command[command.index("--repetition") + 1] == "2"
    assert "--summary" in command


def test_without_accept_cost_the_command_does_not_carry_it():
    """This is the quote: the harness prints the estimate and spends nothing."""
    descriptor = PoCDescriptor(9, "x", Path("py.exe"), "mod")
    command = call_command(
        descriptor, ITEM, summary_path=Path("r.json"), accept_cost=False
    )
    assert "--accept-cost" not in command


def test_the_audit_only_goes_if_requested():
    descriptor = PoCDescriptor(9, "x", Path("py.exe"), "mod")
    sin = call_command(
        descriptor, ITEM, summary_path=Path("r.json"), accept_cost=True
    )
    having = call_command(
        descriptor,
        ITEM,
        summary_path=Path("r.json"),
        accept_cost=True,
        record_audit=True,
    )
    assert "--record-audit" not in sin
    assert "--record-audit" in having


# --- real runs ---


def test_a_green_call_returns_its_summary(fake_harness, tmp_path):
    descriptor = fake_harness(code_value=0, summary={"cost_usd": 0.08})
    resultado = _launch(descriptor, tmp_path)
    assert resultado.code_value == 0
    assert resultado.motivo == ""
    assert resultado.summary["cost_usd"] == 0.08
    assert resultado.summary["scenario"] == 3
    assert resultado.summary["accept_cost"] is True


def test_code_one_with_summary_is_not_a_batch_failure(fake_harness, tmp_path):
    """The harness exits 1 when the run ended red. That is data, not an error."""
    descriptor = fake_harness(code_value=1, summary={"cost_usd": 0.08})
    resultado = _launch(descriptor, tmp_path)
    assert resultado.code_value == 1
    assert resultado.motivo == ""
    assert resultado.had_summary


def test_an_odd_code_is_noted_as_reason(fake_harness, tmp_path):
    descriptor = fake_harness(code_value=3, summary={})
    resultado = _launch(descriptor, tmp_path)
    assert "code 3" in resultado.motivo


def test_finishing_without_a_summary_is_noted_as_reason(fake_harness, tmp_path):
    descriptor = fake_harness(code_value=0, writes_summary=False)
    resultado = _launch(descriptor, tmp_path)
    assert resultado.summary is None
    assert "without leaving a summary" in resultado.motivo


def test_a_hung_up_call_is_killed_and_raises_no_exception(fake_harness, tmp_path):
    descriptor = fake_harness(sleep=30, code_value=0)
    resultado = _launch(descriptor, tmp_path, tope_seg=1.0)
    assert "cap" in resultado.motivo
    assert resultado.duration_s < 20


def test_the_hung_up_call_ends_up_dead_and_not_orphaned(fake_harness, tmp_path):
    """Leaving it alive would have it keep talking over the next call."""
    import time

    descriptor = fake_harness(sleep=4, code_value=0)
    _launch(descriptor, tmp_path, tope_seg=1.0)
    time.sleep(6)  # longer than the child was going to sleep
    assert not (tmp_path / "sobrevivi.txt").exists()


def test_an_old_summary_is_not_confused_with_that_of_this_call(
    fake_harness, tmp_path
):
    ruta = tmp_path / "summary.json"
    ruta.write_text('{"cost_usd": 99.0}', encoding="utf-8")
    descriptor = fake_harness(code_value=0, writes_summary=False)
    resultado = _launch(descriptor, tmp_path, summary_path=ruta)
    assert resultado.summary is None


def test_the_whole_output_goes_to_the_log_and_not_the_console(fake_harness, tmp_path, capfd):
    descriptor = fake_harness(code_value=0, summary={})
    resultado = _launch(descriptor, tmp_path)
    text = resultado.log_file_path.read_text(encoding="utf-8")
    assert "--scenario" in text  # the command was written at the top
    assert "argv:" in text  # and what the child printed
    assert "argv:" not in capfd.readouterr().out


# --- reading the summary ---


def test_an_unreadable_summary_is_read_as_absent(tmp_path):
    ruta = tmp_path / "r.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert read_summary(ruta) is None


def test_a_summary_that_is_not_an_object_is_read_as_absent(tmp_path):
    ruta = tmp_path / "r.json"
    ruta.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_summary(ruta) is None


# --- time cap ---


def test_cap_leaves_margin_to_start():
    """PoC1 loads two ONNX models before accepting the first connection."""
    assert time_cap(0.0) >= 300.0


def test_cap_grows_with_the_call_bound():
    assert time_cap(200.0) > time_cap(100.0)


def test_without_a_bound_the_cap_is_still_finite():
    assert 0 < time_cap(None) < 10_000


# --- descriptors ---


def test_each_poc_points_to_its_own_interpreter():
    """A single interpreter cannot import both pipecat and LiveKit."""
    descriptors = default_descriptors(Path("/repo"))
    assert "poc1_pipecat" in str(descriptors[1].interpreter)
    assert "poc2_livekit" in str(descriptors[2].interpreter)
    assert descriptors[1].interpreter != descriptors[2].interpreter


def test_only_poc2_needs_a_worker():
    descriptors = default_descriptors(Path("/repo"))
    assert not descriptors[1].needs_worker
    assert descriptors[2].needs_worker
