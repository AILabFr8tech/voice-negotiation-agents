"""The PoC2 worker as seen from the batch: up, down, and brought back up."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from shared.lote.worker import READY_SIGNAL, PoC2Worker

# A fake worker: prints whatever it is told and stays alive until killed.
FAKE_WORKER = '''
import json, sys, time
from pathlib import Path
config = json.loads(Path("worker_config.json").read_text(encoding="utf-8"))
if config.get("is_recorded", True):
    print(config.get("sig", "registered worker"), flush=True)
if config.get("dies"):
    sys.exit(1)
time.sleep(60)
'''


@pytest.fixture
def fake_worker(tmp_path: Path):
    packet = tmp_path / "poc2_livekit" / "app"
    packet.mkdir(parents=True)
    (tmp_path / "poc2_livekit" / "__init__.py").write_text("", encoding="utf-8")
    (packet / "__init__.py").write_text("", encoding="utf-8")
    (packet / "agent.py").write_text(FAKE_WORKER, encoding="utf-8")

    def build(**config) -> PoC2Worker:
        (tmp_path / "worker_config.json").write_text(
            __import__("json").dumps(config), encoding="utf-8"
        )
        return PoC2Worker(
            interpreter=Path(sys.executable),
            root=tmp_path,
            log_file_path=tmp_path / "worker.log",
        )

    return build


def test_start_waits_for_it_to_register(fake_worker):
    worker = fake_worker(is_recorded=True)
    try:
        assert worker.start(tope_seg=30.0) is True
        assert worker.alive
    finally:
        worker.halt()


def test_a_worker_that_never_registers_returns_false_and_does_not_blow_up(fake_worker):
    """It has to let the PoC2 calls fail, not bring down the campaign."""
    worker = fake_worker(is_recorded=False)
    try:
        assert worker.start(tope_seg=2.0) is False
    finally:
        worker.halt()


def test_a_worker_that_dies_on_startup_does_not_exhaust_the_cap(fake_worker):
    import time

    worker = fake_worker(is_recorded=False, dies=True)
    try:
        onset = time.monotonic()
        assert worker.start(tope_seg=30.0) is False
        assert time.monotonic() - onset < 20
    finally:
        worker.halt()


def test_stopping_is_idempotent(fake_worker):
    worker = fake_worker(is_recorded=True)
    worker.start(tope_seg=30.0)
    worker.halt()
    worker.halt()
    assert not worker.alive


def test_ensure_does_not_relaunch_one_that_is_already_up(fake_worker):
    worker = fake_worker(is_recorded=True)
    try:
        worker.start(tope_seg=30.0)
        assert worker.ensure() is True
        assert worker.onsets == 1
    finally:
        worker.halt()


def test_relaunching_it_does_not_erase_the_trace_of_the_previous_start(fake_worker):
    """Why the 3 a.m. worker died is in the log of the one that died."""
    worker = fake_worker(is_recorded=True)
    try:
        worker.start(tope_seg=30.0)
        worker.process.kill()
        worker.process.wait(timeout=20)
        worker.ensure(tope_seg=30.0)
        raw_value = worker.log_file_path.read_text(encoding="utf-8")
        assert "=== startup 1 —" in raw_value
        assert "=== startup 2 —" in raw_value
    finally:
        worker.halt()


def test_ensure_relaunches_it_if_it_went_down(fake_worker):
    worker = fake_worker(is_recorded=True)
    try:
        worker.start(tope_seg=30.0)
        worker.process.kill()
        worker.process.wait(timeout=20)
        assert worker.ensure(tope_seg=30.0) is True
        assert worker.onsets == 2
    finally:
        worker.halt()


def test_the_signal_of_an_old_start_does_not_count_for_the_new_one(tmp_path):
    """Without this, a worker that just died would be taken as ready."""
    log = tmp_path / "worker.log"
    log.write_text(
        f"=== startup 1 — yesterday ===\n{READY_SIGNAL}\n=== startup 2 — today ===\n",
        encoding="utf-8",
    )
    worker = PoC2Worker(
        interpreter=Path(sys.executable), root=tmp_path, log_file_path=log
    )
    worker.onsets = 2
    assert worker.was_recorded() is False
    worker.onsets = 1
    # With the marker of startup 1, the signal is in its segment.
    assert worker.was_recorded() is True


def test_without_log_yet_it_has_not_registered(tmp_path):
    worker = PoC2Worker(
        interpreter=Path(sys.executable),
        root=tmp_path,
        log_file_path=tmp_path / "does_not_exist.log",
    )
    assert worker.was_recorded() is False
