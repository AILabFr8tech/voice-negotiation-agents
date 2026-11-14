"""Killing the batch the hard way must take the in-flight call down with it.

The real test sets up the full situation: a parent process that adopts a
child, and the parent is killed by `Stop-Process -Force` (not a signal it could
catch, not a `finally`). If the child writes its file, the mechanism does not
work.

This is the test that has to be run for real and not faked: on 20 September
2026 the version without this left three orphans, and no in-memory test would
have caught it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from shared.lote.children import IS_WINDOWS, adopt, batch_job

# The parent: adopts a child that writes a file within N seconds, reports that
# it is done and stays waiting to be killed.
PADRE = '''
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from shared.lote.children import adopt

child = subprocess.Popen([
    sys.executable, "-c",
    "import time, pathlib; time.sleep(6); "
    "pathlib.Path('{sentinela}').write_text('sobrevivi')",
])
print("adoptado" if adopt(child) else "sin adoptar", flush=True)
time.sleep(120)
'''

REPO = str(Path(__file__).resolve().parents[3])


@pytest.mark.skipif(not IS_WINDOWS, reason="the job object is Windows-only")
def test_killing_the_parent_takes_the_child(tmp_path):
    sentinel = tmp_path / "sobrevivi.txt"
    script = tmp_path / "padre.py"
    # `as_posix` and not the Windows path: the text ends up inside a Python
    # literal inside another, and a `\U` from `C:\Users` would be eaten as an escape.
    script.write_text(PADRE.format(sentinel=sentinel.as_posix()), encoding="utf-8")

    padre = subprocess.Popen(
        [sys.executable, str(script), REPO],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert padre.stdout is not None
        assert padre.stdout.readline().strip() == "adoptado"
        # The hard way: `kill` on Windows is `TerminateProcess`, which does not
        # let the parent run a single line more.
        padre.kill()
        padre.wait(timeout=30)
        # Longer than the grandchild was going to sleep before writing.
        time.sleep(9)
        assert not sentinel.exists()
    finally:
        padre.kill()


@pytest.mark.skipif(not IS_WINDOWS, reason="the job object is Windows-only")
def test_without_adopting_the_child_does_survive(tmp_path):
    """The control for the test above: without the job, the orphan lives.

    Without this, an `adopt` that did nothing would pass the previous test if
    the child were slow to start, and nobody would notice.
    """
    sentinel = tmp_path / "sobrevivi.txt"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import time, pathlib; time.sleep(2); "
            f"pathlib.Path('{sentinel.as_posix()}').write_text('sobrevivi')",
        ]
    )
    try:
        child.wait(timeout=30)
        assert sentinel.exists()
    finally:
        if child.poll() is None:
            child.kill()


@pytest.mark.skipif(not IS_WINDOWS, reason="the job object is Windows-only")
def test_the_job_is_created_only_once():
    """Its open handle IS the mechanism: releasing it is what kills the children."""
    assert batch_job() is not None
    assert batch_job() is batch_job()


def test_adopting_a_process_without_handle_says_no():
    class NoHandle:
        pass

    assert adopt(NoHandle()) is False
