"""PoC3 joins the batch without moving PoC1 or PoC2.

`shared/lote/executor.default_descriptors` is the only authorized exception to
"do not touch `shared/`" in the whole of step 3, and the authorization came with
conditions: a purely additive diff and a test proving that the descriptors of
the other two come out identical. This test is that condition, and it lives
**here** and not in `shared/lote/tests/` so as not to extend the exception by
even one more file.

The PoC1 and PoC2 values are written as literals on purpose. An `assert`
against what the code itself returns would prove nothing: if someone changed
PoC1's interpreter, the test would change along with it. Frozen here, a change
in either of the two breaks it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from shared.lote.executor import PoCDescriptor, default_descriptors

RAIZ = Path("/raiz-de-prueba")

# Copied from `default_descriptors` **before** PoC3 was added, with the run
# that recorded that they come out the same afterwards.
POC1 = PoCDescriptor(
    poc=1,
    name="Pipecat",
    interpreter=RAIZ / "poc1_pipecat" / ".venv" / "Scripts" / "python.exe",
    modulo="poc1_pipecat.app.harness",
    needs_worker=False,
)
POC2 = PoCDescriptor(
    poc=2,
    name="LiveKit Agents",
    interpreter=RAIZ / "poc2_livekit" / ".venv" / "Scripts" / "python.exe",
    modulo="poc2_livekit.app.harness",
    needs_worker=True,
)


class TestTheOldDescriptorsDoNotMove:
    def test_poc1_comes_out_field_by_field_the_same(self) -> None:
        assert default_descriptors(RAIZ)[1] == POC1

    def test_poc2_comes_out_field_by_field_the_same(self) -> None:
        assert default_descriptors(RAIZ)[2] == POC2

    def test_neither_of_the_two_changes_interpreter_or_module(self) -> None:
        """Explicit on top of the equality: these are the two fields that get launched."""
        descriptors = default_descriptors(RAIZ)
        assert descriptors[1].modulo == "poc1_pipecat.app.harness"
        assert descriptors[2].modulo == "poc2_livekit.app.harness"
        assert descriptors[1].interpreter.parts[-4] == "poc1_pipecat"
        assert descriptors[2].interpreter.parts[-4] == "poc2_livekit"

    def test_only_one_key_was_added(self) -> None:
        assert sorted(default_descriptors(RAIZ)) == [1, 2, 3]


class TestThePoC3Descriptor:
    def test_points_to_the_poc3_harness_with_its_own_interpreter(self) -> None:
        """Each PoC runs with its own `uv`: that is what makes each architecture's
        environment reproducible in the residency report."""
        descriptor = default_descriptors(RAIZ)[3]

        assert descriptor.poc == 3
        assert descriptor.modulo == "poc3_gpt_live.app.harness"
        assert descriptor.interpreter == (
            RAIZ / "poc3_gpt_live" / ".venv" / "Scripts" / "python.exe"
        )

    def test_does_not_need_a_worker(self) -> None:
        """PoC3's bridge lives in the harness process, like PoC1's.

        `needs_worker` is `True` only in PoC2, whose agent runs separately and
        stays up between calls. If this came out `True`, the batch would try to
        start and monitor a worker that does not exist.
        """
        assert default_descriptors(RAIZ)[3].needs_worker is False
        assert default_descriptors(RAIZ)[2].needs_worker is True

    @pytest.mark.skipif(sys.platform != "win32", reason="the batch runner uses the Windows venv layout (.venv/Scripts/python.exe)")
    def test_the_poc3_interpreter_exists_in_this_repo(self) -> None:
        """`exists` is what the CLI checks before letting a batch run."""
        root = Path(__file__).resolve().parents[2]
        assert default_descriptors(root)[3].exists is True
