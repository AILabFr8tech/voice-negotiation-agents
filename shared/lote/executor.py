"""Run one harness call as a subprocess and collect its summary.

It is a subprocess and not a function call for a reason that cannot be worked
around: **each PoC has its own environment**. `poc1_pipecat/.venv` has pipecat
and `poc2_livekit/.venv` has the LiveKit SDK, and no single interpreter can
import both. A batch that alternates between the two architectures has to
launch processes, period.

It comes with a side effect one would want anyway, for free: a call that
crashes, hangs or takes the interpreter down with it cannot bring down the
batch. It gets a time cap, the reason is recorded, and the batch moves on to
the next one.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.lote.children import adopt
from shared.lote.plan import BatchItem

# Margins of a call's time cap. The duration bound is quoted by the harness
# itself; this adds what the harness does not quote, which is startup.
#
# In PoC1, startup means bringing up the pipeline server in another process,
# which loads two ONNX models (Silero and the turn analyzer) before accepting
# the first connection. On a cold machine that has taken more than a minute.
# The factor of 2 over the bound covers the fact that the bound assumes the
# longest branch of the script but does not assume the agent sits thinking.
FACTOR_OVER_BOUND: float = 2.0
ONSET_MARGIN_S: float = 300.0


@dataclass(frozen=True)
class PoCDescriptor:
    """How a PoC's harness is launched. The only thing specific to each PoC."""

    poc: int
    name: str
    interpreter: Path
    modulo: str
    # Whether the batch has to keep a separate worker running for it. Only
    # PoC2: its agent runs in another process and stays up between calls.
    needs_worker: bool = False

    @property
    def exists(self) -> bool:
        return self.interpreter.is_file()


def default_descriptors(root: Path) -> dict[int, PoCDescriptor]:
    """Each PoC's interpreter, as `uv` leaves it in the repo."""
    return {
        1: PoCDescriptor(
            poc=1,
            name="Pipecat",
            interpreter=root / "poc1_pipecat" / ".venv" / "Scripts" / "python.exe",
            modulo="poc1_pipecat.app.harness",
        ),
        2: PoCDescriptor(
            poc=2,
            name="LiveKit Agents",
            interpreter=root / "poc2_livekit" / ".venv" / "Scripts" / "python.exe",
            modulo="poc2_livekit.app.harness",
            needs_worker=True,
        ),
        3: PoCDescriptor(
            poc=3,
            name="GPT-Live",
            interpreter=root / "poc3_gpt_live" / ".venv" / "Scripts" / "python.exe",
            modulo="poc3_gpt_live.app.harness",
            # No separate worker: the PoC3 bridge lives inside the process that
            # the harness itself starts, as in PoC1. The one that needs a worker
            # is PoC2, whose agent runs in another process and stays up between
            # calls.
            needs_worker=False,
        ),
    }


def call_command(
    descriptor: PoCDescriptor,
    item: BatchItem,
    *,
    summary_path: Path,
    accept_cost: bool,
    record_audit: bool = False,
    interruptions: bool = False,
) -> list[str]:
    """The exact command line of a call.

    Without `--accept-cost` the harness spends nothing: it prints the warning,
    writes the estimate into the summary and exits. That is the quote the batch
    uses to build its cost table before touching anything.
    """
    parsed_args = [
        str(descriptor.interpreter),
        "-m",
        descriptor.modulo,
        "--scenario",
        str(item.scenario),
        "--repetition",
        str(item.repetition),
        "--summary",
        str(summary_path),
    ]
    if record_audit:
        parsed_args.append("--record-audit")
    if interruptions:
        parsed_args.append("--interruptions")
    if accept_cost:
        parsed_args.append("--accept-cost")
    return parsed_args


@dataclass
class CallResult:
    """How the subprocess exited, and what it left written."""

    code_value: int | None
    summary: dict[str, Any] | None
    motivo: str
    duration_s: float
    log_file_path: Path

    @property
    def had_summary(self) -> bool:
        return self.summary is not None


def read_summary(ruta: Path) -> dict[str, Any] | None:
    """Read the summary the harness left, or `None` if there is no usable one."""
    if not ruta.is_file():
        return None
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def time_cap(seconds_cap: float | None) -> float:
    """How long a call is tolerated before it is killed."""
    bound = float(seconds_cap or 0.0)
    return bound * FACTOR_OVER_BOUND + ONSET_MARGIN_S


def run_call(
    descriptor: PoCDescriptor,
    item: BatchItem,
    *,
    root: Path,
    summary_path: Path,
    log_file_path: Path,
    tope_seg: float,
    accept_cost: bool,
    record_audit: bool = False,
    environment: dict[str, str] | None = None,
    interruptions: bool = False,
) -> CallResult:
    """Launch the harness, wait for it to finish and return what it left.

    All output goes to a file and not to the batch's console: with a hundred
    calls, each harness's report would bury the progress, and with one file per
    call the one that failed can be read afterwards without having had to watch.

    No failure of the child propagates. A nonzero exit code, an expired time cap
    or a missing summary are three different reasons, and all three are
    returned as a result, not as an exception: the batch continues.
    """
    # A stale summary with the same name would be read as if it belonged to
    # this call. That happens when manually retrying one that had already run.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.unlink(missing_ok=True)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_args = call_command(
        descriptor,
        item,
        summary_path=summary_path,
        accept_cost=accept_cost,
        record_audit=record_audit,
        interruptions=interruptions,
    )
    onset = time.monotonic()
    motivo = ""
    code_value: int | None = None
    with open(log_file_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(parsed_args)}\n\n")
        log.flush()
        process = subprocess.Popen(  # noqa: S603 - our own arguments, no shell
            parsed_args,
            cwd=str(root),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        # Before waiting on it: if the batch dies now, this call has to die
        # with it. See `shared/lote/children.py`.
        adopt(process)
        try:
            code_value = process.wait(timeout=tope_seg)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            motivo = f"exceeded the {tope_seg:.0f} s cap and was killed"
        except KeyboardInterrupt:
            # Ctrl+C with a call in flight: kill the child and let it propagate.
            # The batch writes no line, so this call is rerun on resume.
            process.kill()
            process.wait(timeout=30)
            raise

    duration = time.monotonic() - onset
    summary = read_summary(summary_path)
    if not motivo and code_value not in (0, 1):
        motivo = f"the harness exited with code {code_value}"
    if not motivo and summary is None:
        motivo = "the harness finished without leaving a summary"
    return CallResult(
        code_value=code_value,
        summary=summary,
        motivo=motivo,
        duration_s=duration,
        log_file_path=log_file_path,
    )


def get_quote(
    descriptor: PoCDescriptor,
    item: BatchItem,
    *,
    root: Path,
    folder: Path,
    environment: dict[str, str] | None = None,
    interruptions: bool = False,
) -> dict[str, Any] | None:
    """Ask the harness what that call would cost. Spends nothing.

    The batch's cost table is built by the harness, not by the batch: if the
    batch carried its own copy of the rates it would be a second source of
    truth about money, and the one that goes stale is always the copy.
    """
    ruta = folder / "quotes" / f"poc{descriptor.poc}-e{item.scenario}.json"
    log = folder / "quotes" / f"poc{descriptor.poc}-e{item.scenario}.log"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resultado = run_call(
        descriptor,
        item,
        root=root,
        summary_path=ruta,
        log_file_path=log,
        # Quoting calls nobody: it loads the script and multiplies. If it takes
        # five minutes, something is wrong and it is better to find out here.
        tope_seg=300.0,
        accept_cost=False,
        environment=environment,
        # With interruptions, quoting also checks that their clips are ready:
        # if they are missing, the harness fails here and not mid-campaign.
        interruptions=interruptions,
    )
    if resultado.summary and resultado.summary.get("estimate_only"):
        return resultado.summary
    return None
