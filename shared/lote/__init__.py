"""Batch mode: many harness calls, alternating PoCs, unattended.

What it solves, and why each piece lives where it does, is in
`docs/METRICS_CONTRACT.md` §2.6. In short:

- **One call, one file.** Each run writes its summary when it finishes. If
  the batch dies at call 40, the previous 39 are complete on disk.
- **Alternates the PoCs.** Running 100 of PoC1 today and 100 of PoC2 tomorrow
  confounds anything that changes between the two days (the network, the
  provider's mood, a model version) with the architecture. The plan interleaves
  them by scenario, so each comparable pair runs minutes apart.
- **Resumable.** `progress.jsonl` is the single source of truth for what has
  already been done. Killing the process corrupts nothing: the worst case is
  that the in-flight call left no line and gets run again.
- **Spend cap.** Checked before launching each call, using the estimate the
  harness itself quoted. The batch does not carry its own copy of the rates.
- **A failing call does not bring down the batch.** It is recorded with its
  reason and the batch moves on.

The batch **measures nothing**. It launches processes, reads the summaries they
write and keeps count. Every number that comes out of here came from the harness.
"""

from __future__ import annotations

from shared.lote.children import adopt
from shared.lote.executor import (
    PoCDescriptor,
    CallResult,
    call_command,
    run_call,
    default_descriptors,
)
from shared.lote.plan import BatchItem, build_plan, plan_fingerprint, span
from shared.lote.progress import (
    append_row,
    finished_keys,
    cumulative_spend,
    read_progress,
    progress_row,
)
from shared.lote.worker import PoC2Worker

__all__ = [
    "PoCDescriptor",
    "BatchItem",
    "CallResult",
    "PoC2Worker",
    "adopt",
    "append_row",
    "build_plan",
    "finished_keys",
    "call_command",
    "run_call",
    "default_descriptors",
    "cumulative_spend",
    "plan_fingerprint",
    "read_progress",
    "span",
    "progress_row",
]
