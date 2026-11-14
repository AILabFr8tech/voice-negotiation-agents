"""Computation of the comparison metrics. **Identical in all 3 PoCs.**

If each PoC measured in its own way, the numbers would not be comparable and
the technical decision would have no backing. That is why the computation lives
here and not inside each PoC: the only thing that changes between architectures
is the **capture** of the samples.

## The workflow

    1. You run the call            -> the pipeline leaves `call_<sid>.json`
    2. `reference.generate()`      -> file pre-filled with the script
    3. You correct the `yo:` lines -> 2-3 min, the only manual part
    4. `calls.enrich()`            -> percentiles and WER inside the same JSON
    5. `report.consolidate()`      -> Markdown + CSV per PoC

Everything numeric is automatic. The only manual part: correcting the reference
transcript and entering the two 1-5 ratings for naturalness and turn-taking.

See `python -m shared.metrics.cli --help`, and `... cli demo` to validate the
whole computation with synthetic data without spending a call.

## What is measured

- **Latency** (`latency.py`): full turn and per-stage breakdown, with
  percentiles and quartiles, separating cold start, steady state and
  interrupted turns.
- **STT quality** (`wer.py`, `text.py`): WER, CER and amount extraction error
  rate, with Spanish normalization — "dieciocho mil" and "18,000" count the
  same.
- **Negotiation outcome and compliance** (`calls.py`): close rate, margin vs.
  target, and the financial-limit binaries.
- **Naturalness and turn-taking**: manual, 1-5, with the script's rubric.
"""

from shared.metrics.scenarios import (
    SCENARIOS_PATH,
    Scenario,
    Script,
    ScriptTurn,
    load_script,
)
from shared.metrics.latency import (
    BUCKET_OPENING,
    BUCKET_INTERRUPTED,
    BUCKET_REGIMEN,
    BUCKETS,
    TurnSample,
    percentile,
    summarize_samples,
    summarize_values,
)
from shared.metrics.calls import (
    METRICS_KEY,
    SAMPLES_KEY,
    CallMetrics,
    compute,
    compute_from_path,
    enrich,
    list_calls,
)
from shared.metrics.reference import (
    ReferenceFile,
    ManualNotes,
    ReferenceTurn,
)
from shared.metrics.report import (
    Consolidated,
    consolidate,
    write_csv,
    write_markdown,
    generate_markdown,
)
from shared.metrics.text import (
    CONFIG_DEFAULT,
    NormalizationConfig,
    extract_amounts,
    normalize,
    tokenize_text,
)
from shared.metrics.wer import (
    Alignment,
    AmountsResult,
    TextResult,
    align,
    compute_amounts,
    compute_text,
)

__all__ = [
    # text normalization
    "CONFIG_DEFAULT",
    "NormalizationConfig",
    "extract_amounts",
    "normalize",
    "tokenize_text",
    # WER / CER / amounts
    "Alignment",
    "AmountsResult",
    "TextResult",
    "align",
    "compute_amounts",
    "compute_text",
    # latency
    "BUCKETS",
    "BUCKET_OPENING",
    "BUCKET_INTERRUPTED",
    "BUCKET_REGIMEN",
    "TurnSample",
    "percentile",
    "summarize_samples",
    "summarize_values",
    # script
    "SCENARIOS_PATH",
    "Scenario",
    "Guion",
    "ScriptTurn",
    "load_script",
    # reference
    "ReferenceFile",
    "ManualNotes",
    "ReferenceTurn",
    # per call
    "METRICS_KEY",
    "SAMPLES_KEY",
    "CallMetrics",
    "compute",
    "compute_from_path",
    "enrich",
    "list_calls",
    # consolidation
    "Consolidated",
    "consolidate",
    "write_csv",
    "write_markdown",
    "generate_markdown",
]
