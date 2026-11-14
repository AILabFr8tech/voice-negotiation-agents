"""Reanalysis of the validation along the PoC3 path, without running anything new.

The PoC3 validation (`validate_harness_poc3`, 25 September 2026) left in
`diagnostics/harness_validation/poc3/` everything from each run: the audit WAV,
the marks, the log and the fake model's ground truth. Here each turn is measured
again in two ways:

- **Bridge method:** the one the campaign used for most PoC3 turns
  (`external_latency_poc3`), as reported by the validation.
- **Direct method:** the one for the campaign's recovered turns
  (`export_for_report.recovered_turns`), applied to all turns.

It also separates an effect of the fake model itself, which does not exist in the
campaign: the fake model cuts its speech at 2.6 s and delivers it in 100 ms
blocks, so the last block of each response ends with `100 - offset` ms of zeros
(the offset is where the speech started within its first block). If that filler
is short, the bridge's resampler, which keeps state across blocks, releases the
tail of that response at the start of the next one, and the harness takes that
blip as the onset: about 100 ms early. Those turns are marked `old_tail`.

Writes to `docs/report/data/`: `harness_validation_poc3_direct.csv` and
`harness_validation_poc3_direct_summary.txt`.

    uv run --project poc3_gpt_live --with scipy python -m validation.reanalysis_validation_poc3
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Final

from shared.metrics import external_latency_poc3 as ext3
from shared.metrics.export_for_report import recovered_turns

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
WORK: Final[Path] = RAIZ / "diagnostics" / "harness_validation" / "poc3"
CLIPS: Final[Path] = RAIZ / "shared" / "test_scenarios" / "clips"
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"

CONDITIONS: Final[tuple[str, ...]] = (
    "retardo_300", "retardo_1000", "retardo_2500", "nivel_-6", "nivel_-35", "arranque_suave",
)
# The fake model delivers its speech in 100 ms blocks and cuts it at 2.6 s
# (a multiple of 100 ms), so the last block carries 100 - offset ms of zeros.
BLOCK_MS: Final[float] = 100.0
# With less filler than this, the tail of the previous response comes out at the
# start of the next one (see the docstring).
FILLER_WITHOUT_TAIL_MS: Final[float] = 60.0
# The offset comes in samples at 24 kHz, the model's rate.
SAMPLES_PER_MS: Final[float] = 24.0

COLUMNS: Final[list[str]] = [
    "condition", "scenario", "turn", "clip_id", "real_latency_ms",
    "bridge_error_ms", "direct_error_ms", "prior_filler_ms", "old_tail",
]


def run(condition: str, scenario: int, live_bridge: dict[tuple[str, int, str], float | None]) -> list[dict]:
    sid = f"ARNES-call-validacion-poc3-{condition}-e{scenario}"
    marks = ext3.marks_de(WORK / f"zeros_{sid}.jsonl")
    direct_values = {
        t.turn: t.line_latency_ms
        for t in recovered_turns(marks, WORK / f"auditoria_{sid}.wav", CLIPS,
                                    measured_items=set(), event_log=WORK / f"event_log_{sid}.json")
    }
    truth = [
        json.loads(l)
        for l in (WORK / f"verdad_validacion-poc3-{condition}-e{scenario}.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # The responses in the order the fake model gave them (greeting included):
    # the one before each turn is the one that may have left a tail.
    replies = [v for v in truth if v.get("tipo") in ("greeting", "turn")]
    rows = []
    used: set[int] = set()
    for m in marks:
        k = next((k for k, v in enumerate(replies)
                  if k not in used and v.get("tipo") == "turn" and v["clip_id"] == m.get("clip_id")), None)
        if k is None:
            continue
        used.add(k)
        v = replies[k]
        real = v.get("real_latency_ms")
        filler = BLOCK_MS - replies[k - 1]["offset_samples"] / SAMPLES_PER_MS if k > 0 else None
        direct_value = direct_values.get(str(m["turn"]))
        reported = live_bridge.get((condition, scenario, str(m["turn"])))
        rows.append({
            "condition": condition, "scenario": scenario, "turn": m["turn"], "clip_id": m.get("clip_id"),
            "real_latency_ms": real,
            "bridge_error_ms": None if reported is None or real is None else round(reported - real, 1),
            "direct_error_ms": None if direct_value is None or real is None else round(direct_value - real, 1),
            "prior_filler_ms": None if filler is None else round(filler, 1),
            "old_tail": None if filler is None else int(filler <= FILLER_WITHOUT_TAIL_MS),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    global WORK, SAMPLES_PER_MS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=DATA)
    p.add_argument("--entry", type=Path, default=DATA,
                   help="folder with the run's harness_validation_poc3.csv")
    p.add_argument("--work", type=Path, default=WORK,
                   help="folder with the run's artifacts")
    p.add_argument("--format-kind", choices=("pcm24", "pcmu"), default="pcm24",
                   help="session format in that run (sets the fake model's rate)")
    args = p.parse_args(argv)
    WORK = args.work
    if args.format_kind == "pcmu":
        SAMPLES_PER_MS = 8.0
    with (args.entry / "harness_validation_poc3.csv").open(encoding="utf-8") as fh:
        live_bridge = {
            (r["condition"], int(r["scenario"]), r["turn"]):
                float(r["reported_latency_ms"]) if r["reported_latency_ms"] else None
            for r in csv.DictReader(fh)
        }
    rows = [f for c in CONDITIONS for e in range(1, 6) for f in run(c, e, live_bridge)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "harness_validation_poc3_direct.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in rows:
            w.writerow({k: ("" if f.get(k) is None else f[k]) for k in COLUMNS})

    def med(v: list[float]) -> str:
        return f"{statistics.median(v):7.1f} (n={len(v)})" if v else "    --- (n=0)"

    text_lines = ["condición        método del puente: todos / sin cola vieja   método directo: todos / sin cola vieja"]
    for c in CONDITIONS:
        fs = [f for f in rows if f["condition"] == c]
        pu = [f["bridge_error_ms"] for f in fs if f["bridge_error_ms"] is not None]
        pl = [f["bridge_error_ms"] for f in fs if f["bridge_error_ms"] is not None and f["old_tail"] == 0]
        di = [f["direct_error_ms"] for f in fs if f["direct_error_ms"] is not None]
        dl = [f["direct_error_ms"] for f in fs if f["direct_error_ms"] is not None and f["old_tail"] == 0]
        text_lines.append(f"{c:15s}  {med(pu)} / {med(pl)}        {med(di)} / {med(dl)}")
    text_lines.append(f"turnos con cola vieja: {sum(f['old_tail'] == 1 for f in rows)} de {len(rows)}")
    (args.output / "harness_validation_poc3_direct_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
