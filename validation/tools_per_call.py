"""How many times each PoC called each tool, in the analyzed calls.

The function calls come from the same records the exporter uses
(`shared.metrics.round_trips`):

- PoC1: the log of each run (`Calling function [name:…]`).
- PoC2: the campaign's `worker.log`, trimmed to each call's window (from the
  carrier's first audio minus 20 s to the last plus 30 s).
- PoC3: the call's census; counts one per function result.

Population: the calls with `in_the_pool = 1` in `populations.csv`. Writes
`tools_per_call.csv` (poc, clave, function, calls) to `--output`.

    python -m validation.tools_per_call
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Final

from shared.metrics import round_trips as rt
from shared.metrics.campaigns import CAMPAIGN_POC1_POC2, CAMPAIGN_POC3
from shared.metrics.export_for_report import (
    CLIPS_DEFAULT,
    LOTE_DEFAULT,
    _marks_of,
    read_summary,
    marks_from_lines,
)

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
RE_POC1: Final[re.Pattern[str]] = re.compile(r"Calling function \[([a-z_]+):")
ANTES_SEG: Final[float] = 20.0
AFTER_S: Final[float] = 30.0


def analyzed(data: Path) -> list[dict[str, str]]:
    with (data / "populations.csv").open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["estado"] == "ok" and r["in_the_pool"] == "1"]


def of_poc1(lote: Path, clave: str) -> Counter:
    text = (lote / CAMPAIGN_POC1_POC2 / "calls" / f"{clave}.log").read_text(encoding="utf-8", errors="replace")
    return Counter(RE_POC1.findall(text))


def of_poc2(lote: Path, clave: str, worker: list[rt.FunctionCall]) -> Counter:
    res = read_summary(lote / CAMPAIGN_POC1_POC2, clave) or {}
    art = res.get("artifacts") or {}
    candidates = [Path(v).with_name(f"zeros_{res.get('label')}.jsonl") for v in art.values() if isinstance(v, str)]
    zeros = next((c for c in candidates if c.is_file()), None)
    block = (res.get("lines") or {}).get("per_line") or []
    marks = _marks_of(zeros) or marks_from_lines(block, CLIPS_DEFAULT)
    if not marks:
        return Counter()
    since = min(float(m["start_of_clip"]) for m in marks) - ANTES_SEG
    until = max(float(m.get("clip_end", m["measured_end_of_speech"])) for m in marks) + AFTER_S
    return Counter(f.name for f in worker if since <= f.when <= until)


def of_poc3(lote: Path, clave: str) -> Counter:
    art = (read_summary(lote / CAMPAIGN_POC3, clave) or {}).get("artifacts") or {}
    live_bridge = json.loads(Path(art["bridge_json"]).read_text(encoding="utf-8"))
    events = rt.poc3_calls(Path(art["census"]), float(live_bridge["clock"]["anchor_wall_s"]))
    return Counter(e.function for e in events if e.is_result and e.function)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    worker = rt.worker_calls(args.lote / CAMPAIGN_POC1_POC2 / "worker.log")
    rows = []
    totals: dict[str, Counter] = {}
    for r in analyzed(args.data):
        poc, clave = r["poc"], r["clave"]
        if poc == "1":
            c = of_poc1(args.lote, clave)
        elif poc == "2":
            c = of_poc2(args.lote, clave, worker)
        else:
            c = of_poc3(args.lote, clave)
        totals.setdefault(poc, Counter()).update(c)
        rows += [{"poc": poc, "clave": clave, "function": f, "calls": n} for f, n in sorted(c.items())]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tools_per_call.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["poc", "clave", "function", "calls"])
        w.writeheader()
        w.writerows(rows)
    for poc in sorted(totals):
        print(f"PoC{poc}: {sum(totals[poc].values())} · {dict(totals[poc])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
