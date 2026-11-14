"""Sheet for counting what the agent does when it resumes after a backchannel.

The harness metrics say whether the agent stopped and when it spoke again; they
do not say **what** it said on resuming. Counting that requires reading: this
sheet lists each resumption (a backchannel in which the agent stopped and spoke
again) with what it said before and after the event, in **random order and
without the PoC**, to classify it with one of four keys:

- **C — continued**: it carried on with the sentence or idea where it left off,
  without saying again what it had already said.
- **R — repeated**: it said again what it had already said, verbatim or in
  other words, or restarted the sentence from the beginning.
- **N — new**: it said something else, neither continuing nor repeating (for
  example, it answered the "ajá" or changed topic).
- **? — can't tell**: the text is not enough to decide.

The PoC goes in a separate file (`*_clave.csv`) and is joined only when
counting. It is not a fully blind classification (each agent's style is
recognizable) and the report says so. In PoC1 the "before" text is what was
synthesized and may carry words that never got to play; when in doubt, the
audio cut of the same event (`validation.interruption_cuts`) is in the key.

    # 1. Build the sheet from the campaign's exported CSV
    uv run --project poc3_gpt_live python -m validation.resumption_sheet prepare \\
        --events docs/report/data/interruption_campaign/interruptions_events.csv \\
        --output docs/report/data/interruption_campaign/resumption_sheet.csv
    # 2. Fill in the `classification` column (C, R, N or ?)
    # 3. Count
    uv run --project poc3_gpt_live python -m validation.resumption_sheet tally \\
        --sheet docs/report/data/interruption_campaign/resumption_sheet.csv \\
        --events docs/report/data/interruption_campaign/interruptions_events.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

KEYS: Final[dict[str, str]] = {"C": "continuó", "R": "repitió", "N": "new_item", "?": "no se sabe"}
SEED: Final[int] = 20261006
SHEET_COLUMNS: Final[list[str]] = ["id", "over_the_agent", "before_event", "after_the_event",
                                   "classification", "note"]
KEY_COLUMNS: Final[list[str]] = ["id", "poc", "clave", "turn", "estado", "cut"]


def resumptions(events: list[dict[str, str]], *, kinds: tuple[str, ...]) -> list[dict[str, str]]:
    """The events in which the agent stopped and spoke again, with their text split."""
    return [
        e for e in events
        if e.get("tipo") in kinds and e.get("fired") == "1" and e.get("stopped") == "1"
        and e.get("endless_pause") == "0"
        and (e.get("agent_before_event") or e.get("agent_after_event"))
    ]


def prepare(events_csv: Path, output: Path, *, kinds: tuple[str, ...], seed: int = SEED) -> int:
    with events_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = resumptions(list(csv.DictReader(f)), kinds=kinds)
    random.Random(seed).shuffle(rows)
    sheet: list[dict[str, Any]] = []
    clave: list[dict[str, Any]] = []
    for n, e in enumerate(rows, start=1):
        ident = f"R{n:03d}"
        sheet.append({
            "id": ident, "over_the_agent": e.get("text", ""),
            "before_event": e.get("agent_before_event", ""),
            "after_the_event": e.get("agent_after_event", ""),
            "classification": "", "note": "",
        })
        clave.append({
            "id": ident, "poc": e["poc"], "clave": e["clave"], "turn": e["turn"],
            "estado": e.get("estado", ""),
            "cut": f"poc{e['poc']}/{e['clave']}_{e['turn']}_{e['tipo']}.wav",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel opens it with the accented characters.
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(sheet)
    with _ruta_clave(output).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        w.writeheader()
        w.writerows(clave)
    return len(sheet)


def _ruta_clave(sheet: Path) -> Path:
    return sheet.with_name(sheet.stem + "_clave.csv")


def tally(sheet_csv: Path, *, events_csv: Path | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Count per PoC. Fails if any row is left unclassified or has an invalid key.

    With `events_csv`, it leaves out the resumptions of events that fell on the
    call's last turn (`final_turn = 1`, the goodbye), as the report does.
    """
    with sheet_csv.open(encoding="utf-8-sig", newline="") as f:
        sheet = {r["id"]: r for r in csv.DictReader(f)}
    with _ruta_clave(sheet_csv).open(encoding="utf-8-sig", newline="") as f:
        clave = {r["id"]: r for r in csv.DictReader(f)}
    if events_csv is not None:
        with events_csv.open(encoding="utf-8-sig", newline="") as f:
            finals = {(e["clave"], e["turn"]) for e in csv.DictReader(f) if e.get("final_turn") == "1"}
        sheet = {i: r for i, r in sheet.items() if (clave[i]["clave"], clave[i]["turn"]) not in finals}
    bad = [i for i, r in sheet.items() if r["classification"].strip().upper() not in KEYS]
    if bad:
        raise SystemExit(f"{len(bad)} row(s) unclassified or with an invalid key: {', '.join(bad[:10])}")
    per_poc: dict[str, Counter[str]] = {}
    for i, r in sheet.items():
        per_poc.setdefault(clave[i]["poc"], Counter())[r["classification"].strip().upper()] += 1
    rows = [
        {"poc": poc, "resumptions": sum(c.values()), **{KEYS[k]: c.get(k, 0) for k in KEYS}}
        for poc, c in sorted(per_poc.items())
    ]
    text_lines = ["Qué hizo el agente al retomar después de un apoyo verbal (clasificación de la hoja)", "",
              f"{'':6}{'resumptions':>8}" + "".join(f"{v:>12}" for v in KEYS.values())]
    for f in rows:
        text_lines.append(f"PoC{f['poc']:<3}{f['resumptions']:>8}" + "".join(f"{f[v]:>12}" for v in KEYS.values()))
    return "\n".join(text_lines) + "\n", rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--events", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--kinds", default="apoyo")
    b = sub.add_parser("tally")
    b.add_argument("--sheet", type=Path, required=True)
    b.add_argument("--events", type=Path, default=None,
                   help="exporter CSV: leaves out the resumptions in the goodbye (final_turn)")
    args = p.parse_args(argv)
    if args.action == "prepare":
        kinds = tuple(t.strip() for t in args.kinds.split(",") if t.strip())
        n = prepare(args.events, args.output, kinds=kinds)
        print(f"{n} resumptions in {args.output} (the PoC of each one, in {_ruta_clave(args.output)})")
        return 0
    text, rows = tally(args.sheet, events_csv=args.events)
    output = args.sheet.with_name("resumptions_summary.txt")
    output.write_text(text, encoding="utf-8")
    with args.sheet.with_name("resumptions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["poc", "resumptions", *KEYS.values()])
        w.writeheader()
        w.writerows(rows)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
