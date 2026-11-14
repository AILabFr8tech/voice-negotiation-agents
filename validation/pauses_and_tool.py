"""PoC3 in the interruption scenario: do its pauses coincide with a tool query?

GPT-Live pauses even when nobody is talking to it, and some of those pauses are
the wait for the tool ("déjame revisar…", "let me check…"). For each PoC3
backchannel and control, the script looks for whether the model opened a request
(`delegation.created` in the census) near the event:

- if it went quiet: between 4 s before the trigger and the end of the pause (the
  query usually starts 1 to 3 s before the event is played);
- if it kept talking: between 4 s before and 3 s after the trigger.

The event timeline (seconds of the harness run) is mapped to epoch with the
external instrument's anchor (`external_latency_poc3.anchor_on`); the census
timeline, with the bridge's `clock.anchor_wall_s`. It is exploratory: the window
is fixed.

Writes `pauses_and_tool.csv` to `--output`.

    uv run --project poc3_gpt_live --with scipy python -m validation.pauses_and_tool
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

from shared.metrics import external_latency_poc3 as ext3
from shared.metrics import round_trips as rt
from shared.metrics.campaigns import INTERRUPTIONS_POC3
from shared.metrics.export_for_report import CLIPS_DEFAULT, LOTE_DEFAULT, _a_8khz

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data" / "interruption_campaign_v3"
ANTES_SEG: Final[float] = 4.0
AFTER_S: Final[float] = 3.0
ENDLESS_PAUSE_S: Final[float] = 8.0
COLUMNS: Final[list[str]] = ["tipo", "clave", "turn", "text", "stopped", "query_nearby"]


def events_of(folder: Path, row: dict) -> list[dict]:
    res = json.loads((folder / "summaries" / f"{row['clave']}.json").read_text(encoding="utf-8"))
    art = res["artifacts"]
    harness = Path(art["event_log"]).parent
    marks = ext3.marks_de(harness / f"zeros_{res['label']}.jsonl")
    audio, sample_rate = ext3.leer_wav(Path(art["auditoria_wav"]))
    a8, t8 = _a_8khz(audio, sample_rate)
    anchor = ext3.anchor_on(marks, a8[:, 0], t8, clips_path=CLIPS_DEFAULT)
    live_bridge = json.loads(Path(art["bridge_json"]).read_text(encoding="utf-8"))
    queries = sorted(
        e.when for e in rt.poc3_calls(Path(art["census"]), float(live_bridge["clock"]["anchor_wall_s"]))
        if e.name == "delegation.created"
    )
    output = []
    for t in json.loads(Path(art["event_log"]).read_text(encoding="utf-8"))["turns"]:
        ev = t.get("event") or {}
        if ev.get("tipo") not in ("apoyo", "control") or not ev.get("fired") or t.get("final_turn"):
            continue
        near = ""
        if anchor is not None:
            beg = ev["t_trigger"] + anchor.displacement
            if ev.get("stopped"):
                alto = ev["last_speech_before_stopping"] + anchor.displacement
                end = alto + (ev.get("pause_duration_ms") or ENDLESS_PAUSE_S * 1000) / 1000.0
            else:
                end = beg + AFTER_S
            near = int(any(beg - ANTES_SEG <= c <= end for c in queries))
        output.append({"tipo": ev["tipo"], "clave": row["clave"], "turn": t["turn"], "text": ev.get("text", ""),
                       "stopped": int(bool(ev.get("stopped"))), "query_nearby": near})
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args(argv)
    folder = args.lote / INTERRUPTIONS_POC3
    rows = []
    for linea in (folder / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if linea.strip():
            rows += events_of(folder, json.loads(linea))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "pauses_and_tool.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for tipo in ("apoyo", "control"):
        of = [f for f in rows if f["tipo"] == tipo]
        print(f"{tipo}: {len(of)} · went quiet {sum(f['stopped'] for f in of)} "
              f"(with a query {sum(1 for f in of if f['stopped'] and f['query_nearby'] == 1)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
