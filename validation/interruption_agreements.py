"""What the carrier said right before the agent recorded an agreement.

In the interruption scenario, a backchannel ("ok", "sí, sí", "ándele") is played
over the agent. If the agent takes it as an answer, it may close a deal the
carrier did not accept. This script changes no figure: for each call of the
campaign (`interruption_campaign_v3`: PoC1 and PoC2 from 5 October, PoC3 from
the 9th), it takes the outcome the agent recorded (`registrar_acuerdo` or
`finalizar_sin_acuerdo`), its time, and the carrier's last lines before that
time, and joins them with the reading of each call
(`interruption_agreements_reading.csv`, one class and one note per call).

Reading classes:
- `acepto`: recorded after a script line that accepts the deal.
- `apoyo`: recorded after a backchannel, without the carrier having accepted
  since the last offer.
- `linea_incompleta`: recorded after a fragment of a line that, in full, did not
  accept (the recognizer lost words or the end of turn split it).
- `anticipado`: recorded after a complete script line that does not accept
  ("Órale, con eso ya cambia la cosa", "Si me lo deja parejo, lo tomo").
- `sin_acuerdo`: the agent closed without an agreement.

The record's time is mapped onto the transcript's timeline using the start time
carried in the call identifier (1 s resolution): it is good for ordering, not for
measuring.

    python -m validation.interruption_agreements
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote"
CAMPAIGN_OF: Final[dict[str, str]] = {
    "1": "interruptions_full_v2", "2": "interruptions_full_v2", "3": "interruptions_poc3_pcmu",
}
READING: Final[Path] = Path(__file__).with_name("interruption_agreements_reading.csv")
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data" / "interruption_campaign_v3" / "interruption_agreements.csv"
KINDS: Final[tuple[str, ...]] = ("acepto", "apoyo", "linea_incompleta", "anticipado", "sin_acuerdo")
LOCAL_TIME: Final[timezone] = timezone(timedelta(hours=-6))
COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "estado", "resultado", "monto_final", "motivo",
    "record_second", "carrier_second_to_last", "last_carrier",
    "call_backchannels", "kind", "note",
]


def _record(clave: str, poc: str) -> dict[str, Any]:
    folder = LOTE / CAMPAIGN_OF[poc]
    summary = json.loads((folder / "summaries" / f"{clave}.json").read_text(encoding="utf-8"))
    art = summary["artifacts"]
    ruta = art.get("pipeline_json") or art.get("worker_json") or art.get("bridge_json")
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    event_log = json.loads(Path(art["event_log"]).read_text(encoding="utf-8"))
    backchannels = [t["event"].get("text", "") for t in event_log["turns"]
              if (t.get("event") or {}).get("tipo") == "apoyo"]
    return {"data": data, "backchannels": backchannels}


def row_of(poc: str, row: dict[str, Any], reading: dict[str, dict[str, str]]) -> dict[str, Any]:
    reg = _record(row["clave"], poc)
    data = reg["data"]
    res = (data.get("summary") or {}).get("resultado") or {}
    t0 = datetime.strptime(data["call_sid"][-15:], "%Y%m%d-%H%M%S").replace(tzinfo=LOCAL_TIME)
    second = (datetime.fromisoformat(res["timestamp"]) - t0).total_seconds() if res.get("timestamp") else None
    priors = [t["text"] for t in data.get("transcript") or []
               if t["role"] == "carrier" and (second is None or t["t"] <= second)]
    read_value = reading.get(row["clave"], {})
    return {
        "poc": poc, "clave": row["clave"], "scenario": row["scenario"], "repetition": row.get("repetition"),
        "estado": row.get("estado"), "resultado": res.get("resultado"), "monto_final": res.get("monto_final"),
        "motivo": res.get("motivo"), "record_second": None if second is None else round(second, 1),
        "carrier_second_to_last": priors[-2] if len(priors) > 1 else "",
        "last_carrier": priors[-1] if priors else "",
        "call_backchannels": " | ".join(reg["backchannels"]),
        "kind": read_value.get("kind", ""), "note": read_value.get("note", ""),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args(argv)
    with READING.open(encoding="utf-8") as fh:
        reading = {r["clave"]: r for r in csv.DictReader(fh)}
    bad = {r["clave"]: r["kind"] for r in reading.values() if r["kind"] not in KINDS}
    if bad:
        raise SystemExit(f"unknown classes in the reading: {bad}")
    rows: list[dict[str, Any]] = []
    for poc in "123":
        progress = (LOTE / CAMPAIGN_OF[poc] / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        for linea in progress:
            f = json.loads(linea)
            if str(f["poc"]) == poc:
                rows.append(row_of(poc, f, reading))
    unread = [f["clave"] for f in rows if not f["kind"]]
    if unread:
        raise SystemExit(f"calls without a reading: {unread}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for poc in "123":
        of = [f for f in rows if f["poc"] == poc]
        print(f"PoC{poc}: {len(of)} calls · " + " · ".join(
            f"{c} {sum(f['kind'] == c for f in of)}" for c in KINDS))
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
