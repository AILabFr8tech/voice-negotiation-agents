"""Which amount the agent understood in scenario 5's self-correction (R6).

Line E5-T4 says "Yo andaba pensando en dieciocho mil… no, dieciocho mil
quinientos" ("I was thinking eighteen thousand… no, eighteen thousand five
hundred"). It is the only line in the script where misunderstanding leaves a
trace that can be read in the response:

- **Understood 18,500** if it names or offers it. No agent reaches 18,500 on its
  own: its steps are 500 to 1,000 pesos from 16,000, and 18,500 exceeds the
  target price. Offering it means accepting what it heard.
- **Understood 18,000** if it attributes that amount to the carrier ("entiendo
  que buscas 18,000", "I understand you're looking for 18,000").
- **Not visible** if it only counteroffers (17,000, 17,500, or 18,000 as its own
  offer) or states no amount.

Why comparing the `proponer_precio` argument with the carrier's amount across all
lines does not work: that argument is the agent's offer, not what it heard. When
the carrier asks for 18,000 and the agent proposes 17,500, the agent is
negotiating, not misunderstanding; and in scenario 3 the carrier asks for
28,000, which `proponer_precio` always rejects. That match measures how much the
agent concedes.

The reading of each response (corregido / retractado / no_se_ve) is a
classification, and it lives separately in `amount_comprehension_reading.csv`,
together with who made it. Everything else comes from the harness logs: the line
that was played, what the PoC transcribed and what the agent said itself.

Reads the logs of the analyzed calls (costs nothing) and writes to
`docs/report/data/`: `amount_comprehension.csv` (one row per response) and
`amount_comprehension_summary.txt`.

    uv run --project poc3_gpt_live python -m validation.amount_comprehension
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Final

from shared.metrics.text import extract_amounts, fold_accents

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote"
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
READING: Final[Path] = Path(__file__).with_name("amount_comprehension_reading.csv")
FOLDER: Final[dict[int, str]] = {1: "poc1_pipecat", 2: "poc2_livekit", 3: "poc3_gpt_live"}
CAMPAIGNS: Final[tuple[str, ...]] = ("campana200", "campana_poc3_pcmu", "campana_poc3_pcmu_reintentos")

CLIP: Final[str] = "e5_t4"
RETRACTADO: Final[float] = 18_000.0
CORRECTED: Final[float] = 18_500.0

COLUMNS: Final[list[str]] = [
    "poc", "clave", "transcript", "transcript_with_corrected", "response",
    "response_amounts", "audio_track", "reading", "who_read",
]


def with_the_corrected(transcript: str) -> bool:
    """The transcript contains the "quinientos". In PoC3 it arrives in fragments ("quin ientos")."""
    joined = fold_accents(transcript).lower().replace(" ", "")
    return "quinientos" in joined or CORRECTED in extract_amounts(transcript)


def audio_track(amounts: list[float]) -> str:
    """What can be seen without reading the sentence; the manual reading decides."""
    if CORRECTED in amounts:
        return "menciona_el_corregido"
    if RETRACTADO in amounts:
        return "menciona_el_retractado"
    return "otros_montos" if amounts else "sin_monto"


def replies() -> list[dict]:
    with (DATA / "populations.csv").open(encoding="utf-8") as fh:
        valid_ones = {r["clave"] for r in csv.DictReader(fh) if r["estado"] == "ok" and r["in_the_pool"] == "1"}
    reading: dict[str, dict] = {}
    if READING.is_file():
        with READING.open(encoding="utf-8") as fh:
            reading = {r["clave"]: r for r in csv.DictReader(fh)}
    output = []
    for campana in CAMPAIGNS:
        for linea in (LOTE / campana / "progress.jsonl").read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            p = json.loads(linea)
            if campana.endswith("_reintentos"):
                p["clave"] += "-bis"
            if p["clave"] not in valid_ones:
                continue
            ruta = RAIZ / FOLDER[p["poc"]] / "diagnostics" / "harness" / f"event_log_{p['label']}.json"
            for t in json.loads(ruta.read_text(encoding="utf-8"))["turns"]:
                if t["clip_id"].split("_alt")[0] != CLIP:
                    continue
                amounts = [m for m in extract_amounts(t.get("agent") or "") if m >= 5_000]
                read_value = reading.get(p["clave"], {})
                output.append({
                    "poc": p["poc"], "clave": p["clave"], "transcript": t.get("stt") or "",
                    "transcript_with_corrected": int(with_the_corrected(t.get("stt") or "")),
                    "response": t.get("agent") or "",
                    "response_amounts": " ".join(f"{m:.0f}" for m in amounts),
                    "audio_track": audio_track(amounts), "reading": read_value.get("reading", ""),
                    "who_read": read_value.get("who", ""),
                })
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    rows = replies()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "amount_comprehension.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text_lines = []
    for poc in (1, 2, 3):
        of = [f for f in rows if f["poc"] == poc]
        reading_item = Counter(f["reading"] or "unread" for f in of)
        stt = sum(f["transcript_with_corrected"] for f in of)
        bad_with_bad_stt = sum(f["reading"] == "retractado" and not f["transcript_with_corrected"] for f in of)
        text_lines.append(
            f"PoC{poc}: {len(of)} respuestas · transcripción con el corregido {stt} · "
            f"lectura {dict(reading_item)} · retractado con la transcripción sin el corregido {bad_with_bad_stt}"
        )
    (args.output / "amount_comprehension_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
