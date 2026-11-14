"""The campaign's speech overlaps, one by one: who caused each one.

The report's count (`shared/metrics/overlap.py`) counts an overlap as "the agent
came in over" when the agent's streak starts after the harness's. That does not
distinguish two situations:

- **The harness spoke during a pause of the agent.** The agent pauses midway
  through its response, the harness takes it as end of turn (2 s of silence) and
  plays the next line, and the agent resumes right as the line starts. The agent
  is not reacting to the line: it is resuming its own response.
- **The agent started midway through the line.** It began speaking with the line
  already under way: a premature end of turn.

They are separated by how long after the start of the line the agent came in. No
response in the campaign arrived earlier than 1.1 s after the END of the line,
so an entry within the first second of the line cannot be a response to it: it
is the agent resuming. It also records how long the agent had been silent and
whether its previous streak was part of the response to the preceding line.

Reads the audit WAVs of the analyzed calls (costs nothing) and writes to
`docs/report/data/`: `overlaps.csv` (one row per overlap) and
`overlaps_summary.txt`.

    uv run --project poc3_gpt_live python -m validation.classified_overlaps
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

from shared.audio import read_wav_stereo
from shared.metrics.overlap import MIN_OVERLAP_S, speech_streaks, overlaps_between

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote"
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"

# An agent entry within this time from the start of the line cannot be a
# response to the line: it is the agent resuming its own response.
RESUME_S: Final[float] = 1.0

COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "overlap_start_s", "duration_ms",
    "entered_after_line_start_ms", "agent_silent_before_ms", "kind",
]


def analyzed() -> dict[str, dict]:
    with (DATA / "populations.csv").open(encoding="utf-8") as fh:
        return {r["clave"]: r for r in csv.DictReader(fh) if r["estado"] == "ok" and r["in_the_pool"] == "1"}


def call_overlaps(wav: Path) -> list[dict]:
    harness_channel, agent_channel, sample_rate = read_wav_stereo(wav)
    of_harness = speech_streaks(harness_channel, sample_rate)
    of_agent = speech_streaks(agent_channel, sample_rate)
    output = []
    for s in overlaps_between(of_harness, of_agent, min_s=MIN_OVERLAP_S):
        if s.who_entered != "agent":
            continue
        linea = max((r for r in of_harness if r.start <= s.start), key=lambda r: r.start)
        agent = min((r for r in of_agent if r.start >= linea.start and r.end > s.start),
                     key=lambda r: r.start)
        priors = [r for r in of_agent if r.end <= agent.start]
        silent = (agent.start - priors[-1].end) if priors else None
        entered = agent.start - linea.start
        output.append({
            "overlap_start_s": round(s.start, 2),
            "duration_ms": round((s.end - s.start) * 1000.0),
            "entered_after_line_start_ms": round(entered * 1000.0),
            "agent_silent_before_ms": None if silent is None else round(silent * 1000.0),
            "kind": "retomo_en_su_pausa" if entered <= RESUME_S else "arranco_a_mitad_de_la_linea",
        })
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE)
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    valid_ones = analyzed()
    rows = []
    for campana in ("campana200", "campana_poc3_pcmu", "campana_poc3_pcmu_reintentos"):
        folder = args.lote / campana
        labels = {
            e["clave"]: e.get("label")
            for e in map(json.loads, (folder / "progress.jsonl").read_text(encoding="utf-8").splitlines())
        } if (folder / "progress.jsonl").is_file() else {}
        for clave, r in valid_ones.items():
            if r["campana"] != campana:
                continue
            # PoC3 repetitions carry `-bis` in the analysis key.
            file = clave.removesuffix("-bis")
            ruta = folder / "summaries" / f"{file}.json"
            summary = json.loads(ruta.read_text(encoding="utf-8"))
            art = summary.get("artifacts") or {}
            wav = art.get("auditoria_wav")
            if not wav and art.get("event_log"):
                # PoC3 does not name its WAV in the summary: it sits next to the log.
                wav = str(Path(art["event_log"]).with_name(f"auditoria_{summary.get('label') or labels.get(file)}.wav"))
            if not wav or not Path(wav).is_file():
                raise SystemExit(f"{clave}: audit WAV missing")
            for s in call_overlaps(Path(wav)):
                rows.append({"poc": r["poc"], "clave": clave, "scenario": r["scenario"],
                              "repetition": r["repetition"], **s})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "overlaps.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in rows:
            w.writerow({k: ("" if f.get(k) is None else f[k]) for k in COLUMNS})
    text_lines = ["poc  solapes  retomo_en_su_pausa  arranco_a_mitad_de_la_linea"]
    for poc in ("1", "2", "3"):
        of = [f for f in rows if f["poc"] == poc]
        text_lines.append(f"{poc}    {len(of):>3}      {sum(f['kind'] == 'retomo_en_su_pausa' for f in of):>3}"
                      f"                 {sum(f['kind'] == 'arranco_a_mitad_de_la_linea' for f in of):>3}")
    (args.output / "overlaps_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
