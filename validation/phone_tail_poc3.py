"""Two effects of how the harness heard PoC3 in the campaign, measured without running anything.

**1. A phone plays in sequence; the harness timestamps on arrival.** The harness
recording places each GPT-Live speech block at the instant it arrived. A phone
cannot play ahead: if the API stalls and then sends several seconds of audio at
once, the phone plays them one after another and what follows sounds late. Here
that queued player is simulated with the arrival times and duration of each block
in the census (`output_audio.delta`), and the script records how far behind the
arrival the onset of each response would sound. It favors PoC3 in the published
figures.

**2. Streaks the harness discarded live that last longer in the recording.**
Live, the harness saw each ~100 ms block as a single frame; a short streak
("Claro, te entiendo") could look shorter than 0.6 s to it and be discarded, even
though it lasts longer in the recording. That streak was already the response. It
goes against PoC3.

Reads the artifacts of the analyzed calls (costs nothing) and writes to
`docs/report/data/`: `phone_tail_poc3.csv` (one row per turn with a value) and
`phone_tail_poc3_summary.txt`.

    uv run --project poc3_gpt_live --with scipy python -m validation.phone_tail_poc3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

import numpy as np

from shared.carrier_simulator.player import MIN_SUBSTANTIVE_SPEECH_S
from shared.metrics import external_latency_poc3 as ext3
from shared.metrics.export_for_report import recovered_turns

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote" / "campana_poc3_pcmu"
HARNESS: Final[Path] = RAIZ / "poc3_gpt_live" / "diagnostics" / "harness"
RECORDINGS: Final[Path] = RAIZ / "poc3_gpt_live" / "recordings"
CLIPS: Final[Path] = RAIZ / "shared" / "test_scenarios" / "clips"
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"

# A streak discarded live is recognized in the recording if it starts within
# this distance of where the harness timestamped it.
DISCARD_TOLERANCE_S: Final[float] = 0.10
# The first sound may fall up to three windows before the end of speech.
SLACK_S: Final[float] = 0.06

COLUMNS: Final[list[str]] = [
    "clave", "turn", "origin", "latency_ms", "lead_in_row_ms", "latency_without_discard_ms",
]


def queued_player(census: Path, anchor: float) -> tuple[np.ndarray, np.ndarray]:
    """`(arrival, lead)` of each block: how long after arriving a phone would play it."""
    from shared.metrics.census_poc3 import events

    arrivals, durations = [], []
    for e in events(census):
        if e.get("type") != "output_audio.delta":
            continue
        ev = e["event"]
        arrivals.append(anchor + e["t_local_seg"])
        # The alpha timestamped each block (`start_ms`/`end_ms`); the public API
        # does not, and the duration comes from its bytes (`census_poc3`).
        if "audio_ms" in ev:
            durations.append(ev["audio_ms"] / 1000.0)
        else:
            durations.append((ev["end_ms"] - ev["start_ms"]) / 1000.0)
    t = np.array(arrivals)
    playhead = None
    output = np.empty_like(t)
    for k, (arrives, dura) in enumerate(zip(t, durations)):
        playhead = arrives if playhead is None else max(arrives, playhead)
        output[k] = playhead - arrives
        playhead += dura
    return t, output


def call(row: dict) -> list[dict]:
    sid = row["label"]
    paths = dict(
        zeros_path=HARNESS / f"zeros_{sid}.jsonl",
        bridge_json_path=RECORDINGS / f"call_{sid}.json",
        wav_path=HARNESS / f"auditoria_{sid}.wav",
        path_of_log=HARNESS / f"event_log_{sid}.json",
    )
    anchor_wall = (json.loads(paths["bridge_json_path"].read_text(encoding="utf-8")).get("clock") or {}).get(
        "anchor_wall_s")
    turns, anchor, _ = ext3.call_turns(
        call_sid=sid, scenario=row["scenario"], clips_path=CLIPS, **paths)
    if anchor is None or anchor_wall is None:
        return []
    marks = ext3.marks_de(paths["zeros_path"])
    recovered_items = recovered_turns(
        marks, paths["wav_path"], CLIPS, measured_items={str(t.turn) for t in turns},
        event_log=paths["path_of_log"])
    arrivals, lead = queued_player(RECORDINGS / f"census_{sid}.jsonl", float(anchor_wall))

    # The recording's streaks and what the harness discarded live as too short.
    d = anchor.displacement
    audio, sample_rate = ext3.leer_wav(paths["wav_path"])
    streaks = [(i + d, f + d) for i, f in ext3.speech_streaks(ext3.envelope(audio[:, 1], sample_rate))]
    event_log = json.loads(paths["path_of_log"].read_text(encoding="utf-8"))
    descartes = [
        float(x["start"]) + d
        for t in event_log.get("turns", []) for x in (t.get("discarded_fragments") or [])
        if str(x.get("motivo", "")).startswith("m")  # "menos de 0.6 s" (shorter than 0.6 s): by duration, not by text
    ]
    output = []
    for t, origin in [*((t, "instrumento_externo") for t in turns), *((t, "recuperado") for t in recovered_items)]:
        lat = t.line_latency_ms
        k = int(np.searchsorted(arrivals, t.first_speech_heard + 0.005)) - 1
        in_row = float(lead[min(max(k, 0), len(lead) - 1)] * 1000.0) if len(lead) else None
        no_discard = lat
        for start in descartes:
            long_ones = [r for r in streaks if abs(r[0] - start) <= DISCARD_TOLERANCE_S
                      and (r[1] - r[0]) >= MIN_SUBSTANTIVE_SPEECH_S]
            if long_ones and t.end_of_speech - SLACK_S <= long_ones[0][0] < t.first_speech_heard:
                no_discard = min(no_discard, round((long_ones[0][0] - t.end_of_speech) * 1000.0, 1))
        output.append({
            "clave": row["clave"], "turn": str(t.turn), "origin": origin, "latency_ms": lat,
            "lead_in_row_ms": None if in_row is None else round(in_row, 1),
            "latency_without_discard_ms": no_discard,
        })
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    with (DATA / "populations.csv").open(encoding="utf-8") as fh:
        valid_ones = {r["clave"] for r in csv.DictReader(fh)
                   if r["campana"].startswith("campana_poc3_pcmu") and r["estado"] == "ok" and r["in_the_pool"] == "1"}
    from shared.metrics.campaigns import poc3_for_analysis

    # The first attempt, or its repetition if it failed (key `-bis`).
    progress = [f for _, f in poc3_for_analysis(LOTE.parent)]
    rows = [f for r in progress if r["clave"] in valid_ones for f in call(r)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "phone_tail_poc3.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in rows:
            w.writerow({k: ("" if f.get(k) is None else f[k]) for k in COLUMNS})
    lat = np.array([f["latency_ms"] for f in rows])
    row_ = np.array([f["latency_ms"] + (f["lead_in_row_ms"] or 0.0) for f in rows])
    sin = np.array([f["latency_without_discard_ms"] for f in rows])
    lead_values = np.array([f["lead_in_row_ms"] or 0.0 for f in rows])
    text_lines = [
        f"turns: {len(rows)}",
        f"mediana publicada {np.median(lat):.1f} ms · con reproductor en fila {np.median(row_):.1f} ms",
        f"p95 publicado {np.percentile(lat, 95):.1f} ms · con reproductor en fila {np.percentile(row_, 95):.1f} ms",
        f"adelanto en fila: p50 {np.median(lead_values):.0f} ms, p95 {np.percentile(lead_values, 95):.0f} ms, "
        f"máximo {lead_values.max():.0f} ms, turnos con más de 500 ms: {int((lead_values > 500).sum())}",
        f"turnos con un arranque descartado en vivo que en la grabación dura 0.6 s o más: "
        f"{int((sin < lat).sum())} · mediana sin ese descarte {np.median(sin):.1f} ms",
    ]
    (args.output / "phone_tail_poc3_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
