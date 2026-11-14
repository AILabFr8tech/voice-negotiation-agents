"""From the first audible sound to full speech, in each PoC's real responses.

The harness timestamps the agent's response at the first 20 ms segment whose RMS
reaches 200 int16 counts (−44 dBFS). The validation with the fake agent showed
that a "full speech" criterion (the first 10 ms window within 25 dB of the peak,
the same one used to timestamp the clips' end of speech) marks the onset a few
tens of milliseconds later, and that this distance depends on how each voice
starts. This script measures it in the campaign's audit recordings, for each PoC.

Method, on the agent channel (channel 1) of each recording, at its native rate:

1. 20 ms segments with RMS ≥ 200 (the harness rule), grouped into streaks with
   the same 0.3 s maximum pause the harness uses.
2. Streaks of at least 0.6 s (the harness's minimum substantive speech) are
   kept, and the first one of each call, which is the greeting, is left out.
3. In each streak: the streak's peak, threshold = peak − 25 dB (with the
   −55 dBFS floor of `shared/audio/level.py`), and the first 10 ms window that
   reaches it. The distance is that window minus the start of the streak.

No clock alignment is needed: both marks come from the same audio.

Writes `speech_onset.csv` (one row per response) and
`speech_onset.txt` (summary per PoC) to `--output`.

    uv run --project poc3_gpt_live --with scipy python -m validation.speech_onset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio.level import dbfs_pico, levels_per_window, speech_threshold
from shared.carrier_simulator.observed import STREAK_HANGOVER_S, RMS_WITH_SPEECH
from shared.carrier_simulator.player import MIN_SUBSTANTIVE_SPEECH_S
from shared.metrics import external_latency_poc3 as ext3
from shared.metrics.export_for_report import (
    LOTE_DEFAULT,
    DEFAULT_OUTPUT,
    read_progress,
    read_summary,
    campaign200_pairs,
)

COLUMNS: Final[list[str]] = [
    "poc", "clave", "start_s", "duration_s", "pico_dbfs", "distance_ms",
]


def harness_streaks(canal: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Streaks `(start_sample, end_sample)` under the harness's energy rule."""
    per_frame = sample_rate * 20 // 1000
    n = len(canal) // per_frame
    if n == 0:
        return []
    frames = canal[: n * per_frame].astype(np.float64).reshape(n, per_frame)
    actives = np.flatnonzero(np.sqrt(np.mean(frames**2, axis=1)) >= RMS_WITH_SPEECH)
    streaks: list[tuple[int, int]] = []
    pause = STREAK_HANGOVER_S / 0.02
    for i in actives:
        if streaks and i - streaks[-1][1] <= pause:
            streaks[-1] = (streaks[-1][0], int(i))
        else:
            streaks.append((int(i), int(i)))
    return [(a * per_frame, (b + 1) * per_frame) for a, b in streaks]


def distance_to_full_speech(segment: np.ndarray, sample_rate: int) -> tuple[float, float] | None:
    """`(distance_ms, pico_dbfs)` of a streak, or `None` if no window reaches the threshold."""
    data = np.asarray(segment, dtype=np.int16)
    umbral = speech_threshold(data)
    levels, per_window = levels_per_window(data, sample_rate)
    arriba = np.flatnonzero(levels >= umbral)
    if arriba.size == 0:
        return None
    return round(float(arriba[0]) * per_window / sample_rate * 1000.0, 1), round(dbfs_pico(data), 2)


def recording_responses(wav: Path, poc: int, clave: str) -> list[dict[str, Any]]:
    audio, sample_rate = ext3.leer_wav(wav)
    canal = audio[:, 1]
    rows = []
    for k, (a, b) in enumerate(harness_streaks(canal, sample_rate)):
        duration = (b - a) / sample_rate
        if k == 0 or duration < MIN_SUBSTANTIVE_SPEECH_S:
            continue
        measured_value = distance_to_full_speech(canal[a:b], sample_rate)
        if measured_value is None:
            continue
        rows.append({
            "poc": poc, "clave": clave, "start_s": round(a / sample_rate, 3),
            "duration_s": round(duration, 3), "pico_dbfs": measured_value[1], "distance_ms": measured_value[0],
        })
    return rows


def calls(lote: Path) -> list[tuple[Path, dict[str, Any]]]:
    c200 = lote / "campana200"
    c3 = lote / "campana_poc3_pcmu"
    output = [(c200, f) for f in campaign200_pairs(read_progress(c200))]
    from shared.metrics.campaigns import poc3_for_analysis

    # The first attempt, or its repetition if it failed.
    output += [
        (c, {**f, "clave": f["batch_key"]}) for c, f in poc3_for_analysis(lote)
        if f.get("estado") == "ok"
    ]
    return output


def summary(rows: list[dict[str, Any]], no_recording: dict[int, int]) -> str:
    text_lines = [
        "Distancia del primer tramo audible (RMS ≥ 200, 20 ms) a la voz plena "
        "(pico − 25 dB, 10 ms), respuestas del agente",
        "",
        f"{'PoC':<5}{'calls':>9}{'replies':>11}{'p25':>8}{'median_value':>9}{'p75':>8}{'p95':>8}{'mean':>8}",
    ]
    medians = {}
    for poc in (1, 2, 3):
        d = np.array([f["distance_ms"] for f in rows if f["poc"] == poc], dtype=float)
        n_calls = len({f["clave"] for f in rows if f["poc"] == poc})
        if d.size == 0:
            text_lines.append(f"PoC{poc:<2}{n_calls:>9}{0:>11}")
            continue
        p25, p50, p75, p95 = np.percentile(d, [25, 50, 75, 95])
        medians[poc] = p50
        text_lines.append(
            f"PoC{poc:<2}{n_calls:>9}{d.size:>11}{p25:>8.1f}{p50:>9.1f}{p75:>8.1f}{p95:>8.1f}{d.mean():>8.1f}"
        )
    if no_recording:
        text_lines.append(f"\nllamadas sin grabación de auditoría: {no_recording}")
    if 1 in medians and 3 in medians:
        text_lines.append(
            f"\nCon el criterio de voz plena, la diferencia PoC1 − PoC3 se movería del orden de "
            f"{medians[1] - medians[3]:+.1f} ms (mediana PoC1 − mediana PoC3)."
        )
    if 1 in medians and 2 in medians:
        text_lines.append(
            f"La diferencia PoC1 − PoC2 se movería del orden de {medians[1] - medians[2]:+.1f} ms."
        )
    return "\n".join(text_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    rows: list[dict[str, Any]] = []
    no_recording: dict[int, int] = {}
    items = calls(args.lote)
    for k, (folder, row) in enumerate(items, start=1):
        if k % 20 == 0 or k == len(items):
            print(f"  {k}/{len(items)} calls", flush=True)
        poc, clave = int(row["poc"]), str(row["clave"])
        art = (read_summary(folder, clave) or {}).get("artifacts") or {}
        wav = Path(art["auditoria_wav"]) if art.get("auditoria_wav") else None
        if wav is None or not wav.is_file():
            no_recording[poc] = no_recording.get(poc, 0) + 1
            continue
        rows += recording_responses(wav, poc, clave)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "speech_onset.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows, no_recording)
    (args.output / "speech_onset.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
