"""The fastest responses of each PoC, cut and transcribed for review.

PoC3 has a tail of very low latencies (down to 0.2 s) that PoC1 and PoC2 do not
have. This script changes no figure: for each response with an observed latency
below `--umbral` (and, in PoC3, each turn recovered by the direct method), it
locates in the audit recording the carrier's end of speech and the onset measured
by the report, and leaves:

- `audio_files/`: the stereo segment (left the carrier, right the agent), from
  slightly before the carrier's line until 4 s after the onset, for listening.
- `low_latencies.csv`: one row per response with what the agent said in its
  first seconds (Deepgram nova-3 on the agent channel), whether it was already
  speaking when the carrier's line started and how long it had been silent, and
  blank columns for classification.

The end of speech is mapped onto the WAV's timeline with the external
instrument's anchor (`shared.metrics.external_latency_poc3.anchor_on`), which
correlates the carrier clips against channel 0; it works the same for all three
PoCs because all three record with the same harness.

    poc1_pipecat/.venv/Scripts/python -m validation.low_latencies
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.metrics import external_latency_poc3 as ext3
from shared.metrics.campaigns import CAMPAIGN_POC1_POC2, CAMPAIGN_POC3, CAMPAIGN_POC3_RETRIES, RETRY_SUFFIX
from shared.metrics.export_for_report import CLIPS_DEFAULT, LOTE_DEFAULT, _a_8khz, read_summary
from validation.resumption_sheet_audio import transcribe, wav_mono
from validation.turns_without_latency_poc3 import _write_wav, _manifest

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
DEFAULT_OUTPUT: Final[Path] = RAIZ / "diagnostics" / "low_latencies"
UMBRAL_MS: Final[float] = 1500.0
BEFORE_CLIP_S: Final[float] = 1.0
AFTER_THE_ONSET_S: Final[float] = 4.0

COLUMNS: Final[list[str]] = [
    "file", "poc", "clave", "turn", "clip_id", "origin", "observed_latency_ms",
    "first_sound_latency_ms", "carrier_line", "agent_was_speaking_at_line_start",
    "agent_silence_before_onset_ms", "onset_second_in_audio",
    "agent_speech_during_line_ms", "said_during_the_line", "said_at_start", "kind", "note",
]


def _folder(lote: Path, poc: str, clave: str) -> tuple[Path, str]:
    if poc != "3":
        return lote / CAMPAIGN_POC1_POC2, clave
    if clave.endswith(RETRY_SUFFIX):
        return lote / CAMPAIGN_POC3_RETRIES, clave[: -len(RETRY_SUFFIX)]
    return lote / CAMPAIGN_POC3, clave


def _paths(folder: Path, clave: str) -> tuple[Path, Path] | None:
    summary = read_summary(folder, clave) or {}
    art = summary.get("artifacts") or {}
    sid = summary.get("label")
    if not sid or not art.get("event_log"):
        return None
    harness = Path(art["event_log"]).parent
    wav = Path(art["auditoria_wav"]) if art.get("auditoria_wav") else harness / f"auditoria_{sid}.wav"
    zeros = harness / f"zeros_{sid}.jsonl"
    return (wav, zeros) if wav.is_file() and zeros.is_file() else None


def selection(rows: list[dict[str, str]], umbral: float) -> list[dict[str, str]]:
    """Responses below the threshold, plus all recovered PoC3 turns."""
    return [
        f for f in rows
        if f["observed_latency_ms"]
        and (float(f["observed_latency_ms"]) < umbral or (f["poc"] == "3" and f["origin"] == "recuperado"))
    ]


def locate(wav: Path, zeros: Path, clips: Path, turn: str, latency_ms: float) -> dict[str, Any] | None:
    """The measured onset, in WAV seconds, and what the agent was doing before it."""
    marks = sorted(ext3.marks_de(zeros), key=lambda m: m["measured_end_of_speech"])
    audio, sample_rate = ext3.leer_wav(wav)
    a8, t8 = _a_8khz(audio, sample_rate)
    anchor = ext3.anchor_on(marks, a8[:, 0], t8, clips_path=clips)
    mark = next((m for m in marks if str(m["turn"]) == turn), None)
    if anchor is None or mark is None:
        return None
    shift = anchor.displacement
    end = float(mark["measured_end_of_speech"]) - shift
    line_start = float(mark.get("start_of_clip", mark["measured_end_of_speech"])) - shift
    onset = end + latency_ms / 1000.0
    streaks = sorted(ext3.speech_streaks(ext3.envelope(a8[:, 1], t8)))
    priors = [r for r in streaks if r[1] <= onset - 0.05]
    was_speaking = any(r[0] < line_start < r[1] for r in streaks)
    silence = (onset - priors[-1][1]) * 1000.0 if priors else None
    # Agent speech between the start of the carrier's line and the measured onset.
    durante = sum(max(0.0, min(r[1], onset) - max(r[0], line_start)) for r in streaks)
    k = marks.index(mark)
    following = float(marks[k + 1]["start_of_clip"]) - shift if k + 1 < len(marks) else len(audio) / sample_rate
    return {
        "audio": audio, "sample_rate": sample_rate, "arranque": onset, "line_start": line_start,
        "until": min(onset + AFTER_THE_ONSET_S, max(following, onset + 0.5), len(audio) / sample_rate),
        "was_speaking": int(was_speaking), "silence": None if silence is None else round(silence),
        "durante": round(durante * 1000.0),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--clips", type=Path, default=CLIPS_DEFAULT)
    p.add_argument("--lines", type=Path, default=DATA / "lines.csv")
    p.add_argument("--umbral", type=float, default=UMBRAL_MS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--no-transcribe", action="store_true")
    args = p.parse_args(argv)

    key = ""
    if not args.no_transcribe:
        from dotenv import load_dotenv

        load_dotenv(RAIZ / ".env", override=False)
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise SystemExit("DEEPGRAM_API_KEY missing from .env")

    with args.lines.open(encoding="utf-8") as fh:
        rows = selection(list(csv.DictReader(fh)), args.umbral)
    rows.sort(key=lambda f: (f["poc"], float(f["observed_latency_ms"])))
    texts = _manifest(args.clips)
    output: list[dict[str, Any]] = []
    for n, f in enumerate(rows, start=1):
        folder, clave = _folder(args.lote, f["poc"], f["clave"])
        paths = _paths(folder, clave)
        place = locate(*paths, args.clips, f["turn"], float(f["observed_latency_ms"])) if paths else None
        row = {c: "" for c in COLUMNS} | {
            "poc": f["poc"], "clave": f["clave"], "turn": f["turn"], "clip_id": f["clip_id"],
            "origin": f["origin"], "observed_latency_ms": f["observed_latency_ms"],
            "first_sound_latency_ms": f["first_sound_latency_ms"],
            "carrier_line": texts.get(f["clip_id"], ""),
        }
        if place is None:
            row["note"] = "no se pudo ubicar en la grabación"
            output.append(row)
            continue
        audio, sample_rate = place["audio"], place["sample_rate"]
        since = max(0, int((place["line_start"] - BEFORE_CLIP_S) * sample_rate))
        until = int(place["until"] * sample_rate)
        name = f"{n:03d}_poc{f['poc']}_{f['clave']}_{f['turn']}_{round(float(f['observed_latency_ms']))}ms.wav"
        _write_wav(args.output / "audio_files" / name, audio[since:until], sample_rate)
        if key:
            a = int(max(0.0, place["arranque"] - 0.1) * sample_rate)
            row["said_at_start"] = transcribe(wav_mono(audio[a:until, 1], sample_rate), key)
            if place["durante"] >= 200:
                i = int(max(0.0, place["line_start"]) * sample_rate)
                row["said_during_the_line"] = transcribe(wav_mono(audio[i:a, 1], sample_rate), key)
        row |= {
            "file": name,
            "agent_was_speaking_at_line_start": place["was_speaking"],
            "agent_speech_during_line_ms": place["durante"],
            "agent_silence_before_onset_ms": place["silence"],
            "onset_second_in_audio": round(place["arranque"] - since / sample_rate, 2),
        }
        output.append(row)
        print(f"  {n}/{len(rows)} {f['clave']} {f['turn']}: {row['said_at_start'][:70]}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "low_latencies.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(output)
    print(f"{len(output)} responses in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
