"""Audio cuts of each interruption event, for review by ear.

Rates nothing and produces no report figures: from each call's audit WAV it cuts
a short segment around each event (control, backchannel and interruption) and
writes an index with what the harness said and what the audio measures. It is
used to check, before spending calls, that the scenario does what it claims: that
the event lands mid-sentence, that the clip is heard, and that the harness's
"stop" corresponds to something audible.

Each cut starts `ANTES_SEG` before the event's reference (the speech onset of the
clip; for the control, the trigger), so **the event always lands at second 2 of
the cut**. Left channel = harness (carrier), right = agent, same as the audit
WAV.

The index marks `discrepa = 1` when the harness reported "stopped" and the agent
channel has no 0.7 s silence (the measure from `interruption_pauses`), or the
other way round. In PoC3 this happens with 0.6-0.7 s pauses: the harness
timestamps the agent's audio on arrival, in ~100 ms blocks, and the pause looks
one block longer to it than it sounds.

    uv run --project poc3_gpt_live python -m validation.interruption_cuts \\
        --campana diagnostico/lote/interrupciones --campana diagnostico/lote/interrupciones_poc3 \\
        --include-failed --output diagnostico/recortes_interrupciones

The cuts are audio and are not committed to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio.wav import write_wav_stereo, read_wav_stereo
from shared.carrier_simulator.interruptions import NO_REPLY_WAIT_S
from validation.interruption_pauses import KINDS, measure_pause, frames_with_speech

ANTES_SEG: Final[float] = 2.0
AFTER_S: Final[float] = 6.0

COLUMNS: Final[list[str]] = [
    "file", "poc", "estado", "clave", "scenario", "turn", "tipo", "clip_id", "text",
    "second_in_call", "harness_stopped", "stop_time_ms",
    "pause_audio_ms", "discrepa",
]


def trim(canal: np.ndarray, sample_rate: int, *, since: float, until: float) -> np.ndarray:
    """The segment `[since, until)` in seconds, zero-filled where it falls outside the file."""
    a, b = int(round(since * sample_rate)), int(round(until * sample_rate))
    output = np.zeros(max(0, b - a), dtype=np.int16)
    beg, end = max(a, 0), min(b, len(canal))
    if end > beg:
        output[beg - a:end - a] = canal[beg:end]
    return output


def cuts_of(campana: Path, output: Path, *, include_failed: bool) -> list[dict[str, Any]]:
    valid_items = ("ok", "roja") if include_failed else ("ok",)
    rows: list[dict[str, Any]] = []
    for linea in (campana / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        p = json.loads(linea)
        if p.get("estado") not in valid_items:
            continue
        ruta = campana / "summaries" / f"{p['clave']}.json"
        if not ruta.is_file():
            continue
        summary = json.loads(ruta.read_text(encoding="utf-8"))
        wav = (summary.get("artifacts") or {}).get("auditoria_wav")
        events = [e for e in summary.get("events") or []
                   if e.get("tipo") in KINDS and e.get("fired") and e.get("t_speech_onset") is not None]
        if not events or not wav or not Path(wav).is_file():
            continue
        harness, agent, sample_rate = read_wav_stereo(wav)
        with_voice = frames_with_speech(agent, sample_rate)
        folder = output / f"poc{p.get('poc')}"
        folder.mkdir(parents=True, exist_ok=True)
        for ev in events:
            ref = float(ev["t_speech_onset"])
            end_of_speech = float(ev.get("t_end_of_speech") or ref)
            pause, _, _ = measure_pause(
                with_voice, reference=ref, end_of_speech=end_of_speech, until=end_of_speech + NO_REPLY_WAIT_S)
            name = f"{p['clave']}_{ev.get('turn')}_{ev['tipo']}.wav"
            write_wav_stereo(
                folder / name,
                trim(harness, sample_rate, since=ref - ANTES_SEG, until=ref + AFTER_S),
                trim(agent, sample_rate, since=ref - ANTES_SEG, until=ref + AFTER_S),
                sample_rate,
            )
            harness_says = ev.get("stopped")
            rows.append({
                "file": f"poc{p.get('poc')}/{name}", "poc": p.get("poc"), "estado": p.get("estado"),
                "clave": p["clave"], "scenario": p.get("scenario"), "turn": ev.get("turn"),
                "tipo": ev["tipo"], "clip_id": ev.get("clip_id") or "", "text": ev.get("text") or "",
                "second_in_call": round(ref, 2),
                "harness_stopped": "" if harness_says is None else int(bool(harness_says)),
                "stop_time_ms": ev.get("stop_time_ms", ""),
                "pause_audio_ms": "" if pause is None else pause,
                "discrepa": "" if harness_says is None else int(bool(harness_says) != (pause is not None)),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campana", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-failed", action="store_true")
    args = p.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for c in args.campana:
        rows.extend(cuts_of(c, args.output, include_failed=args.include_failed))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "indice.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    disagree = [f for f in rows if f["discrepa"] == 1]
    print(f"{len(rows)} cuts in {args.output} (the event lands at second {ANTES_SEG:.0f} of each one)")
    print(f"{len(disagree)} where the harness and the audio disagree:")
    for f in disagree:
        print(f"  {f['file']}  harness stopped={f['harness_stopped']}  pause in the audio={f['pause_audio_ms'] or 'none ≥0.7 s'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
