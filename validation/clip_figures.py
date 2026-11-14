"""One image per stimulus audio, with its end-of-speech mark and, on request, Whisper.

For each clip that some script can play:

- plots the waveform and the level per 10 ms window, with the speech threshold
  (25 dB below the peak) and the manifest's end-of-speech mark;
- **recomputes** the end of speech with the same function that generated the
  manifest and reports whether it matches;
- verifies the file's sha256 against the manifest;
- with `--whisper MODEL`, transcribes with per-word timestamps and compares the
  end of the last word with the end-of-speech mark.

Writes `end_of_speech_clips.csv` (in `--output`) and a PDF with one page per clip
(in `--figures`). With `--png`, also one standalone image per clip.

    uv run --project poc1_pipecat --with matplotlib --with scipy \
        --with faster-whisper==1.2.1 python -m validation.clip_figures --whisper large-v3

Costs nothing: Whisper runs locally. Without `--whisper` it takes seconds; with
`large-v3` on CPU, on the order of a minute per clip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.audio.manifest import MANIFEST_NAME, load_manifest
from shared.audio.level import end_of_speech, levels_per_window
from shared.carrier_simulator.script import load_clip
from validation.validate_harness import RAIZ, DEFAULT_OUTPUT, reproducible_ones

DEFAULT_FIGURES: Final[Path] = RAIZ / "docs" / "report" / "figures"

COLUMNS: Final[list[str]] = [
    "clip_id", "scenario", "turn", "variant", "text",
    "duration_s", "end_of_speech_s", "recomputed_end_of_speech_s", "coincide",
    "cola_seg", "speech_threshold_dbfs", "sha256_ulaw", "verified_sha256",
    "whisper_model", "whisper_text", "whisper_last_word",
    "whisper_last_word_end_s", "whisper_minus_end_of_speech_ms",
]


def transcriber(model_name: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def whisper_words(model_name: Any, pcm: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
    """Same configuration as stage 1 of the WER reference."""
    from scipy.signal import resample_poly

    audio = resample_poly(pcm.astype(np.float32) / 32768.0, 2, 1).astype(np.float32)
    segments, _ = model_name.transcribe(
        audio, language="es", beam_size=5, temperature=0.0, vad_filter=False,
        word_timestamps=True, condition_on_previous_text=False,
    )
    text, words = [], []
    for s in segments:
        text.append(s.text)
        for w in s.words or []:
            words.append((w.word.strip(), float(w.start), float(w.end)))
    return "".join(text).strip(), words


def draw(ax_wave: Any, ax_level: Any, pcm: np.ndarray, entry: Any,
            recalculated: float, words: list[tuple[str, float, float]]) -> None:
    t = np.arange(len(pcm)) / TELEPHONY_SAMPLE_RATE
    ax_wave.plot(t, pcm / 32768.0, lw=0.4, color="#555555")
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_ylabel("amplitude")
    levels, per_window = levels_per_window(pcm, TELEPHONY_SAMPLE_RATE)
    tv = (np.arange(len(levels)) + 0.5) * per_window / TELEPHONY_SAMPLE_RATE
    ax_level.plot(tv, levels, lw=0.8, color="#1f5f9f")
    ax_level.axhline(entry.speech_threshold_dbfs, ls="--", lw=0.8, color="#888888",
                     label=f"speech threshold ({entry.speech_threshold_dbfs:.1f} dBFS)")
    ax_level.set_ylim(-80, 0)
    ax_level.set_ylabel("level (dBFS)")
    ax_level.set_xlabel("time since clip start (s)")
    for ax in (ax_wave, ax_level):
        ax.axvline(entry.end_of_speech_s, color="#c0392b", lw=1.2,
                   label=f"end of speech ({entry.end_of_speech_s:.2f} s)")
        ax.set_xlim(0, len(pcm) / TELEPHONY_SAMPLE_RATE)
    if words:
        for _, beg, end in words:
            ax_wave.axvspan(beg, end, color="#2e86c1", alpha=0.08)
        last, _, end = words[-1]
        ax_wave.axvline(end, color="#2e86c1", ls=":", lw=1.2,
                        label=f"end of «{last}» per Whisper ({end:.2f} s)")
    ax_wave.legend(loc="upper right", fontsize=7)
    ax_level.legend(loc="upper right", fontsize=7)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--figures", default=str(DEFAULT_FIGURES))
    p.add_argument("--whisper", default=None, metavar="MODEL", help="e.g. large-v3")
    p.add_argument("--png", action="store_true", help="also one standalone image per clip")
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    clips = Path(args.clips)
    by_id = load_manifest(clips / MANIFEST_NAME).by_id()
    ids = [i for i in reproducible_ones() if i in by_id]
    model_name = transcriber(args.whisper) if args.whisper else None
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with PdfPages(figures / "clips_fin_de_voz.pdf") as pdf:
        for n, clip_id in enumerate(ids, start=1):
            entry = by_id[clip_id]
            ruta = clips / f"{clip_id}.ulaw"
            sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
            pcm, _ = load_clip(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
            recalculated = end_of_speech(pcm, TELEPHONY_SAMPLE_RATE).end_s
            text_w, words = whisper_words(model_name, pcm) if model_name else ("", [])
            row = {
                "clip_id": clip_id, "scenario": entry.scenario, "turn": entry.turn,
                "variant": entry.variant, "text": entry.text,
                "duration_s": entry.duration_s, "end_of_speech_s": entry.end_of_speech_s,
                "recomputed_end_of_speech_s": recalculated,
                "coincide": int(abs(recalculated - entry.end_of_speech_s) < 0.0005),
                "cola_seg": entry.cola_seg, "speech_threshold_dbfs": entry.speech_threshold_dbfs,
                "sha256_ulaw": entry.sha256_ulaw, "verified_sha256": int(sha == entry.sha256_ulaw),
                "whisper_model": args.whisper or "",
                "whisper_text": text_w,
                "whisper_last_word": words[-1][0] if words else "",
                "whisper_last_word_end_s": round(words[-1][2], 3) if words else "",
                "whisper_minus_end_of_speech_ms": (
                    round((words[-1][2] - entry.end_of_speech_s) * 1000.0, 1) if words else ""
                ),
            }
            rows.append(row)
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
            draw(a1, a2, pcm, entry, recalculated, words)
            fig.suptitle(f"{clip_id} — «{entry.text}»", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            if args.png:
                fig.savefig(figures / f"clip_{clip_id}.png", dpi=110)
            plt.close(fig)
            extra = (f"  whisper {row['whisper_minus_end_of_speech_ms']} ms" if words else "")
            print(f"[{n}/{len(ids)}] {clip_id:<12} end of speech {entry.end_of_speech_s:.3f} s"
                  f"  recomputed {recalculated:.3f}  sha {'ok' if row['verified_sha256'] else 'NO'}{extra}",
                  flush=True)

    with (output / "end_of_speech_clips.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    diffs = [float(f["whisper_minus_end_of_speech_ms"]) for f in rows if f["whisper_minus_end_of_speech_ms"] != ""]
    print(f"\n{len(rows)} clips · recomputed end of speech equal to the manifest's in "
          f"{sum(f['coincide'] for f in rows)} · sha256 verified in {sum(f['verified_sha256'] for f in rows)}")
    if diffs:
        d = np.array(diffs)
        print(f"Whisper − end of speech (ms): median {np.median(d):.0f}, p25 {np.percentile(d, 25):.0f}, "
              f"p75 {np.percentile(d, 75):.0f}, min {d.min():.0f}, max {d.max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
