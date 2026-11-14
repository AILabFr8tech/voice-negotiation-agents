"""Who talked over whom, measured on the two-channel WAV.

## What it measures and why it is needed

`docs/METRICS_CONTRACT.md` calls for turn-taking: unwanted interruptions, false
ends of turn, long silences. Until now that was counted by hand by listening to
the call, and by hand it does not scale to a campaign of 50. The audit WAV
already has both sides in separate channels aligned to the same zero, so the
count can be done afterwards and at no cost.

**Two** things are distinguished that are not the same and that mixing them
would confuse:

- **The agent started over the carrier.** It is the unwanted interruption that
  matters to the business: the agent cut off someone who was still talking. A
  premature end of turn sounds like this.
- **The carrier started over the agent.** Here the carrier is the harness,
  which has an explicit advance rule —wait for the agent to go quiet—, so this
  does not describe the framework: it describes that the advance rule failed,
  and that run must be inspected before using its latency.

## What it is NOT

**It is not a latency and it enters no percentile.** Channel 1 is timestamped
when the frame **arrives**, with the transport's jitter inside; channel 0 comes
out exact because the player knows its deadline. That is more than enough to
say "these two sounded at the same time for half a second" and not enough to say
"the overlap lasted 487 ms". That is why **counts** and rounded durations are
reported, never a p95.

And for that same reason the threshold below is not cosmetic: without it, the
tail of a delayed frame would count as an interruption.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import read_wav_stereo
from shared.audio.level import DROP_FOR_TRIM_DB, windows_with_speech

# Channel 0 is the harness and 1 the agent. Fixed in `carrier_simulator/auditoria`,
# and repeated here because reading the file backwards would swap who interrupted
# whom without anything failing.
HARNESS_CHANNEL: Final[int] = 0
AGENT_CHANNEL: Final[int] = 1

# How long both have to sound at the same time to count as an overlap. It is the
# same value PoC3 uses (`poc3_gpt_live/app/latencia.py`,
# `VENTANA_DE_SOLAPE_SEG`), and it is repeated here on purpose instead of being
# imported: `shared/` does not depend on any PoC. If it changes, change it in both.
#
# Below this, an interruption cannot be told apart from the tail of a frame that
# arrived late, which is the error channel 1 can make.
MIN_OVERLAP_S: Final[float] = 0.15

# How much silence separates two speech streaks on the same side. Short on
# purpose: over-splitting a long sentence does no harm —any of its pieces is
# enough to detect the overlap—, whereas gluing together two sentences two
# seconds apart would invent an overlap with whatever sounded in between.
STREAK_GAP_S: Final[float] = 0.3

# How far below its channel's peak a window counts as speech. **The trim drop
# (35 dB) is used, not the level drop (25)**, and the difference is not
# cosmetic: with 25 dB, the harness channel yields much less speech than what
# actually sounds. That threshold exists to *measure the level* of a recording,
# where including the low-energy trailing bits sinks the RMS; here the opposite
# is needed —knowing whether **someone was talking**— and for that the repo
# already has the trim drop, which is written precisely so as not to clip word
# endings. With 35 dB the threshold (-36 dBFS) stays 24 dB above the comfort
# noise floor (-60 dBFS), which is what must not count as carrier speech.
DROP_FOR_ACTIVITY_DB: Final[float] = DROP_FOR_TRIM_DB


@dataclass(frozen=True)
class Streak:
    """A contiguous stretch of speech in one channel, in seconds from the WAV's zero."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Overlap:
    """A stretch in which both channels had speech at the same time."""

    start: float
    end: float
    # Who was already sounding when the other came in: "harness" or "agent".
    who_was_there: str
    # Who came in over the other. It is the one interrupting.
    who_entered: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def agent_cut_it_off(self) -> bool:
        """The unwanted interruption: the agent came in over the carrier."""
        return self.who_entered == "agent"

    def a_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start, 2),
            "duration_ms": round(self.duration * 1000.0, -1),
            "who_was_there": self.who_was_there,
            "who_entered": self.who_entered,
        }


def speech_streaks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    umbral_dbfs: float | None = None,
    drop_db: float = DROP_FOR_ACTIVITY_DB,
    gap_s: float = STREAK_GAP_S,
) -> list[Streak]:
    """The speech stretches of one channel, merging those that almost touch."""
    with_voice, per_window, _ = windows_with_speech(
        samples, sample_rate, umbral_dbfs=umbral_dbfs, drop_db=drop_db
    )
    if with_voice.size == 0:
        return []
    step = per_window / float(sample_rate)
    jump = max(1, int(round(gap_s / step)))

    streaks: list[Streak] = []
    start = int(with_voice[0])
    anterior = start
    for indice in map(int, with_voice[1:]):
        if indice - anterior > jump:
            streaks.append(Streak(start * step, (anterior + 1) * step))
            start = indice
        anterior = indice
    streaks.append(Streak(start * step, (anterior + 1) * step))
    return streaks


def overlaps_between(
    of_harness: list[Streak],
    of_agent: list[Streak],
    *,
    min_s: float = MIN_OVERLAP_S,
) -> list[Overlap]:
    """The stretches where the two lists overlap, with who came in over whom.

    Who came in over the other is decided by **which one started later**, not
    by who talks louder or who lasts longer: the interruption is committed by
    whoever opens their mouth knowing the other already had theirs open.
    """
    overlaps: list[Overlap] = []
    for una in of_harness:
        for other in of_agent:
            start = max(una.start, other.start)
            end = min(una.end, other.end)
            if end - start < min_s:
                continue
            if other.start > una.start:
                who_was_there, who_entered = "harness", "agent"
            elif una.start > other.start:
                who_was_there, who_entered = "agent", "harness"
            else:
                # They started in the same window: nobody came in over the other.
                who_was_there, who_entered = "a la vez", "a la vez"
            overlaps.append(
                Overlap(
                    start=start,
                    end=end,
                    who_was_there=who_was_there,
                    who_entered=who_entered,
                )
            )
    return sorted(overlaps, key=lambda s: s.start)


def summary(overlaps: list[Overlap], *, duration_s: float) -> dict[str, Any]:
    """The counts that go into the report. Counts, never percentiles."""
    of_agent = [s for s in overlaps if s.agent_cut_it_off]
    of_harness = [s for s in overlaps if s.who_entered == "harness"]
    seconds = sum(s.duration for s in overlaps)
    return {
        "what_it_is": (
            "voz simultánea en los dos canales del WAV de auditoría. Conteos de "
            "turn-taking, NO latencia: el canal del agente se fecha al llegar y "
            "lleva el jitter del transporte dentro. No entra a ningún percentil"
        ),
        "run_duration_s": round(duration_s, 1),
        "overlaps": len(overlaps),
        "the_agent_talked_over": len(of_agent),
        "the_harness_talked_over": len(of_harness),
        "seconds_in_overlap": round(seconds, 1),
        "run_fraction": (
            round(seconds / duration_s, 4) if duration_s else None
        ),
        "min_overlap_ms": round(MIN_OVERLAP_S * 1000.0),
        "how_to_read": (
            "`el_agente_entro_encima` son interrupciones no deseadas y describen "
            "al framework. `el_arnes_entro_encima` NO: el arnés tiene regla de "
            "avance explícita, así que si entró encima es que la regla falló y "
            "esa corrida hay que mirarla antes de usar su latencia"
        ),
    }


def measure(
    ruta_wav: Path | str, *, min_s: float = MIN_OVERLAP_S
) -> dict[str, Any]:
    """Measure the overlap of a two-channel audit WAV."""
    ruta_wav = Path(ruta_wav)
    harness_channel, agent_channel, sample_rate = read_wav_stereo(ruta_wav)
    duration = len(harness_channel) / float(sample_rate)

    # The threshold is re-derived from the peak of **each channel separately**.
    # It has to be so: the harness channel carries a continuous comfort-noise
    # floor the agent's does not have, and a common threshold would count that
    # noise as carrier speech for the whole call.
    of_harness = speech_streaks(harness_channel, sample_rate)
    of_agent = speech_streaks(agent_channel, sample_rate)
    overlaps = overlaps_between(of_harness, of_agent, min_s=min_s)

    data = summary(overlaps, duration_s=duration)
    data["file"] = ruta_wav.name
    # Coverage is ALWAYS included, and above all when the result is zero. A
    # detector blind in one channel produces zero overlaps and reads the same as
    # a perfectly turn-based conversation. This tells them apart: if a channel
    # carries 0 s of speech, the zero means nothing.
    data["detected_speech"] = {
        "harness_s": round(sum(r.duration for r in of_harness), 1),
        "agent_s": round(sum(r.duration for r in of_agent), 1),
        "harness_streaks": len(of_harness),
        "agent_streaks": len(of_agent),
        "drop_db": DROP_FOR_ACTIVITY_DB,
        "why_it_matters": (
            "si un canal sale con 0 s, el conteo de solapes de arriba no "
            "significa 'no hubo': significa que no se detectó nada que solapar"
        ),
    }
    data["detail"] = [s.a_dict() for s in overlaps]
    return data


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="shared.metrics.overlap",
        description=(
            "Counts simultaneous speech in the two-channel audit WAVs. "
            "Offline and free."
        ),
    )
    arg_parser.add_argument("wavs", nargs="+", help="one or more `auditoria_*.wav`")
    arg_parser.add_argument("--json", action="store_true")
    args = arg_parser.parse_args(argv)

    measured_items = []
    for ruta in args.wavs:
        try:
            measured_items.append(measure(ruta))
        except Exception as error:  # noqa: BLE001 - diagnostics: report and continue
            print(f"[!] {Path(ruta).name}: {error}")
    if not measured_items:
        print("[FAIL] no file could be measured")
        return 1

    if args.json:
        print(json.dumps(measured_items, ensure_ascii=False, indent=2))
        return 0

    print(
        f"\n  {'file':<32}{'dur':>6}{'harness sp':>11}{'agent sp':>12}"
        f"{'overlaps':>9}{'agent':>8}{'harness':>7}"
    )
    for data in measured_items:
        voice = data["detected_speech"]
        print(
            f"  {data['file'][-30:]:<32}"
            f"{data['run_duration_s']:>6.0f}"
            f"{voice['harness_s']:>10.1f}s"
            f"{voice['agent_s']:>11.1f}s"
            f"{data['overlaps']:>9}"
            f"{data['the_agent_talked_over']:>8}"
            f"{data['the_harness_talked_over']:>7}"
        )
    blind = [
        d
        for d in measured_items
        if not d["detected_speech"]["harness_s"] or not d["detected_speech"]["agent_s"]
    ]
    for data in blind:
        print(
            f"\n  [!] {data['file']}: one channel came out with no speech. Its zero "
            "overlaps does not mean 'there were none'."
        )
    agent_total = sum(d["the_agent_talked_over"] for d in measured_items)
    print(
        f"\n  {agent_total} agent interruption(s) in "
        f"{len(measured_items) - len(blind)} measurable run(s)."
    )
    print("  This is turn-taking, not latency: the agent channel is timestamped on arrival.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
