"""What PoC2's zero bias is made of. Two benches, neither of which spends money.

PoC2's harness publishes a `speech_bias_ms` of between 307 and 753 ms depending
on the run, and for two runs it was read as LiveKit not subtracting its VAD's
wait — because it exceeds the 550 ms of `min_silence_duration`. This module
measures the pieces of that number separately, instead of arguing about them:

    bias = criterion mismatch + AudioSource queue + SFU + Opus/jitter

- **`cola`** (queue) spins up a real `rtc.AudioSource` and pushes frames into
  it with the harness's pacing, to see how much audio is left waiting.
  `queue_size_ms=200` is a **cap**, not an occupancy, and a cap is only reached
  by pushing faster than real time.
- **`vad`** feeds the campaign clips to LiveKit's same Silero **in the same
  process**, with no room, no Opus and no SFU. Whatever comes out of that is
  the floor of the bias: the part that is not network path but disagreement
  between the criterion `shared/audio` uses to timestamp the end of speech
  (drop in dB) and the VAD's criterion (its own probability).

Both run locally, open no room and **cost $0**. The numbers they produce are
cited in `docs/METRICS_CONTRACT.md` §2.2.2; if someone reruns them and gets
different values, that section is out of date.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from shared.carrier_simulator.audio import MS_PER_FRAME, comfort_noise_frame
from shared.carrier_simulator.script import RUTA_CLIPS_DEFAULT
from shared.carrier_simulator.player import high_clock_resolution

SAMPLE_RATE = 8000
SAMPLES_PER_FRAME = SAMPLE_RATE * MS_PER_FRAME // 1000

# Comfort noise around the clip. Before, so Silero has a floor to decide
# against; after, so it manages to accumulate its `min_silence_duration` and
# declare end of speech — without it the event never arrives.
NOISE_BEFORE_S = 2.0
NOISE_AFTER_S = 3.0


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    v = sorted(values)
    n = len(v)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "min": round(v[0], 3),
        "q1": round(v[n // 4], 3),
        "median_value": round(v[n // 2], 3),
        "q3": round(v[(3 * n) // 4], 3),
        "p95": round(v[min(n - 1, int(0.95 * n))], 3),
        "max": round(v[-1], 3),
    }


# --------------------------------------------------------------------------
# Bench 1: the AudioSource queue
# --------------------------------------------------------------------------


async def measure_tail(*, seconds: float, cola_ms: int) -> dict[str, Any]:
    """How much audio stays in the `AudioSource` with the harness's pacing.

    It connects to no room: the source is the real one and the Rust FFI is the
    same, and what we want to know —whether the cap is reached— does not depend
    on someone listening on the other side.

    Two things are measured and both are needed:

    - **`queued_duration`**, the occupancy the SDK tracks in Python. It says
      how much audio is ahead of the frame about to be pushed.
    - **how long `capture_frame` takes to return**. It is the independent
      witness: the Rust side applies backpressure when its buffer fills up, so
      if it never blocks, it never filled up. Running this with a `cola_ms` of
      a single frame and seeing that it does not block either proves that the
      residence does not go beyond that.
    """
    from livekit import rtc

    source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=cola_ms)
    silence = b"\x00\x00" * SAMPLES_PER_FRAME

    antes: list[float] = []
    after: list[float] = []
    capture: list[float] = []
    skew: list[float] = []

    total = int(seconds * 1000 / MS_PER_FRAME)
    with high_clock_resolution():
        base = time.perf_counter()
        for i in range(total):
            target = base + i * (MS_PER_FRAME / 1000.0)
            wait = target - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            skew.append((time.perf_counter() - target) * 1000.0)

            antes.append(source.queued_duration * 1000.0)
            t0 = time.perf_counter()
            await source.capture_frame(
                rtc.AudioFrame(
                    data=silence,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
            )
            capture.append((time.perf_counter() - t0) * 1000.0)
            after.append(source.queued_duration * 1000.0)

    await source.aclose()

    maximum = max(after)
    return {
        "declared_tail_ms": cola_ms,
        "frames": total,
        "occupancy_before_push_ms": _summary(antes),
        "occupancy_after_push_ms": _summary(after),
        "capture_frame_ms": _summary(capture),
        "skew_against_deadline_ms": _summary(skew),
        "cap_usage_pct": round(100.0 * maximum / cola_ms, 1),
        "how_to_read": (
            "if `capture_frame` never blocks, the Rust buffer never filled up; "
            "and if the occupancy stays at one frame, the declared queue is a "
            "CAP and not a delay. It would only fill up by pushing faster than "
            "real time, and the player pushes against an absolute deadline"
        ),
    }


# --------------------------------------------------------------------------
# Bench 2: the floor of the bias, without transport
# --------------------------------------------------------------------------


def _pcm8k(clip_id: str, folder: Path) -> np.ndarray:
    with wave.open(str(folder / f"{clip_id}.wav"), "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{clip_id} is at {w.getframerate()} Hz and the harness publishes at {SAMPLE_RATE}"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _profiles(folder: Path) -> dict[str, dict[str, Any]]:
    from shared.audio.manifest import MANIFEST_NAME

    data = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["clips"]}


async def _un_clip(vad: Any, clip_id: str, card: dict[str, Any], folder: Path):
    from livekit import rtc

    samples = _pcm8k(clip_id, folder)
    stream = vad.stream()
    events: list[tuple[float, Any]] = []

    async def _listen() -> None:
        async for ev in stream:
            events.append((time.time(), ev))

    tarea = asyncio.create_task(_listen())

    antes = int(NOISE_BEFORE_S * 1000 / MS_PER_FRAME)
    of_the_clip = int(np.ceil(len(samples) / SAMPLES_PER_FRAME))
    after = int(NOISE_AFTER_S * 1000 / MS_PER_FRAME)

    # The real instant each frame was released, anchored the same way as
    # `FramePacing`: `time.time()` once and `perf_counter` to advance.
    reales: list[float] = []
    base_perf = time.perf_counter()
    base_wall = time.time()

    for i in range(antes + of_the_clip + after):
        target = base_perf + i * (MS_PER_FRAME / 1000.0)
        wait = target - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)

        if antes <= i < antes + of_the_clip:
            j = (i - antes) * SAMPLES_PER_FRAME
            piece = samples[j : j + SAMPLES_PER_FRAME]
            if len(piece) < SAMPLES_PER_FRAME:
                piece = np.concatenate(
                    [piece, np.zeros(SAMPLES_PER_FRAME - len(piece), dtype="<i2")]
                )
            raw_value = piece.tobytes()
        else:
            raw_value = comfort_noise_frame(SAMPLES_PER_FRAME)

        reales.append(base_wall + (time.perf_counter() - base_perf))
        stream.push_frame(
            rtc.AudioFrame(
                data=raw_value,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
        )

    stream.end_input()
    await asyncio.sleep(1.0)
    await stream.aclose()
    tarea.cancel()

    ends = [(t, ev) for t, ev in events if str(ev.type).endswith("END_OF_SPEECH")]
    if not ends:
        return None

    t_event, ev = ends[-1]
    # The SDK's formula, verbatim: `voice/audio_recognition.py`, END_OF_SPEECH
    # branch. Applying it here is what makes this floor comparable with the
    # bias the worker publishes in a call.
    estimated = t_event - ev.silence_duration - ev.inference_duration
    final_frame = antes + int(card["end_of_speech_s"] * 1000 / MS_PER_FRAME)
    t_end_of_speech = reales[min(final_frame, len(reales) - 1)]

    return {
        "clip": clip_id,
        "speech_bias_ms": round((estimated - t_end_of_speech) * 1000.0, 1),
        "silence_duration_ms": round(ev.silence_duration * 1000.0, 1),
        "inference_duration_ms": round(ev.inference_duration * 1000.0, 1),
        "cola_ms": round(card["cola_seg"] * 1000.0, 1),
        "drop_db": card["drop_db"],
    }


async def measure_floor(*, clips: Sequence[str], folder: Path) -> dict[str, Any]:
    """The zero bias with ZERO transport: the harness talks to the VAD in-process.

    It is the argument that settles the open question of whether LiveKit
    subtracts its VAD's wait. If it did not subtract it, no measurement could
    come out below `silence_duration`; what comes out is ~155 ms.
    """
    from livekit.plugins import silero

    from poc2_livekit.app.agent import build_vad

    del silero  # the import checks that the plugin is present; the agent builds the VAD.
    vad = build_vad()
    profiles = _profiles(folder)

    rows = []
    with high_clock_resolution():
        for clip_id in clips:
            row = await _un_clip(vad, clip_id, profiles[clip_id], folder)
            if row is not None:
                rows.append(row)

    silences = {f["silence_duration_ms"] for f in rows}
    return {
        "clips": rows,
        "speech_bias_ms": _summary([f["speech_bias_ms"] for f in rows]),
        "silence_duration_ms": sorted(silences),
        "inference_duration_ms": sorted({f["inference_duration_ms"] for f in rows}),
        "how_to_read": (
            "there is no AudioSource, no Opus, no SFU and no jitter buffer here: "
            "what remains is the disagreement between the criterion shared/audio "
            "uses to timestamp the end of speech (drop in dB) and the VAD's (its "
            "probability). It is the FLOOR of the bias, and it correlates with "
            "each clip's tail. Coming out far below silence_duration proves that "
            "the SDK does subtract it"
        ),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.zero_diagnostics",
        description="What PoC2's zero bias is made of. No room, $0.",
    )
    sub = arg_parser.add_subparsers(dest="banco", required=True)

    cola = sub.add_parser("cola", help="the AudioSource queue")
    cola.add_argument("--seconds", type=float, default=30.0)
    cola.add_argument(
        "--cola-ms",
        type=int,
        nargs="+",
        default=[20, 200, 1000],
        help=(
            "caps to try. With one of 20 ms (one frame) and no backpressure it "
            "is proven that the residence does not exceed one frame"
        ),
    )

    piso = sub.add_parser("vad", help="the floor of the bias, without transport")
    piso.add_argument(
        "--clips", nargs="+", default=["e1_t2", "e1_t3", "e1_t4", "e1_t5", "e1_t6", "e1_t7"]
    )
    piso.add_argument("--folder", type=Path, default=RUTA_CLIPS_DEFAULT)

    args = arg_parser.parse_args(argv)

    if args.banco == "cola":
        for tope in args.cola_ms:
            data = asyncio.run(measure_tail(seconds=args.seconds, cola_ms=tope))
            print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        data = asyncio.run(measure_floor(clips=args.clips, folder=args.folder))
        print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
