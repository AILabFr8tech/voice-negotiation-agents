"""When each frame **was due** to go out and when it really went out.

The turn's zero is anchored on a single frame (the clip's last one), so the
skew of that frame against its deadline is not averaged with anything: it goes entirely into
that turn's latency. That is why the player is not satisfied with the nominal
deadline and measures each frame's real instant.

**Both** are kept, per frame and in every run. The nominal one is the axis
on which it was planned; the real one is the one used to timestamp the zero. The
difference is the skew, and publishing it per campaign is what allows defending (or
correcting) the number the metrics contract declares in §1.

## What exactly "the send instant" is

**The instant at which the player releases the frame to its transport**, taken in
the shared loop right before `await publish(...)`, and therefore
**conceptually identical in the 3 PoCs**.

It is not taken afterwards, and that is not a detail: what comes after that point is
each transport's own work (building the JSON and the base64 and pushing it to the
WebSocket in PoC1, building the `AudioFrame` and handing it to the `AudioSource` in
PoC2), and in PoC2 `capture_frame` also **blocks** when its queue is full,
so the instant would be governed by the draining of a LiveKit buffer and
not by the harness's clock. Timestamping there would put the transport's cost inside the
zero, which is exactly what the comparison wants to measure separately.

What the transport takes from that point until the audio reaches the agent
**does not disappear**: it shows up measured in the zero's bias, which is where it belongs.

## Two clocks and a single anchor

`time.time()` on Windows declares 15.625 ms resolution and is not monotonic;
`time.monotonic()` is `GetTickCount64`, with the same granularity. The only
fine-grained clock is `time.perf_counter()` (QueryPerformanceCounter, 100 ns), and it is
monotonic but does not tell the time of day.

So it is anchored **once** at startup (`(perf_counter, time())` taken
together) and from then on everything is measured with `perf_counter` and translated to the
wall axis by adding the difference. The drift between QPC and the system clock is
parts per million: over a three-minute call it stays far below a
millisecond, and far below what is being measured.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from shared.carrier_simulator.audio import MS_PER_FRAME


@dataclass
class FramePacing:
    """The (nominal deadline, real instant) pair of each published frame.

    Both on the same axis as the rest of the harness: seconds since
    `Observed.start`. Annotating is an `append` to two lists; it does not touch the
    audio path.
    """

    nominal_values: list[float] = field(default_factory=list)
    reales: list[float] = field(default_factory=list)

    def annotate(self, nominal: float, real: float) -> None:
        self.nominal_values.append(nominal)
        self.reales.append(real)

    def __len__(self) -> int:
        return len(self.reales)

    @property
    def frames(self) -> int:
        return len(self.reales)

    def skews_ms(self) -> list[float]:
        """Real − nominal, per frame. Positive = the frame went out late."""
        return [
            (real - nominal) * 1000.0
            for nominal, real in zip(self.nominal_values, self.reales, strict=True)
        ]

    def intervals_ms(self) -> list[float]:
        """Real time between consecutive frames. It should be around 20 ms."""
        return [
            (b - a) * 1000.0
            for a, b in zip(self.reales, self.reales[1:], strict=False)
        ]

    def summary(self) -> dict[str, Any]:
        """What goes into the run JSON. Diagnostics, not a metric."""
        if not self.reales:
            return {"frames": 0, "suficientes": False}
        skews = self.skews_ms()
        intervals = self.intervals_ms()
        absolutes = [abs(d) for d in skews]
        summary: dict[str, Any] = {
            "frames": self.frames,
            "suficientes": bool(intervals),
            "nominal_ms": float(MS_PER_FRAME),
            "mean_skew_ms": round(statistics.fmean(skews), 4),
            "max_abs_skew_ms": round(max(absolutes), 4),
            "p95_abs_skew_ms": round(sorted(absolutes)[int(0.95 * (len(absolutes) - 1))], 4),
            # The accumulated drift: how far the last frame shifted relative to
            # where the plan put it. With an absolute deadline it does not grow.
            "final_skew_ms": round(skews[-1], 4),
        }
        if intervals:
            ordered_items = sorted(intervals)
            summary.update(
                {
                    "interval_mean_ms": round(statistics.fmean(intervals), 4),
                    "interval_p50_ms": round(ordered_items[len(ordered_items) // 2], 4),
                    "interval_min_ms": round(min(intervals), 4),
                    "interval_max_ms": round(max(intervals), 4),
                    "interval_std_dev_ms": round(
                        statistics.pstdev(intervals) if len(intervals) > 1 else 0.0, 4
                    ),
                    "actual_duration_s": round(self.reales[-1] - self.reales[0], 4),
                    "nominal_duration_s": round(
                        self.nominal_values[-1] - self.nominal_values[0], 4
                    ),
                }
            )
        return summary

    def as_text(self) -> str:
        r = self.summary()
        if not r.get("suficientes"):
            return f"  pacing: {r['frames']} trama(s), no alcanzan para medir"
        return (
            f"  tramas               {r['frames']}\n"
            f"  intervalo            medio {r['interval_mean_ms']:.3f} ms · "
            f"p50 {r['interval_p50_ms']:.3f} · "
            f"min {r['interval_min_ms']:.3f} · max {r['interval_max_ms']:.3f} · "
            f"σ {r['interval_std_dev_ms']:.3f}\n"
            f"  desvío vs. deadline  medio {r['mean_skew_ms']:+.3f} ms · "
            f"p95 abs {r['p95_abs_skew_ms']:.3f} · "
            f"máx abs {r['max_abs_skew_ms']:.3f} · final {r['final_skew_ms']:+.3f}\n"
            f"  duración             {r['actual_duration_s']:.4f} s reales vs. "
            f"{r['nominal_duration_s']:.4f} s nominales"
        )
