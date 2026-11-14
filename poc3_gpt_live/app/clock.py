"""The bridge's wall axis: `time.time()` anchored ONCE + `perf_counter`.

This is `marcas.STEP_3_REQUIREMENT` implemented. The problem it solves
is written out in full in the header of `app/marks.py`; here is just enough
to read this file without opening that one:

- **Origin.** The harness timestamps its two zeros with `time.time()` (epoch) and
  `ColectorDeLatencia` timestamps the acknowledgement with `time.monotonic()` (uptime,
  `GetTickCount64` on Windows). Subtracting across different axes does not give a latency,
  it gives the machine's age.
- **Resolution.** `time.monotonic()` jumps in 15 ms steps and **does not improve**
  even if another process refines the global timer; `time.time()` does follow it and is around
  a millisecond. So the coarse end would be the acknowledgement, and 15 ms is on the
  order of the 20 ms window on which `bridge.has_speech` decides.

Both are fixed by the same mechanism, the one already used by
`ReproductorDeMediaStreams` (§1 of the contract) and PoC1's `SondaDelCero`:
`time.time()` is read only once, `perf_counter()` is saved at that same instant, and
from then on every mark is `anchor_wall + (perf_counter() - ancla_perf)`. The
origin stays in epoch time and the resolution becomes that of
`QueryPerformanceCounter`, 100 ns.

## Why the anchor must be audited, not just set

`time.time()` is adjustable and `perf_counter` is not. If NTP corrects the time
mid-call, the anchor goes stale and everything timestamped afterwards is shifted
with nothing to warn about it — and the shift is exactly the size of what is
being measured. That is why this module measures the **drift** between the two:

    drift = (time.time() - anchor_wall) - (perf_counter() - ancla_perf)

With a healthy clock that is zero plus quantization noise. A jump is a
clock adjustment, and a call with an adjustment **is declared non-interpretable instead
of being corrected**: correcting without knowing at what instant the jump happened would be
distributing the error by eye.

The statistic is the call's **maximum absolute value**, not the final one. A
jump that is corrected later leaves the marks timestamped while it lasted just as
unusable, and with only two readings —on open and on hang-up— the criterion
would bound the end and not the worst moment. That is why it is sampled periodically.

## Resolution is measured, not manipulated

`time.get_clock_info("time").resolution` returns 15.625 ms on Windows, which is
the declared worst case and **not** what actually happens: the real granularity of
`GetSystemTimeAsFileTime` follows the global timer, which any process can
raise, and it ranges between half a millisecond and a millisecond and a half depending on what is running
on the machine (measurements in `docs/campaign_poc3_estado_20260923.md`).

Hence `R` is measured **within the call** and twice —on open and on hang-up— and
the coarser one is used: the threshold has to cover the worst clock the
call had, not the best. Both are always published, and the "the clock was
unstable" flag compares **thresholds** and not resolutions (`THRESHOLD_TOLERANCE`): below
the floor the threshold does not move, so there the difference means
nothing. Separately there is `la_eleccion_de_R_cambia_el_veredicto`, which answers the
sharper question — not "did the clock vary?" but "does that variation decide anything?".

**The timer is not pinned with `timeBeginPeriod`.** It would change the granularity of
`asyncio` and with it the run conditions relative to the PoC1 and PoC2
campaigns, which ran without touching it.
"""

from __future__ import annotations

import time
from typing import Any, Final

from poc3_gpt_live.app.marks import (
    DRIFT_TO_RECORD,
    HARNESS_CLOCK,
    STEP_3_REQUIREMENT,
    drift_threshold_ms,
)

__all__ = [
    "RESOLUTION_SAMPLES",
    "THRESHOLD_TOLERANCE",
    "SAMPLING_PERIOD_S",
    "AnchoredClock",
    "MEASUREMENT_CAP_S",
    "measure_effective_resolution_ms",
]

# How many non-zero jumps of `time.time()` are collected before keeping the
# smallest. 32 is plenty: what is sought is the minimum of a quantization, not a
# mean, and the minimum stabilizes within the first few dozen.
RESOLUTION_SAMPLES: Final[int] = 32

# And the cap on the real time measuring it may cost. It exists because the two
# limits behave in opposite ways depending on the clock: with a 0.5 ms grain the 32
# samples take ~16 ms, but with a 15.6 ms grain they would take half a second, and
# this runs on the opening path of a call that is already ringing. With the
# cap, as many as fit are taken —with a coarse grain they are few and enough, because the
# jump is always the same— and the opening pays no more than this.
MEASUREMENT_CAP_S: Final[float] = 0.05

# How often a drift sample is taken during the call. It is a subtraction of
# two clock readings: at this period that is ~160 samples in a
# 157 s call and the cost is not measurable.
SAMPLING_PERIOD_S: Final[float] = 1.0

# How far apart the two thresholds must be for it to be worth saying that
# the clock was unstable. The THRESHOLD is compared, not the resolution: while
# `4R <= 2.0` —i.e. `R <= 0.5 ms`— the floor dominates and the threshold is constant **by
# construction of the formula**, so there any variation in R is immaterial and
# a flag that turns on is noise. Above the floor the threshold is linear
# in R, and so the band on the threshold is the same band on R.
#
# The 10% comes from the margin the formula already has: the `4` in `4R` is twice the
# quantization noise (which reaches `2R`), so a 10% error in R stays
# an order of magnitude inside that margin and cannot loosen a threshold that
# was tight. With the old criterion —`R_abrir != R_colgar`— the flag
# turned on with 0.3676 vs 0.5083 ms, which move the threshold by 34 µs.
THRESHOLD_TOLERANCE: Final[float] = 0.10


def measure_effective_resolution_ms() -> float:
    """The smallest non-zero jump between two readings of `time.time()`, in ms.

    It is **not** `time.get_clock_info`, which on Windows declares the worst case
    (15.625 ms) and not what the clock actually does. The budget is kept by
    `perf_counter`, not by the clock being measured: using the measured clock as the
    stopwatch of its own measurement would tie the cap to the quantization being
    sought.
    """
    limit = time.perf_counter() + MEASUREMENT_CAP_S
    jumps: list[float] = []
    prior = time.time()
    while len(jumps) < RESOLUTION_SAMPLES and time.perf_counter() < limit:
        now = time.time()
        if now != prior:
            jumps.append(now - prior)
            prior = now
    if not jumps:
        # A clock so coarse that it did not jump even once within the budget. Fall back to
        # the declared worst case, which is an upper bound and makes the threshold looser, never
        # tighter.
        return float(time.get_clock_info("time").resolution) * 1000.0
    return min(jumps) * 1000.0


class AnchoredClock:
    """Wall-clock mark with `perf_counter` resolution, plus its audit.

    One instance per call. It is built on open —that is where it measures `R` for the first
    time— and closed on hang-up, which is where it measures it again.

    `ahora()` is the only thing the hot path calls, and it is one addition and one
    subtraction: it does not allocate, does not block and does not touch disk.
    """

    def __init__(self) -> None:
        self.resolution_at_open_ms = measure_effective_resolution_ms()
        self.resolution_at_hang_up_ms: float | None = None
        # The two anchor readings, taken as close together as possible. Whatever
        # falls between them is the fixed error of the origin, and it is on the order of a
        # nanosecond.
        self._anchor_wall = time.time()
        self._anchor_perf = time.perf_counter()

        self.drift_at_open_ms = 0.0
        self.drift_at_hang_up_ms: float | None = None
        self._max_abs_drift_ms = 0.0
        self._samples = 0
        self._closed = False

    # -- What the hot path uses --

    def now(self) -> float:
        """Seconds since the epoch, with `perf_counter` granularity."""
        return self._anchor_wall + (time.perf_counter() - self._anchor_perf)

    # -- Audit --

    def drift_ms(self) -> float:
        """`(time.time() - anchor) - (perf_counter() - anchor)`, in ms.

        Positive means the wall clock moved ahead relative to the
        counter; negative, that it fell behind. With a healthy clock it oscillates within the
        quantization noise of `time.time()`.
        """
        return (
            (time.time() - self._anchor_wall)
            - (time.perf_counter() - self._anchor_perf)
        ) * 1000.0

    def take_samples(self) -> float:
        """Take a drift sample and keep the absolute maximum."""
        drift = self.drift_ms()
        self._samples += 1
        if abs(drift) > abs(self._max_abs_drift_ms):
            self._max_abs_drift_ms = drift
        return drift

    def shut(self) -> None:
        """On hang-up: last sample and second measurement of `R`. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.drift_at_hang_up_ms = self.take_samples()
        self.resolution_at_hang_up_ms = measure_effective_resolution_ms()

    # -- Derived readings --

    @property
    def used_resolution_ms(self) -> float:
        """The coarser of the two measurements. The threshold covers the worst clock."""
        if self.resolution_at_hang_up_ms is None:
            return self.resolution_at_open_ms
        return max(self.resolution_at_open_ms, self.resolution_at_hang_up_ms)

    @property
    def finest_resolution_ms(self) -> float:
        """The other one. It decides nothing: it exists to bound how much R could vary."""
        if self.resolution_at_hang_up_ms is None:
            return self.resolution_at_open_ms
        return min(self.resolution_at_open_ms, self.resolution_at_hang_up_ms)

    @property
    def umbral_ms(self) -> float:
        return drift_threshold_ms(self.used_resolution_ms)

    @property
    def unstable_resolution(self) -> bool:
        """Do the two measurements of R separate the thresholds materially?

        Compares thresholds, not resolutions (see `THRESHOLD_TOLERANCE`), so as not to
        turn on with any jitter.
        """
        alto = self.umbral_ms
        low = drift_threshold_ms(self.finest_resolution_ms)
        return (alto - low) > THRESHOLD_TOLERANCE * alto

    @property
    def choice_of_R_changes_the_verdict(self) -> bool:
        """Would the call be interpretable with one R and not with the other?

        Turns on only when the observed drift falls **between** the two thresholds.
        It is sharper than `unstable_resolution` and answers a different question: not
        "did the clock vary?" but "does that variation decide anything?". It can turn on with
        `unstable_resolution` false —two nearly equal thresholds with the drift
        right in between— and that is exactly the case that must be checked by hand.
        """
        drift = abs(self._max_abs_drift_ms)
        alto = self.umbral_ms
        low = drift_threshold_ms(self.finest_resolution_ms)
        return (drift <= low) != (drift <= alto)

    @property
    def max_drift_ms(self) -> float:
        return self._max_abs_drift_ms

    @property
    def interpretable(self) -> bool:
        """Can this call's marks be subtracted against the harness's?"""
        return abs(self._max_abs_drift_ms) <= self.umbral_ms

    def block(self) -> dict[str, Any]:
        """What goes into the call JSON. It is always published, healthy or not.

        It goes in full and not just the verdict: if `R` comes out so coarse that the clock
        cannot resolve what is needed, that **is declared with `R` in plain view** —the
        threshold is not tightened to hide it— and whoever reads the file has to
        be able to see which clock it was measured with.
        """
        resolutions: dict[str, Any] = {
            "on_open_ms": round(self.resolution_at_open_ms, 4),
            "on_hang_up_ms": (
                None
                if self.resolution_at_hang_up_ms is None
                else round(self.resolution_at_hang_up_ms, 4)
            ),
            "used_ms": round(self.used_resolution_ms, 4),
            "declared_by_get_clock_info_ms": round(
                float(time.get_clock_info("time").resolution) * 1000.0, 4
            ),
            "threshold_tolerance": THRESHOLD_TOLERANCE,
            # Materiality on the threshold, not equality of resolutions: two Rs
            # below the floor give the same threshold and their difference
            # means nothing.
            "inestable": self.unstable_resolution,
        }
        return {
            "requisito": STEP_3_REQUIREMENT,
            "axis": "time.time anclado una vez + perf_counter",
            "harness_clock": HARNESS_CLOCK,
            "same_axis_as_harness": True,
            "anchor_wall_s": round(self._anchor_wall, 6),
            "effective_resolution": resolutions,
            "umbral_ms": round(self.umbral_ms, 4),
            DRIFT_TO_RECORD: {
                "absolute_max_ms": round(self._max_abs_drift_ms, 4),
                "on_open_ms": round(self.drift_at_open_ms, 4),
                "on_hang_up_ms": (
                    None
                    if self.drift_at_hang_up_ms is None
                    else round(self.drift_at_hang_up_ms, 4)
                ),
                "samples": self._samples,
                "period_s": SAMPLING_PERIOD_S,
            },
            "interpretable": self.interpretable,
            # The sharper question, separate from the previous one: does the verdict depend
            # on which of the two measurements of R was used? If it turns on, the
            # call is borderline and is decided by hand, not by the formula.
            "the_choice_of_R_changes_the_verdict": (
                self.choice_of_R_changes_the_verdict
            ),
            "what_it_means": (
                "`interpretable: false` means the wall clock "
                "moved relative to `perf_counter` by more than `umbral_ms` at some "
                "point during the call — a time adjustment. Marks timestamped "
                "while it lasted are shifted and are NOT corrected: it is unknown "
                "at what instant the jump happened. This call is excluded from the "
                "cross-check against the harness zeros. The threshold is max(2.0, 4R) "
                "with R = `resolucion_efectiva.usada_ms`, fixed by formula on "
                "September 22, 2026, before running"
            ),
        }
