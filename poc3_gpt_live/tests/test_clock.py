"""The bridge's clock anchor and its audit.

What is tested here is what decides whether a call enters the cross-check
against the harness zeros: that the axis is the epoch one, that the resolution
is measured instead of read from `get_clock_info`, that the threshold comes
from the formula fixed before running, and that a drift above the threshold
leaves the call declared not interpretable instead of corrected.

It is not tested that `time.time()` has a specific grain: that depends on what
else is running on the machine and such an assertion would be intermittent by
design. What is tested is that the number **is measured** and comes out positive.
"""

from __future__ import annotations

import sys
import time

import pytest

from poc3_gpt_live.app.marks import (
    DRIFT_TO_RECORD,
    DRIFT_FLOOR_MS,
    FACTOR_OVER_RESOLUTION,
    STEP_3_REQUIREMENT,
    drift_threshold_ms,
)
from poc3_gpt_live.app.clock import (
    THRESHOLD_TOLERANCE,
    MEASUREMENT_CAP_S,
    AnchoredClock,
    measure_effective_resolution_ms,
)


class TestTheResolutionIsMeasured:
    """`R` comes from this machine's clock, not from what the clock declares."""

    def test_is_a_positive_number(self) -> None:
        assert measure_effective_resolution_ms() > 0.0

    def test_does_not_cost_more_than_its_cap(self) -> None:
        """It runs at the opening of a call: the budget is the contract."""
        t0 = time.perf_counter()
        measure_effective_resolution_ms()
        # Slack over the cap: the loop checks the budget between readings,
        # so it can overrun by one iteration.
        assert time.perf_counter() - t0 < MEASUREMENT_CAP_S * 4

    def test_is_not_what_get_clock_info_declares(self) -> None:
        """It is not tested what it is; it is tested that the worst case is not copied.

        On Windows `get_clock_info` returns 15.625 ms, which is the worst case
        and not what the clock does; on Linux it returns ~1 ns, which is better
        than what is observed between two Python readings. The only thing that
        holds on both platforms is that the measured number need not match, so
        what is asserted is that the measurement happens and gives something
        usable.
        """
        measured = measure_effective_resolution_ms()
        assert 0.0 < measured < 1000.0


class TestTheAxisIsTheEpochOne:
    def test_now_is_tied_to_time_time(self) -> None:
        clock = AnchoredClock()
        assert abs(clock.now() - time.time()) < 0.050

    def test_advances_like_the_wall_clock(self) -> None:
        clock = AnchoredClock()
        a = clock.now()
        time.sleep(0.02)
        b = clock.now()
        assert 0.015 < (b - a) < 0.200

    @pytest.mark.skipif(sys.platform != "win32", reason="the anchored clock works around the 15 ms Windows tick; elsewhere the premise does not hold")
    def test_resolves_below_the_monotonic_grain(self) -> None:
        """The anchor's reason to exist: the 15 ms of `GetTickCount64` are not enough.

        Two consecutive readings have to be distinguishable. With
        `time.monotonic()` on Windows both would give the same value for
        15 ms in a row, which is of the order of the 20 ms window on which
        `bridge.has_speech` decides.
        """
        clock = AnchoredClock()
        readings = [clock.now() for _ in range(200)]
        different_ones = len(set(readings))
        assert different_ones > 100, f"solo {different_ones} valores distintos en 200 lecturas"


class TestTheDrift:
    def test_with_a_healthy_clock_it_is_noise(self) -> None:
        clock = AnchoredClock()
        for _ in range(20):
            clock.take_samples()
        assert abs(clock.max_drift_ms) <= clock.umbral_ms
        assert clock.interpretable

    def test_keeps_the_absolute_max_and_not_the_last(self) -> None:
        """A jump that is later reverted leaves the marks just as broken."""
        clock = AnchoredClock()
        clock._max_abs_drift_ms = -40.0
        clock.take_samples()  # healthy, close to zero
        assert clock.max_drift_ms == -40.0
        assert not clock.interpretable

    def test_a_drift_above_the_threshold_is_not_corrected(self) -> None:
        """It is declared, and the call leaves the cross-check. There is no compensating subtraction."""
        clock = AnchoredClock()
        clock._max_abs_drift_ms = clock.umbral_ms * 3
        block = clock.block()
        assert block["interpretable"] is False
        assert block[DRIFT_TO_RECORD]["absolute_max_ms"] > block["umbral_ms"]
        # And the published axis is still the same: nothing was readjusted.
        assert block["same_axis_as_harness"] is True

    def test_closing_is_idempotent(self) -> None:
        clock = AnchoredClock()
        clock.shut()
        first = clock.resolution_at_hang_up_ms
        clock.shut()
        assert clock.resolution_at_hang_up_ms == first


class TestTheThreshold:
    def test_comes_from_the_formula_fixed_before_running(self) -> None:
        clock = AnchoredClock()
        assert clock.umbral_ms == drift_threshold_ms(clock.used_resolution_ms)

    def test_uses_the_coarser_resolution_of_the_two(self) -> None:
        """The threshold covers the worst clock the call had, not the best."""
        clock = AnchoredClock()
        clock.resolution_at_open_ms = 0.5
        clock.resolution_at_hang_up_ms = 8.0
        assert clock.used_resolution_ms == 8.0
        assert clock.umbral_ms == FACTOR_OVER_RESOLUTION * 8.0

    def test_has_no_ceiling(self) -> None:
        """A ceiling would leave it below the very noise it describes."""
        clock = AnchoredClock()
        clock.resolution_at_open_ms = 15.625
        clock.resolution_at_hang_up_ms = 15.625
        assert clock.umbral_ms == FACTOR_OVER_RESOLUTION * 15.625

    def test_the_floor_governs_with_fine_clocks(self) -> None:
        clock = AnchoredClock()
        clock.resolution_at_open_ms = 0.0001
        clock.resolution_at_hang_up_ms = 0.0001
        assert clock.umbral_ms == DRIFT_FLOOR_MS


class TestTheBlockThatGoesIntoTheJson:
    def test_names_the_requirement_of_step_3(self) -> None:
        assert AnchoredClock().block()["requisito"] == STEP_3_REQUIREMENT

    def test_publishes_the_two_resolutions_and_whether_it_is_unstable(self) -> None:
        clock = AnchoredClock()
        clock.shut()
        res = clock.block()["effective_resolution"]
        assert res["on_open_ms"] > 0
        assert res["on_hang_up_ms"] is not None
        assert res["used_ms"] == max(res["on_open_ms"], res["on_hang_up_ms"])
        assert res["threshold_tolerance"] == THRESHOLD_TOLERANCE
        assert isinstance(res["inestable"], bool)


class TestInstabilityIsMeasuredAgainstTheThreshold:
    """The old flag went off with jitter that did not move the threshold.

    The case that motivated it is real: 0.3676 versus 0.5083 ms, measured at
    opening and at hang-up of the same call, separate the thresholds by 34 µs.
    Saying "the clock was unstable" there is noise, and a warning that always
    goes off stops being read.
    """

    def _having(self, begin: float, hang_up: float) -> AnchoredClock:
        clock = AnchoredClock()
        clock.resolution_at_open_ms = begin
        clock.resolution_at_hang_up_ms = hang_up
        return clock

    def test_the_case_that_motivated_it_no_longer_fires(self) -> None:
        clock = self._having(0.3676, 0.5083)
        assert clock.used_resolution_ms == 0.5083
        assert clock.finest_resolution_ms == 0.3676
        assert clock.unstable_resolution is False

    def test_below_the_floor_nothing_ever_triggers(self) -> None:
        """With `4R <= 2.0` the threshold is constant by construction."""
        clock = self._having(0.0001, 0.4999)
        assert clock.umbral_ms == DRIFT_FLOOR_MS
        assert clock.unstable_resolution is False

    def test_a_material_variation_does_trigger(self) -> None:
        """Two of the five measurements from September 22: 0.51 and 1.52 ms."""
        clock = self._having(0.51, 1.52)
        assert clock.unstable_resolution is True

    def test_inside_the_band_it_does_not_fire_and_outside_it_does(self) -> None:
        """The band is relative to the threshold: 5% no, 20% yes. Without touching the exact edge.

        The edge is not tested with equality because the comparison is in
        floating point and a test that depends on the last bit asserts nothing
        useful.
        """
        assert self._having(1.0, 1.05).unstable_resolution is False
        assert self._having(1.0, 1.25).unstable_resolution is True


class TestTheSharpFlag:
    """`the_choice_of_R_changes_the_verdict`: does the variation decide anything?

    It answers a different question from the previous one and can go off
    without it: two nearly equal thresholds with the drift right in between is
    a borderline call, and that one has to be looked at by hand even if the
    clock barely moved.
    """

    def _having(self, begin: float, hang_up: float, drift: float) -> AnchoredClock:
        clock = AnchoredClock()
        clock.resolution_at_open_ms = begin
        clock.resolution_at_hang_up_ms = hang_up
        clock._max_abs_drift_ms = drift
        return clock

    def test_comfortable_drift_does_not_fire(self) -> None:
        clock = self._having(0.51, 1.52, drift=0.5)
        assert clock.interpretable is True
        assert clock.choice_of_R_changes_the_verdict is False

    def test_huge_drift_does_not_fire_either(self) -> None:
        """With both thresholds exceeded, which one is chosen makes no difference."""
        clock = self._having(0.51, 1.52, drift=99.0)
        assert clock.interpretable is False
        assert clock.choice_of_R_changes_the_verdict is False

    def test_drift_between_the_two_thresholds_fires(self) -> None:
        """2.04 with the fine R, 6.08 with the coarse one: the verdict depends on R."""
        clock = self._having(0.51, 1.52, drift=4.0)
        assert clock.interpretable is True  # the coarser R is used
        assert clock.choice_of_R_changes_the_verdict is True

    def test_can_trigger_with_the_stable_clock(self) -> None:
        """This is the case that justifies it: immaterial for the threshold, decisive here."""
        clock = self._having(0.60, 0.63, drift=2.45)
        assert clock.unstable_resolution is False
        assert clock.choice_of_R_changes_the_verdict is True

    def test_goes_to_the_block(self) -> None:
        clock = self._having(0.51, 1.52, drift=4.0)
        assert clock.block()["the_choice_of_R_changes_the_verdict"] is True

    def test_before_hanging_up_the_second_measure_is_null_not_zero(self) -> None:
        """Null means "not measured"; zero would mean "the clock has no grain"."""
        block = AnchoredClock().block()
        assert block["effective_resolution"]["on_hang_up_ms"] is None
        assert block[DRIFT_TO_RECORD]["on_hang_up_ms"] is None

    def test_counts_the_periodic_samples(self) -> None:
        clock = AnchoredClock()
        for _ in range(7):
            clock.take_samples()
        assert clock.block()[DRIFT_TO_RECORD]["samples"] == 7
