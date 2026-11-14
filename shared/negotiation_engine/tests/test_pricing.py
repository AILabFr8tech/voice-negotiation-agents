"""Tests for price validation.

The critical case is `TestFinancialCeiling`: compliance with the financial
limit is a binary metric that must come out at 100%, so no path may return
"aceptado" above `max_price`.
"""

from __future__ import annotations

import math

import pytest

from shared.negotiation_engine.pricing import (
    ACCEPTED,
    RECHAZADO,
    proponer_precio,
    last_increment,
    validate_decreasing_concession,
)

TARGET = 20_000.0
MAXIMUM = 24_000.0


class TestFinancialCeiling:
    """`max_price` is never exceeded, by any path."""

    @pytest.mark.parametrize(
        "monto",
        [
            MAXIMUM + 0.01,  # one centavo over
            MAXIMUM + 1,  # one peso over
            MAXIMUM + 100,
            MAXIMUM * 2,
            1_000_000,  # absurd
            float("inf"),
        ],
        ids=[
            "one_centavo_over",
            "one_peso_over",
            "hundred_pesos_over",
            "the_double",
            "absurd",
            "infinite",
        ],
    )
    def test_any_amount_above_the_max_is_rejected(self, monto):
        assert proponer_precio(monto, TARGET, MAXIMUM) == RECHAZADO

    @pytest.mark.parametrize(
        "historial",
        [
            (),
            (18_000,),
            (18_000, 21_000),
            (18_000, 21_000, 23_000),
            (23_999,),  # right under the ceiling
        ],
    )
    def test_the_max_is_respected_with_any_history(self, historial):
        """Neither a 'favorable' history nor a missing history opens the ceiling."""
        assert proponer_precio(MAXIMUM + 1, TARGET, MAXIMUM, historial) == RECHAZADO

    def test_the_exact_max_is_accepted(self):
        """The ceiling is inclusive: `max_price` is payable, exceeding it is not."""
        assert proponer_precio(MAXIMUM, TARGET, MAXIMUM) == ACCEPTED

    def test_sequence_of_attempts_escalating_toward_the_max(self):
        """Simulates the LLM pushing the price up offer after offer."""
        historial: list[float] = []
        monto = 19_000.0
        accepted: list[float] = []

        while monto <= MAXIMUM + 5_000:
            verdict = proponer_precio(monto, TARGET, MAXIMUM, historial)
            if verdict == ACCEPTED:
                accepted.append(monto)
                historial.append(monto)
            monto += 500

        assert accepted, "the scenario must accept at least the first offer"
        assert max(accepted) <= MAXIMUM
        assert all(offer <= MAXIMUM for offer in historial)

    def test_verdict_does_not_leak_the_max(self):
        """The return value travels to the LLM: it can only be one of two strings."""
        verdicts = {
            proponer_precio(19_000, TARGET, MAXIMUM),
            proponer_precio(30_000, TARGET, MAXIMUM),
            proponer_precio(-5, TARGET, MAXIMUM),
        }
        assert verdicts <= {ACCEPTED, RECHAZADO}
        for verdict in verdicts:
            assert str(int(MAXIMUM)) not in verdict
            assert str(int(TARGET)) not in verdict


class TestInvalidAmounts:
    @pytest.mark.parametrize(
        "monto", [0, -1, -20_000, float("nan"), "20000", None, True, [20_000]]
    )
    def test_are_rejected(self, monto):
        assert proponer_precio(monto, TARGET, MAXIMUM) == RECHAZADO


class TestInvalidConfig:
    """An inconsistent configuration is an integration bug: it must blow up."""

    @pytest.mark.parametrize(
        ("target", "maximum"),
        [
            (20_000, 19_999),  # maximum below the target
            (0, 24_000),
            (-1, 24_000),
            (20_000, 0),
            (20_000, -1),
            (20_000, float("nan")),
            ("20000", 24_000),
            (20_000, None),
        ],
    )
    def test_raises_value_error(self, target, maximum):
        with pytest.raises(ValueError):
            proponer_precio(19_000, target, maximum)

    def test_the_error_message_does_not_include_the_amounts(self):
        """The exception may end up serialized to the LLM."""
        with pytest.raises(ValueError) as excinfo:
            proponer_precio(19_000, 20_000, 19_999)
        message = str(excinfo.value)
        assert "19999" not in message and "19,999" not in message
        assert "20000" not in message and "20,000" not in message


class TestLastIncrement:
    @pytest.mark.parametrize(
        ("historial", "expected"),
        [
            ((), None),
            ((18_000,), None),
            ((18_000, 18_000), None),  # repetition: there was no increment
            ((18_000, 20_000), 2_000),
            ((18_000, 20_000, 21_000), 1_000),
            ((18_000, 20_000, 19_000), 2_000),  # went down: the last real increase counts
        ],
    )
    def test_computes_the_last_positive_increase(self, historial, expected):
        assert last_increment(historial) == expected

    def test_history_with_invalid_value_blows_up(self):
        with pytest.raises(ValueError):
            last_increment((18_000, "20000"))


class TestDecreasingConcessions:
    def test_first_offer_always_valid(self):
        assert validate_decreasing_concession(18_000, ()) is True

    def test_first_increment_always_valid(self):
        """With a single previous offer there is no earlier increment to compare."""
        assert validate_decreasing_concession(21_000, (18_000,)) is True

    def test_increment_smaller_than_the_previous_is_valid(self):
        # 18000 -> 20000 (+2000), now +1000: concedes less. Valid.
        assert validate_decreasing_concession(21_000, (18_000, 20_000)) is True

    def test_increment_equal_to_the_previous_is_rejected(self):
        """Strictly smaller: repeating the same jump is not decreasing."""
        assert validate_decreasing_concession(22_000, (18_000, 20_000)) is False

    def test_increment_greater_than_the_previous_is_rejected(self):
        assert validate_decreasing_concession(23_000, (18_000, 20_000)) is False

    def test_repeating_the_last_amount_is_valid(self):
        """Step 6 of the prompt requires repeating the final amount before closing."""
        assert validate_decreasing_concession(20_000, (18_000, 20_000)) is True

    def test_lowering_is_valid(self):
        """Going down is not a concession and can never break the ceiling."""
        assert validate_decreasing_concession(19_000, (18_000, 20_000)) is True

    def test_lowering_and_raising_again_does_not_reset_the_margin(self):
        """The increment is measured against the highest offer already spoken.

        If it were measured against the last one (19_000), a jump to 22_000
        would look like an admissible +3_000 after a +2_000, and an increasing
        raise would slip through.
        """
        historial = (18_000, 20_000, 19_000)
        assert validate_decreasing_concession(22_000, historial) is False
        assert validate_decreasing_concession(21_500, historial) is True

    def test_invalid_amount_does_not_pass(self):
        assert validate_decreasing_concession(-1, (18_000,)) is False
        assert validate_decreasing_concession("20000", (18_000,)) is False


class TestProposePriceIntegration:
    def test_negotiation_with_decreasing_concessions(self):
        """Happy path: anchors low and concedes less each time, always under the ceiling."""
        historial: list[float] = []
        for offer in (18_000, 21_000, 22_500, 23_200):
            assert proponer_precio(offer, TARGET, MAXIMUM, historial) == ACCEPTED
            historial.append(offer)

        # A jump that breaks the concession rule (+800 > +700) is rejected...
        assert proponer_precio(24_000, TARGET, MAXIMUM, historial) == RECHAZADO
        # ...and one that respects it and stays under the ceiling is accepted.
        assert proponer_precio(23_800, TARGET, MAXIMUM, historial) == ACCEPTED

    def test_exceeding_the_target_is_allowed(self):
        """The target is a reference, not a barrier: the hard ceiling is the maximum."""
        assert proponer_precio(TARGET + 1_000, TARGET, MAXIMUM) == ACCEPTED

    def test_default_history_is_empty(self):
        assert proponer_precio(19_000, TARGET, MAXIMUM) == ACCEPTED

    def test_the_rejection_log_does_not_include_the_max(self, caplog):
        """The reason is logged on the server, but without revealing the ceiling."""
        with caplog.at_level("INFO", logger="shared.negotiation_engine.pricing"):
            proponer_precio(30_000, TARGET, MAXIMUM)
        text = caplog.text
        assert "exceeds_max_price" in text
        assert "24000" not in text and "24000.0" not in text

    def test_decimal_amounts_work(self):
        assert proponer_precio(23_999.99, TARGET, MAXIMUM) == ACCEPTED
        assert proponer_precio(24_000.01, TARGET, MAXIMUM) == RECHAZADO

    def test_there_are_no_side_effects_on_the_history(self):
        """The function is pure: it does not mutate its inputs."""
        historial = [18_000, 20_000]
        clone = list(historial)
        proponer_precio(21_000, TARGET, MAXIMUM, historial)
        assert historial == clone

    def test_max_equal_to_target_is_a_valid_config(self):
        """No margin over the target: valid, but the ceiling holds firm."""
        assert proponer_precio(TARGET, TARGET, TARGET) == ACCEPTED
        assert proponer_precio(TARGET + 0.01, TARGET, TARGET) == RECHAZADO

    def test_nan_does_not_slip_through_the_comparison(self):
        """`nan > x` is False: without the finiteness check it would pass the ceiling."""
        assert math.isnan(float("nan"))
        assert proponer_precio(float("nan"), TARGET, MAXIMUM) == RECHAZADO
