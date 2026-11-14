"""Negotiation session state on top of the shared engine."""

from __future__ import annotations

import json

import pytest

from shared.negotiation_engine import (
    ACCEPTED,
    RECHAZADO,
    AGREEMENT_RESULT,
    NO_AGREEMENT_RESULT,
    TripData,
    CloseReason,
    NegotiationSession,
    compensation_ids,
)

TARGET = 18_000.0
MAXIMUM = 21_000.0


@pytest.fixture()
def trip() -> TripData:
    return TripData(
        company="Freight Technologies",
        origin="Monterrey",
        destination="Laredo",
        cargo_type="caja seca",
        target_price=TARGET,
        max_price=MAXIMUM,
        carrier_name="Carlos",
    )


@pytest.fixture()
def session(trip: TripData) -> NegotiationSession:
    return NegotiationSession(trip=trip, call_id="CA-prueba")


# --- Price validation ---


def test_offer_below_the_max_is_accepted_and_enters_the_history(session):
    assert session.proponer(16_000) == ACCEPTED
    assert session.offer_history == [16_000.0]


def test_offer_above_the_max_is_rejected_and_does_not_enter_the_history(session):
    assert session.proponer(MAXIMUM + 1) == RECHAZADO
    assert session.offer_history == []


def test_non_decreasing_concession_is_rejected(session):
    session.proponer(16_000)
    session.proponer(17_000)  # increment of 1000
    assert session.proponer(19_000) == RECHAZADO  # would raise by 2000
    assert session.offer_history == [16_000.0, 17_000.0]


def test_decreasing_concession_is_accepted(session):
    session.proponer(16_000)
    session.proponer(17_000)
    assert session.proponer(17_500) == ACCEPTED
    assert session.offer_history == [16_000.0, 17_000.0, 17_500.0]


def test_revalidating_the_same_amount_does_not_duplicate_the_history(session):
    """Repeating the amount aloud is part of the close (step 6 of the prompt).

    If every revalidation entered the history, the metric's counteroffer count
    would be inflated.
    """
    session.proponer(17_000)
    assert session.proponer(17_000) == ACCEPTED
    assert session.offer_history == [17_000.0]


def test_non_numeric_amount_is_rejected_without_blowing_up(session):
    assert session.proponer("diecisiete mil") == RECHAZADO
    assert session.proponer(None) == RECHAZADO
    assert session.offer_history == []


# --- Compensations ---


def test_catalog_compensation_returns_its_description(session):
    primer_id = compensation_ids()[0]
    compensation = session.offer_compensation(primer_id)
    assert compensation is not None
    assert compensation.id == primer_id
    assert compensation.descripcion
    assert session.offered_compensations == [primer_id]


def test_made_up_compensation_is_not_recorded(session):
    assert session.offer_compensation("secret_discount") is None
    assert session.offer_compensation(42) is None
    assert session.offered_compensations == []


def test_the_same_compensation_is_not_recorded_twice(session):
    primer_id = compensation_ids()[0]
    session.offer_compensation(primer_id)
    session.offer_compensation(primer_id)
    assert session.offered_compensations == [primer_id]


# --- Closing ---


def test_closing_with_validated_amount_is_flagged_as_validated(session):
    session.proponer(17_500)
    resultado = session.close_with_agreement(17_500, ["dos horas de espera pagadas"])

    assert resultado.resultado == AGREEMENT_RESULT
    assert resultado.monto_final == 17_500.0
    assert resultado.validated_final_amount is True
    assert resultado.call_id == "CA-prueba"
    assert session.is_closed


def test_closing_with_never_validated_amount_is_flagged(session):
    """The agent spoke a number on its own: it is recorded, not hidden."""
    session.proponer(17_000)
    resultado = session.close_with_agreement(19_500, "sin condiciones")
    assert resultado.validated_final_amount is False


def test_closing_without_agreement_classifies_the_reason(session):
    resultado = session.close_without_agreement(CloseReason.FINAL_REJECTION)
    assert resultado.resultado == NO_AGREEMENT_RESULT
    assert resultado.reason_category == CloseReason.FINAL_REJECTION.value


def test_a_second_close_does_not_rewrite_the_result(session):
    session.proponer(17_000)
    session.close_with_agreement(17_000, "condiciones acordadas")
    session.close_without_agreement(CloseReason.HUNG_UP)

    assert session.resultado is not None
    assert session.resultado.resultado == AGREEMENT_RESULT
    assert session.resultado.monto_final == 17_000.0


def test_the_pending_close_is_taken_only_once(session):
    assert session.take_pending_close() is None  # not closed yet
    session.close_without_agreement(CloseReason.NO_RESPONSE)
    assert session.take_pending_close() is not None
    assert session.take_pending_close() is None


# --- Leaks ---


def test_the_summary_does_not_contain_the_max_price(session):
    session.proponer(17_500)
    session.close_with_agreement(17_500, "dos horas de espera")

    dumped = json.dumps(session.summary(), ensure_ascii=False, default=str)
    assert "max_price" not in dumped
    assert "21000" not in dumped
    assert "21,000" not in dumped
