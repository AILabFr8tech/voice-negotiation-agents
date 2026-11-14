"""Tests for the data contract of closing the negotiation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from shared.negotiation_engine.outcomes import (
    AGREEMENT_RESULT,
    NO_AGREEMENT_RESULT,
    CloseReason,
    finalizar_sin_acuerdo,
    registrar_acuerdo,
)

MOMENT = datetime(2026, 7, 27, 18, 30, tzinfo=timezone.utc)
HISTORIAL = (18_000.0, 21_000.0, 22_500.0)


class TestRecordAgreement:
    def test_complete_record(self):
        record = registrar_acuerdo(
            22_500,
            ["pago a 15 días", "2 horas de espera pagadas"],
            offer_history=HISTORIAL,
            offered_compensations=["tiempo_espera_carga"],
            call_id="CA123",
            timestamp=MOMENT,
        )

        assert record.resultado == AGREEMENT_RESULT
        assert record.monto_final == 22_500.0
        assert record.condiciones == ("pago a 15 días", "2 horas de espera pagadas")
        assert record.offered_compensations == ("tiempo_espera_carga",)
        assert record.offer_history == HISTORIAL
        assert record.counteroffers == 2
        assert record.validated_final_amount is True
        assert record.call_id == "CA123"
        assert record.timestamp == "2026-07-27T18:30:00+00:00"
        assert record.motivo is None

    def test_a_single_condition_is_normalized_to_a_tuple(self):
        record = registrar_acuerdo(
            22_500, "pago a 15 días", offer_history=HISTORIAL
        )
        assert record.condiciones == ("pago a 15 días",)

    def test_empty_conditions_are_discarded(self):
        record = registrar_acuerdo(
            22_500, ["pago a 15 días", "", "   "], offer_history=HISTORIAL
        )
        assert record.condiciones == ("pago a 15 días",)

    def test_a_non_textual_condition_blows_up(self):
        with pytest.raises(ValueError):
            registrar_acuerdo(22_500, [22_500], offer_history=HISTORIAL)

    def test_the_record_is_serializable_to_json(self):
        record = registrar_acuerdo(
            22_500, "sin condiciones extra", offer_history=HISTORIAL,
            timestamp=MOMENT,
        )
        raw_value = json.dumps(record.to_dict())
        assert json.loads(raw_value)["monto_final"] == 22_500.0

    def test_the_record_does_not_contain_the_max_price(self):
        """The financial ceiling is neither persisted nor exposed in the result."""
        record = registrar_acuerdo(
            22_500, "sin condiciones", offer_history=HISTORIAL, timestamp=MOMENT
        )
        assert "max_price" not in record.to_dict()

    @pytest.mark.parametrize("monto", [0, -1, float("nan"), float("inf"), "22500", None, True])
    def test_invalid_final_amount_blows_up(self, monto):
        with pytest.raises(ValueError):
            registrar_acuerdo(monto, "x", offer_history=HISTORIAL)

    def test_the_record_is_immutable(self):
        record = registrar_acuerdo(22_500, "x", offer_history=HISTORIAL)
        with pytest.raises(FrozenInstanceError):
            record.monto_final = 99_999  # type: ignore[misc]


class TestUnvalidatedAmount:
    """Safeguard: the agreed amount must come from a validated offer."""

    def test_amount_outside_the_history_is_flagged(self, caplog):
        with caplog.at_level("WARNING", logger="shared.negotiation_engine.outcomes"):
            record = registrar_acuerdo(
                23_900, "x", offer_history=HISTORIAL, call_id="CA999"
            )
        assert record.validated_final_amount is False
        assert "does not match any offer validated" in caplog.text

    def test_without_history_it_cannot_be_considered_validated(self):
        record = registrar_acuerdo(22_500, "x")
        assert record.validated_final_amount is False

    def test_tolerates_differences_of_cents(self):
        """A rounding by the LLM must not flag a real amount as unvalidated."""
        record = registrar_acuerdo(22_500.004, "x", offer_history=HISTORIAL)
        assert record.validated_final_amount is True

    def test_does_not_tolerate_differences_of_pesos(self):
        record = registrar_acuerdo(22_501, "x", offer_history=HISTORIAL)
        assert record.validated_final_amount is False


class TestFinishWithoutAgreement:
    def test_complete_record(self):
        record = finalizar_sin_acuerdo(
            CloseReason.NO_RESPONSE,
            offer_history=HISTORIAL,
            offered_compensations=["tiempo_espera_carga"],
            call_id="CA123",
            timestamp=MOMENT,
        )

        assert record.resultado == NO_AGREEMENT_RESULT
        assert record.monto_final is None
        assert record.condiciones == ()
        assert record.motivo == "sin_respuesta"
        assert record.reason_category == CloseReason.NO_RESPONSE.value
        assert record.offer_history == HISTORIAL
        assert record.counteroffers == 2
        assert record.validated_final_amount is None

    @pytest.mark.parametrize(
        "motivo",
        [
            "sin_respuesta",
            "rechazo_definitivo",
            "sin_acuerdo_tras_contraofertas",
            "colgado",
            "falla_tecnica",
        ],
    )
    def test_known_reasons_are_classified(self, motivo):
        record = finalizar_sin_acuerdo(motivo)
        assert record.reason_category == motivo
        assert record.motivo == motivo

    def test_a_free_reason_falls_into_other_without_losing_the_text(self):
        record = finalizar_sin_acuerdo("el transportista ya tomó otro viaje")
        assert record.reason_category == CloseReason.OTHER.value
        assert record.motivo == "el transportista ya tomó otro viaje"

    @pytest.mark.parametrize("motivo", ["", "   ", None, 42])
    def test_invalid_reason_blows_up(self, motivo):
        with pytest.raises(ValueError):
            finalizar_sin_acuerdo(motivo)

    def test_is_json_serializable(self):
        record = finalizar_sin_acuerdo(CloseReason.HUNG_UP, timestamp=MOMENT)
        assert json.loads(json.dumps(record.to_dict()))["resultado"] == (
            NO_AGREEMENT_RESULT
        )

    def test_default_timestamp_is_utc(self):
        record = finalizar_sin_acuerdo(CloseReason.HUNG_UP)
        assert datetime.fromisoformat(record.timestamp).tzinfo is not None
