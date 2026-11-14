"""PoC2's 4 negotiation functions.

Includes the **anti-drift** test against PoC1: the two schemas must be the
same JSON. If they drift apart, the LLM sees different contracts in each PoC
and the tool-calling comparison stops measuring the framework.
"""

from __future__ import annotations

import pytest

from poc2_livekit.app.tools import (
    END_WITHOUT_AGREEMENT,
    OFFER_COMPENSATION,
    PROPOSE_PRICE,
    RECORD_AGREEMENT,
    build_schemas,
    execute_end_without_agreement,
    execute_offer_compensation,
    execute_propose_price,
    execute_register_agreement,
)
from shared.negotiation_engine import (
    TEST_TRIP,
    CloseReason,
    NegotiationSession,
    compensation_ids,
)


@pytest.fixture()
def session() -> NegotiationSession:
    return NegotiationSession(trip=TEST_TRIP, call_id="prueba")


class TestSchemas:
    def test_are_the_four_of_the_prompt(self):
        names = [e["name"] for e in build_schemas()]
        assert names == [
            PROPOSE_PRICE,
            OFFER_COMPENSATION,
            RECORD_AGREEMENT,
            END_WITHOUT_AGREEMENT,
        ]

    def test_the_enums_come_from_the_real_sources(self):
        by_name = {e["name"]: e for e in build_schemas()}
        tipo = by_name[OFFER_COMPENSATION]["parameters"]["properties"]["tipo"]
        motivo = by_name[END_WITHOUT_AGREEMENT]["parameters"]["properties"]["motivo"]
        assert tipo["enum"] == list(compensation_ids())
        assert motivo["enum"] == [m.value for m in CloseReason]

    def test_no_schema_hints_that_a_ceiling_exists(self):
        text = str(build_schemas()).lower()
        for forbidden in ("maximum", "máximo", "tope", "limite superior"):
            assert forbidden not in text

    def test_without_drift_against_poc1(self):
        """The LLM must see the SAME JSON in both PoCs.

        Skipped if pipecat is not in the environment (PoC2's does not have it):
        the test runs in the root suite, which uses PoC1's environment.
        """
        pytest.importorskip("pipecat", reason="requires PoC1's environment")

        from poc1_pipecat.app.tools import build_schemas as schemas_poc1

        of_poc1 = {
            e.name: {
                "description": e.description,
                "properties": e.properties,
                "required": list(e.required),
            }
            for e in schemas_poc1()
        }
        of_poc2 = {
            e["name"]: {
                "description": e["description"],
                "properties": e["parameters"]["properties"],
                "required": list(e["parameters"]["required"]),
            }
            for e in build_schemas()
        }
        assert of_poc2 == of_poc1


class TestHandlers:
    def test_propose_only_returns_the_verdict(self, session: NegotiationSession):
        """Any extra field would be a hint about where the ceiling is."""
        response = execute_propose_price(session, {"monto": 16_000})
        assert response == {"resultado": "aceptado"}

    def test_an_amount_above_the_max_is_rejected_and_does_not_enter_the_history(
        self, session: NegotiationSession
    ):
        response = execute_propose_price(
            session, {"monto": TEST_TRIP.max_price + 1}
        )
        assert response == {"resultado": "rechazado"}
        assert session.offer_history == []

    def test_the_compensation_returns_the_authorized_wording(
        self, session: NegotiationSession
    ):
        first = next(iter(compensation_ids()))
        response = execute_offer_compensation(session, {"tipo": first})
        assert response["resultado"] == "ofrecida"
        assert response["descripcion"]

    def test_an_invented_compensation_returns_the_catalog(
        self, session: NegotiationSession
    ):
        response = execute_offer_compensation(session, {"tipo": "does_not_exist"})
        assert response["resultado"] == "no_disponible"
        assert response["compensaciones_validas"] == list(compensation_ids())

    def test_an_unreadable_amount_does_not_take_down_the_call(self, session: NegotiationSession):
        response = execute_register_agreement(
            session, {"monto_final": "no soy un número", "condiciones": []}
        )
        assert response["resultado"] == "error"
        assert not session.is_closed

    def test_a_reason_outside_the_enum_is_kept_and_classified_as_other(
        self, session: NegotiationSession
    ):
        """The free text is not lost; what gets normalized is the category.

        This matters for aggregating across PoCs: `reason_category` is the
        comparable part, and `motivo` keeps what the LLM said so it can be audited.
        """
        response = execute_end_without_agreement(session, {"motivo": "inventado"})
        assert response["resultado"] == "cierre_registrado"
        assert response["motivo"] == "inventado"
        assert session.resultado is not None
        assert session.resultado.reason_category == CloseReason.OTHER.value

    def test_the_repeated_close_keeps_the_first(self, session: NegotiationSession):
        execute_propose_price(session, {"monto": 17_000})
        execute_register_agreement(session, {"monto_final": 17_000, "condiciones": ["x"]})
        execute_end_without_agreement(session, {"motivo": CloseReason.OTHER.value})
        assert session.resultado is not None
        assert session.resultado.resultado == "acuerdo"


class TestClose:
    def test_the_close_is_notified_only_once(self, session: NegotiationSession):
        """`take_pending_close` is what prevents hanging up twice."""
        execute_propose_price(session, {"monto": 17_000})
        execute_register_agreement(session, {"monto_final": 17_000, "condiciones": []})
        assert session.take_pending_close() is not None
        assert session.take_pending_close() is None
