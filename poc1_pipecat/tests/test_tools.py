"""Contrato de tool-calling: schemas que ve el LLM y payloads que le vuelven."""

from __future__ import annotations

import json

import pytest

from poc1_pipecat.app.tools import (
    END_WITHOUT_AGREEMENT,
    FUNCTION_NAMES,
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
    ACCEPTED,
    RECHAZADO,
    AGREEMENT_RESULT,
    TripData,
    CloseReason,
    NegotiationSession,
    compensation_ids,
)

TARGET = 18_000.0
MAXIMUM = 21_000.0


@pytest.fixture()
def session() -> NegotiationSession:
    return NegotiationSession(
        trip=TripData(
            company="Freight Technologies",
            origin="Monterrey",
            destination="Laredo",
            cargo_type="caja seca",
            target_price=TARGET,
            max_price=MAXIMUM,
        ),
        call_id="CA-prueba",
    )


@pytest.fixture()
def schemas():
    pytest.importorskip("pipecat.adapters.schemas.function_schema")
    return {e.name: e for e in build_schemas()}


# --- Schemas ---


def test_the_four_names_are_those_of_the_contract(schemas):
    assert set(schemas) == set(FUNCTION_NAMES)


def test_proponer_precio_only_exposes_amount(schemas):
    """El LLM no debe poder tocar el objetivo, el máximo ni el historial."""
    schema = schemas[PROPOSE_PRICE]
    assert set(schema.properties) == {"monto"}
    assert schema.required == ["monto"]


def test_the_compensations_enum_comes_from_the_catalog(schemas):
    enum = schemas[OFFER_COMPENSATION].properties["tipo"]["enum"]
    assert list(enum) == list(compensation_ids())


def test_the_reasons_enum_comes_from_close_reason(schemas):
    enum = schemas[END_WITHOUT_AGREEMENT].properties["motivo"]["enum"]
    assert list(enum) == [m.value for m in CloseReason]


def test_registrar_acuerdo_asks_for_amount_and_conditions(schemas):
    schema = schemas[RECORD_AGREEMENT]
    assert set(schema.required) == {"monto_final", "condiciones"}
    assert schema.properties["condiciones"]["type"] == "array"


@pytest.mark.parametrize("word", ["maximum", "máximo", "tope", "limit", "límite"])
def test_no_schema_hints_that_a_ceiling_exists(schemas, word):
    """Las descripciones viajan al contexto del LLM igual que el prompt."""
    for name, schema in schemas.items():
        text = json.dumps(schema.to_default_dict(), ensure_ascii=False).lower()
        assert word not in text, f"{name} menciona {word!r}"


# --- Payloads que vuelven al LLM ---


def test_proponer_precio_returns_only_the_verdict(session):
    aceptado = execute_propose_price(session, {"monto": 16_000})
    rechazado = execute_propose_price(session, {"monto": MAXIMUM + 5_000})

    assert aceptado == {"resultado": ACCEPTED}
    assert rechazado == {"resultado": RECHAZADO}


def test_proponer_precio_without_arguments_does_not_blow_up(session):
    assert execute_propose_price(session, {}) == {"resultado": RECHAZADO}


def test_valid_compensation_returns_the_authorized_wording(session):
    primer_id = compensation_ids()[0]
    response = execute_offer_compensation(session, {"tipo": primer_id})

    assert response["resultado"] == "ofrecida"
    assert response["descripcion"]


def test_invalid_compensation_returns_the_valid_options(session):
    response = execute_offer_compensation(session, {"tipo": "secret_discount"})

    assert response["resultado"] == "no_disponible"
    assert response["compensaciones_validas"] == list(compensation_ids())


def test_conditions_as_loose_text_are_normalized(session):
    execute_propose_price(session, {"monto": 17_000})
    execute_register_agreement(
        session, {"monto_final": 17_000, "condiciones": "pago a 15 días"}
    )

    assert session.resultado is not None
    assert session.resultado.condiciones == ("pago a 15 días",)


def test_empty_conditions_do_not_break_the_close(session):
    execute_propose_price(session, {"monto": 17_000})
    response = execute_register_agreement(session, {"monto_final": 17_000})

    assert response["resultado"] == "acuerdo_registrado"
    assert session.resultado is not None
    assert session.resultado.condiciones == ()


def test_unreadable_final_amount_returns_error_and_does_not_close(session):
    """El LLM debe poder reintentar en lugar de tirar una llamada en curso."""
    response = execute_register_agreement(
        session, {"monto_final": "diecisiete mil", "condiciones": []}
    )

    assert response["resultado"] == "error"
    assert not session.is_closed


def test_reason_outside_the_enum_is_classified_as_other(session):
    response = execute_end_without_agreement(session, {"motivo": "se aburrió"})

    assert response["resultado"] == "cierre_registrado"
    assert session.resultado is not None
    assert session.resultado.motivo == "se aburrió"
    assert session.resultado.reason_category == CloseReason.OTHER.value


def test_missing_reason_is_closed_as_other(session):
    execute_end_without_agreement(session, {})

    assert session.resultado is not None
    assert session.resultado.reason_category == CloseReason.OTHER.value


def test_no_payload_mentions_the_max(session):
    """Barrido sobre todos los payloads de una negociación completa."""
    payloads = [
        execute_propose_price(session, {"monto": 16_000}),
        execute_propose_price(session, {"monto": MAXIMUM + 1_000}),
        execute_offer_compensation(session, {"tipo": compensation_ids()[0]}),
        execute_register_agreement(
            session, {"monto_final": 16_000, "condiciones": ["nada"]}
        ),
    ]
    dumped = json.dumps(payloads, ensure_ascii=False)

    assert "21000" not in dumped
    assert "21,000" not in dumped
    assert payloads[-1]["resultado"] == "acuerdo_registrado"
    assert session.resultado is not None
    assert session.resultado.resultado == AGREEMENT_RESULT
