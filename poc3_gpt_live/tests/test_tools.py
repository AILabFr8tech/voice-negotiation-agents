"""El contrato de tool-calling de PoC3.

Lo que se protege aquí es que el modelo vea **el mismo JSON** que en PoC1 y PoC2
—si no, la comparación del tool calling mide la redacción de una descripción— y
que la delegación quede en el modo que permite validar los montos.
"""

from __future__ import annotations

import json

import pytest

from poc3_gpt_live.app import bridge
from poc3_gpt_live.app.tools import (
    END_WITHOUT_AGREEMENT,
    FUNCTION_NAMES,
    OFFER_COMPENSATION,
    PROPOSE_PRICE,
    RECORD_AGREEMENT,
    build_delegation,
    build_schemas,
    build_output_event,
    execute_function,
    parse_arguments,
)
from shared.negotiation_engine import (
    ACCEPTED,
    RECHAZADO,
    TEST_TRIP,
    CloseReason,
    NegotiationSession,
    compensation_ids,
)


@pytest.fixture
def session() -> NegotiationSession:
    return NegotiationSession(trip=TEST_TRIP, call_id="prueba")


class TestSchemas:
    def test_are_the_4_functions_of_the_contract(self):
        assert [e["name"] for e in build_schemas()] == list(FUNCTION_NAMES)

    def test_all_declare_type_function(self):
        """La Responses API declara los tools planos, no anidados en `function`."""
        assert all(e["type"] == "function" for e in build_schemas())

    def test_the_enums_come_from_the_real_sources(self):
        by_name = {e["name"]: e for e in build_schemas()}
        kinds = by_name[OFFER_COMPENSATION]["parameters"]["properties"]["tipo"]
        assert kinds["enum"] == list(compensation_ids())
        reasons = by_name[END_WITHOUT_AGREEMENT]["parameters"]["properties"]["motivo"]
        assert reasons["enum"] == [m.value for m in CloseReason]

    def test_no_schema_hints_that_a_ceiling_exists(self):
        """Las descripciones viajan al contexto del modelo igual que el prompt."""
        text = json.dumps(build_schemas(), ensure_ascii=False).lower()
        for word in ("máximo", "maximum", "tope", "límite", "limit"):
            assert word not in text

    def test_proponer_precio_only_exposes_the_amount(self):
        by_name = {e["name"]: e for e in build_schemas()}
        properties = by_name[PROPOSE_PRICE]["parameters"]["properties"]
        assert set(properties) == {"monto"}

    def test_registrar_acuerdo_requires_amount_and_conditions(self):
        by_name = {e["name"]: e for e in build_schemas()}
        required = by_name[RECORD_AGREEMENT]["parameters"]["required"]
        assert set(required) == {"monto_final", "condiciones"}

    def test_descriptions_are_identical_to_those_of_poc2(self):
        """El contrato que ve el modelo tiene que ser el mismo en las 3 PoC.

        Se compara contra PoC2 y no contra PoC1 porque PoC2 ya guarda los schemas
        como dicts crudos; los de PoC1 son `FunctionSchema` de pipecat, con la
        misma redacción. Si esta prueba falla, alguien cambió una descripción en
        una sola PoC y la comparación del tool calling dejó de medir el framework.
        """
        from poc2_livekit.app.tools import build_schemas as schemas_poc2

        mine = {e["name"]: e for e in build_schemas()}
        theirs = {e["name"]: e for e in schemas_poc2()}
        assert set(mine) == set(theirs)
        for name, schema in mine.items():
            assert schema["description"] == theirs[name]["description"]
            assert schema["parameters"] == theirs[name]["parameters"]


class TestDelegation:
    def test_runs_in_responses_mode(self):
        """Con `client` no hay call_id ni argumentos: no se podrían validar montos."""
        assert build_delegation(model_name="gpt-4.1-mini")["type"] == "responses"

    def test_carries_the_model_passed_to_it(self):
        delegation = build_delegation(model_name="un-modelo")
        assert delegation["responses"]["model"] == "un-modelo"

    def test_parallel_calls_are_turned_off(self):
        """`proponer_precio` evalúa contra el historial: en paralelo se pisarían."""
        delegation = build_delegation(model_name="gpt-4.1-mini")
        assert delegation["responses"]["parallel_tool_calls"] is False

    def test_does_not_declare_hosted_tools(self):
        """Un `web_search` metería latencia que ninguna otra PoC paga."""
        tools = build_delegation(model_name="gpt-4.1-mini")["responses"]["tools"]
        assert {t["type"] for t in tools} == {"function"}
        assert len(tools) == 4


class TestArgumentParsing:
    def test_reads_the_json_string_the_alpha_sends(self):
        assert parse_arguments('{"monto": 16000}') == {"monto": 16000}

    @pytest.mark.parametrize(
        "raw_value", ["", "   ", None, "{no es json", "[1,2,3]", '"texto"', 42]
    )
    def test_the_unreadable_comes_out_empty_and_does_not_blow_up(self, raw_value):
        """Reventar aquí colgaría una llamada ya pagada por un error del modelo."""
        assert parse_arguments(raw_value) == {}

    def test_accepts_an_already_parsed_dict(self):
        assert parse_arguments({"monto": 1}) == {"monto": 1}


class TestExecution:
    def test_a_valid_amount_enters_the_history(self, session):
        response = execute_function(session, PROPOSE_PRICE, '{"monto": 16000}')
        assert response == {"resultado": ACCEPTED}
        assert session.offer_history == [16_000.0]

    def test_an_amount_above_the_max_is_rejected_without_details(self, session):
        """El payload no puede traer ninguna pista de dónde está el techo."""
        monto = TEST_TRIP.max_price + 5_000
        response = execute_function(session, PROPOSE_PRICE, json.dumps({"monto": monto}))
        assert response == {"resultado": RECHAZADO}

    def test_unreadable_arguments_are_rejected(self, session):
        assert execute_function(session, PROPOSE_PRICE, "{roto") == {
            "resultado": RECHAZADO
        }

    def test_an_invented_function_is_answered_instead_of_blowing_up(self, session):
        response = execute_function(session, "delete_all", "{}")
        assert response["resultado"] == "unknown_function"
        assert response["valid_functions"] == list(FUNCTION_NAMES)


class TestOutputEvent:
    def test_has_the_shape_of_the_public_api(self):
        event = build_output_event("call_1", {"resultado": ACCEPTED})
        assert event["type"] == bridge.EV_RESPONSE_ITEM_CREATE == "response.item.create"
        assert event["item"]["type"] == "function_call_output"
        assert event["item"]["call_id"] == "call_1"

    def test_the_output_travels_as_a_json_string(self):
        """No como objeto: es el detalle que más fácil se implementa mal."""
        event = build_output_event("call_1", {"resultado": ACCEPTED})
        output = event["item"]["output"]
        assert isinstance(output, str)
        assert json.loads(output) == {"resultado": ACCEPTED}

    def test_accents_are_not_escaped(self):
        event = build_output_event("c", {"descripcion": "día adicional"})
        assert "día" in event["item"]["output"]
