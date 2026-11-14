"""Carga e interpolación del system prompt v0."""

from __future__ import annotations

import pytest

from shared.negotiation_engine import (
    TEST_TRIP,
    TripData,
    load_compensations,
)
from shared.system_prompt import (
    PROMPT_VARIABLES,
    MaxPriceLeak,
    InvalidPrompt,
    load_template,
    amount_representations,
    render_system_prompt,
    variables_in_template,
    verify_absence_of_max_price,
)


@pytest.fixture()
def minimal_template() -> str:
    """Plantilla de prueba con las 7 variables y nada más."""
    return "\n".join(f"{name}: {{{name}}}" for name in PROMPT_VARIABLES)


def test_v0_has_exactly_the_seven_variables():
    assert variables_in_template(load_template()) == set(PROMPT_VARIABLES)


def test_render_leaves_no_live_placeholders():
    render = render_system_prompt(TEST_TRIP)
    assert variables_in_template(render) == set()


def test_render_injects_the_trip_data():
    render = render_system_prompt(TEST_TRIP)
    assert TEST_TRIP.company in render
    assert TEST_TRIP.origin in render
    assert TEST_TRIP.destination in render
    assert TEST_TRIP.cargo_type in render
    assert TEST_TRIP.carrier_name in render
    assert "18,000 pesos mexicanos" in render


def test_render_injects_the_catalog_compensations():
    render = render_system_prompt(TEST_TRIP)
    for compensation in load_compensations():
        assert compensation.descripcion in render


def test_render_does_not_contain_the_max_price():
    render = render_system_prompt(TEST_TRIP)
    for shape in amount_representations(TEST_TRIP.max_price):
        assert shape not in render


@pytest.mark.parametrize("shape", sorted(amount_representations(21_000.0)))
def test_the_leak_detector_recognizes_each_representation(shape: str):
    """Control negativo: un detector roto pasaría el test de arriba siempre."""
    with pytest.raises(MaxPriceLeak):
        verify_absence_of_max_price(f"el tope es {shape}", TEST_TRIP)


def test_the_leak_message_does_not_repeat_the_amount():
    """El texto de la excepción puede terminar serializado hacia el LLM."""
    with pytest.raises(MaxPriceLeak) as error:
        verify_absence_of_max_price("el tope es 21,000", TEST_TRIP)
    assert "21" not in str(error.value)


def test_the_detector_is_skipped_if_max_equals_the_target():
    """El objetivo sí va en el prompt: sin margen, no hay nada que distinguir."""
    trip = TripData(
        company="X",
        origin="A",
        destination="B",
        cargo_type="caja",
        target_price=18_000.0,
        max_price=18_000.0,
    )
    verify_absence_of_max_price("la tarifa es 18,000", trip)


def test_template_missing_a_variable_fails(tmp_path, minimal_template):
    mutilated = minimal_template.replace("{ORIGEN}", "Monterrey")
    ruta = tmp_path / "v_mutilated.md"
    ruta.write_text(mutilated, encoding="utf-8")

    with pytest.raises(InvalidPrompt, match="ORIGEN"):
        render_system_prompt(TEST_TRIP, ruta_prompt=ruta)


def test_a_literal_brace_does_not_break_the_interpolation(tmp_path, minimal_template):
    """El caso que CLAUDE.md pide blindar: un ejemplo con JSON en el prompt.

    El sustituidor solo reconoce los 7 nombres conocidos, así que cualquier
    otra llave del archivo pasa intacta en lugar de reventar.
    """
    ruta = tmp_path / "v_con_json.md"
    ruta.write_text(
        minimal_template + '\n\nEjemplo: {"monto": 1000} y {OTRA_COSA}\n',
        encoding="utf-8",
    )

    render = render_system_prompt(TEST_TRIP, ruta_prompt=ruta)
    assert '{"monto": 1000}' in render
    assert "{OTRA_COSA}" in render


def test_a_value_with_a_placeholder_inside_fails(tmp_path, minimal_template):
    """Cierra la puerta a inyección desde los datos del viaje."""
    ruta = tmp_path / "v0.md"
    ruta.write_text(minimal_template, encoding="utf-8")

    trip = TripData(
        company="{PRECIO_OBJETIVO}",
        origin="A",
        destination="B",
        cargo_type="caja",
        target_price=18_000.0,
        max_price=21_000.0,
    )
    with pytest.raises(InvalidPrompt, match="EMPRESA"):
        render_system_prompt(trip, ruta_prompt=ruta)
