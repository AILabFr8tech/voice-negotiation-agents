"""Tests for the catalog of non-monetary compensations."""

from __future__ import annotations

import pytest

from shared.negotiation_engine.compensations import (
    COMPENSATIONS_PATH,
    Compensation,
    load_compensations,
    compensation_ids,
    get_compensation,
    render_compensations,
    validate_compensation,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The catalog is cached per path; each test starts from scratch."""
    load_compensations.cache_clear()
    yield
    load_compensations.cache_clear()


def _write_yaml(tmp_path, content: str):
    destination = tmp_path / "compensations.yaml"
    destination.write_text(content, encoding="utf-8")
    return destination


class TestRepoCatalog:
    """On the real `compensations.yaml` (placeholder pending business confirmation)."""

    def test_loads_the_three_examples(self):
        catalog = load_compensations()
        assert len(catalog) == 3
        assert all(isinstance(c, Compensation) for c in catalog)

    def test_the_ids_are_the_expected_ones(self):
        assert compensation_ids() == (
            "tiempo_espera_carga",
            "prioridad_proxima_ruta",
            "flexibilidad_fecha_entrega",
        )

    def test_the_file_lives_next_to_the_module(self):
        assert COMPENSATIONS_PATH.exists()
        assert COMPENSATIONS_PATH.name == "compensations.yaml"


class TestValidateCompensation:
    def test_a_kind_from_the_catalog_is_valid(self):
        assert validate_compensation("tiempo_espera_carga") is True

    @pytest.mark.parametrize(
        "tipo",
        [
            "fuel_discount",  # invented by the LLM
            "LOAD_WAIT_TIME",  # uppercase: not the id
            "tiempo espera carga",  # with spaces
            "",
            None,
            123,
            ["tiempo_espera_carga"],
        ],
    )
    def test_a_kind_outside_the_catalog_is_not_valid(self, tipo):
        """The agent cannot promise a compensation Spots does not offer."""
        assert validate_compensation(tipo) is False

    def test_get_returns_the_description(self):
        compensation = get_compensation("prioridad_proxima_ruta")
        assert compensation is not None
        assert "siguiente ruta" in compensation.descripcion

    def test_get_a_nonexistent_kind_returns_none(self):
        assert get_compensation("does_not_exist") is None


class TestRendering:
    def test_joins_the_prose_descriptions(self):
        prose = render_compensations()
        assert prose.startswith("hasta 2 horas")
        assert " y un día adicional" in prose
        # It is synthesized as audio: no bullets or technical ids.
        assert "\n" not in prose
        for identifier in compensation_ids():
            assert identifier not in prose

    def test_a_single_compensation_carries_no_conjunction(self, tmp_path):
        ruta = _write_yaml(
            tmp_path, '- id: unica\n  descripcion: "una sola cosa"\n'
        )
        assert render_compensations(ruta) == "una sola cosa"

    def test_two_compensations_are_joined_with_and(self, tmp_path):
        ruta = _write_yaml(
            tmp_path,
            '- id: uno\n  descripcion: "cosa uno"\n- id: dos\n  descripcion: "cosa dos"\n',
        )
        assert render_compensations(ruta) == "cosa uno y cosa dos"

    def test_the_prompt_and_the_function_do_not_go_out_of_sync(self, tmp_path):
        """Same source: every rendered description has a validatable id."""
        ruta = _write_yaml(
            tmp_path,
            '- id: uno\n  descripcion: "cosa uno"\n- id: dos\n  descripcion: "cosa dos"\n',
        )
        prose = render_compensations(ruta)
        for identifier in compensation_ids(ruta):
            compensation = get_compensation(identifier, ruta)
            assert compensation is not None
            assert compensation.descripcion in prose
            assert validate_compensation(identifier, ruta) is True


class TestInvalidCatalog:
    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_compensations(tmp_path / "no_esta.yaml")

    @pytest.mark.parametrize(
        ("content", "case"),
        [
            ("", "empty"),
            ("[]", "empty_list"),
            ("id: uno\ndescripcion: hola\n", "not_a_list"),
            ("- solo_texto\n", "record_not_mapped"),
            ('- descripcion: "sin id"\n', "no_id"),
            ('- id: uno\n', "no_description"),
            ('- id: uno\n  descripcion: "   "\n', "empty_description"),
            ('- id: "Con Mayusculas"\n  descripcion: "x"\n', "invalid_id"),
            ('- id: "1_empieza_con_numero"\n  descripcion: "x"\n', "invalid_id_2"),
            ('- id: "con-guion"\n  descripcion: "x"\n', "id_with_hyphen"),
            (
                '- id: uno\n  descripcion: "a"\n- id: uno\n  descripcion: "b"\n',
                "id_duplicate",
            ),
        ],
    )
    def test_invalid_structure_blows_up(self, tmp_path, content, case):
        ruta = _write_yaml(tmp_path, content)
        with pytest.raises(ValueError):
            load_compensations(ruta)

    def test_the_duplicate_id_is_named_in_the_error(self, tmp_path):
        """The ids are the valid values of `tipo`: a duplicate is ambiguous."""
        ruta = _write_yaml(
            tmp_path, '- id: uno\n  descripcion: "a"\n- id: uno\n  descripcion: "b"\n'
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_compensations(ruta)
