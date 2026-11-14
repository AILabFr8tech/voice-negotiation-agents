"""Pruebas de la normalización de español.

El caso que da sentido a todo el módulo es `TestMontosLetraVsDigito`: si
"dieciocho mil" y "18,000" no colapsan al mismo token, el WER mide formato de
salida del STT en vez de calidad de escucha, y el bake-off Deepgram vs.
ElevenLabs Scribe no compara nada.
"""

from __future__ import annotations

import pytest

from shared.metrics.text import (
    NormalizationConfig,
    extract_amounts,
    normalize,
    fold_accents,
    tokenize_text,
)


class TestAmountsWordVsDigit:
    """Un monto vale lo mismo dicho con letra que escrito con dígito."""

    @pytest.mark.parametrize(
        ("letter", "digit"),
        [
            ("dieciocho mil", "18,000"),
            ("dieciocho mil", "18000"),
            ("dieciocho mil", "18.000"),
            ("dieciocho mil", "$18,000"),
            ("veintiocho mil", "28,000"),
            ("veinticinco mil", "25,000"),
            ("diecisiete mil quinientos", "17,500"),
            ("veintiun mil", "21,000"),
            ("ciento cincuenta mil", "150,000"),
            ("18 mil", "18,000"),
        ],
    )
    def test_collapse_to_the_same_token(self, letter: str, digit: str) -> None:
        assert tokenize_text(letter) == tokenize_text(digit)

    def test_in_a_full_sentence(self) -> None:
        # El caso del guion: el mismo monto, una vez con letra y otra con dígito.
        with_letter = "Yo ando pensando en dieciocho mil."
        with_digit = "yo ando pensando en $18,000"
        assert normalize(with_letter) == normalize(with_digit)

    def test_different_amounts_do_not_collapse(self) -> None:
        assert tokenize_text("dieciocho mil") != tokenize_text("diecinueve mil")


class TestNumberParser:
    """Gramática de los numerales en español."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("treinta y cinco", ["35"]),
            ("cuarenta y dos", ["42"]),
            ("cien", ["100"]),
            ("ciento veintiocho", ["128"]),
            ("quinientos", ["500"]),
            ("mil", ["1000"]),
            ("mil quinientos", ["1500"]),
            ("dos millones", ["2000000"]),
            ("dieciocho mil quinientos", ["18500"]),
            ("veintiocho", ["28"]),
            ("cero", ["0"]),
        ],
    )
    def test_values(self, text: str, expected: list[str]) -> None:
        assert tokenize_text(text) == expected

    def test_the_conjunction_y_is_not_swallowed(self) -> None:
        # "y" une decena con unidad, pero aquí es conjunción de verdad.
        assert tokenize_text("el diesel y las casetas") == [
            "el",
            "diesel",
            "y",
            "las",
            "casetas",
        ]

    def test_consecutive_numbers_do_not_merge(self) -> None:
        assert tokenize_text("18000 21000") == ["18000", "21000"]

    def test_teen_does_not_combine_with_unit(self) -> None:
        # "dieciocho ocho" son dos números, no uno.
        assert tokenize_text("dieciocho ocho") == ["18", "8"]

    def test_decimal_is_preserved(self) -> None:
        assert tokenize_text("1.5 toneladas") == ["1.5", "toneladas"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Punto o coma seguidos de exactamente 3 dígitos = separador de miles.
            ("18.000", ["18000"]),
            ("18,000", ["18000"]),
            ("150,000", ["150000"]),
            # Cualquier otra cantidad de dígitos = decimal de verdad.
            ("1.5", ["1.5"]),
            ("18000.0001", ["18000.0001"]),
        ],
    )
    def test_thousands_separator_vs_decimal(
        self, text: str, expected: list[str]
    ) -> None:
        """La regla es "punto o coma + exactamente 3 dígitos = miles".

        Es una heurística con un costo conocido: "18000.001" se lee 18000001. En
        este dominio los fletes son pesos enteros, así que el caso no ocurre;
        queda anotado aquí para que sea una decisión visible y no una sorpresa.
        """
        assert tokenize_text(text) == expected

    def test_elided_amount_stays_literal(self) -> None:
        """"no, dieciocho" (queriendo decir 18,000) se lee 18, y está bien.

        Es habla real de la llamada del 3 de agosto ("Diecinueve mil, bueno, no,
        dieciocho"). Inferir el "mil" que el hablante se comió sería adivinar, y
        adivinar en una herramienta de medición puede tapar un error del STT.
        Como la misma regla se aplica a la referencia y a la hipótesis, la
        comparación sigue siendo justa.
        """
        assert tokenize_text("Diecinueve mil, bueno, no, dieciocho.") == [
            "19000",
            "bueno",
            "no",
            "18",
        ]


class TestAccentsAndCase:
    def test_folds_accents(self) -> None:
        assert fold_accents("dieciséis") == "dieciseis"

    def test_preserves_the_enie(self) -> None:
        """"año" y "ano" son palabras distintas: confundirlas SÍ es un error."""
        assert fold_accents("año") == "año"
        assert tokenize_text("año") != tokenize_text("ano")

    def test_accents_do_not_count_as_errors(self) -> None:
        assert tokenize_text("dieciséis") == tokenize_text("dieciseis")

    def test_uppercase_does_not_count(self) -> None:
        assert tokenize_text("Sí, Soy CARLOS") == tokenize_text("sí, soy carlos")

    def test_punctuation_does_not_count(self) -> None:
        assert tokenize_text("¿Sabe qué? Si me lo deja parejo, lo tomo.") == tokenize_text(
            "sabe que si me lo deja parejo lo tomo"
        )


class TestAmountExtraction:
    def test_takes_the_prices(self) -> None:
        assert extract_amounts("Le doy dieciocho mil por el viaje") == [18_000.0]

    def test_ignores_numbers_that_are_not_prices(self) -> None:
        """El umbral evita que cantidades del dominio pasen por montos."""
        assert extract_amounts("hasta 2 horas de espera y un día extra") == []
        assert extract_amounts("caja seca de 53 pies") == []

    def test_preserves_order_and_repetition(self) -> None:
        text = "primero dieciséis mil, luego 17,000, y otra vez dieciséis mil"
        assert extract_amounts(text) == [16_000.0, 17_000.0, 16_000.0]

    def test_two_consecutive_amounts_are_not_added(self) -> None:
        """Un segundo "mil" tras un "mil" empieza otro número (sumarlos daría 36,000)."""
        assert extract_amounts("dieciocho mil, dieciocho mil") == [18_000.0, 18_000.0]
        assert extract_amounts("dieciocho mil dieciocho mil quinientos") == [18_000.0, 18_500.0]
        assert extract_amounts("dos millones trescientos mil") == [2_300_000.0]
        assert extract_amounts("mil ciento veinte pesos") == [1_120.0]

    def test_configurable_threshold(self) -> None:
        config = NormalizationConfig(amount_threshold=50.0)
        assert extract_amounts("caja seca de 53 pies", config) == [53.0]


class TestConfig:
    def test_fillers_off_by_default(self) -> None:
        """Las muletillas son habla que el STT debe reconocer: no se borran."""
        assert "este" in tokenize_text("Este, mañana")

    def test_fillers_can_be_removed(self) -> None:
        config = NormalizationConfig(remove_fillers=True)
        assert "este" not in tokenize_text("Este, mañana", config)

    def test_description_documents_the_config(self) -> None:
        text = NormalizationConfig().descripcion()
        assert "acentos=plegados" in text
        assert "numeros=canonicos" in text

    def test_without_canonicalizing_numbers_stay_literal(self) -> None:
        config = NormalizationConfig(canonicalize_numbers=False)
        assert tokenize_text("dieciocho mil", config) == ["dieciocho", "mil"]


class TestEdges:
    @pytest.mark.parametrize("text", ["", "   ", "...", "¿?"])
    def test_text_without_content(self, text: str) -> None:
        assert tokenize_text(text) == []
