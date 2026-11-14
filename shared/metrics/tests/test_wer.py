"""Pruebas de WER, CER y error de extracción de montos.

Los valores esperados están calculados a mano: si el alineador cambia, estas
pruebas fallan con un número concreto en vez de con una sensación.
"""

from __future__ import annotations

import pytest

from shared.metrics.wer import (
    Alignment,
    AccumulatorSTT,
    align,
    compute_amounts,
    compute_text,
    combine,
)


class TestAlignment:
    """Conteos de sustitución, borrado e inserción."""

    def test_identical(self) -> None:
        tokens = ["a", "b", "c"]
        resultado = align(tokens, tokens)
        assert resultado.errors == 0
        assert resultado.hits == 3
        assert resultado.sample_rate == 0.0

    def test_a_deletion(self) -> None:
        resultado = align(
            "the cat sat on the mat".split(), "the cat sat on mat".split()
        )
        assert (resultado.substitutions, resultado.deleted, resultado.insertions) == (
            0,
            1,
            0,
        )
        assert resultado.sample_rate == pytest.approx(1 / 6)

    def test_a_substitution(self) -> None:
        resultado = align(
            "the cat sat on the mat".split(), "the cat sit on the mat".split()
        )
        assert (resultado.substitutions, resultado.deleted, resultado.insertions) == (
            1,
            0,
            0,
        )

    def test_an_insertion(self) -> None:
        resultado = align("the cat".split(), "the big cat".split())
        assert (resultado.substitutions, resultado.deleted, resultado.insertions) == (
            0,
            0,
            1,
        )
        assert resultado.sample_rate == pytest.approx(0.5)

    def test_reference_length_is_consistent(self) -> None:
        resultado = align(list("abcdef"), list("abxdef"))
        assert resultado.longitud_reference == 6

    def test_empty_reference_with_hypothesis(self) -> None:
        """No se puede dividir entre cero, pero inventar texto sí es un error."""
        resultado = align([], ["a", "b"])
        assert resultado.insertions == 2
        assert resultado.sample_rate == 1.0

    def test_everything_empty(self) -> None:
        assert align([], []).sample_rate == 0.0

    def test_rate_can_exceed_one(self) -> None:
        """Inventar más de lo que había no se recorta: esconderlo tapa un fallo."""
        resultado = align(["a"], ["x", "y", "z"])
        assert resultado.sample_rate > 1.0


class TestSum:
    def test_sums_counts(self) -> None:
        a = Alignment(hits=5, substitutions=1)
        b = Alignment(hits=3, deleted=2)
        total = a + b
        assert total.hits == 8
        assert total.substitutions == 1
        assert total.deleted == 2
        assert total.longitud_reference == 11

    def test_combine_empty(self) -> None:
        assert combine([]).errors == 0


class TestWERWithNormalization:
    def test_amount_in_words_vs_digit_is_not_an_error(self) -> None:
        resultado = compute_text(
            "Yo ando pensando en dieciocho mil.", "yo ando pensando en $18,000"
        )
        assert resultado.wer.sample_rate == 0.0
        assert resultado.cer.sample_rate == 0.0

    def test_symmetric(self) -> None:
        """Da igual de qué lado esté el dígito: el bake-off no debe premiar formato."""
        a = compute_text("déjelo en 18,000 parejo", "dejelo en dieciocho mil parejo")
        b = compute_text("dejelo en dieciocho mil parejo", "déjelo en 18,000 parejo")
        assert a.wer.sample_rate == b.wer.sample_rate == 0.0

    def test_real_error_does_count(self) -> None:
        resultado = compute_text("le doy dieciocho mil", "le doy diecinueve mil")
        assert resultado.wer.substitutions == 1
        assert resultado.wer.sample_rate == pytest.approx(1 / 3)

    def test_counts_words(self) -> None:
        resultado = compute_text("uno dos tres", "uno dos")
        assert resultado.words_reference == 3
        assert resultado.hypothesis_words == 2


class TestAmounts:
    def test_misheard_amount(self) -> None:
        resultado = compute_amounts(
            "le doy dieciocho mil", "le doy diecinueve mil"
        )
        assert resultado.alignment.substitutions == 1
        assert resultado.error_rate == 1.0
        assert resultado.expected_items == (18_000.0,)
        assert resultado.obtained == (19_000.0,)

    def test_missing_amount(self) -> None:
        resultado = compute_amounts("quedamos en 18,000", "quedamos en")
        assert resultado.alignment.deleted == 1

    def test_spurious_amount(self) -> None:
        """Inventar un monto que nadie dijo es el error más peligroso."""
        resultado = compute_amounts("quedamos así", "quedamos en 18,000")
        assert resultado.alignment.insertions == 1

    def test_without_amounts_there_is_no_error(self) -> None:
        resultado = compute_amounts("sí, soy Carlos", "si soy carlos")
        assert resultado.error_rate == 0.0

    def test_letter_vs_digit_is_correct(self) -> None:
        resultado = compute_amounts(
            "ando pensando en dieciocho mil", "ando pensando en 18,000"
        )
        assert resultado.alignment.hits == 1
        assert resultado.error_rate == 0.0

    def test_cent_tolerance(self) -> None:
        """Ruido de serialización de float no cuenta como monto distinto."""
        resultado = compute_amounts("son 18000", "son 18000.0001")
        assert resultado.alignment.hits == 1


class TestAggregation:
    """Entre llamadas se suman conteos, nunca se promedian tasas."""

    def test_sums_not_mean(self) -> None:
        # Llamada corta con un error de dos palabras: WER = 1.0
        corta = compute_text("hola mundo", "adios planeta")
        # Llamada larga sin errores: WER = 0.0
        largo = " ".join(["word"] * 98)
        long = compute_text(largo, largo)

        accumulator = AccumulatorSTT()
        accumulator.add(corta, compute_amounts("hola mundo", "adios planeta"))
        accumulator.add(long, compute_amounts(largo, largo))

        # Promediar daría (1.0 + 0.0) / 2 = 0.5, que sería absurdo: son 2
        # errores en 100 palabras.
        assert accumulator.wer.longitud_reference == 100
        assert accumulator.wer.errors == 2
        assert accumulator.wer.sample_rate == pytest.approx(0.02)

    def test_counts_calls(self) -> None:
        accumulator = AccumulatorSTT()
        for _ in range(3):
            accumulator.add(
                compute_text("a b", "a b"), compute_amounts("a b", "a b")
            )
        assert accumulator.to_dict()["calls"] == 3
