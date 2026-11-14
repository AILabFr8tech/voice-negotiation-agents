"""Verificación de montos hablados del chequeo de humo.

Es la comprobación que responde a la métrica binaria de cumplimiento del
límite financiero: ¿el agente pronunció algún monto que `proponer_precio`
nunca validó?
"""

from __future__ import annotations

import pytest

from poc1_pipecat.app.smoke import amounts_in_digits, check_spoken_amounts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Te ofrezco 17,500 pesos por el viaje.", [17_500.0]),
        ("Te ofrezco 17500 pesos.", [17_500.0]),
        ("Serían 17.500 pesos.", [17_500.0]),
        ("Te doy 16000 y subo a 17000 si cierras hoy.", [16_000.0, 17_000.0]),
        # Números chicos: no son montos, son cantidades del acuerdo.
        ("Incluyo hasta 2 horas de espera y 1 día de flexibilidad.", []),
        ("Diecisiete mil quinientos pesos.", []),
    ],
)
def test_amount_extraction(text: str, expected: list[float]):
    assert amounts_in_digits(text) == expected


def test_spoken_amounts_that_came_from_the_history_pass():
    ok, detail = check_spoken_amounts(
        "Te ofrezco 16000. Puedo llegar a 17,000.", [16_000.0, 17_000.0]
    )
    assert ok
    assert "2 montos" in detail


def test_a_never_validated_amount_fails():
    """El caso que la regla no negociable prohíbe."""
    ok, detail = check_spoken_amounts(
        "Te ofrezco 16000, y si insistes le llegamos a 25000.", [16_000.0]
    )
    assert not ok
    assert "25000" in detail


def test_without_amounts_in_digits_compliance_is_not_asserted():
    """Si el agente dijo los montos con letras, hay que revisar a mano."""
    ok, detail = check_spoken_amounts("Quedamos en diecisiete mil.", [17_000.0])
    assert ok
    assert "letras" in detail


def test_one_cent_tolerance():
    ok, _ = check_spoken_amounts("Serían 17500 pesos.", [17_500.004])
    assert ok


# --- Reensamblado del texto que dijo el TTS ---


def _join(fragments):
    pytest.importorskip("pipecat.frames.frames")
    from poc1_pipecat.app.pipeline import join_tts_fragments

    return join_tts_fragments(fragments)


def test_frames_per_word_without_spaces_are_separated():
    """Lo que emite ElevenLabs: una palabra por frame, sin los espacios."""
    fragments = [("Hola,", False), ("¿hablo", False), ("con", False), ("Carlos?", False)]
    assert _join(fragments) == "Hola, ¿hablo con Carlos?"


def test_frames_that_already_have_spaces_are_left_alone():
    fragments = [("Hola, ", True), ("¿hablo con Carlos?", True)]
    assert _join(fragments) == "Hola, ¿hablo con Carlos?"


def test_existing_spaces_are_not_duplicated():
    fragments = [("Hola, ", False), ("¿qué tal?", False)]
    assert _join(fragments) == "Hola, ¿qué tal?"


def test_empty_fragments_are_ignored():
    assert _join([("", False), ("Listo", False), ("", False)]) == "Listo"


def test_without_fragments_returns_empty_string():
    assert _join([]) == ""
