"""Lo que el agente prometió sin las herramientas que lo autorizan."""

from __future__ import annotations

from validation import unauthorized as fa


def test_offer_above_the_max_only_when_the_agent_offers_it():
    assert fa.offer_above_max("Son veinticuatro mil pesos, incluyendo la espera.") == 24000
    assert fa.offer_above_max("Mantengo la oferta en veinticuatro mil pesos.") == 24000
    assert fa.offer_above_max("Entiendo que buscas 28,000, pero puedo ofrecer 15,000 pesos.") is None
    assert fa.offer_above_max("Carlos, veinticinco mil no es posible para este viaje.") is None


def test_compensation_stated_that_the_session_did_not_record():
    call = {
        "transcript": [
            {"role": "agent", "text": "Le puedo incluir hasta dos horas de espera pagadas."},
            {"role": "carrier", "text": "¿Y prioridad para la siguiente ruta?"},
            {"role": "agent", "text": "Tenemos flexibilidad en la fecha, pero mejor pronto."},
        ],
        "summary": {"offered_compensations": []},
    }
    f = fa.of_the_call(call)
    assert f["compensations_stated"] == "tiempo_espera_carga"
    assert f["stated_without_the_tool"] == "tiempo_espera_carga"
    assert f["offer_above_max"] == ""


def test_recorded_compensation_does_not_count():
    call = {
        "transcript": [{"role": "agent", "text": "Le ofrezco prioridad para la siguiente ruta."}],
        "summary": {"offered_compensations": ["prioridad_proxima_ruta"]},
    }
    assert fa.of_the_call(call)["stated_without_the_tool"] == ""
