"""Tras un apoyo o un control en que el agente se calló, el arnés espera a que retome."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.carrier_simulator import interruptions as it

VOICE = (np.ones(160, dtype=np.int16) * 3000).tobytes()


@pytest.fixture(autouse=True)
def scale(monkeypatch):
    """Los umbrales a escala 1:10 para que la prueba dure poco."""
    monkeypatch.setattr(it, "AGENT_SILENCE_S", 0.2)
    monkeypatch.setattr(it, "NO_REPLY_WAIT_S", 0.8)
    monkeypatch.setattr(it, "STOP_SILENCE_S", 0.07)
    monkeypatch.setattr(it, "STOP_WINDOW_S", 0.3)
    monkeypatch.setattr(it, "STREAK_HANGOVER_S", 0.03)
    monkeypatch.setattr(cs.observed, "STREAK_HANGOVER_S", 0.03)


async def _speak(o: cs.Observed, seg: float) -> None:
    end = asyncio.get_running_loop().time() + seg
    while asyncio.get_running_loop().time() < end:
        o.annotate_audio(VOICE)
        await asyncio.sleep(0.01)


async def _launch(tipo: str, script) -> tuple[str, float, dict]:
    o = cs.Observed(silent=True)
    await _speak(o, 0.1)
    reference = o.now()
    agent = asyncio.create_task(script(o))
    t0 = asyncio.get_running_loop().time()
    motivo = await it._wait_after_event(
        o, tipo=tipo, reference=reference, end_of_speech=reference, tope_seg=5.0, closed=None,
    )
    tardo = asyncio.get_running_loop().time() - t0
    await agent
    m = it.measure_event(list(o.speech_streaks), reference=reference, end_of_speech=reference,
                           tipo=tipo, until=o.now())
    return motivo, tardo, m


def test_waits_for_the_agent_that_resumes_after_the_2_s():
    async def script(o):
        await asyncio.sleep(0.5)  # más que el silencio normal (0.2): antes se le hablaba encima
        await _speak(o, 0.1)

    motivo, tardo, m = asyncio.run(_launch("apoyo", script))
    assert motivo == "retomó después de detenerse y terminó"
    assert tardo == pytest.approx(0.8, abs=0.15)
    assert m["endless_pause"] is False and m["pause_duration_ms"] == pytest.approx(500, abs=60)


def test_if_it_does_not_resume_it_waits_for_the_silence_cap():
    async def script(o):
        await asyncio.sleep(0)

    motivo, tardo, m = asyncio.run(_launch("control", script))
    assert motivo.startswith("se detuvo y no volvió a hablar")
    assert tardo == pytest.approx(0.8, abs=0.15)
    assert m["endless_pause"] is True


def test_if_it_did_not_stop_the_normal_rule_applies():
    async def script(o):
        await _speak(o, 0.5)  # habla más allá de la ventana de detención

    motivo, tardo, _ = asyncio.run(_launch("apoyo", script))
    assert motivo == "terminó de hablar"
    assert tardo == pytest.approx(0.7, abs=0.15)
