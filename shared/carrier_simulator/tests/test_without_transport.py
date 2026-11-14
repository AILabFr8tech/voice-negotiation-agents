"""Lo que tiene que valer aunque no haya ningún SDK de voz instalado.

El detalle de cada pieza se prueba desde el arnés de cada PoC, que es donde se
usan. Lo que se prueba aquí es lo único que el arnés no puede probar por su
running_count: que el simulador **no depende de su framework**. Si mañana alguien mete
un `from livekit import …` en este paquete, PoC1 deja de poder reutilizarlo y no
hay nada más que lo detecte — la suite de PoC2 pasaría igual de verde.
"""

from __future__ import annotations

import asyncio
import tokenize
from pathlib import Path

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.carrier_simulator.player import Player

RAIZ = Path(cs.__file__).parent

# Los frameworks de voz de las 3 PoC. Ninguno puede aparecer aquí dentro.
FRAMEWORKS = ("livekit", "pipecat", "twilio", "websockets")


def _modules() -> list[Path]:
    return [r for r in sorted(RAIZ.glob("*.py"))]


def _code_without_text(ruta: Path) -> str:
    """El código sin comentarios ni literales: los docstrings SÍ nombran los
    frameworks, porque explican justamente qué pone cada PoC."""
    with ruta.open("rb") as file:
        return " ".join(
            token.string
            for token in tokenize.tokenize(file.readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )


@pytest.mark.parametrize("ruta", _modules(), ids=lambda r: r.name)
def test_no_module_names_a_voice_framework(ruta: Path) -> None:
    code_value = _code_without_text(ruta).lower()
    for framework in FRAMEWORKS:
        assert framework not in code_value, (
            f"{ruta.name} menciona {framework!r}: el simulador dejó de ser "
            "reutilizable por las otras PoC"
        )


class _TestPlayer(Player):
    """El transporte más tonto posible: guardar la trama en una lista."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frames: list[bytes] = []

    async def publish(self, raw_value: bytes) -> None:
        self.frames.append(raw_value)


def test_the_player_runs_with_any_transport() -> None:
    """Heredar y escribir `publicar` tiene que bastar. Es el contrato con PoC1."""
    observed = cs.Observed(silent=True)
    player = _TestPlayer(sample_rate=8000, observed=observed)
    clip = np.full(1600, 5000, dtype=np.int16)  # 0.2 s = 10 tramas

    async def _launch() -> float:
        halt = asyncio.Event()
        tarea = asyncio.create_task(player.launch(halt))
        end = await player.play(clip)
        halt.set()
        tarea.cancel()
        await asyncio.gather(tarea, return_exceptions=True)
        return end

    end = asyncio.run(_launch())

    assert observed.clip_frames == 10
    assert end == pytest.approx(observed.published_frames * 0.02, abs=0.02)
    # Entre clips va ruido de confort, nunca silencio digital: con DTX activo el
    # encoder dejaría de mandar paquetes y el VAD del agente no acumularía el
    # silencio observado que pide su `min_silence_duration`.
    for trama in player.frames:
        assert np.any(np.frombuffer(trama, dtype=np.int16) != 0)


def test_the_turn_minimum_is_required_over_the_measurable_ones() -> None:
    """La regla que decide si hay que volver a gastar una llamada. Vive aquí
    porque tiene que ser la misma en las 3 PoC: si cada arnés tuviera su propio
    mínimo, las tres poblarían su percentil con llamadas de largo distinto."""
    observed = cs.Observed(silent=True)
    observed.t_agent_first_speech = 1.0
    for i in range(1, 7):
        turn = cs.ObservedTurn(
            step=cs.ScriptStep(turn=f"T{i}", text="x", samples=np.zeros(8, np.int16))
        )
        turn.response = "algo dijo"
        # Los dos últimos: el agente escribió y su TTS no sonó.
        if i <= 4:
            turn.t_agent_first_speech = float(i)
        observed.turns.append(turn)

    resultado = cs.evaluate_run(observed)
    assert resultado.reached == 6
    assert resultado.measurable == 4
    assert not resultado.valid and resultado.retry
    assert resultado.no_audio == ("T5", "T6")
