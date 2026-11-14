"""La perilla de avance: cuánto silencio del agente espera el arnés.

Con menos de 2.0 s el arnés le habla encima al agente, y no al azar, sino en las
respuestas largas, que son las del p95 de latencia. Ver
`docs/METRICS_CONTRACT.md` §2.3.2.

Lo que estas pruebas tienen que sostener, en este orden:

1. que la perilla de verdad gobierna el avance;
2. que **no** entra en la latencia: el cero es el fin de voz del arnés, no el
   instante en que decidió avanzar. Si entrara, subirla inflaría el número y el
   arreglo habría comprado una corrida válida a cambio de un percentil falso.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.carrier_simulator.player import (
    MIN_SUBSTANTIVE_SPEECH_S,
    AGENT_SILENCE_S,
    BACKCHANNEL_CAP_S,
    wait_until_agent_is_silent,
    wait_for_agent_turn,
)

TRAMA = np.full(480, 8000, dtype=np.int16).tobytes()  # 20 ms bien por encima del umbral

# La pausa interna más larga que se le midió al agente en las 10 auditorías.
LONGEST_MEASURED_PAUSE_S = 1.63

# Cuánto habla el agente actuado en cada racha. Tiene que quedar **holgadamente**
# entre los dos umbrales del clasificador: por debajo del mínimo la racha se
# descarta y la espera nunca llega a su segunda fase; por encima del tope de
# apoyo verbal se acepta sin mirar el texto y el clasificador deja de
# ejercitarse. Con 0.7 s el margen inferior era de 0.1 s y un tropiezo del bucle
# de `asyncio` bastaba para volver la prueba intermitente.
SPEECH_S = (MIN_SUBSTANTIVE_SPEECH_S + BACKCHANNEL_CAP_S) / 2.0


async def _speak(observed, seconds: float) -> None:
    """El agente habla `segundos`, con texto, para que cuente como su respuesta.

    El texto hace falta: una racha de menos de `BACKCHANNEL_CAP_S` sin
    texto se descarta por apoyo verbal, y entonces la espera nunca llega a su
    segunda fase — que es la que estas pruebas miran.
    """
    observed.annotate_text("agent", "Le propongo 16,000 pesos por el viaje.")
    end = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < end:
        observed.annotate_audio(TRAMA)
        await asyncio.sleep(0.02)


def _launch(script, *, silence: float, tope: float = 8.0):
    """Corre la espera mientras `guion(observado)` actúa al agente.

    Devuelve `(observado, espera, t_end)`. `t_end` se toma **en el instante en
    que la espera devuelve**, no al final: si se leyera después, el actor habría
    seguido hablando y ninguna prueba podría distinguir un avance temprano de
    uno tardío. El actor se cancela ahí mismo, por lo mismo.
    """
    observed = cs.Observed(silent=True)

    async def _everything():
        actor = asyncio.create_task(script(observed))
        wait = await wait_for_agent_turn(
            # `desde` va en negativo, no en 0: el cero es justo el instante en
            # que arranca el actor, y dejar la frontera pegada a la primera
            # racha hace que a veces cuente y a veces no.
            observed,
            since=-1.0,
            silence_s=silence,
            tope_seg=tope,
        )
        t_end = observed.now()
        actor.cancel()
        await asyncio.gather(actor, return_exceptions=True)
        return wait, t_end

    wait, t_end = asyncio.run(_everything())
    return observed, wait, t_end


# --------------------------------------------------------------------------
# El valor
# --------------------------------------------------------------------------


def test_the_knob_covers_the_longest_pause_measured_on_the_agent():
    """Con margen. Es una cola de 2 casos en 741: ajustada al máximo sería
    elegir el número a ojo con pasos extra."""
    assert AGENT_SILENCE_S > LONGEST_MEASURED_PAUSE_S
    assert AGENT_SILENCE_S >= LONGEST_MEASURED_PAUSE_S * 1.2


def test_the_knob_is_a_single_one_for_the_three_poc():
    """Vive en `shared/` y cada PoC la reexporta. Si una definiera la suya, el
    arnés esperaría distinto en cada arquitectura y el turn-taking dejaría de
    ser comparable."""
    assert cs.AGENT_SILENCE_S is AGENT_SILENCE_S


# --------------------------------------------------------------------------
# Gobierna el avance
# --------------------------------------------------------------------------


def _with_pause_mid_response(returns: list[float], pause: float = 0.6):
    """Lo que pasó en T4 del 20/09: el agente calla, y vuelve.

    Apunta en `vuelve` el instante **real** en que volvió a hablar. Contra un
    tiempo nominal la prueba no discrimina: `asyncio.sleep` se pasa, así que el
    guion corre más lento que su nominal y un avance temprano puede caer después
    del número esperado sin que nadie se entere.
    """

    async def script(observed):
        await _speak(observed, SPEECH_S)
        await asyncio.sleep(pause)
        returns.append(observed.now())
        await _speak(observed, SPEECH_S)
        await asyncio.sleep(10.0)

    return script


def test_the_acted_script_produces_a_response_from_the_first_streak():
    """Guarda de las de abajo, no una prueba del código.

    Si la primera racha no calificara como respuesta —por quedar corta bajo
    carga—, la espera la buscaría en la segunda y las pruebas de avance dirían
    lo contrario de lo que creen decir, de forma intermitente.
    """
    returns: list[float] = []
    _, wait, _ = _launch(_with_pause_mid_response(returns), silence=1.0)
    assert wait.t_first_speech is not None
    assert wait.t_first_speech < SPEECH_S


def test_a_pause_shorter_than_the_knob_does_not_advance():
    """El caso de T4 del 20/09.

    El control de que esto no es vacío no va aquí sino en
    `test_the_knob_lengthens_the_wait_but_not_the_first_speech`, que compara dos
    valores de la perilla entre sí. El control "con la perilla corta sí avanza
    en la pausa" se intentó y se quitó: dependía de que el avance cayera dentro
    de una ventana absoluta, y bajo carga el hangover de la racha más la
    granularidad del sondeo lo empujan fuera. Fallaba una de cada tres veces sin
    que nada estuviera roto.
    """
    returns: list[float] = []
    _, wait, t_end = _launch(_with_pause_mid_response(returns), silence=1.0)
    assert wait.motivo == "contestó y terminó"
    assert returns, "el agente nunca llegó a volver: el guion no probó nada"
    assert t_end > returns[0], "avanzó dentro de la pausa: le habría hablado encima"


def test_the_opening_wait_also_respects_the_knob():
    """Antes del primer clip el arnés usa otra función, con la misma perilla.

    Es el mismo fallo sobre el saludo: si avanza antes de tiempo le habla encima
    a la apertura del agente, y ese turno sale contaminado de barge-in.
    """
    observed = cs.Observed(silent=True)

    async def _everything():
        actor = asyncio.create_task(_speak(observed, SPEECH_S))
        motivo = await wait_until_agent_is_silent(
            observed, silence_s=0.9, tope_seg=8.0
        )
        t_end = observed.now()
        actor.cancel()
        await asyncio.gather(actor, return_exceptions=True)
        return motivo, t_end

    motivo, t_end = asyncio.run(_everything())
    assert motivo == "el agente terminó de hablar"
    assert t_end - observed.t_agent_last_speech >= 0.9


def test_the_harness_waits_at_least_the_knob_after_the_last_speech():
    async def script(observed):
        await _speak(observed, SPEECH_S)
        await asyncio.sleep(10.0)

    observed, wait, t_end = _launch(script, silence=0.9)
    assert wait.motivo == "contestó y terminó"
    assert t_end - observed.t_agent_last_speech >= 0.9


# --------------------------------------------------------------------------
# Y NO entra en la latencia
# --------------------------------------------------------------------------


def test_the_zero_of_the_turn_does_not_move_with_the_knob():
    """La latencia va del fin de voz del arnés a la PRIMERA voz del agente.

    Las dos puntas son anteriores al avance, así que subir la perilla no puede
    tocarlas. Se comprueba con la misma actuación a dos valores muy distintos.
    """

    async def script(observed):
        await asyncio.sleep(0.4)  # el agente tarda en arrancar: eso ES la latencia
        await _speak(observed, SPEECH_S)
        await asyncio.sleep(10.0)

    first_ones = []
    for silence in (0.4, 1.6):
        _, wait, _ = _launch(script, silence=silence)
        assert wait.motivo == "contestó y terminó"
        first_ones.append(wait.t_first_speech)

    # Las dos primeras voces caen en el mismo punto del guion, ±ruido de reloj.
    assert first_ones[0] == pytest.approx(first_ones[1], abs=0.15)


def test_the_knob_lengthens_the_wait_but_not_the_first_speech():
    """El control del anterior: la espera SÍ se alarga.

    Sin esto, `test_the_zero_of_the_turn_does_not_move_with_the_knob` pasaría
    también si la perilla no hiciera nada en absoluto.
    """

    async def script(observed):
        await asyncio.sleep(0.4)
        await _speak(observed, SPEECH_S)
        await asyncio.sleep(10.0)

    ends = []
    for silence in (0.4, 1.6):
        _, _, t_end = _launch(script, silence=silence)
        ends.append(t_end)

    assert ends[1] - ends[0] == pytest.approx(1.2, abs=0.3)
