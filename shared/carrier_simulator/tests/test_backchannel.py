"""El avance de turno sobre habla sustantiva, y el clasificador que lo decide.

La prueba que da la razón de ser del módulo es
`TestElIncidenteDel16DeSeptiembre`: reconstruye, con los tiempos medidos del WAV
de auditoría, el fragmento de 160 ms que corrió el guion un turno entero.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.carrier_simulator.verbal_backchannel import is_backchannel
from shared.carrier_simulator.observed import STREAK_HANGOVER_S
from shared.carrier_simulator.player import (
    MIN_SUBSTANTIVE_SPEECH_S,
    wait_for_agent_turn,
)

TRAMA = np.full(480, 8000, dtype=np.int16).tobytes()  # 20 ms muy por encima del umbral


class TestClassifier:
    @pytest.mark.parametrize(
        "text",
        [
            "Ajá",
            "ajá.",
            "Sí",
            "Claro.",
            "Mhm",
            "De acuerdo",
            "Muy bien",
            "Ok",
            "Órale",
            "Entiendo.",
            "Claro que sí",
            "Perfecto, gracias",
        ],
    )
    def test_backchannels_are_recognized(self, text: str) -> None:
        assert is_backchannel(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Claro, le explico la ruta",
            "Sí, le propongo 16,000 pesos",
            "Entiendo que el precio pueda parecer bajo",
            "De acuerdo con el monto que me pide",
            "16,000",
            "Monterrey",
        ],
    )
    def test_a_real_response_is_not_backchannel(self, text: str) -> None:
        assert not is_backchannel(text)

    def test_without_text_it_is_not_backchannel(self) -> None:
        """"No hay texto" no es "el texto es un apoyo". Confundirlos disfrazaría
        un fallo de transcripción de backchannel, y el arnés se quedaría
        esperando un turno que ya pasó."""
        assert not is_backchannel("")
        assert not is_backchannel("   ")

    def test_accents_do_not_change_the_verdict(self) -> None:
        assert is_backchannel("aja") and is_backchannel("ajá")
        assert is_backchannel("si") and is_backchannel("sí")


def _launch(script, *, tope=6.0, since=0.0, silence=0.4):
    """Corre la espera mientras `guion(observado)` actúa al agente."""
    observed = cs.Observed(silent=True)

    async def _everything():
        tarea = asyncio.create_task(
            wait_for_agent_turn(
                observed, since=since, silence_s=silence, tope_seg=tope
            )
        )
        await script(observed)
        return await tarea

    return observed, asyncio.run(_everything())


async def _speak(observed, seconds: float) -> None:
    end = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < end:
        observed.annotate_audio(TRAMA)
        await asyncio.sleep(0.02)


class TestTheSeptember16Incident:
    """`spots-poc2-arnes-20260916-113508`, turno 2.

    Medido del WAV de auditoría: el clip de T1 terminó en 10.84 s; el agente
    soltó 160 ms de audio a casi fondo de escala en 12.62 s, **sin producir
    transcripción**; su respuesta de verdad empezó en 20.32 s y duró 8.38 s. El
    arnés viejo avanzó a los 13.60 s y reprodujo T2 entero contra nadie.

    Aquí los tiempos van a escala para que la prueba corra en segundos.
    """

    def test_a_short_mute_fragment_does_not_advance_the_turn(self) -> None:
        async def script(observed):
            await asyncio.sleep(0.1)
            await _speak(observed, 0.16)  # el fragmento que descarriló T2
            await asyncio.sleep(STREAK_HANGOVER_S + 0.4)
            await _speak(observed, 1.0)  # la respuesta de verdad

        observed, wait = _launch(script)

        assert wait.motivo == "contestó y terminó"
        # Lo decisivo: el acuse cuelga de la respuesta, no del fragmento.
        assert wait.t_first_speech is not None
        assert wait.t_first_speech > 0.6, (
            "el arnés volvió a anclar el turno en el fragmento corto"
        )
        # Y quedó registrado, que es lo que hace detectable el desfase.
        assert len(wait.discarded_items) == 1
        discarded_one = wait.discarded_items[0]
        assert discarded_one["motivo"] == "más corta que el mínimo de voz sustantiva"
        assert discarded_one["text"] == ""
        assert 100 <= discarded_one["duration_ms"] <= 350

    def test_the_classifier_alone_would_not_have_been_enough(self) -> None:
        """El fragmento real no tenía texto: `is_backchannel` no tenía nada que
        clasificar. La red de duración es la que atrapa este caso, y por eso no
        es opcional."""
        assert not is_backchannel("")


class TestBackchannelWithText:
    def test_a_transcribed_uh_huh_does_not_count_as_a_response(self) -> None:
        async def script(observed):
            await asyncio.sleep(0.1)
            # Dura más que el mínimo, así que la duración sola lo dejaría pasar.
            await _speak(observed, MIN_SUBSTANTIVE_SPEECH_S + 0.3)
            observed.annotate_text("agent", "Ajá.")
            await asyncio.sleep(STREAK_HANGOVER_S + 0.3)
            await _speak(observed, 1.0)
            observed.annotate_text("agent", "Le propongo 16,000 pesos por el viaje.")

        _, wait = _launch(script)

        assert wait.motivo == "contestó y terminó"
        assert [d["motivo"] for d in wait.discarded_items] == [
            "apoyo verbal, no una respuesta"
        ]
        assert wait.discarded_items[0]["text"] == "Ajá."

    def test_a_short_response_with_content_does_advance(self) -> None:
        async def script(observed):
            await asyncio.sleep(0.1)
            await _speak(observed, MIN_SUBSTANTIVE_SPEECH_S + 0.2)
            observed.annotate_text("agent", "Claro, le explico la ruta.")

        _, wait = _launch(script)
        assert wait.motivo == "contestó y terminó"
        assert wait.discarded_items == []

    def test_a_long_streak_advances_even_without_text(self) -> None:
        """Si la transcripción se cae, exigirla convertiría una respuesta real en
        un turno no contestado — y eso corta la cuenta de turnos alcanzados."""

        async def script(observed):
            await asyncio.sleep(0.1)
            await _speak(observed, cs.BACKCHANNEL_CAP_S + 0.3)

        _, wait = _launch(script, tope=8.0)
        assert wait.motivo == "contestó y terminó"
        assert wait.discarded_items == []


class TestNoNoiseOnTheHealthyPath:
    def test_a_normal_turn_notes_no_discard(self) -> None:
        """La primera trama de cualquier respuesta mide cero de duración. Si se
        juzgara ahí, cada turno sano traería un descarte falso."""

        async def script(observed):
            await asyncio.sleep(0.1)
            await _speak(observed, 1.2)
            observed.annotate_text("agent", "Le propongo 16,000 pesos por el viaje.")

        _, wait = _launch(script)
        assert wait.discarded_items == []
        assert wait.motivo == "contestó y terminó"


class TestOnlyLooseFragments:
    def test_an_agent_that_only_clicks_is_distinguished_from_a_mute_one(self) -> None:
        async def script(observed):
            await asyncio.sleep(0.1)
            await _speak(observed, 0.1)

        _, wait = _launch(script, tope=1.6)
        assert "no eran su respuesta" in wait.motivo
        assert wait.t_first_speech is None
        assert len(wait.discarded_items) == 1

    def test_a_truly_mute_agent_says_something_else(self) -> None:
        async def script(observed):
            await asyncio.sleep(0.1)

        _, wait = _launch(script, tope=0.8)
        assert wait.motivo == "el agente no contestó: venció el tope de seguridad"
        assert wait.discarded_items == []


class TestStreaks:
    def test_the_contiguous_speech_is_a_single_streak(self) -> None:
        observed = cs.Observed(silent=True)
        for _ in range(5):
            observed.annotate_audio(TRAMA)
        assert len(observed.speech_streaks) == 1

    def test_a_long_silence_opens_a_new_streak(self) -> None:
        observed = cs.Observed(silent=True)
        observed.annotate_audio(TRAMA)
        first = observed.speech_streaks[0]
        # Se simula el hueco moviéndole el final a la racha abierta.
        first.end -= STREAK_HANGOVER_S + 0.2
        first.start -= STREAK_HANGOVER_S + 0.2
        observed.annotate_audio(TRAMA)
        assert len(observed.speech_streaks) == 2

    def test_only_the_streaks_after_the_zero_count(self) -> None:
        observed = cs.Observed(silent=True)
        observed.annotate_audio(TRAMA)
        assert observed.streaks_after(observed.now() + 1.0) == []
        assert len(observed.streaks_after(-1.0)) == 1


class TestWhatGetsWritten:
    def test_the_log_lists_the_turns_with_discard(self) -> None:
        observed = cs.Observed(silent=True)
        for numero, descartes in ((1, []), (2, [{"start": 12.62, "duration_ms": 160.0,
                                                 "motivo": "más corta que el mínimo "
                                                           "de voz sustantiva",
                                                 "text": ""}])):
            turn = cs.ObservedTurn(
                step=cs.ScriptStep(turn=f"T{numero}", text="x")
            )
            turn.discarded_fragments = descartes
            observed.turns.append(turn)

        event_log = cs.event_log_de_la_corrida(observed, "sala-x", scenario=1)

        assert event_log["turns_with_discard"] == ["T2"]
        assert event_log["turns"][0]["discarded_fragments"] == []
        assert event_log["turns"][1]["discarded_fragments"][0]["duration_ms"] == 160.0
        assert "corrido" in event_log["what_the_discarded_fragments_are"]

    def test_a_turn_without_discards_is_not_out_of_sync(self) -> None:
        turn = cs.ObservedTurn(step=cs.ScriptStep(turn="T1", text="x"))
        assert not turn.out_of_sync
        turn.discarded_fragments = [{"start": 1.0}]
        assert turn.out_of_sync
