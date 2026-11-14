"""Pruebas de la captura de latencia y de la voz real del agente.

Las cosas que más fácil se rompen sin que se note:

- **el cero del turno.** Si no se resta el `stop_secs` del VAD, toda la latencia
  sale sistemáticamente 200 ms más corta de lo que la persona percibe. Es un
  sesgo silencioso: los números se ven razonables y están mal.
- **la frontera del turno.** Si cada pausa del VAD abre un turno, la llamada sale
  con el doble de muestras y dos tercios vacías.
- **la bandera de interrupción.** Pipecat emite `InterruptionFrame` en cada
  arranque de turno del usuario, hable o no el agente. Usarla como señal de
  barge-in marca la llamada entera como interrumpida.
- **el reensamblado de la voz del agente.** `TTSTextFrame` hereda de
  `AggregatedTextFrame`, así que anotar detrás del TTS sin reensamblar produce
  una fila por palabra en vez de una por frase.

El reloj se inyecta para que los milisegundos esperados sean exactos y no
dependan de la máquina.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterruptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from poc1_pipecat.app.latency import (
    FIRST_TOKEN_MARK,
    AgentSpeechAnnotator,
    LatencyCollector,
)


class FakeClock:
    """Reloj monotónico controlado a mano, en segundos."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def collector(clock: FakeClock) -> LatencyCollector:
    return LatencyCollector(clock=clock)


def open_call(collector: LatencyCollector, clock: FakeClock, greeting: float = 1.0) -> None:
    """Conecta y deja que el agente termine su saludo. Cierra el turno 0."""
    collector.begin()
    clock.advance(greeting)
    collector.primer_audio()


class TestOpeningTurn:
    """El saludo del agente no tiene "fin de habla" antes: se mide desde conectar."""

    def test_is_measured_from_when_the_call_connects(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        collector.begin()
        clock.advance(2.1)  # el agente tarda 2.1 s en soltar el primer audio
        collector.primer_audio()

        samples = collector.samples
        assert len(samples) == 1
        assert samples[0].indice == 0
        assert samples[0].is_opening is True
        assert samples[0].full_turn_ms == pytest.approx(2100.0, abs=1)

    def test_goes_to_the_opening_bucket(self, collector: LatencyCollector, clock: FakeClock) -> None:
        collector.begin()
        clock.advance(1.8)
        collector.primer_audio()
        assert collector.samples[0].bucket == "apertura"

    def test_without_starting_it_does_not_invent_a_sample(self, collector: LatencyCollector) -> None:
        """Sin cero de llamada no hay nada que restar: mejor ninguna muestra."""
        collector.primer_audio()
        assert collector.samples == []


class TestTurnZero:
    def test_subtracts_the_vad_stop_secs(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """El VAD confirma 0.2 s tarde; el cero es cuando se dejó de hablar."""
        open_call(collector, clock)

        # El transportista deja de hablar en t=101.0, el VAD avisa en t=101.2.
        clock.advance(0.2)
        collector.speech_end(ago=0.2)
        collector.end_of_turn()
        clock.advance(0.8)  # el agente contesta 0.8 s después del aviso
        collector.primer_audio()

        regimen = collector.samples[1]
        # 0.2 (retraso del VAD) + 0.8 = 1.0 s desde que dejó de hablar.
        assert regimen.full_turn_ms == pytest.approx(1000.0, abs=1)

    def test_without_subtracting_it_would_come_out_shorter(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Control: con hace=0 el mismo turno mide 200 ms menos."""
        open_call(collector, clock)

        clock.advance(0.2)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.8)
        collector.primer_audio()

        assert collector.samples[1].full_turn_ms == pytest.approx(800.0, abs=1)

    def test_going_negative_does_not_advance_the_zero(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)
        collector.speech_end(ago=-5.0)
        collector.end_of_turn()
        clock.advance(0.5)
        collector.primer_audio()
        assert collector.samples[1].full_turn_ms == pytest.approx(500.0, abs=1)

    def test_the_last_end_of_speech_wins(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Una pausa a media frase no puede quedarse con el cero del turno.

        Con `stop_secs=0.2` el VAD emite un fin de habla cada vez que alguien
        respira. El cero es el último, que es cuando de verdad calló; si ganara
        el primero, `full_turn_ms` traería sumada la pausa.
        """
        open_call(collector, clock)

        collector.speech_end(ago=0.0)  # pausa a media frase
        clock.advance(1.5)  # sigue hablando
        collector.speech_end(ago=0.0)  # fin real
        collector.end_of_turn()
        clock.advance(0.6)
        collector.primer_audio()

        assert len(collector.samples) == 2
        assert collector.samples[1].full_turn_ms == pytest.approx(600.0, abs=1)


class TestGhostTurns:
    """Regresión: una pausa a media frase no abre un turno.

    Abrir un turno por cada fin de habla del VAD haría que cada pausa a media
    frase produjera una muestra vacía. El turno lo define el fin de turno del
    analizador.
    """

    def test_three_pauses_produce_a_single_turn(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        for _ in range(3):
            collector.speech_end(ago=0.2, clave=None)
            clock.advance(0.4)
        collector.end_of_turn()
        clock.advance(0.5)
        collector.primer_audio()

        assert len(collector.samples) == 2  # apertura + un turno
        assert all(m.full_turn_ms is not None for m in collector.samples)

    def test_without_end_of_turn_there_is_no_sample(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Si el analizador nunca cerró, pipecat no lo consideró un turno.

        Inventarlo aquí sería volver a inflar la cuenta. Solo queda la apertura.
        """
        open_call(collector, clock)
        for _ in range(4):
            collector.speech_end(ago=0.2)
            clock.advance(0.3)
        collector.close_pending()

        assert len(collector.samples) == 1
        assert collector.samples[0].is_opening is True


class TestStages:
    def test_full_breakdown(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """El orden real: la transcripción llega ANTES que el fin de turno."""
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        clock.advance(0.35)
        collector.transcript()
        clock.advance(0.05)
        collector.end_of_turn()
        clock.advance(0.55)
        collector.set_mark(FIRST_TOKEN_MARK)
        clock.advance(0.5)
        collector.primer_audio()

        sample = collector.samples[1]
        assert sample.stt_ms == pytest.approx(350.0, abs=1)
        assert sample.end_of_turn_ms == pytest.approx(400.0, abs=1)
        # El TTFT del LLM cuelga del fin de turno, no de la transcripción: entre
        # las dos está la decisión del analizador, que no es tiempo del LLM.
        assert sample.llm_ttft_ms == pytest.approx(550.0, abs=1)
        assert sample.tts_ttfb_ms == pytest.approx(500.0, abs=1)
        assert sample.full_turn_ms == pytest.approx(1450.0, abs=1)
        assert sample.notes == ()

    def test_the_stt_never_comes_out_negative(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Regresión: `stt_ms` negativo.

        Sale de anclar el STT en el fin de turno, que en pipecat 1.6 ocurre
        DESPUÉS de la transcripción porque el analizador la espera.
        """
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        clock.advance(0.372)
        collector.transcript()
        clock.advance(0.063)  # el analizador tarda 63 ms en decidir
        collector.end_of_turn()
        clock.advance(2.265)
        collector.set_mark(FIRST_TOKEN_MARK)
        clock.advance(0.469)
        collector.primer_audio()

        sample = collector.samples[1]
        assert sample.stt_ms == pytest.approx(372.0, abs=1)
        assert sample.stt_ms > 0

    def test_a_late_transcript_does_not_overwrite_the_turn_one(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Deepgram sigue mandando finales después de que el turno cerró."""
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        clock.advance(0.3)
        collector.transcript()
        collector.end_of_turn()
        clock.advance(2.0)
        collector.transcript()  # final tardío, ya no es de este turno
        clock.advance(0.1)
        collector.primer_audio()

        assert collector.samples[1].stt_ms == pytest.approx(300.0, abs=1)

    def test_the_last_transcript_before_end_of_turn_wins(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Deepgram parte una intervención en varios finales; cuenta el último."""
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        clock.advance(0.2)
        collector.transcript()  # "Va,"
        clock.advance(0.25)
        collector.transcript()  # "suena interesante..."
        collector.end_of_turn()
        clock.advance(0.5)
        collector.primer_audio()

        assert collector.samples[1].stt_ms == pytest.approx(450.0, abs=1)

    def test_the_marks_are_idempotent(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """La misma sonda puesta en varios puntos no debe mover la marca."""
        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.3)
        collector.set_mark(FIRST_TOKEN_MARK)
        clock.advance(1.0)
        collector.set_mark(FIRST_TOKEN_MARK)  # segunda sonda, más tarde
        clock.advance(0.1)
        collector.primer_audio()
        assert collector.samples[1].llm_ttft_ms == pytest.approx(300.0, abs=1)

    def test_in_regime_without_end_of_turn_there_is_no_ttft(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Anclar en el fin de habla le sumaría al LLM la espera del analizador.

        La caída al fin de habla existe **solo** para la apertura, donde el cero
        es el instante en que conectó la llamada.
        """
        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        # Se descarta la marca de fin de turno para simular que no llegó.
        collector._in_measurement.marks.pop("t_end_of_turn")
        clock.advance(0.4)
        collector.set_mark(FIRST_TOKEN_MARK)
        clock.advance(0.3)
        collector.primer_audio()

        assert collector.samples[1].llm_ttft_ms is None
        assert collector.samples[1].tts_ttfb_ms == pytest.approx(300.0, abs=1)

    def test_unmeasured_stage_stays_none(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """None, no cero: un cero silencioso arruinaría el p50."""
        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(1.0)
        collector.primer_audio()
        sample = collector.samples[1]
        assert sample.stt_ms is None
        assert sample.llm_ttft_ms is None
        assert sample.full_turn_ms is not None

    def test_marking_without_an_open_turn_does_not_blow_up(
        self, collector: LatencyCollector
    ) -> None:
        collector.set_mark(FIRST_TOKEN_MARK)
        assert collector.samples == []


class TestOpening:
    """El arranque en frío, incluido el caso en que contestan "¿bueno?"."""

    def test_also_measures_the_greeting_stages(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        collector.begin()
        clock.advance(2.0)
        collector.set_mark(FIRST_TOKEN_MARK)
        clock.advance(1.6)
        collector.primer_audio()

        opening = collector.samples[0]
        assert opening.is_opening is True
        assert opening.full_turn_ms == pytest.approx(3600.0, abs=1)
        # El arranque en frío del LLM se mide desde que conectó la llamada.
        assert opening.llm_ttft_ms == pytest.approx(2000.0, abs=1)
        assert opening.tts_ttfb_ms == pytest.approx(1600.0, abs=1)

    def test_answering_hello_does_not_split_the_opening_in_two(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """El transportista contesta mientras el agente todavía sintetiza.

        No se emite una muestra vacía por la apertura, y el costo del arranque en
        frío no se cuela a la cubeta de régimen: lo hereda el turno que sí sonó.
        """
        collector.begin()
        clock.advance(0.5)
        collector.speech_end(ago=0.0)  # "¿bueno?"
        collector.end_of_turn()
        clock.advance(3.1)
        collector.primer_audio()

        assert len(collector.samples) == 1
        sample = collector.samples[0]
        assert sample.is_opening is True
        assert sample.bucket == "apertura"
        assert sample.full_turn_ms == pytest.approx(3100.0, abs=1)


class TestSuccessiveTurns:
    def test_consecutive_indices(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock, greeting=2.0)
        for _ in range(3):
            collector.speech_end(ago=0.2)
            collector.end_of_turn()
            clock.advance(1.0)
            collector.primer_audio()
        assert [m.indice for m in collector.samples] == [0, 1, 2, 3]
        assert [m.bucket for m in collector.samples] == [
            "apertura",
            "regimen",
            "regimen",
            "regimen",
        ]

    def test_second_audio_of_the_same_turn_is_ignored(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Un tool call parte el turno del agente en dos tramos de voz.

        `full_turn` es hasta el **primer** audio; el segundo tramo no puede
        abrir una muestra nueva ni sobreescribir la anterior.
        """
        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.9)
        collector.primer_audio()
        clock.advance(2.0)
        collector.primer_audio()  # segundo tramo, tras la función

        assert len(collector.samples) == 2
        assert collector.samples[1].full_turn_ms == pytest.approx(900.0, abs=1)

    def test_turn_without_response_is_left_without_measurement(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """El transportista habló dos veces y el agente no contestó a la primera.

        El audio que llegue después responde al segundo turno: el LLM del primero
        quedó superado.
        """
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(3.0)
        collector.speech_end(ago=0.0)  # vuelve a hablar sin respuesta
        collector.end_of_turn()
        clock.advance(1.0)
        collector.primer_audio()

        assert len(collector.samples) == 3
        assert collector.samples[1].full_turn_ms is None
        assert collector.samples[2].full_turn_ms == pytest.approx(1000.0, abs=1)

    def test_closing_pending_on_hang_up(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        collector.close_pending()
        assert len(collector.samples) == 2
        assert collector.samples[1].full_turn_ms is None
        # Segunda vez no duplica.
        collector.close_pending()
        assert len(collector.samples) == 2


class TestSeveralProbesTheSameEvent:
    """El pipeline telefónico tiene tres sondas y cada frame las cruza todas."""

    def test_the_same_key_is_counted_once(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        # Tres sondas ven el mismo evento.
        for _ in range(3):
            collector.speech_end(ago=0.2, clave=4242)
        collector.end_of_turn(clave=99)
        clock.advance(1.0)
        collector.primer_audio()

        assert len(collector.samples) == 2
        assert collector.samples[1].full_turn_ms == pytest.approx(1200.0, abs=1)

    def test_the_same_end_of_turn_does_not_close_twice(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)
        collector.speech_end(ago=0.0, clave=1)
        for _ in range(3):
            collector.end_of_turn(clave=7)
        clock.advance(0.8)
        collector.primer_audio()

        assert len(collector.samples) == 2

    def test_different_keys_only_move_the_zero(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Dos fines de habla distintos son una intervención, no dos turnos."""
        open_call(collector, clock)

        collector.speech_end(ago=0.0, clave=1)
        clock.advance(1.0)
        collector.speech_end(ago=0.0, clave=2)
        collector.end_of_turn()
        clock.advance(1.0)
        collector.primer_audio()

        assert len(collector.samples) == 2
        assert collector.samples[1].full_turn_ms == pytest.approx(1000.0, abs=1)

    def test_frame_key_matches_the_two_copies_of_a_broadcast(self) -> None:
        """Pipecat emite dos instancias cruzadas por `broadcast_sibling_id`."""
        from pipecat.frames.frames import VADUserStoppedSpeakingFrame

        from poc1_pipecat.app.latency import frame_key

        arriba = VADUserStoppedSpeakingFrame(stop_secs=0.2)
        below = VADUserStoppedSpeakingFrame(stop_secs=0.2)
        arriba.broadcast_sibling_id = below.id
        below.broadcast_sibling_id = arriba.id

        assert arriba.id != below.id
        assert frame_key(arriba) == frame_key(below)

    def test_frame_key_without_sibling_uses_the_id(self) -> None:
        from pipecat.frames.frames import VADUserStoppedSpeakingFrame

        from poc1_pipecat.app.latency import frame_key

        frame = VADUserStoppedSpeakingFrame(stop_secs=0.2)
        assert frame_key(frame) == frame.id


class TestBargeIn:
    def test_talking_over_the_agent_marks_the_turn(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        collector.bot_started_speaking()
        collector.user_started_speaking()  # barge-in
        collector.speech_end(ago=0.2)
        collector.end_of_turn()
        clock.advance(0.4)
        collector.primer_audio()

        interrupted = collector.samples[1]
        assert interrupted.interrupted is True
        assert interrupted.bucket == "interrupted"

    def test_speaking_in_silence_is_not_barge_in(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        collector.bot_started_speaking()
        collector.bot_stopped_speaking()
        collector.user_started_speaking()  # el agente ya había callado
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(1.0)
        collector.primer_audio()

        assert collector.samples[1].interrupted is False

    def test_the_mark_does_not_carry_over_to_the_following_turn(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        collector.bot_started_speaking()
        collector.user_started_speaking()
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.5)
        collector.primer_audio()

        collector.bot_stopped_speaking()
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.9)
        collector.primer_audio()

        assert collector.samples[1].interrupted is True
        assert collector.samples[2].interrupted is False

    def test_the_interruption_frame_no_longer_marks_anything(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Regresión: turnos marcados como interrumpidos sin un solo barge-in.

        Pipecat emite `InterruptionFrame` en **cada** arranque de turno del
        usuario (`llm_response_universal.py:1209`), hable o no el agente. La
        sonda ya no lo escucha; el único barge-in real es hablar mientras el
        agente tiene la palabra.
        """
        from poc1_pipecat.app.latency import LatencyProbe

        probe = LatencyProbe(collector)
        open_call(collector, clock)

        # El agente está callado y llega el InterruptionFrame de rutina.
        asyncio.run(
            _forward(probe, InterruptionFrame(), UserStartedSpeakingFrame())
        )
        collector.speech_end(ago=0.0)
        collector.end_of_turn()
        clock.advance(0.6)
        collector.primer_audio()

        assert collector.samples[1].interrupted is False
        assert collector.samples[1].bucket == "regimen"


class TestAnalyzerVerdicts:
    """La evidencia para calibrar el turn-taking sin gastar llamadas."""

    def test_are_stored_in_order_in_the_turn_that_produced_them(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        open_call(collector, clock)

        collector.speech_end(ago=0.0)
        collector.end_of_turn_verdict(0.31)  # pausa: incompleto
        clock.advance(0.5)
        collector.speech_end(ago=0.0)
        collector.end_of_turn_verdict(0.88)  # fin real: completo
        collector.end_of_turn()
        clock.advance(0.4)
        collector.primer_audio()

        assert collector.samples[1].end_of_turn_verdicts == (0.31, 0.88)
        assert collector.samples[0].end_of_turn_verdicts == ()

    def test_do_not_enter_any_percentile(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Son evidencia, no una métrica: el resumen no debe inventar columnas."""
        from shared.metrics.latency import load_samples, summarize_samples

        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn_verdict(0.42)
        collector.end_of_turn()
        clock.advance(0.5)
        collector.primer_audio()

        summary = summarize_samples(load_samples(collector.to_json()))
        stages = summary["buckets"]["regimen"]
        assert "end_of_turn_verdicts" not in stages

    def test_a_frame_with_two_verdicts_does_not_lose_the_second(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """La clave se consume por frame, no por dato."""
        from pipecat.frames.frames import MetricsFrame
        from pipecat.metrics.metrics import TurnMetricsData

        from poc1_pipecat.app.latency import LatencyProbe

        probe = LatencyProbe(collector)
        open_call(collector, clock)
        collector.speech_end(ago=0.0)

        frame = MetricsFrame(
            data=[
                TurnMetricsData(
                    processor="BaseSmartTurn",
                    is_complete=False,
                    probability=0.2,
                    e2e_processing_time_ms=9.0,
                ),
                TurnMetricsData(
                    processor="BaseSmartTurn",
                    is_complete=True,
                    probability=0.7,
                    e2e_processing_time_ms=9.0,
                ),
            ]
        )
        asyncio.run(_forward(probe, frame))
        collector.end_of_turn()
        clock.advance(0.3)
        collector.primer_audio()

        assert collector.samples[1].end_of_turn_verdicts == (0.2, 0.7)


class TestSerialization:
    def test_to_json_is_read_by_shared_metrics(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """El contrato entre la captura y el cálculo, de punta a punta."""
        from shared.metrics.latency import load_samples, summarize_samples

        open_call(collector, clock, greeting=2.2)
        for _ in range(4):
            collector.speech_end(ago=0.2)
            collector.end_of_turn()
            clock.advance(1.1)
            collector.primer_audio()

        raw_value = collector.to_json()
        assert len(raw_value) == 5
        assert all("full_turn_ms" in m for m in raw_value)

        summary = summarize_samples(load_samples(raw_value))
        assert summary["buckets"]["apertura"]["full_turn"]["n"] == 1
        assert summary["buckets"]["regimen"]["full_turn"]["n"] == 4
        assert summary["buckets"]["regimen"]["full_turn"]["p50"] == pytest.approx(
            1300.0, abs=5
        )

    def test_the_json_survives_the_round_trip(
        self, collector: LatencyCollector, clock: FakeClock
    ) -> None:
        """Los veredictos y las notas también, aunque el JSON los vuelva listas."""
        import json

        from shared.metrics.latency import load_samples

        open_call(collector, clock)
        collector.speech_end(ago=0.0)
        collector.end_of_turn_verdict(0.55)
        collector.end_of_turn()
        clock.advance(0.4)
        collector.primer_audio()

        outbound = load_samples(json.loads(json.dumps(collector.to_json())))
        assert outbound[1].end_of_turn_verdicts == (0.55,)
        assert outbound[1] == collector.samples[1]


def _tts_text(text: str) -> TTSTextFrame:
    """Un `TTSTextFrame` como el que emite ElevenLabs: una palabra, sin espacios."""
    return TTSTextFrame(text=text, aggregated_by="word")


async def _forward(annotator: AgentSpeechAnnotator, *frames: object) -> None:
    for frame in frames:
        await annotator.process_frame(frame, FrameDirection.DOWNSTREAM)


class TestAgentSpeechAnnotator:
    """Reensambla lo que se pronunció, una fila por frase y no por palabra."""

    def test_reassembles_the_sentence(self) -> None:
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(
            _forward(
                annotator,
                _tts_text("Hola,"),
                _tts_text("habla"),
                _tts_text("Carlos"),
                TTSStoppedFrame(),
            )
        )
        # Sin reponer espacios saldría "Hola,hablaCarlos".
        assert stated_items == ["Hola, habla Carlos"]

    def test_one_row_per_sentence_not_per_word(self) -> None:
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(
            _forward(
                annotator,
                _tts_text("Primera"),
                _tts_text("frase."),
                TTSStoppedFrame(),
                _tts_text("Segunda"),
                _tts_text("frase."),
                TTSStoppedFrame(),
            )
        )
        assert stated_items == ["Primera frase.", "Segunda frase."]

    def test_dumping_twice_does_not_duplicate(self) -> None:
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(
            _forward(
                annotator,
                _tts_text("Solo"),
                _tts_text("una."),
                TTSStoppedFrame(),
                BotStoppedSpeakingFrame(),
            )
        )
        assert stated_items == ["Solo una."]

    def test_without_text_it_notes_nothing(self) -> None:
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(_forward(annotator, TTSStoppedFrame(), BotStoppedSpeakingFrame()))
        assert stated_items == []

    @pytest.mark.parametrize(
        "closure", [TTSStoppedFrame, BotStoppedSpeakingFrame, InterruptionFrame, EndFrame]
    )
    def test_dumps_with_any_close(self, closure: type) -> None:
        """Si el TTSStoppedFrame no llega, el tramo hablado no se pierde."""
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(_forward(annotator, _tts_text("algo"), closure()))
        assert stated_items == ["algo"]

    def test_the_interrupted_records_only_what_was_spoken(self) -> None:
        """El punto de anotar después del TTS.

        Si el LLM iba a decir un monto y lo cortan antes, ese monto no debe
        quedar registrado como pronunciado: produciría una falla dura falsa en la
        auditoría del límite financiero.
        """
        stated_items: list[str] = []
        annotator = AgentSpeechAnnotator(stated_items.append)
        asyncio.run(
            _forward(
                annotator,
                _tts_text("Le"),
                _tts_text("ofrezco"),
                InterruptionFrame(),  # lo cortan antes del monto
            )
        )
        assert stated_items == ["Le ofrezco"]
        assert not any("22" in d for d in stated_items)


class TestBotSpeaking:
    def test_bot_started_marks_first_audio(self) -> None:
        """Si no hubiera frames de audio del TTS, el transporte sirve de respaldo."""
        clock = FakeClock()
        collector = LatencyCollector(clock=clock)
        collector.begin()
        clock.advance(1.5)
        collector.bot_started_speaking()
        collector.primer_audio()
        assert collector.samples[0].full_turn_ms == pytest.approx(1500.0, abs=1)
