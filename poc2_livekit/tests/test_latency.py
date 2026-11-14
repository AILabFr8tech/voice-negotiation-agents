"""Captura de latencia de PoC2: emparejado de turnos y veredicto del cero.

Todo con métricas sintéticas: ninguna prueba necesita LiveKit, una sala ni una
llamada. Es lo que permite confiar en la instrumentación **antes** de gastar la
llamada de diagnóstico.
"""

from __future__ import annotations

import pytest

from poc2_livekit.app.latency import (
    MIN_SILENCE_DURATION,
    LatencyCollector,
)


def _user(
    *,
    stopped: float = 1_000.0,
    transcript: float = 0.18,
    end_of_turn: float = 0.9,
) -> dict[str, float]:
    """Métricas típicas de un mensaje del transportista (en segundos)."""
    return {
        "started_speaking_at": stopped - 3.0,
        "stopped_speaking_at": stopped,
        "transcription_delay": transcript,
        "end_of_turn_delay": end_of_turn,
    }


def _assistant(
    *,
    e2e: float = 1.9,
    ttft: float = 0.62,
    ttfb: float = 0.35,
    speech_start: float = 1_001.9,
) -> dict[str, float]:
    """Métricas típicas de una respuesta del agente (en segundos)."""
    return {
        "llm_node_ttft": ttft,
        "tts_node_ttfb": ttfb,
        "e2e_latency": e2e,
        "playback_latency": 0.0,
        "started_speaking_at": speech_start,
    }


class TestPairing:
    def test_a_complete_turn_maps_the_five_stages(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert sample.full_turn_ms == 1900.0
        assert sample.end_of_turn_ms == 900.0
        assert sample.stt_ms == 180.0
        assert sample.llm_ttft_ms == 620.0
        assert sample.tts_ttfb_ms == 350.0

    def test_the_greeting_without_a_prior_user_is_the_opening(self):
        """El agente habla primero: se mide del cero de la llamada al audio."""
        collector = LatencyCollector(clock=lambda: 1_000.0)
        collector.begin()
        collector.record_message(
            "assistant", _assistant(speech_start=1_002.4), interrupted=False
        )

        (sample,) = collector.samples
        assert sample.is_opening is True
        assert sample.full_turn_ms == 2400.0
        # El saludo no tiene turno de usuario antes: esas dos etapas no existen.
        assert sample.end_of_turn_ms is None
        assert sample.stt_ms is None

    def test_if_the_carrier_speaks_before_the_greeting_it_inherits_the_opening(self):
        """El "¿bueno?" mientras el agente sintetiza no debe perder el arranque en frío."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert sample.is_opening is True
        # Y aun así se mide como turno normal, con sus etapas.
        assert sample.full_turn_ms == 1900.0
        assert sample.end_of_turn_ms == 900.0

    def test_a_second_speech_segment_does_not_open_a_sample(self):
        """Tras un tool call el agente vuelve a hablar: sigue siendo el mismo turno."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("assistant", _assistant(), interrupted=False)
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)
        # Segundo tramo, sin usuario en medio.
        collector.record_message("assistant", _assistant(), interrupted=False)

        assert len(collector.samples) == 2

    def test_user_without_anchor_produces_no_sample(self):
        """LiveKit descarta el ancla cuando queda inconsistente; sin cero no hay muestra."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", {}, interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert sample.is_opening is True
        assert sample.end_of_turn_ms is None


class TestInterruption:
    def test_the_mark_goes_to_the_turn_that_interrupted(self):
        """LiveKit marca el mensaje del AGENTE que se cortó.

        En el contrato, `interrumpido` describe el turno del transportista que
        habló encima — que es el siguiente. Es el mismo criterio de PoC1.
        """
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        # Este mensaje del agente quedó cortado.
        collector.record_message("assistant", _assistant(), interrupted=True)
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        first, second_one = collector.samples
        assert first.interrupted is False
        assert second_one.interrupted is True


class TestZeroVerdict:
    """La verificación crítica, decidida con los datos de la propia llamada."""

    def _collector_with(self, end_of_turn_s: float, *, turns: int = 4) -> LatencyCollector:
        collector = LatencyCollector()
        collector.begin()
        # Un turno de apertura primero, para que los demás sean de régimen.
        collector.record_message("assistant", _assistant(), interrupted=False)
        for _ in range(turns):
            collector.record_message(
                "user", _user(end_of_turn=end_of_turn_s), interrupted=False
            )
            collector.record_message("assistant", _assistant(), interrupted=False)
        return collector

    def test_anchor_corrected_when_the_delta_includes_the_silence_wait(self):
        """0.9 s > 0.55 s de VAD: el cero ya está en el fin de habla real."""
        verdict = self._collector_with(0.9).zero_verdict()
        assert verdict["verdict"] == "anchored_on_real_end_of_speech"
        assert verdict["n_turns"] == 4
        assert "NO hay que restar nada" in verdict["detail"]

    def test_anchor_at_confirmation_when_the_delta_is_only_the_endpointing(self):
        """0.3 s: la espera de silencio ya había transcurrido. Habría que restar."""
        verdict = self._collector_with(0.3).zero_verdict()
        assert verdict["verdict"] == "anchored_on_vad_confirmation"
        assert "recalcular" in verdict["detail"]

    def test_without_regime_turns_no_verdict_is_invented(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("assistant", _assistant(), interrupted=False)
        assert collector.zero_verdict()["verdict"] == "no_data"

    def test_the_opening_does_not_enter_the_verdict(self):
        """La apertura no tiene end_of_turn_ms: no puede contaminar la decisión."""
        verdict = self._collector_with(0.9, turns=2).zero_verdict()
        assert verdict["n_turns"] == 2

    @pytest.mark.parametrize("end_of_turn_s", [0.0, 0.2, 0.49])
    def test_deltas_below_the_vad_silence_trigger_the_alarm(
        self, end_of_turn_s: float
    ):
        verdict = self._collector_with(end_of_turn_s).zero_verdict()
        assert verdict["verdict"] == "anchored_on_vad_confirmation"
        assert verdict["min_silence_duration_s"] == MIN_SILENCE_DURATION


class TestExtraBlock:
    def test_records_the_detector_version_and_whether_it_fell_back_to_local(self):
        """El detector puede caer solo de remoto a local a media llamada."""
        collector = LatencyCollector()
        collector.begin()
        collector.requested_detector = "v1"
        collector.initial_detector = "turn-detector-v1"
        collector.final_detector = "turn-detector-v1-mini"

        detector = collector.extra_block()["turn_detector"]
        assert detector["requested"] == "v1"
        assert detector["cayo_a_local"] is True

    def test_the_extras_do_not_contaminate_the_samples(self):
        """`playback_latency` y compañía van fuera de `latencias`.

        `MuestraDeTurno.from_dict` ignora las claves que no conoce, así que
        meterlas ahí las tiraría en silencio.
        """
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.to_json()
        assert "playback_latency" not in sample
        assert "e2e_latency" not in sample

        extra = collector.extra_block()
        assert extra["per_turn"][0]["assistant"]["playback_latency"] == 0.0

    def test_the_end_of_turn_verdicts_are_left_empty(self):
        """LiveKit no expone la probabilidad del audio turn detector."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert sample.end_of_turn_verdicts == ()


def _fragment(*, start: float, end: float, td: float, eot: float) -> dict[str, float]:
    """Un fragmento del turno del transportista, con sus dos anclas de habla."""
    return {
        "started_speaking_at": start,
        "stopped_speaking_at": end,
        "transcription_delay": td,
        "end_of_turn_delay": eot,
    }


class TestMultiFragmentTurn:
    """El transportista suelta su intervención en 3 pedazos con 2 pausas.

    Es el caso del escenario 5 (la autocorrección de monto a media frase). Si el
    pendiente se sobrescribiera, solo sobreviviría el último fragmento y los
    huecos de silencio quedarían irreconstruibles.
    """

    @staticmethod
    def _collector_with_three_fragments() -> LatencyCollector:
        collector = LatencyCollector()
        collector.begin()
        for frag in (
            _fragment(start=100.0, end=101.5, td=0.21, eot=0.62),
            _fragment(start=102.3, end=103.0, td=0.19, eot=0.58),  # hueco 0.80
            _fragment(start=104.1, end=105.0, td=0.24, eot=0.71),  # hueco 1.10
        ):
            collector.record_message("user", frag, interrupted=False)
        collector.record_message(
            "assistant", _assistant(e2e=1.42, ttft=0.55, ttfb=0.31), interrupted=False
        )
        return collector

    def test_the_three_fragments_land_in_per_turn(self):
        (turn,) = self._collector_with_three_fragments().extra_block()["per_turn"]
        assert len(turn["user_fragments"]) == 3

    def test_the_TWO_silence_gaps_are_captured(self):
        """Lo que de verdad se está protegiendo: con un solo fragmento guardado
        no hay ningún hueco; con el primero y el último habría uno solo, y
        además falso, porque se comería el habla del de en medio."""
        (turn,) = self._collector_with_three_fragments().extra_block()["per_turn"]
        frags = turn["user_fragments"]

        gaps = [
            round(b["started_speaking_at"] - a["stopped_speaking_at"], 3)
            for a, b in zip(frags, frags[1:])
        ]
        assert gaps == [0.8, 1.1]

    def test_the_stages_hang_off_the_last_fragment(self):
        """El cero del turno es el fin de habla del último fragmento, que es
        desde donde LiveKit mide `e2e_latency`. Si el desglose colgara del
        primero, las etapas no cuadrarían contra `full_turn_ms`."""
        (sample,) = self._collector_with_three_fragments().samples

        assert sample.full_turn_ms == 1420.0
        assert sample.end_of_turn_ms == 710.0  # f3, no los 620.0 de f1
        assert sample.stt_ms == 240.0  # f3, no los 210.0 de f1

    def test_a_single_fragment_behaves_the_same_as_before(self):
        """Regresión: el caso de un turno normal no se movió."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (turn,) = collector.extra_block()["per_turn"]
        assert len(turn["user_fragments"]) == 1
        assert turn["user"] == turn["user_fragments"][0]

    def test_a_fragment_without_anchor_is_stored_anyway(self):
        """Si se tirara, los huecos vecinos se leerían como una sola pausa larga
        cuando en realidad hubo habla en medio."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message(
            "user",
            _fragment(start=100.0, end=101.5, td=0.21, eot=0.62),
            interrupted=False,
        )
        collector.record_message("user", {"stt_metadata": {}}, interrupted=False)
        collector.record_message(
            "user",
            _fragment(start=104.1, end=105.0, td=0.24, eot=0.71),
            interrupted=False,
        )
        collector.record_message("assistant", _assistant(), interrupted=False)

        (turn,) = collector.extra_block()["per_turn"]
        frags = turn["user_fragments"]
        assert len(frags) == 3
        assert "stopped_speaking_at" not in frags[1]
        # y la muestra sigue colgando del último ANCLADO
        assert turn["user"]["stopped_speaking_at"] == 105.0


class TestTailTurn:
    def test_fragments_without_response_are_not_lost(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("assistant", _assistant(), interrupted=False)
        collector.record_message(
            "user",
            _fragment(start=200.0, end=201.0, td=0.2, eot=0.6),
            interrupted=False,
        )
        collector.record_message(
            "user",
            _fragment(start=202.5, end=203.0, td=0.2, eot=0.6),
            interrupted=False,
        )
        collector.close_pending_ones()  # la llamada se cortó aquí

        cola = collector.extra_block()["per_turn"][-1]
        assert cola["no_agent_response"] is True
        assert cola["indice"] is None
        assert len(cola["user_fragments"]) == 2
        assert len(collector.samples) == 1  # solo la apertura: no se inventó muestra

    def test_without_pending_it_adds_nothing(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)
        collector.close_pending_ones()

        assert len(collector.extra_block()["per_turn"]) == 1


class TestProviderFailures:
    """Un atorón del LLM tiene que dejar rastro en el JSON.

    Estas pruebas fijan que lo deje, y que la muestra del turno
    afectado quede marcada — sin borrarla ni anularla, porque el número que
    midió LiveKit sigue siendo cierto; lo que deja de ser cierto es que
    describa al framework de voz.
    """

    def test_the_failure_is_recorded_with_its_second(self):
        collector = LatencyCollector(clock=lambda: 1_010.0)
        collector.begin()
        collector.record_failure(
            "llm_error",
            recuperable=True,
            provider="openai",
            model_name="gpt-4.1-mini",
            detail="Request timed out.",
        )

        (failure,) = collector.failures
        assert failure["tipo"] == "llm_error"
        assert failure["recuperable"] is True
        assert failure["provider"] == "openai"
        assert failure["model_name"] == "gpt-4.1-mini"
        assert failure["t"] == 0.0

    def test_marks_the_sample_of_the_affected_turn(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        for _ in range(3):
            collector.record_failure("llm_error", recuperable=True, detail="timeout")
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert any("3 fallo(s) del proveedor" in n for n in sample.notes)
        assert any("llm_error" in n for n in sample.notes)
        # El número medido no se toca: la nota advierte, no borra.
        assert sample.full_turn_ms == 1900.0

    def test_following_turn_comes_out_clean(self):
        """Las notas no se arrastran: cada fallo marca un solo turno."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_failure("llm_error", recuperable=True, detail="timeout")
        collector.record_message("assistant", _assistant(), interrupted=False)
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        first, second_one = collector.samples
        assert any("fallo(s) del proveedor" in n for n in first.notes)
        assert not any("fallo(s) del proveedor" in n for n in second_one.notes)
        # Pero el fallo sigue en la lista de la llamada, que es la evidencia.
        assert len(collector.failures) == 1

    def test_a_failure_in_the_opening_marks_the_opening(self):
        collector = LatencyCollector(clock=lambda: 1_000.0)
        collector.begin()
        collector.record_failure("tts_error", recuperable=False, detail="ws cerrado")
        collector.record_message(
            "assistant", _assistant(speech_start=1_002.4), interrupted=False
        )

        (sample,) = collector.samples
        assert sample.is_opening is True
        assert any("tts_error" in n for n in sample.notes)

    def test_without_failures_there_are_no_failure_notes(self):
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)

        (sample,) = collector.samples
        assert collector.failures == []
        assert not any("provider" in n for n in sample.notes)


class TestHarnessZero:
    """El cableado del arnés: `livekit_extra.arnes`, nunca `latencias`."""

    def _collector_with_marks(self, *, actual_end: float = 999.88):
        from poc2_livekit.app.config import CARRIER_IDENTITY
        from poc2_livekit.app.marks import (
            MARKS_TOPIC,
            ZeroMark,
            MarksRecord,
            encode_mark,
        )

        record = MarksRecord(clock=lambda: 1_000.2, host="misma-maquina")
        record.receive(
            encode_mark(
                ZeroMark(
                    sequence=1,
                    turn="T1",
                    clip_id="scen1_t1",
                    start_of_clip=actual_end - 2.5,
                    clip_end=actual_end,
                    # 80 ms de cola: la voz termina antes que el archivo.
                    measured_end_of_speech=actual_end - 0.08,
                    drop_db=25.0,
                    speech_threshold_dbfs=-27.0,
                    publish_sample_rate=8000,
                    host="misma-maquina",
                    sent_at=actual_end + 0.001,
                )
            ),
            topic=MARKS_TOPIC,
            identity=CARRIER_IDENTITY,
        )

        collector = LatencyCollector(clock=lambda: 998.0, marks=record)
        collector.begin()
        # Apertura: el saludo, sin turno de usuario antes.
        collector.record_message(
            "assistant", _assistant(speech_start=998.6), interrupted=False
        )
        # El turno del clip.
        collector.record_message("user", _user(stopped=1_000.0), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)
        return collector

    def test_publishes_the_two_turn_biases(self):
        block = self._collector_with_marks().extra_block()["harness"]
        (turn,) = [t for t in block["per_turn"] if t["paired"]]
        # stopped_speaking_at 1000.0, fin del clip 999.88, última voz 999.80.
        assert turn["clip_bias_ms"] == pytest.approx(120.0, abs=0.5)
        assert turn["speech_bias_ms"] == pytest.approx(200.0, abs=0.5)
        assert turn["drop_db"] == 25.0

    def test_the_opening_is_not_paired_with_any_mark(self):
        """El saludo no tiene turno del transportista antes: no hay cero del
        arnés que le corresponda, y colgarle uno daría un sesgo inventado."""
        block = self._collector_with_marks().extra_block()["harness"]
        assert len(block["per_turn"]) == 1
        assert block["per_turn"][0]["indice"] == 1

    def test_without_harness_the_key_does_not_appear(self):
        """Las 15 llamadas de la campaña no tienen arnés: publicar un bloque
        vacío invitaría a leerlo como captura fallida."""
        collector = LatencyCollector()
        collector.begin()
        collector.record_message("user", _user(), interrupted=False)
        collector.record_message("assistant", _assistant(), interrupted=False)
        assert "harness" not in collector.extra_block()

    def test_the_latency_contract_does_not_change(self):
        """Lo del arnés vive en `livekit_extra`. Si se colara a `latencias`,
        `MuestraDeTurno.from_dict` lo tiraría en silencio."""
        import dataclasses

        collector = self._collector_with_marks()
        for sample in collector.samples:
            fields = set(dataclasses.asdict(sample))
            assert not any("harness" in c or "bias" in c or "cero" in c for c in fields)
            assert sample.end_of_turn_verdicts == ()
