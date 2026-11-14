"""Pruebas de la capa de telefonía. Nada de red, nada de Twilio, costo cero.

Lo que se puede probar sin llamada: el TwiML, el registro del diálogo, la
persistencia de la corrida y la estimación de costo. El pipeline de voz en sí
necesita un WebSocket real de Twilio, así que su verificación vive en la
llamada de humo, no aquí.
"""

from __future__ import annotations

import json
from xml.etree import ElementTree

import pytest

from poc1_pipecat.app.call import (
    AI_COST_PER_CALL,
    DEFAULT_LIMIT_SECONDS,
    estimate_cost,
    run_twiml_url,
)
from poc1_pipecat.app.telephony import (
    CallRecord,
    build_twiml,
    save_run,
    read_identification,
)
from shared.negotiation_engine import TripData, NegotiationSession


def _fake_settings():
    """Ajustes mínimos para armar URLs. No toca `.env` ni la red."""
    from poc1_pipecat.app.config import Settings

    return Settings(
        openai_api_key="x",
        openai_model="gpt-4.1-mini",
        deepgram_api_key="x",
        deepgram_model="nova-3",
        deepgram_language="multi",
        eleven_api_key="x",
        eleven_model="eleven_flash_v2_5",
        public_url="https://ejemplo.ngrok.app",
    )


TRIP = TripData(
    company="Freight Technologies",
    origin="Monterrey, Nuevo León",
    destination="Laredo, Texas",
    cargo_type="caja seca de 53 pies",
    target_price=18_000.0,
    max_price=21_000.0,
    carrier_name="Carlos",
)


class TestTwiML:
    def test_is_valid_xml(self):
        root = ElementTree.fromstring(build_twiml("wss://ejemplo.dev/ws"))
        assert root.tag == "Response"

    def test_connects_the_bidirectional_stream(self):
        """`Connect` es bidireccional; con `Start` el transportista no oiría nada."""
        root = ElementTree.fromstring(build_twiml("wss://ejemplo.dev/ws"))
        connect = root.find("Connect")
        assert connect is not None, "el TwiML debe usar <Connect>, no <Start>"
        assert root.find("Start") is None
        stream = connect.find("Stream")
        assert stream is not None
        assert stream.get("url") == "wss://ejemplo.dev/ws"

    def test_escapes_the_url(self):
        """Una URL con `&` rompería el XML si se pegara sin escapar."""
        xml = build_twiml("wss://ejemplo.dev/ws?a=1&b=2")
        root = ElementTree.fromstring(xml)  # no debe levantar
        stream = root.find("Connect/Stream")
        assert stream is not None
        assert stream.get("url") == "wss://ejemplo.dev/ws?a=1&b=2"

    def test_does_not_include_verbs_that_speak(self):
        """Ningún `Say` ni `Play`: la voz la pone ElevenLabs por el stream."""
        xml = build_twiml("wss://ejemplo.dev/ws")
        assert "<Say" not in xml and "<Play" not in xml


class TestRunStamping:
    """La etiqueta del escenario viaja con la llamada, no en la memoria de nadie.

    Con 45 llamadas (15 × 3 PoC) todas nombradas por Call SID, reconstruir después
    cuál fue cuál es una fuente de error garantizada. La cadena es: marcador ->
    query string del TwiML -> `<Parameter>` del `<Stream>` -> `customParameters`
    del WebSocket -> JSON de la corrida.
    """

    def test_emits_the_parameters_in_the_stream(self):
        xml = build_twiml(
            "wss://ejemplo.dev/ws",
            {"poc": "1", "scenario": "2", "repetition": "3"},
        )
        stream = ElementTree.fromstring(xml).find("Connect/Stream")
        assert stream is not None
        parameters = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
        assert parameters == {"poc": "1", "scenario": "2", "repetition": "3"}

    def test_without_parameters_the_stream_is_clean(self):
        stream = ElementTree.fromstring(
            build_twiml("wss://ejemplo.dev/ws")
        ).find("Connect/Stream")
        assert stream is not None
        assert stream.findall("Parameter") == []

    def test_omits_the_empties(self):
        xml = build_twiml(
            "wss://ejemplo.dev/ws", {"poc": "1", "scenario": "", "repetition": "  "}
        )
        stream = ElementTree.fromstring(xml).find("Connect/Stream")
        assert [p.get("name") for p in stream.findall("Parameter")] == ["poc"]

    def test_escapes_the_values(self):
        """Un valor con comillas o `&` rompería el XML si se pegara sin escapar."""
        xml = build_twiml("wss://ejemplo.dev/ws", {"scenario": 'a"&b'})
        stream = ElementTree.fromstring(xml).find("Connect/Stream")
        assert stream.find("Parameter").get("value") == 'a"&b'

    def test_reads_the_twilio_custom_parameters(self):
        label = read_identification(
            {"customParameters": {"poc": "1", "scenario": "2", "repetition": "3"}}
        )
        assert label == {"poc": "1", "scenario": "2", "repetition": "3"}

    def test_tolerates_uppercase_and_extra_keys(self):
        """Un error leyendo esto no debe tumbar una llamada que ya está sonando."""
        label = read_identification(
            {"customParameters": {"Scenario": "4", "POC": "1", "otherThing": "x"}}
        )
        assert label == {"poc": "1", "scenario": "4"}

    @pytest.mark.parametrize(
        "start",
        [{}, {"customParameters": {}}, {"customParameters": None}, {"customParameters": "odd"}],
    )
    def test_without_parameters_returns_empty(self, start):
        assert read_identification(start) == {}

    def test_the_dialer_url_carries_the_query_string(self):
        from urllib.parse import parse_qs, urlparse

        settings = _fake_settings()
        url = run_twiml_url(settings, poc=1, scenario=2, repetition=3)
        parts = urlparse(url)
        assert parse_qs(parts.query) == {
            "poc": ["1"],
            "scenario": ["2"],
            "repetition": ["3"],
        }

    def test_without_a_label_the_url_stays_the_same(self):
        settings = _fake_settings()
        url = run_twiml_url(settings, poc=None, scenario=None, repetition=None)
        assert "?" not in url

    def test_the_json_is_labeled(self, tmp_path):
        """Es lo que `shared.metrics` lee para saber que la llamada es la 2.3."""
        session = NegotiationSession(trip=TRIP, call_id="CAetiquetada")
        record = CallRecord()
        record.annotate("agent", "Buenas tardes.")

        destination = save_run(
            session,
            record,
            base_path=tmp_path,
            identification={"poc": "1", "scenario": "2", "repetition": "3"},
        )
        data = json.loads(destination.read_text(encoding="utf-8"))
        assert (data["poc"], data["scenario"], data["repetition"]) == (1, 2, 3)

        from shared.metrics.calls import identify

        assert identify(data, None) == (1, 2, 3)

    def test_non_numeric_value_is_kept(self, tmp_path):
        """Perder trazabilidad es peor que guardar un valor raro."""
        session = NegotiationSession(trip=TRIP, call_id="CAraro")
        destination = save_run(
            session,
            CallRecord(),
            base_path=tmp_path,
            identification={"scenario": "calentamiento"},
        )
        data = json.loads(destination.read_text(encoding="utf-8"))
        assert data["scenario"] == "calentamiento"


class TestLatencyInTheJSON:
    """El bloqueante número uno: sin esto no hay criterio de decisión de latencia."""

    def _collector_with_turns(self):
        from poc1_pipecat.app.latency import LatencyCollector

        clock = {"t": 100.0}

        def now() -> float:
            return clock["t"]

        collector = LatencyCollector(clock=now)
        collector.begin()
        clock["t"] += 2.0
        collector.primer_audio()  # apertura
        for _ in range(3):
            # Dos fines de habla por intervención, como en una llamada real: la
            # pausa a media frase no debe producir un turno de más.
            collector.speech_end(ago=0.2, clave=None)
            clock["t"] += 0.3
            collector.speech_end(ago=0.2, clave=None)
            collector.transcript()
            collector.end_of_turn()
            clock["t"] += 1.0
            collector.primer_audio()
        return collector

    def test_writes_the_latencies_key(self, tmp_path):
        session = NegotiationSession(trip=TRIP, call_id="CAlatencia")
        destination = save_run(
            session,
            CallRecord(),
            base_path=tmp_path,
            collector=self._collector_with_turns(),
        )
        data = json.loads(destination.read_text(encoding="utf-8"))
        assert "latencies" in data
        assert len(data["latencies"]) == 4
        assert all(m["full_turn_ms"] is not None for m in data["latencies"])

    def test_without_a_collector_it_does_not_write_the_key(self, tmp_path):
        """Compatibilidad: un JSON viejo sin latencias sigue siendo válido."""
        session = NegotiationSession(trip=TRIP, call_id="CAsinlatencia")
        destination = save_run(session, CallRecord(), base_path=tmp_path)
        data = json.loads(destination.read_text(encoding="utf-8"))
        assert "latencies" not in data

    def test_shared_metrics_extracts_percentiles_from_the_json(self, tmp_path):
        """De punta a punta: captura -> JSON -> cálculo, sin intervención manual."""
        from shared.metrics.calls import compute

        session = NegotiationSession(trip=TRIP, call_id="CAcompleta")
        record = CallRecord()
        record.annotate("agent", "Le ofrezco dieciséis mil pesos.")
        record.annotate("carrier", "está muy bajo")

        destination = save_run(
            session,
            record,
            base_path=tmp_path,
            collector=self._collector_with_turns(),
            identification={"poc": "1", "scenario": "1", "repetition": "1"},
        )
        data = json.loads(destination.read_text(encoding="utf-8"))
        metrics = compute(data)

        buckets = metrics.latency["buckets"]
        assert buckets["apertura"]["full_turn"]["n"] == 1
        assert buckets["apertura"]["full_turn"]["p50"] == pytest.approx(2000, abs=5)
        assert buckets["regimen"]["full_turn"]["n"] == 3
        assert buckets["regimen"]["full_turn"]["p50"] == pytest.approx(1200, abs=5)
        assert metrics.label == "1.1"
        # Ya no debe quejarse de que falta la latencia ni el agente.
        assert not any("sin muestras de latencia" in a for a in metrics.warning_list)
        assert not any("turnos del agente" in a for a in metrics.warning_list)


class TestCallRecord:
    def test_logs_in_order_with_its_time(self):
        record = CallRecord()
        record.annotate("agent", "Hola, ¿hablo con Carlos?")
        record.annotate("carrier", "Sí, soy Carlos.")

        assert [t["role"] for t in record.turns] == ["agent", "carrier"]
        assert all(isinstance(t["t"], float) for t in record.turns)
        assert record.turns[0]["t"] <= record.turns[1]["t"]

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_discards_the_empty(self, text):
        record = CallRecord()
        record.annotate("agent", text)
        assert record.turns == []

    def test_trims_the_edges(self):
        record = CallRecord()
        record.annotate("agent", "  Hola  ")
        assert record.turns[0]["text"] == "Hola"

    def test_as_text_includes_both_parts(self):
        record = CallRecord()
        record.annotate("agent", "Ofrezco dieciséis mil")
        record.annotate("carrier", "Está bajo")
        text = record.as_text()
        assert "agent" in text and "carrier" in text
        assert "Ofrezco dieciséis mil" in text

    def test_without_turns_gives_empty_string(self):
        assert CallRecord().as_text() == ""


class TestSaveRun:
    def _session_closed(self) -> NegotiationSession:
        session = NegotiationSession(trip=TRIP, call_id="CAtest123")
        session.proponer(16_000)
        session.proponer(17_000)
        session.offer_compensation("tiempo_espera_carga")
        session.close_with_agreement(17_000, ["dos horas de espera pagadas"])
        return session

    def test_escribe_json_legible(self, tmp_path):
        session = self._session_closed()
        record = CallRecord()
        record.annotate("agent", "Quedamos en diecisiete mil")

        destination = save_run(session, record, base_path=tmp_path)

        assert destination.name == "call_CAtest123.json"
        content = json.loads(destination.read_text(encoding="utf-8"))
        assert content["call_sid"] == "CAtest123"
        assert content["summary"]["offer_history"] == [16_000.0, 17_000.0]
        assert content["summary"]["resultado"]["monto_final"] == 17_000.0
        assert content["summary"]["resultado"]["validated_final_amount"] is True
        assert content["transcript"][0]["role"] == "agent"

    def test_does_not_persist_the_max_price(self):
        """El techo financiero no viaja al disco. Ni la llave ni el valor."""
        session = self._session_closed()
        raw_value = json.dumps(session.summary(), ensure_ascii=False)
        assert "max_price" not in raw_value
        assert "21000" not in raw_value and "21,000" not in raw_value

    def test_creates_the_folder_if_it_does_not_exist(self, tmp_path):
        destination = save_run(
            self._session_closed(),
            CallRecord(),
            base_path=tmp_path / "new_item" / "honda",
        )
        assert destination.exists()

    def test_without_call_sid_does_not_blow_up(self, tmp_path):
        session = NegotiationSession(trip=TRIP)
        destination = save_run(session, CallRecord(), base_path=tmp_path)
        assert destination.name == "call_no_sid.json"


class TestCostEstimation:
    def test_rounds_to_the_minute_like_twilio(self):
        """Twilio factura minutos completos: 1:47 se cobra como 2 minutos."""
        assert estimate_cost(1.78, is_recording=True) == estimate_cost(2.0, is_recording=True)

    def test_charges_a_minimum_of_one_minute(self):
        assert estimate_cost(0.1, is_recording=True) > AI_COST_PER_CALL

    def test_recording_costs_more(self):
        assert estimate_cost(4, is_recording=True) > estimate_cost(4, is_recording=False)

    def test_a_four_minute_call_costs_cents(self):
        """Sirve de red: si una tarifa se teclea mal por 10x, esto lo caza."""
        estimated = estimate_cost(4, is_recording=True)
        assert 0.15 < estimated < 0.45

    def test_ten_minute_cap_is_still_cheap(self):
        estimated = estimate_cost(DEFAULT_LIMIT_SECONDS / 60, is_recording=True)
        assert estimated < 1.00

    def test_grows_with_the_duration(self):
        assert estimate_cost(8, is_recording=True) > estimate_cost(4, is_recording=True)
