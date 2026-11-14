"""TwiML, run label, initial session and persistence.

The telephony layer is identical to PoC1's on purpose, so these tests are
almost the same: what is protected is that it **stays** identical.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock

import numpy as np
import pytest

from poc3_gpt_live.app import bridge, telephony

from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    DEFAULT_LIVE_MODEL,
    Settings,
)
from poc3_gpt_live.app.latency import LatencyCollector
from poc3_gpt_live.app.marks import DRIFT_TO_RECORD
from poc3_gpt_live.app.clock import AnchoredClock
from poc3_gpt_live.diagnostics import audit_recorder, energy_probe
from poc3_gpt_live.app.telephony import (
    CallRecord,
    diagnostics_block,
    CallBridge,
    build_initial_session,
    build_twiml,
    save_run,
    read_identification,
)
from shared.negotiation_engine import TEST_TRIP, NegotiationSession
from shared.system_prompt import amount_representations


class _Clock:
    """Fake monotonic clock, so that the monitoring does not really sleep."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="sk-prueba",
        live_model=DEFAULT_LIVE_MODEL,
        voice="marin",
        delegation_model=DEFAULT_DELEGATION_MODEL,
    )


class TestTwiml:
    def test_connects_the_bidirectional_stream(self):
        """`<Connect>`, not `<Start>`: otherwise the carrier hears nothing."""
        twiml = build_twiml("wss://ejemplo.dev/ws")
        assert "<Connect>" in twiml
        assert 'url="wss://ejemplo.dev/ws"' in twiml

    def test_the_run_label_travels_as_parameter(self):
        twiml = build_twiml(
            "wss://ejemplo.dev/ws", {"poc": "3", "scenario": "2", "repetition": "1"}
        )
        assert '<Parameter name="poc" value="3" />' in twiml
        assert '<Parameter name="scenario" value="2" />' in twiml

    def test_empty_values_do_not_generate_parameter(self):
        twiml = build_twiml("wss://ejemplo.dev/ws", {"poc": "3", "scenario": "  "})
        assert "scenario" not in twiml

    def test_a_value_with_quotes_does_not_break_the_xml(self):
        """Unescaped, a value with quotes would break the XML and drop the call.

        It is checked by really parsing and reading the attribute back, instead
        of looking for `&quot;` by hand: `quoteattr` may escape with a single
        quote —which is also valid XML— and a textual test would deem it broken.
        """
        from xml.etree import ElementTree

        twiml = build_twiml("wss://ejemplo.dev/ws", {"scenario": 'a"b'})
        root = ElementTree.fromstring(twiml)
        parameter = root.find(".//Parameter")
        assert parameter is not None
        assert parameter.get("name") == "scenario"
        assert parameter.get("value") == 'a"b'

    def test_twiml_is_always_well_formed_xml(self):
        from xml.etree import ElementTree

        twiml = build_twiml(
            "wss://ejemplo.dev/ws?a=1&b=2", {"poc": "3", "scenario": "<2>"}
        )
        root = ElementTree.fromstring(twiml)
        stream = root.find(".//Stream")
        assert stream is not None
        assert stream.get("url") == "wss://ejemplo.dev/ws?a=1&b=2"


class TestIdentification:
    def test_reads_the_twilio_custom_parameters(self):
        label = read_identification(
            {"customParameters": {"poc": "3", "scenario": "2", "repetition": "1"}}
        )
        assert label == {"poc": "3", "scenario": "2", "repetition": "1"}

    def test_tolerates_keys_in_any_case(self):
        label = read_identification({"customParameters": {"Scenario": "4"}})
        assert label == {"scenario": "4"}

    @pytest.mark.parametrize(
        "start", [{}, {"customParameters": None}, {"customParameters": "no es dict"}]
    )
    def test_an_odd_start_does_not_take_down_a_call_that_is_already_ringing(self, start):
        assert read_identification(start) == {}


class TestInitialSession:
    def test_carries_the_model(self, settings):
        """In the public API the model goes in `session.start`, not in the URL."""
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        assert build_initial_session(settings, negotiation)["model"] == settings.live_model

    def test_only_sends_the_four_schema_fields(self, settings):
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        session = build_initial_session(settings, negotiation)
        assert set(session) == {"model", "instructions", "audio", "delegation"}

    def test_requests_pcm16_at_24_khz_explicitly(self, settings):
        """The format the measurement was validated with; it is not left to the default."""
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        audio = build_initial_session(settings, negotiation)["audio"]
        assert audio["format"] == {"type": "audio/pcm", "rate": 24_000}
        assert audio["output"] == {"voice": settings.voice}

    def test_with_pcmu_asks_for_ulaw_at_8_khz(self, settings):
        """The October campaign: Twilio's format, without resampling."""
        import dataclasses

        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        with_ulaw = dataclasses.replace(settings, audio_format="pcmu")
        audio = build_initial_session(with_ulaw, negotiation)["audio"]
        assert audio["format"] == {"type": "audio/pcmu", "rate": 8_000}

    def test_the_instructions_are_the_interpolated_prompt_v0(self, settings):
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        instructions = build_initial_session(settings, negotiation)["instructions"]
        assert TEST_TRIP.company in instructions
        assert TEST_TRIP.origin in instructions
        assert "{" not in instructions.replace("{", "", 0) or "{EMPRESA}" not in instructions

    def test_the_max_price_does_not_reach_the_session(self, settings):
        """The non-negotiable guardrail, on what really goes out over the wire."""
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="x")
        dumped = json.dumps(
            build_initial_session(settings, negotiation), ensure_ascii=False
        )
        for shape in amount_representations(TEST_TRIP.max_price):
            assert shape not in dumped


class TestRecord:
    def test_ignores_empty_texts(self):
        record = CallRecord()
        record.annotate("agent", "   ")
        record.annotate("agent", "hola")
        assert len(record.turns) == 1

    def test_records_the_two_roles(self):
        record = CallRecord()
        record.annotate("agent", "hola")
        record.annotate("carrier", "bueno")
        assert record.roles() == {"agent", "carrier"}


class TestPersistence:
    def _run(self, tmp_path):
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="CA123")
        negotiation.proponer(16_000)
        record = CallRecord()
        record.annotate("agent", "Hola, ¿hablo con Carlos?")
        record.annotate("carrier", "Sí, soy Carlos.")

        collector = LatencyCollector(clock=lambda: 0.0)
        collector.begin()
        collector.anchor_server({"model": DEFAULT_LIVE_MODEL})
        collector.agent_first_audio(start_ms=0)

        destination = save_run(
            negotiation,
            record,
            base_path=tmp_path,
            collector=collector,
            identification={"poc": "3", "scenario": "2", "repetition": "1"},
        )
        return destination, json.loads(destination.read_text(encoding="utf-8"))

    def test_the_json_carries_the_keys_shared_metrics_looks_for(self, tmp_path):
        _, content = self._run(tmp_path)
        assert content["poc"] == 3
        assert content["scenario"] == 2
        assert content["repetition"] == 1
        for key in ("summary", "transcript", "latencies", "gpt_live_extra"):
            assert key in content

    def test_the_alpha_extra_goes_outside_latencies(self, tmp_path):
        """`TurnSample.from_dict` silently drops the keys it does not know."""
        _, content = self._run(tmp_path)
        assert "gpt_live_extra" in content
        for row in content["latencies"]:
            assert "clock_verdict" not in row

    def test_the_max_price_does_not_reach_the_file(self, tmp_path):
        destination, _ = self._run(tmp_path)
        dumped = destination.read_text(encoding="utf-8")
        assert "max_price" not in dumped
        for shape in amount_representations(TEST_TRIP.max_price):
            assert shape not in dumped

    def test_a_non_numeric_label_is_kept_instead_of_being_lost(self, tmp_path):
        negotiation = NegotiationSession(trip=TEST_TRIP, call_id="CA9")
        destination = save_run(
            negotiation,
            CallRecord(),
            base_path=tmp_path,
            identification={"scenario": "dos"},
        )
        content = json.loads(destination.read_text(encoding="utf-8"))
        assert content["scenario"] == "dos"


class TestCloseWatching:
    """The hang-up cannot fire in the middle of a delegation.

    After `registrar_acuerdo` the agent says a filler ("Un segundo...") and goes
    quiet while the delegate model composes the goodbye; with only the silence
    timer, the call is hung up before the carrier hears it.

    The event loop is driven by hand with `asyncio.run` so as not to add
    pytest-asyncio as a dependency just for this.
    """

    def _bridge(self, clock):
        """A bridge with the bare minimum to exercise the close monitoring."""
        live_bridge = CallBridge.__new__(CallBridge)
        live_bridge._closing = asyncio.Event()
        live_bridge._t_last_outbound_audio = clock()
        live_bridge._in_flight_responses = 0
        live_bridge._pending_continuations = 0
        live_bridge._responses = bridge.ResponseTracker()
        live_bridge._t_delegation_end = clock()
        # The delegation tracking also feeds the collector.
        live_bridge._collector = LatencyCollector(clock=clock)
        return live_bridge

    def _hung_up_at(self, clock, live_bridge, *, releases_at: float | None = None) -> float:
        """Runs the monitoring with the fake clock and returns when it hung up.

        Each tick advances the clock, and `releases_at` simulates the delegate
        model's `response.completed` arriving at that second.
        """

        async def sleep(_: float) -> None:
            clock.t = round(clock.t + 0.25, 6)
            if releases_at is not None and clock.t >= releases_at:
                live_bridge._track_delegation({"type": bridge.EV_RESPONSE_COMPLETED})
            if clock.t > 300.0:
                raise AssertionError("la vigilancia nunca colgó")

        async def launch() -> float:
            live_bridge._closing.set()
            with mock.patch.object(telephony.asyncio, "sleep", sleep):
                with mock.patch.object(telephony.time, "monotonic", clock):
                    await live_bridge._watch_the_close()
            return clock.t

        return asyncio.run(launch())

    def test_waits_for_the_delegation_to_finish_before_hanging_up(self):
        clock = _Clock()
        live_bridge = self._bridge(clock)
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_CREATED})

        hung_up_at = self._hung_up_at(clock, live_bridge, releases_at=5.0)

        assert hung_up_at >= 5.0 + telephony.SILENCE_TO_HANG_UP_S

    def test_a_stuck_delegation_does_not_leave_the_call_open(self):
        """The hard cap: a delegation that never finishes does not bill forever."""
        clock = _Clock()
        live_bridge = self._bridge(clock)
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_CREATED})

        hung_up_at = self._hung_up_at(clock, live_bridge)

        assert telephony.HANG_UP_CAP_S <= hung_up_at < (
            telephony.HANG_UP_CAP_S + 1.0
        )

    def test_without_delegation_it_hangs_up_on_the_silence(self):
        clock = _Clock()
        live_bridge = self._bridge(clock)

        hung_up_at = self._hung_up_at(clock, live_bridge)

        assert telephony.SILENCE_TO_HANG_UP_S <= hung_up_at < (
            telephony.SILENCE_TO_HANG_UP_S + 1.0
        )

    def test_the_delegation_count_does_not_drop_below_zero(self):
        """An orphan `response.completed` must not leave the counter negative."""
        live_bridge = self._bridge(_Clock())
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_COMPLETED})
        assert live_bridge._in_flight_responses == 0
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_CREATED})
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_CREATED})
        assert live_bridge._in_flight_responses == 2

    def test_the_continuation_after_a_function_leaves_no_gap_at_zero(self):
        """Between the continuation's `completed` and `created` it is still in flight.

        With the public API the continuation is requested by the bridge
        (`response.create`) after the `completed`; if the counter dropped to
        zero there, the hang-up could fire in between and the round trip would
        be closed halfway.
        """
        live_bridge = self._bridge(_Clock())
        item = {
            "type": bridge.EV_OUTPUT_ITEM_DONE,
            "delegation_id": "of_1",
            "item": {"type": "function_call", "status": "completed",
                     "call_id": "call_1", "name": "proponer_precio", "arguments": "{}"},
        }
        assert live_bridge._track_delegation(
            {"type": bridge.EV_RESPONSE_CREATED, "delegation_id": "of_1"}) is False
        assert live_bridge._responses.new_call(item) is not None
        assert live_bridge._responses.answered("call_1") is False

        proceed = live_bridge._track_delegation(
            {"type": bridge.EV_RESPONSE_COMPLETED, "delegation_id": "of_1"})
        assert proceed is True
        assert live_bridge._in_flight_responses == 1

        # The continuation's `created` does not count twice, and its `completed` closes it.
        live_bridge._track_delegation({"type": bridge.EV_RESPONSE_CREATED, "delegation_id": "of_1"})
        assert live_bridge._in_flight_responses == 1
        assert live_bridge._track_delegation(
            {"type": bridge.EV_RESPONSE_COMPLETED, "delegation_id": "of_1"}) is False
        assert live_bridge._in_flight_responses == 0


class TestWhitespaceFragments:
    """The alpha sends the separator as a fragment of its own: `" a"`, `" "`, `"28"`.

    Dropping it glues the word to the number. In call 3.2 of August 12, 2026
    `"a28,000"` and `"bajo25,000"` came out, and the amount extractor lost 2 of
    4 amounts: **50% error where the STT got nothing wrong**. That call's WER
    went down from 6.33% to 1.27% when they were kept.
    """

    def test_a_space_fragment_does_not_glue_the_word_to_the_amount(self):
        record = CallRecord()
        for fragment in (" Ese viaje me lo pagan", " a", " ", "28", ",000"):
            record.annotate("carrier", fragment)
        record.normalize()

        assert record.turns[0]["text"] == "Ese viaje me lo pagan a 28,000"

    def test_a_space_fragment_does_not_open_an_empty_turn(self):
        record = CallRecord()
        record.annotate("agent", " ")
        record.annotate("agent", " Hola")
        record.normalize()

        assert len(record.turns) == 1
        assert record.turns[0]["text"] == "Hola"

    def test_the_space_does_not_cross_the_role_change(self):
        """When the speaker changes, a stray space must not open a turn."""
        record = CallRecord()
        record.annotate("agent", " ¿Carlos?")
        record.annotate("carrier", " ")
        record.annotate("carrier", "Sí")
        record.normalize()

        assert [t["role"] for t in record.turns] == ["agent", "carrier"]
        assert record.turns[1]["text"] == "Sí"


class TestBridgeConstruction:
    """The bridge is built through `__init__`, not through `__new__`.

    Regression: the wiring of the energy probe and that of the audit recorder
    cannot read `record.call_id`, because `CallRecord` only has `start` and
    `turns`. The `AttributeError` would be raised inside `__init__` —before the
    environment-variable guardrail even gets checked, because the argument is
    evaluated before entering `begin()`—, so the WebSocket would die as soon
    as Twilio connects, with the diagnostics on or off.

    No test saw it: they all went in through `__new__` and set up by hand the
    fields they were going to exercise. Hence these two, which check no logic —
    they check that the object can be created.
    """

    def _build(self) -> CallBridge:
        return CallBridge(
            websocket=mock.Mock(),
            client=mock.Mock(),
            negotiation=NegotiationSession(trip=TEST_TRIP, call_id="CAdeprueba"),
            record=CallRecord(),
            collector=LatencyCollector(),
            stream_sid="MZdeprueba",
        )

    def test_is_built_with_diagnostics_off(self, monkeypatch):
        monkeypatch.delenv(energy_probe.TURN_ON_VARIABLE, raising=False)
        monkeypatch.delenv(audit_recorder.TURN_ON_VARIABLE, raising=False)

        live_bridge = self._build()

        assert live_bridge._probe is None
        assert live_bridge._auditoria is None

    def test_is_built_with_diagnostics_on(self, monkeypatch, tmp_path):
        """And both artifacts are dated with the call's SID, not the stream's."""
        monkeypatch.setenv(energy_probe.TURN_ON_VARIABLE, "1")
        monkeypatch.setenv(audit_recorder.TURN_ON_VARIABLE, "1")
        monkeypatch.setattr(energy_probe, "RUTA_DEFAULT", tmp_path)
        monkeypatch.setattr(audit_recorder, "RUTA_DEFAULT", tmp_path)

        live_bridge = self._build()

        assert live_bridge._probe is not None
        assert live_bridge._auditoria is not None
        assert "CAdeprueba" in live_bridge._probe.destination.name
        assert "CAdeprueba" in live_bridge._auditoria.destination.name


class TestTheDiagnosticsCannotCutTheAudio:
    """The audio loop's guardrail.

    Both diagnostics sites run inside the audio loop: without protection, an
    exception there would bring down the whole call, which is exactly what the
    closing `try/except` already prevents at hang-up. With the recorder on for
    the whole campaign, that is not hypothetical.

    The three things each test checks are the three that matter, in order: the
    audio went out byte for byte, the diagnostics were turned off, and the
    failure was written down.
    """

    def _build(self, monkeypatch, tmp_path) -> CallBridge:
        monkeypatch.setenv(energy_probe.TURN_ON_VARIABLE, "1")
        monkeypatch.setenv(audit_recorder.TURN_ON_VARIABLE, "1")
        monkeypatch.setattr(energy_probe, "RUTA_DEFAULT", tmp_path)
        monkeypatch.setattr(audit_recorder, "RUTA_DEFAULT", tmp_path)
        return CallBridge(
            websocket=mock.Mock(),
            client=mock.Mock(),
            negotiation=NegotiationSession(trip=TEST_TRIP, call_id="CAdeprueba"),
            record=CallRecord(),
            collector=LatencyCollector(),
            stream_sid="MZdeprueba",
        )

    # -- incoming site: what goes to the alpha -----------------------------

    def test_incoming_the_audio_keeps_flowing_out_to_the_alpha(
        self, monkeypatch, tmp_path
    ):
        import asyncio
        import base64
        import json as _json

        live_bridge = self._build(monkeypatch, tmp_path)

        def _raises(*_a, **_k):
            raise RuntimeError("disco lleno a media llamada")

        monkeypatch.setattr(live_bridge._auditoria, "annotate_carrier", _raises)

        # Twelve 20 ms frames of tone: enough for the resampler to release
        # several blocks towards the alpha.
        tone = np.sin(np.arange(160 * 12) / 8_000.0 * 2 * np.pi * 300)
        ulaw = bridge.pcm16_a_ulaw((tone * 6_000).astype(np.int16))
        frames = [ulaw[i : i + 160] for i in range(0, len(ulaw), 160)]

        sent_items: list[bytes] = []

        async def _send(pcm24: bytes) -> None:
            sent_items.append(pcm24)

        live_bridge._client.send_audio = _send
        messages = [
            _json.dumps(
                {
                    "event": "mean",
                    "mean": {"payload": base64.b64encode(t).decode("ascii")},
                }
            )
            for t in frames
        ] + [_json.dumps({"event": "stop"})]

        pending_items = list(messages)

        async def _receive_text() -> str:
            return pending_items.pop(0)

        live_bridge._ws.receive_text = _receive_text
        asyncio.run(live_bridge._from_twilio_to_live())

        # 1. The audio went out, and it went out whole: what the alpha receives
        #    has to be the resampling of the 12 frames, not a prefix.
        #
        #    **It is not compared byte for byte against a reference**, even
        #    though that is what one would want: `soxr` is not bit-reproducible
        #    across instances — the same audio through two instances gives up to
        #    **2 LSB** of difference from one run to another (measured: 2
        #    failures in 6 runs with a cap of 1). What the guardrail has to
        #    guarantee is that **no frame is dropped or skipped**, and the exact
        #    length proves that; the 4 LSB tolerance on a ±6,000 signal is 0.07%
        #    and could not hide an altered frame.
        assert sent_items
        expected = bridge.AudioBridge()
        full = b"".join(expected.from_twilio(t) for t in frames)
        exited = b"".join(sent_items)
        assert len(exited) == len(full)
        assert (
            np.abs(
                np.frombuffer(exited, dtype="<i2").astype(np.int32)
                - np.frombuffer(full, dtype="<i2").astype(np.int32)
            ).max()
            <= 4
        )

        # 2. The diagnostics were turned off and not retried per frame.
        assert live_bridge._auditoria is None
        assert live_bridge._probe is None
        assert len(live_bridge.diagnostics_failures) == 1

        # 3. And the failure was recorded with site, exception and instant.
        fault = live_bridge.diagnostics_failures[0]
        assert fault["sitio"] == "incoming"
        assert fault["excepcion"] == "RuntimeError"
        assert "disco lleno" in fault["message"]
        assert fault["call_second"] >= 0.0

    # -- outgoing site: what goes to Twilio --------------------------------

    def test_outgoing_the_audio_keeps_flowing_out_to_twilio(
        self, monkeypatch, tmp_path
    ):
        import asyncio
        import base64
        import json as _json

        live_bridge = self._build(monkeypatch, tmp_path)
        monkeypatch.setattr(
            live_bridge._auditoria,
            "annotate_agent",
            mock.Mock(side_effect=ValueError("buffer corrupto")),
        )

        sent: list[str] = []

        async def _send_text(raw_value: str) -> None:
            sent.append(raw_value)

        live_bridge._ws.send_text = _send_text

        pcm24 = (
            np.sin(np.arange(2_400) / 24_000.0 * 2 * np.pi * 300) * 6_000
        ).astype(np.int16).tobytes()
        event = {
            "type": bridge.EV_OUTPUT_AUDIO_DELTA,
            "delta": base64.b64encode(pcm24).decode("ascii"),
        }
        asyncio.run(live_bridge._handle_audio(event))
        asyncio.run(live_bridge._handle_audio(dict(event)))

        # 1. Both deliveries went out to Twilio, of the expected size and
        #    without losing any. Same note as above about `soxr`: length and
        #    equality within 1 LSB are checked, not byte for byte.
        assert len(sent) == 2
        expected = bridge.AudioBridge()
        reference = expected.a_twilio(pcm24)
        exited = base64.b64decode(_json.loads(sent[0])["mean"]["payload"])
        assert len(exited) == len(reference)
        assert (
            np.abs(
                bridge.ulaw_a_pcm16(exited).astype(np.int32)
                - bridge.ulaw_a_pcm16(reference).astype(np.int32)
            ).max()
            <= 8  # one mu-law quantization step in the low range
        )

        # 2. Turned off, and the second delivery no longer touched it.
        assert live_bridge._auditoria is None
        assert live_bridge._probe is None
        assert len(live_bridge.diagnostics_failures) == 1

        # 3. Recorded.
        fault = live_bridge.diagnostics_failures[0]
        assert fault["sitio"] == "saliente"
        assert fault["excepcion"] == "ValueError"
        assert "buffer corrupto" in fault["message"]

    # -- and that the exclusion is put in writing in the JSON --------------

    def test_the_call_with_the_diagnostics_down_is_excluded_in_writing(self):
        healthy_one = diagnostics_block(None)
        assert healthy_one["full_capture"] is True
        assert healthy_one["eligible_for_hang_up_analysis"] is True
        assert healthy_one["faults"] == []

        roto = diagnostics_block(
            [{"sitio": "saliente", "excepcion": "ValueError"}]
        )
        assert roto["full_capture"] is False
        assert roto["eligible_for_hang_up_analysis"] is False
        assert roto["faults"][0]["sitio"] == "saliente"

    def test_the_block_always_goes_even_with_no_failures(self, tmp_path):
        """A key that only appears on failure reads as absent when the code that
        writes it is missing, and then the broken call counts as healthy
        again."""
        destination = save_run(
            NegotiationSession(trip=TEST_TRIP, call_id="CAsano"),
            CallRecord(),
            base_path=tmp_path,
        )
        data = json.loads(destination.read_text(encoding="utf-8"))
        assert data["diagnostics"]["full_capture"] is True

    def test_the_test_does_not_pass_by_chance(self, monkeypatch, tmp_path):
        """That the guardrail is WHAT saves the call, and not something else.

        If `_turn_off_diagnostics` raises again, the audio loop has to die.
        Without this test, the two above would pass just the same if the
        exception never reached the `except` —for instance if the test double
        were misplaced— and they would be verifying nothing.
        """
        import asyncio
        import base64
        import json as _json

        live_bridge = self._build(monkeypatch, tmp_path)
        monkeypatch.setattr(
            live_bridge._auditoria,
            "annotate_carrier",
            mock.Mock(side_effect=RuntimeError("boom")),
        )

        def _blow_up(sitio, error):
            raise error

        monkeypatch.setattr(live_bridge, "_turn_off_diagnostics", _blow_up)

        trama = bridge.pcm16_a_ulaw(np.zeros(160, dtype=np.int16))
        pending_items = [
            _json.dumps(
                {
                    "event": "mean",
                    "mean": {"payload": base64.b64encode(trama).decode("ascii")},
                }
            )
        ]

        async def _receive_text() -> str:
            return pending_items.pop(0)

        live_bridge._ws.receive_text = _receive_text
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(live_bridge._from_twilio_to_live())


class TestTheDriftSampling:
    """The task that watches the anchor while the call runs.

    Three things that cannot be seen by looking at `AnchoredClock` alone: that
    the task really samples, that it does **not** end on its own —if it did,
    `asyncio.wait` with `FIRST_COMPLETED` would hang up the call in the first
    second— and that an exception there is recorded without touching the audio.
    """

    def _bridge(self, clock):
        return CallBridge(
            websocket=mock.Mock(),
            client=mock.Mock(),
            negotiation=NegotiationSession(
                trip=TEST_TRIP, call_id="CAdeprueba"
            ),
            record=CallRecord(),
            collector=LatencyCollector(),
            stream_sid="MZdeprueba",
            clock=clock,
        )

    def test_samples_while_the_call_runs(self, monkeypatch):
        monkeypatch.setattr(telephony, "SAMPLING_PERIOD_S", 0.01)
        clock = AnchoredClock()
        live_bridge = self._bridge(clock)

        async def launch():
            tarea = asyncio.create_task(live_bridge._sample_drift())
            await asyncio.sleep(0.08)
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)

        asyncio.run(launch())

        assert clock.block()[DRIFT_TO_RECORD]["samples"] >= 3

    def test_does_not_end_on_its_own_or_without_a_clock(self, monkeypatch):
        """Without an anchor it sleeps forever: returning would hang up the call."""
        monkeypatch.setattr(telephony, "SAMPLING_PERIOD_S", 0.01)
        live_bridge = self._bridge(None)

        async def launch():
            tarea = asyncio.create_task(live_bridge._sample_drift())
            await asyncio.sleep(0.05)
            alive = not tarea.done()
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
            return alive

        assert asyncio.run(launch()) is True

    def test_a_failure_is_recorded_and_the_task_does_not_die(self, monkeypatch):
        monkeypatch.setattr(telephony, "SAMPLING_PERIOD_S", 0.01)
        clock = AnchoredClock()
        monkeypatch.setattr(
            clock, "take_samples", mock.Mock(side_effect=RuntimeError("se rompio"))
        )
        live_bridge = self._bridge(clock)

        async def launch():
            tarea = asyncio.create_task(live_bridge._sample_drift())
            await asyncio.sleep(0.06)
            alive = not tarea.done()
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
            return alive

        alive = asyncio.run(launch())

        assert alive is True, "si termina, `asyncio.wait` cuelga la llamada"
        assert clock.take_samples.call_count == 1, "se apaga a la primera"
        assert [f["sitio"] for f in live_bridge.diagnostics_failures] == [
            "drift_sampling"
        ]
