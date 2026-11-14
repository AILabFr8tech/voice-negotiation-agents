"""The audio bridge: G.711 codec and resampling.

It is the piece that fails silently and expensively — a badly implemented codec
does not crash, it just makes the carrier hear noise and the model understand
nothing, and that is discovered with the phone in hand.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from poc3_gpt_live.app import bridge


class TestCodecUlaw:
    def test_the_256_codes_go_and_come_back(self):
        """Exact round trip, except for the ±0 that μ-law collapses into a single code."""
        codes = bytes(range(256))
        outbound = bridge.pcm16_a_ulaw(bridge.ulaw_a_pcm16(codes))
        different_items = [i for i in range(256) if outbound[i] != codes[i]]
        assert different_items == [127]

    def test_decoding_respects_the_sign(self):
        pcm = bridge.ulaw_a_pcm16(bytes([0xFF, 0x7F]))
        # 0xFF is positive zero and 0x7F negative zero: both are ~0 with a sign.
        assert pcm[0] >= 0
        assert pcm[1] <= 0

    def test_silence_is_encoded_as_the_zero_code(self):
        silence = np.zeros(64, dtype=np.int16)
        encoded = bridge.pcm16_a_ulaw(silence)
        assert set(encoded) == {0xFF}

    @pytest.mark.parametrize("valor", [-32768, -32767, -1, 0, 1, 32766, 32767])
    def test_the_extremes_do_not_overflow(self, valor):
        """The limits of the int16 range have to yield a valid code, not blow up."""
        encoded = bridge.pcm16_a_ulaw(np.array([valor], dtype=np.int16))
        assert len(encoded) == 1
        assert 0 <= encoded[0] <= 255

    def test_the_quantization_error_stays_bounded(self):
        """μ-law is logarithmic: the relative error has to be small across the band.

        Without this test, a codec with a wrong exponent would pass the ones
        above (the extremes and zero come out fine) and still wreck the voice
        in the mid range, which is where speech lives.
        """
        original = np.arange(-32768, 32768, 37, dtype=np.int16)
        recovered = bridge.ulaw_a_pcm16(bridge.pcm16_a_ulaw(original))
        error = np.abs(original.astype(np.int32) - recovered.astype(np.int32))
        # The μ-law step in the highest segment is 256 on 14 bits,
        # i.e. 1024 on the PCM's 16 bits.
        assert error.max() <= 1024


class TestAudioBridge:
    def test_upsamples_to_24k_with_the_correct_ratio(self):
        live_bridge = bridge.AudioBridge()
        # A whole second of audio, in 20 ms frames like Twilio's.
        entry = bridge.pcm16_a_ulaw(np.zeros(bridge.TELEPHONY_SAMPLE_RATE, dtype=np.int16))
        output = bytearray()
        for i in range(0, len(entry), 160):
            output += live_bridge.from_twilio(bytes(entry[i : i + 160]))

        samples = len(output) // bridge.BYTES_PER_SAMPLE
        # The filter eats a tail at the end; the rest has to be there.
        assert 0.9 * bridge.LIVE_SAMPLE_RATE <= samples <= bridge.LIVE_SAMPLE_RATE

    def test_the_pcm_that_comes_out_always_has_an_even_number_of_bytes(self):
        """The alpha rejects any odd-length payload with `invalid_audio`."""
        live_bridge = bridge.AudioBridge()
        entry = bridge.pcm16_a_ulaw(np.zeros(4000, dtype=np.int16))
        for i in range(0, len(entry), 160):
            piece = live_bridge.from_twilio(bytes(entry[i : i + 160]))
            assert len(piece) % bridge.BYTES_PER_SAMPLE == 0

    def test_an_odd_payload_is_sanitized_instead_of_misaligning(self):
        """A stray byte would misalign the rest of the call: that byte is dropped, not everything.

        Checked with a large payload —one of 3 bytes is not even enough to
        fill the filter— so as to tell "it was sanitized and carried on" apart
        from "it was dropped entirely".
        """
        live_bridge = bridge.AudioBridge()
        pair = np.zeros(4_800, dtype=np.int16).tobytes()  # 200 ms at 24 kHz
        output = live_bridge.a_twilio(pair + b"\x7f")  # one extra byte
        assert output != b""

    def test_the_odd_payload_does_not_shift_the_rest_of_the_call(self):
        """Dropping the stray byte has to leave the phase intact from then on."""
        cleaned = bridge.AudioBridge()
        dirty = bridge.AudioBridge()
        samples = np.zeros(4_800, dtype=np.int16).tobytes()
        assert len(dirty.a_twilio(samples + b"\x7f")) == len(cleaned.a_twilio(samples))

    def test_the_empties_do_not_blow_up(self):
        live_bridge = bridge.AudioBridge()
        assert live_bridge.from_twilio(b"") == b""
        assert live_bridge.a_twilio(b"") == b""

    def test_preserves_the_tone_round_trip(self):
        """440 Hz has to still be 440 Hz after 8k -> 24k -> 8k.

        This is what tells resampling with anti-aliasing apart from plain
        interpolation: with aliasing, the energy shifts bins and this fails.
        """
        live_bridge = bridge.AudioBridge()
        t = np.arange(bridge.TELEPHONY_SAMPLE_RATE) / bridge.TELEPHONY_SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 440 * t) * 12_000).astype(np.int16)

        entry = bridge.pcm16_a_ulaw(tone)
        upsampled = bytearray()
        for i in range(0, len(entry), 160):
            upsampled += live_bridge.from_twilio(bytes(entry[i : i + 160]))
        outbound = live_bridge.a_twilio(bytes(upsampled))

        samples = bridge.ulaw_a_pcm16(outbound).astype(np.float64)
        assert len(samples) > bridge.TELEPHONY_SAMPLE_RATE // 2

        spectrum = np.abs(np.fft.rfft(samples))
        pico_hz = np.fft.rfftfreq(len(samples), 1 / bridge.TELEPHONY_SAMPLE_RATE)[
            int(np.argmax(spectrum))
        ]
        assert abs(pico_hz - 440) < 15


class TestLogWithoutAudio:
    def test_the_audio_never_appears_in_full_in_the_logs(self):
        """Guardrail: an audio dump is minutes of voice in a .log."""
        event = {
            "type": bridge.EV_OUTPUT_AUDIO_DELTA,
            "delta": "QUJDRA==" * 500,
            "event_id": "ev_1",
        }
        output = bridge.summarize_for_log(event)
        assert "QUJDRA==" not in output
        assert "<audio" in output
        # What is not audio does have to be visible: it is what gets audited.
        assert '"event_id": "ev_1"' in output

    def test_the_inbound_audio_neither(self):
        event = {"type": bridge.EV_INPUT_AUDIO_APPEND, "audio": "QUJDRA==" * 500}
        output = bridge.summarize_for_log(event)
        assert "QUJDRA==" not in output
        assert "<audio" in output

    def test_events_without_audio_come_out_whole(self):
        event = {"type": bridge.EV_SESSION_STARTED, "session": {"id": "sess_1"}}
        assert "sess_1" in bridge.summarize_for_log(event)


def _function_item(**changes):
    """A `response.output_item.done` of a `function_call`, already unwrapped."""
    item = {
        "id": "fc_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "proponer_precio",
        "arguments": '{"monto": 16000}',
    }
    item.update(changes)
    return {"type": bridge.EV_OUTPUT_ITEM_DONE, "delegation_id": "of_1", "item": item}


class TestEventAccesses:
    def test_reads_a_function_call(self):
        assert bridge.function_call(_function_item()) == (
            "call_1",
            "proponer_precio",
            '{"monto": 16000}',
        )

    def test_ignores_events_that_are_not_calls(self):
        assert bridge.function_call({"type": bridge.EV_RESPONSE_CREATED}) is None
        message = _function_item(type="message")
        assert bridge.function_call(message) is None

    def test_loose_arguments_no_longer_trigger_the_function(self):
        """`function_call_arguments.done` carries no `call_id`: nothing is executed from it."""
        event = {
            "type": bridge.EV_FUNCTION_ARGS_DONE,
            "call_id": "call_1",
            "name": "proponer_precio",
            "arguments": "{}",
        }
        assert bridge.function_call(event) is None

    def test_an_incomplete_call_is_not_actioned(self):
        assert bridge.function_call(_function_item(status="in_progress")) is None

    def test_a_call_without_call_id_is_not_actioned(self):
        """Without a `call_id` there is no one to answer: better to ignore it than to guess."""
        assert bridge.function_call(_function_item(call_id="")) is None

    def test_the_unreadable_base64_does_not_blow_up(self):
        assert bridge.event_audio({"delta": "no es base64 !!!"}) == b""
        assert bridge.event_audio({}) == b""

    def test_the_audio_travels_in_delta(self):
        import base64

        raw_value = b"\x01\x00\x02\x00"
        event = {
            "type": bridge.EV_OUTPUT_AUDIO_DELTA,
            "delta": base64.b64encode(raw_value).decode("ascii"),
        }
        assert bridge.event_audio(event) == raw_value

    def test_reads_the_text_of_a_transcript_without_touching_the_spaces(self):
        """Raw: the separating space comes at the start of the fragment.

        Stripping it here glues the words together and triggers a false WER.
        """
        event = {
            "type": bridge.EV_INPUT_TRANSCRIPT,
            "delta": " soy Carlos.",
            "start_ms": 1200,
            "end_ms": 1400,
        }
        assert bridge.transcript_text(event) == " soy Carlos."

    def test_concatenated_fragments_keep_the_spaces(self):
        fragments = [" Sí,", " soy", " Carlos.", " D", "ígame"]
        joined = "".join(
            bridge.transcript_text({"type": bridge.EV_INPUT_TRANSCRIPT, "delta": f})
            for f in fragments
        )
        assert joined.strip() == "Sí, soy Carlos. Dígame"


class TestUnwrapResponse:
    """The public API wraps the delegation's events in `response.event`."""

    def test_extracts_the_event_and_sets_the_delegation_id(self):
        on_top = {
            "type": bridge.EV_RESPONSE_EVENT,
            "delegation_id": "of_7",
            "event": {"type": "response.created", "response": {"id": "resp_1"}},
        }
        assert bridge.unwrap_response(on_top) == {
            "type": "response.created",
            "response": {"id": "resp_1"},
            "delegation_id": "of_7",
        }

    def test_an_empty_envelope_does_not_become_something_else(self):
        on_top = {"type": bridge.EV_RESPONSE_EVENT, "delegation_id": "of_7"}
        assert bridge.unwrap_response(on_top)["type"] == bridge.EV_RESPONSE_EVENT

    def test_listen_delivers_the_unwrapped_events(self, tmp_path):
        """The census stores the envelope; dispatch sees the `response.*` inside."""
        import asyncio

        on_top = {
            "type": bridge.EV_RESPONSE_EVENT,
            "delegation_id": "of_7",
            "event": {"type": "response.completed"},
        }
        messages = [json.dumps(on_top), json.dumps({"type": bridge.EV_SESSION_CLOSED})]

        class _FakeWs:
            async def recv(self):
                return messages.pop(0)

        census = bridge.EventCensus(tmp_path / "census.jsonl")
        client = bridge.GptLiveClient("wss://x", {}, {}, census=census)
        client._ws = _FakeWs()

        async def _join():
            return [e async for e in client.listen()]

        seen_items = asyncio.run(_join())
        census.shut()
        assert [e["type"] for e in seen_items] == ["response.completed", bridge.EV_SESSION_CLOSED]
        assert seen_items[0]["delegation_id"] == "of_7"
        assert census.count == {bridge.EV_RESPONSE_EVENT: 1, bridge.EV_SESSION_CLOSED: 1}


class TestWhatIsSentToTheServer:
    """The shape of the client events, against a WebSocket that only takes notes."""

    def _client(self):
        sent_items: list[dict] = []

        class _FakeWs:
            async def send(self, text):
                sent_items.append(json.loads(text))

        client = bridge.GptLiveClient("wss://x", {}, {})
        client._ws = _FakeWs()
        return client, sent_items

    def test_the_audio_goes_as_session_input_audio_append(self):
        import asyncio

        client, sent_items = self._client()
        asyncio.run(client.send_audio(b"\x00\x01" * 10))
        assert sent_items[0]["type"] == "session.input_audio.append"
        assert set(sent_items[0]) == {"type", "audio"}

    def test_the_opening_goes_as_commentary_with_null_delegation_id(self):
        import asyncio

        client, sent_items = self._client()
        asyncio.run(client.add_context("hola"))
        assert sent_items[0]["type"] == "session.commentary.append"
        assert sent_items[0]["content"] == "hola"
        # Mandatory in these events, even if null.
        assert "delegation_id" in sent_items[0] and sent_items[0]["delegation_id"] is None

    def test_continuing_the_delegation_is_response_create(self):
        import asyncio

        client, sent_items = self._client()
        asyncio.run(client.proceed_delegation())
        assert sent_items[0]["type"] == "response.create"


class TestEventCensus:
    """The census is instrumentation: it does not change dispatch, it only observes it."""

    def _census(self, tmp_path):
        return bridge.EventCensus(tmp_path / "census.jsonl")

    def test_writes_one_json_per_line_with_local_time(self, tmp_path):
        census = self._census(tmp_path)
        census.registrar({"type": "session.started", "session": {"id": "sess_1"}})
        census.registrar({"type": "turn.done", "turn": {"role": "user"}})
        census.shut()

        text_lines = census.destination.read_text(encoding="utf-8").strip().splitlines()
        assert len(text_lines) == 2
        first = json.loads(text_lines[0])
        assert first["type"] == "session.started"
        assert first["n"] == 1
        assert isinstance(first["t_local_seg"], float)
        assert first["keys"] == ["session", "type"]
        assert first["event"]["session"]["id"] == "sess_1"

    def test_the_audio_is_never_written(self, tmp_path):
        """Same guardrail as the logs: the census does not dump voice either."""
        census = self._census(tmp_path)
        census.registrar(
            {"type": bridge.EV_OUTPUT_AUDIO_DELTA, "delta": "QUJDRA==", "event_id": "ev_1"}
        )
        census.shut()

        dumped = census.destination.read_text(encoding="utf-8")
        assert "QUJDRA" not in dumped
        assert "<audio" in dumped
        # The rest of the event does survive.
        assert json.loads(dumped)["event"]["event_id"] == "ev_1"

    def test_separates_unknown_from_known_unused(self, tmp_path):
        census = self._census(tmp_path)
        census.registrar({"type": bridge.EV_OUTPUT_AUDIO_DELTA})  # handled
        census.registrar({"type": "session.commentary.appended"})  # known, unused
        census.registrar({"type": "response.inventado.nuevo"})      # unknown
        census.shut()

        summary = census.summary()
        assert summary["events"] == 3
        assert summary["unknown_ones"] == {"response.inventado.nuevo": 1}
        assert summary["response_unknown"] == {"response.inventado.nuevo": 1}

    def test_the_summary_preserves_the_order_of_first_appearance(self, tmp_path):
        census = self._census(tmp_path)
        for tipo in ("b", "a", "b", "c", "a"):
            census.registrar({"type": tipo})
        census.shut()

        summary = census.summary()
        assert summary["first_appearance"] == ["b", "a", "c"]
        assert summary["count_by_kind"] == {"a": 2, "b": 2, "c": 1}


class TestResponseTracking:
    """When `response.create` is sent: response finished and everything answered."""

    def _created(self, delegation="of_1"):
        return {"type": bridge.EV_RESPONSE_CREATED, "delegation_id": delegation}

    def _finished(self, delegation="of_1", tipo=bridge.EV_RESPONSE_COMPLETED):
        return {"type": tipo, "delegation_id": delegation}

    def test_answering_before_it_finishes_waits_for_the_end(self):
        """The normal order: the item arrives, it is answered, and then `completed`."""
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        assert seg.new_call(_function_item()) is not None
        assert seg.answered("call_1") is False
        assert seg.finished(self._finished()) is True

    def test_finishing_before_answering_waits_for_the_output(self):
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        seg.new_call(_function_item())
        assert seg.finished(self._finished()) is False
        assert seg.answered("call_1") is True

    def test_a_response_without_functions_is_not_continued(self):
        """If the delegate only spoke, there is nothing to resume."""
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        assert seg.finished(self._finished()) is False

    def test_with_two_calls_it_waits_for_both(self):
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        seg.new_call(_function_item())
        seg.new_call(_function_item(call_id="call_2", id="fc_2"))
        assert seg.finished(self._finished()) is False
        assert seg.answered("call_1") is False
        assert seg.answered("call_2") is True

    def test_delegations_are_not_mixed(self):
        seg = bridge.ResponseTracker()
        seg.created(self._created("of_1"))
        seg.created(self._created("of_2"))
        seg.new_call(_function_item())
        seg.answered("call_1")
        assert seg.finished(self._finished("of_2")) is False
        assert seg.finished(self._finished("of_1")) is True

    def test_a_failed_response_also_closes(self):
        """Same treatment as the pipecat reference: whatever is pending gets resumed."""
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        seg.new_call(_function_item())
        seg.answered("call_1")
        assert seg.finished(self._finished(tipo=bridge.EV_RESPONSE_FAILED)) is True

    def test_continues_only_once(self):
        seg = bridge.ResponseTracker()
        seg.created(self._created())
        seg.new_call(_function_item())
        seg.answered("call_1")
        assert seg.finished(self._finished()) is True
        assert seg.finished(self._finished()) is False

    def test_counts_the_executed_unanswered(self):
        seg = bridge.ResponseTracker()
        seg.new_call(_function_item())
        assert seg.pending_items == 1
        seg.answered("call_1")
        assert seg.pending_items == 0


class TestSpeechEnergy:
    """A measurement instrument, not a VAD: it only sets timestamps."""

    def _tone(self, amplitude: int, n: int = 480) -> bytes:
        t = np.arange(n) / 24_000.0
        return (np.sin(2 * np.pi * 300 * t) * amplitude).astype(np.int16).tobytes()

    def test_silence_is_not_speech(self):
        assert bridge.has_speech(np.zeros(480, dtype=np.int16).tobytes()) is False

    def test_speech_is_indeed_voice(self):
        assert bridge.has_speech(self._tone(8_000)) is True

    def test_the_line_noise_does_not_cross_the_threshold(self):
        assert bridge.has_speech(self._tone(50)) is False

    def test_a_weak_onset_is_not_diluted_in_the_delta(self):
        """The reason the 20 ms window exists.

        A 100 ms delta with 80 ms of silence and 20 ms of speech is the onset
        of a line. Averaging the whole delta divides the RMS by the square
        root of 5, so an onset **near the threshold** stops crossing and the
        mark moves to the next delta.

        For a strong onset the effect is different and also costly: both
        windows cross, but the coarse window's decision is only available once
        the delta has finished arriving, and that delay falls inside the only
        latency metric PoC3 shares with PoC1 and PoC2 (measurement in the
        appendix of `docs/comparativo_results.md`).
        """
        silence = np.zeros(4 * 480, dtype=np.int16).tobytes()
        # Amplitude 400: RMS 283 in its window, 126 averaged over the delta.
        delta = silence + self._tone(400)

        assert bridge.has_speech(delta) is True
        # The average of the whole block, which is how it used to be measured, does not cross.
        assert bridge.rms_pcm16(delta) < bridge.SPEECH_THRESHOLD_RMS

    def test_the_window_is_the_same_as_the_shared_one(self):
        """20 ms: the Media Streams frame, which is what PoC1 measures."""
        assert bridge.SPEECH_WINDOW_S == 0.020
        assert bridge.SAMPLES_PER_SPEECH_WINDOW == int(
            bridge.LIVE_SAMPLE_RATE * bridge.SPEECH_WINDOW_S
        )

    def test_a_block_shorter_than_the_window_is_measured_whole(self):
        """If it is shorter than 20 ms it cannot be invisible by definition."""
        assert bridge.has_speech(self._tone(8_000, n=100)) is True
        assert bridge.has_speech(np.zeros(100, dtype=np.int16).tobytes()) is False

    def test_the_rms_of_a_sine_is_the_amplitude_over_root_two(self):
        rms = bridge.rms_pcm16(self._tone(10_000))
        assert abs(rms - 10_000 / (2**0.5)) < 200

    def test_edges_do_not_blow_up(self):
        assert bridge.rms_pcm16(b"") == 0.0
        assert bridge.rms_pcm16(b"\x01") == 0.0
        # An odd byte is ignored instead of misaligning the read.
        assert bridge.rms_pcm16(np.zeros(4, dtype=np.int16).tobytes() + b"\x7f") == 0.0


class TestDoubleFunctionResponse:
    """The `Unknown handoff function call id` bug from the August 12 call."""

    def test_a_repeated_item_is_not_executed_again(self, caplog):
        seg = bridge.ResponseTracker()
        assert seg.new_call(_function_item()) is not None
        seg.answered("call_1")
        with caplog.at_level(logging.WARNING, logger="poc3_gpt_live.app.bridge"):
            assert seg.new_call(_function_item()) is None
        assert "ya se ejecutó" in caplog.text

    def test_neither_while_it_is_being_answered(self):
        seg = bridge.ResponseTracker()
        assert seg.new_call(_function_item()) is not None
        assert seg.new_call(_function_item()) is None


class TestAnchoredSpeechThreshold:
    """PoC3's 200.0 has to remain the same 200.0 as in `shared`.

    The value lives in three independent places —`shared/carrier_simulator/
    observed.py`, PoC2's room smoke test and `bridge.py`— and **they are not
    unified**: unifying them would touch PoC1's path, which has already
    delivered its campaign. What can be done is to leave a detector in place, so
    that the day someone moves one of the two the divergence shows up here and
    not in a latency table.

    This test does NOT say that 200.0 is the right value for PoC3; what it costs
    is measured by `diagnostics/threshold_calibration.py`. It only says that the
    two numbers are the same number, which is the condition for being able to
    compare them.
    """

    def test_matches_the_one_in_shared(self):
        from shared.carrier_simulator.observed import RMS_WITH_SPEECH

        assert bridge.SPEECH_THRESHOLD_RMS == RMS_WITH_SPEECH

    def test_matches_the_calibration_analyzer_one(self):
        from poc3_gpt_live.diagnostics.threshold_calibration import INHERITED_THRESHOLD_RMS

        assert bridge.SPEECH_THRESHOLD_RMS == INHERITED_THRESHOLD_RMS

    def test_matches_the_poc2_room_smoke_one(self):
        """The third place. It is imported, not touched: PoC2 has already delivered its campaign."""
        from poc2_livekit.app.smoke_room import RMS_WITH_SPEECH as RMS_OF_POC2

        assert bridge.SPEECH_THRESHOLD_RMS == RMS_OF_POC2


class TestOnsetMidDelivery:
    """Where the mark falls when speech starts in the middle of a delivery.

    This is the test that justifies the 20 ms window **and the one that sets its
    ceiling**. `has_speech` returns one `bool` per block and its caller
    timestamps the mark once per block, so the fine window gives no resolution
    *within* the block: what it changes is **which block** the mark falls in.
    The resolution is still that of the block it is handed.

    PoC3's blocks are not the same size in the two directions:

    - outgoing: one `output_audio.delta`, ~100 ms;
    - incoming: whatever the resampler releases, ~61 ms (3 windows);
    - PoC1 and PoC2: the Twilio frame, exactly 20 ms.

    That is why the claim "the mark falls within 20 ms of the onset" only holds
    with 20 ms blocks, and it is tested separately below.
    """

    SAMPLE_RATE = 24_000
    MS_PER_WINDOW = 20

    def _sig(self, onset_ms: int, total_ms: int, amplitude: int) -> np.ndarray:
        """Silence until `onset_ms`, a 300 Hz tone from there on."""
        n = int(self.SAMPLE_RATE * total_ms / 1000)
        since = int(self.SAMPLE_RATE * onset_ms / 1000)
        t = np.arange(n) / float(self.SAMPLE_RATE)
        waveform = (np.sin(2 * np.pi * 300 * t) * amplitude).astype(np.int16)
        waveform[:since] = 0
        return waveform

    # Amplitude 400 = RMS 283 in a window full of tone, which crosses 200;
    # averaged over 100 ms with only the last 40 ms of tone, 179, which does not.
    # It is the level of a speech onset, not that of a sustained vowel.
    ONSET_AMPLITUDE = 400

    @pytest.mark.parametrize("onset_ms", [0, 20, 40, 60, 80])
    def test_the_fine_window_marks_in_the_onset_block(self, onset_ms):
        """Wherever it falls within the delta, that delta crosses."""
        delta = self._sig(onset_ms, 100, self.ONSET_AMPLITUDE).tobytes()
        assert bridge.has_speech(delta) is True

    @pytest.mark.parametrize("onset_ms", [60, 80])
    def test_the_delta_mean_misses_the_late_onset(self, onset_ms):
        """A late onset within the delta gets diluted, and the mark moves to the next one.

        It is not a 20 ms delay: it is a whole delta, ~100 ms, because the
        next chance to mark is the next delivery.
        """
        delta = self._sig(onset_ms, 100, self.ONSET_AMPLITUDE).tobytes()
        assert bridge.rms_pcm16(delta) < bridge.SPEECH_THRESHOLD_RMS
        # The next delta is full of tone: there it does cross, 100 ms late.
        following = self._sig(0, 100, self.ONSET_AMPLITUDE).tobytes()
        assert bridge.rms_pcm16(following) >= bridge.SPEECH_THRESHOLD_RMS

    def test_with_20_ms_blocks_the_mark_falls_within_the_window(self):
        """The literal claim, at the block size where it holds.

        With 20 ms frames —Twilio's, which is what PoC1 and PoC2 see— the first
        block that crosses is the one that **contains** the onset, so the
        mark's error is smaller than the window. In PoC3 this does not apply
        as is: its blocks are ~61 ms incoming and ~100 ms outgoing.
        """
        onset_ms = 50
        sig = self._sig(onset_ms, 200, 600)
        per_block = int(self.SAMPLE_RATE * self.MS_PER_WINDOW / 1000)
        primero = next(
            i
            for i in range(len(sig) // per_block)
            if bridge.has_speech(sig[i * per_block : (i + 1) * per_block].tobytes())
        )
        block_start = primero * self.MS_PER_WINDOW
        assert block_start <= onset_ms < block_start + self.MS_PER_WINDOW

    @pytest.mark.parametrize("onset_ms", [0, 20, 40, 60, 80])
    def test_a_strong_onset_gains_nothing_over_the_coarse_window(self, onset_ms):
        """And this is what actually happened in the calibration run.

        With an onset at full speech level both rules cross in the same delta:
        on `CA221584…` the fine window did not bring forward **any** of the 24
        onsets (nor any of the 8 incoming bursts). What the fine window gains
        is the weak onset, which is the rare case. This test exists so that no
        one reads E4's 80 ms as the acknowledgement arriving earlier: those
        80 ms live inside the delta and the acknowledgement cannot timestamp
        them.
        """
        delta = self._sig(onset_ms, 100, 8_000).tobytes()
        assert bridge.has_speech(delta) is True
        assert bridge.rms_pcm16(delta) >= bridge.SPEECH_THRESHOLD_RMS


class TestFormatoPcmu:
    """8 kHz μ-law: Twilio's audio goes to the session unconverted, and comes back the same."""

    def test_the_session_format_of_each_mode(self):
        assert bridge.session_format("pcmu") == {"type": "audio/pcmu", "rate": 8000}
        assert bridge.session_format("pcm24") == {"type": "audio/pcm", "rate": 24000}
        with pytest.raises(ValueError):
            bridge.session_format("opus")

    def test_the_audio_passes_untouched_in_both_directions(self):
        live_bridge = bridge.AudioBridge("pcmu")
        trama = bytes(range(160))
        assert live_bridge.entry(trama) == trama
        assert live_bridge.output(trama + b"\x7f") == trama + b"\x7f"

    def test_the_speech_is_measured_decoded_and_in_8_khz_windows(self):
        live_bridge = bridge.AudioBridge("pcmu")
        tone = (np.sin(np.arange(160) / 8000 * 2 * np.pi * 300) * 8000).astype(np.int16)
        ulaw = bridge.pcm16_a_ulaw(tone)
        assert live_bridge.samples_per_window == 160
        assert len(live_bridge.to_measure(ulaw)) == 320
        assert bridge.has_speech(live_bridge.to_measure(ulaw), samples_per_window=live_bridge.samples_per_window)
        silence = bridge.pcm16_a_ulaw(np.zeros(160, dtype=np.int16))
        assert not bridge.has_speech(live_bridge.to_measure(silence), samples_per_window=160)

    def test_in_pcm24_still_resamples(self):
        live_bridge = bridge.AudioBridge()
        assert live_bridge.es_pcm and live_bridge.samples_per_window == 480
        assert len(live_bridge.entry(bytes(160) * 10)) > 160 * 10

    def test_in_ulaw_an_odd_payload_is_sent(self):
        import asyncio

        sent_items: list[dict] = []

        class _FakeWs:
            async def send(self, text):
                sent_items.append(json.loads(text))

        client = bridge.GptLiveClient("wss://x", {}, {"audio": {"format": bridge.session_format("pcmu")}})
        client._ws = _FakeWs()
        asyncio.run(client.send_audio(b"\x7f" * 161))
        assert len(sent_items) == 1

        client = bridge.GptLiveClient("wss://x", {}, {})
        client._ws = _FakeWs()
        asyncio.run(client.send_audio(b"\x7f" * 161))
        assert len(sent_items) == 1
