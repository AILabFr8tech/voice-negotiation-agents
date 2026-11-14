"""El gancho de texto del puente hacia el arnés, de punta a punta.

Aparte de `test_telephony.py` porque lo que se prueba cruza tres módulos: el
puente de producción emite, el cliente del arnés despacha y `ObservadoDePoC3`
reensambla. Si el archivo viviera en cualquiera de los tres, la prueba se leería
como "lo mío funciona" en vez de "los tres encajan".

**Dónde emite PoC1 su equivalente, que es lo que hay que igualar.**
`poc1_pipecat/app/harness.py:175` define el `EcoDeTexto` y
`poc1_pipecat/app/telephony.py:417` lo inserta después del TTS y **antes** de
`transporte.output()`. Lleva cinco llaves: `event`, `de`, `texto`,
`has_spaces` y `final`. PoC3 manda esas cinco con los mismos nombres, más
`item_id` (vacío desde la API pública, que no manda `item`), `start_ms` y
`end_ms`, que PoC1 no tiene porque su texto no viene con línea de tiempo del
servidor.

Lo que **no** es equivalente es el momento, y está medido en el encabezado de
`app/media_streams.py`: el texto de PoC3 llega antes que su propio audio.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest import mock

import numpy as np

from poc3_gpt_live.app import bridge, telephony
from poc3_gpt_live.app.latency import LatencyCollector
from poc3_gpt_live.app.telephony import (
    FROM_AGENT,
    FROM_CARRIER,
    TEXT_EVENT,
    CallBridge,
    CallRecord,
    diagnostics_block,
)
from shared.negotiation_engine import TEST_TRIP, NegotiationSession

# --------------------------------------------------------------------------
# Fragmentos reales, no sintéticos
# --------------------------------------------------------------------------
#
# Texto y tiempos copiados tal cual del censo de la corrida de calibración
# `CA221584bae6d0768a1365a8fd2ffb03aa` (22 de septiembre de 2026), con la forma
# de la API pública (`session.*_transcript.delta` con `delta`; el alpha los
# mandaba como `*_transcript.added` con `item.text`). Un fragmento
# inventado "bien portado" no tendría los dos casos que deciden el reensamblado:
# la palabra partida en dos trozos (`" ¿H"` + `"ablo con"`, `" diecis"` +
# `"iete"`) y el separador que viaja al PRINCIPIO del trozo.

AGENT_FRAGMENTS: list[dict[str, Any]] = [
    {"type": "session.output_transcript.delta", "start_ms": 4400, "end_ms": 4600,
     "delta": " ¿H"},
    {"type": "session.output_transcript.delta", "start_ms": 4600, "end_ms": 4800,
     "delta": "ablo con"},
    {"type": "session.output_transcript.delta", "start_ms": 4800, "end_ms": 5000,
     "delta": " Carlos?"},
    {"type": "session.output_transcript.delta", "start_ms": 34200, "end_ms": 34400,
     "delta": " una tarifa"},
    {"type": "session.output_transcript.delta", "start_ms": 34600, "end_ms": 34800,
     "delta": " de"},
    {"type": "session.output_transcript.delta", "start_ms": 34800, "end_ms": 35000,
     "delta": " diecis"},
    {"type": "session.output_transcript.delta", "start_ms": 35000, "end_ms": 35200,
     "delta": "iete"},
    {"type": "session.output_transcript.delta", "start_ms": 35400, "end_ms": 35600,
     "delta": " mil"},
]

# Lo que esas ocho piezas dicen. Es la afirmación entera: con la concatenación
# cruda sale esto, y con cualquier regla que reponga separadores saldría
# "¿H ablo" y "diecis iete".
WHAT_THE_AGENT_SAID = "¿Hablo con Carlos? una tarifa de diecisiete mil"

# Del mismo censo, el lado del transportista. `", hola"` es el caso contrario:
# un fragmento que NO abre con espacio porque abre con coma, y al que reponerle
# uno daría "Sí , hola".
CARRIER_FRAGMENTS: list[dict[str, Any]] = [
    {"type": "session.input_transcript.delta", "start_ms": 14000, "end_ms": 14200,
     "delta": " Sí"},
    {"type": "session.input_transcript.delta", "start_ms": 14400, "end_ms": 14600,
     "delta": ", hola"},
    {"type": "session.input_transcript.delta", "start_ms": 14600, "end_ms": 14800,
     "delta": ", Carlos"},
]
WHAT_THE_CARRIER_SAID = "Sí, hola, Carlos"


# --------------------------------------------------------------------------
# Andamios
# --------------------------------------------------------------------------


def _audio_delta(audio_ms: int = 100) -> dict[str, Any]:
    """Un `session.output_audio.delta` con PCM16 de 24 kHz de verdad."""
    samples = int(bridge.LIVE_SAMPLE_RATE * audio_ms / 1000)
    tone = np.sin(np.arange(samples) / bridge.LIVE_SAMPLE_RATE * 2 * np.pi * 300)
    pcm = (tone * 8_000).astype("<i2").tobytes()
    return {
        "type": bridge.EV_OUTPUT_AUDIO_DELTA,
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


class _WebSocketPointingTo:
    """Un `_ws` que guarda lo que el puente manda. `raises_on` lo revienta."""

    def __init__(self, raises_on: str = "") -> None:
        self.sent: list[dict[str, Any]] = []
        self._raises_on = raises_on
        self.crashed_attempts = 0

    async def send_text(self, raw_value: str) -> None:
        message = json.loads(raw_value)
        if self._raises_on and message.get("event") == self._raises_on:
            self.crashed_attempts += 1
            raise RuntimeError("el socket se cayó a media llamada")
        self.sent.append(message)

    def of_kind(self, event: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("event") == event]


def _bridge(ws: Any, *, text_echo: bool) -> CallBridge:
    """El puente REAL, por `__init__`. Ver `TestConstruccionDelPuente`."""
    return CallBridge(
        websocket=ws,
        client=mock.Mock(),
        negotiation=NegotiationSession(trip=TEST_TRIP, call_id="CAdeprueba"),
        record=CallRecord(),
        collector=LatencyCollector(),
        stream_sid="MZdeprueba",
        text_echo=text_echo,
    )


def _launch(live_bridge: CallBridge, events: list[dict[str, Any]]) -> None:
    """Le da de comer al bucle real `_from_live_to_twilio` y lo deja terminar."""

    async def _listen():
        for event in events:
            yield event

    live_bridge._client.listen = _listen
    asyncio.run(live_bridge._from_live_to_twilio())


# --------------------------------------------------------------------------


class TestTheBridgeEmitsWithThePoC1Protocol:
    def test_disabled_sends_not_a_single_line(self) -> None:
        """Se apaga solo para una llamada telefónica real: ahí sí hay Twilio."""
        ws = _WebSocketPointingTo()
        live_bridge = _bridge(ws, text_echo=False)

        _launch(live_bridge, AGENT_FRAGMENTS)

        assert ws.of_kind(TEXT_EVENT) == []
        # Y la transcripción del JSON sí se llenó: lo que se apaga es el cable.
        assert live_bridge._record.turns

    def test_sends_the_same_keys_as_poc1(self) -> None:
        ws = _WebSocketPointingTo()
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, AGENT_FRAGMENTS[:1])

        message = ws.of_kind(TEXT_EVENT)[0]
        assert message["de"] == FROM_AGENT
        assert message["has_spaces"] is True
        assert message["final"] is False
        assert message["item_id"] == ""
        assert message["start_ms"] == 4400

    def test_the_text_travels_raw_with_its_leading_space(self) -> None:
        """Un `strip()` aquí repondría el defecto del WER falso de 84.5%."""
        ws = _WebSocketPointingTo()
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, AGENT_FRAGMENTS[:1])

        assert ws.of_kind(TEXT_EVENT)[0]["text"].startswith(" ")

    def test_also_sends_the_carrier_side(self) -> None:
        """PoC1 emite los dos (`build_carrier_echo`); éste también."""
        ws = _WebSocketPointingTo()
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, CARRIER_FRAGMENTS)

        assert {m["de"] for m in ws.of_kind(TEXT_EVENT)} == {FROM_CARRIER}

    def test_the_audio_did_not_change_shape(self) -> None:
        """El gancho no toca el único evento que Twilio sí espera."""
        ws = _WebSocketPointingTo()
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, [_audio_delta()])

        mean = ws.of_kind("mean")
        assert len(mean) == 1
        assert mean[0]["streamSid"] == "MZdeprueba"
        assert mean[0]["mean"]["payload"]


class TestTheFullTextPathAgainstTheHarness:
    """Puente real -> cable -> cliente del arnés real -> `ObservadoDePoC3` real.

    La lección de `TestConstruccionDelPuente` otra vez: las piezas pueden estar
    verdes por separado y no encajar. Aquí corren las tres juntas sobre los
    fragmentos que el alpha mandó de verdad, y lo que se afirma es que el texto
    que reensambla el arnés es **idéntico** al que el puente escribe en el JSON
    de la llamada. Si los dos lados se separaran, el WER dependería de qué
    archivo se mirara.
    """

    def _ruta(self, events: list[dict[str, Any]]):
        from poc3_gpt_live.app import media_streams as ms

        observed = ms.ObservedFromPoC3(silent=True)
        harness = ms.MediaStreamsClient(
            "ws://no-se-conecta",
            label="ruta-de-texto",
            on_receive_message=observed.annotate_bridge_text,
        )

        class _WsToTheHarness:
            async def send_text(self, raw_value: str) -> None:
                # El dispatcher REAL del cliente del arnés, el mismo que corre
                # cuando el WebSocket le entrega un frame.
                harness._dispatch(raw_value)

        live_bridge = _bridge(_WsToTheHarness(), text_echo=True)
        _launch(live_bridge, events)
        live_bridge._record.normalize()
        return live_bridge, harness, observed

    def test_the_harness_reassembles_exactly_what_the_json_says(self) -> None:
        live_bridge, harness, observed = self._ruta(AGENT_FRAGMENTS)

        of_harness = observed.assemble_agent(observed.agent_fragments())
        of_json = " ".join(t["text"] for t in live_bridge._record.turns)

        assert of_harness == WHAT_THE_AGENT_SAID
        assert of_harness == of_json
        assert harness.text_messages == len(AGENT_FRAGMENTS)
        assert observed.deaf_to_text is False

    def test_split_words_are_not_separated(self) -> None:
        _, _, observed = self._ruta(AGENT_FRAGMENTS)

        text = observed.assemble_agent(observed.agent_fragments())
        assert "diecisiete" in text and "diecis iete" not in text
        assert "¿Hablo" in text and "¿H ablo" not in text

    def test_the_carrier_side_goes_to_our_stt(self) -> None:
        """Con `own_identity`: es lo que el WER compara contra el clip."""
        _, _, observed = self._ruta(CARRIER_FRAGMENTS)

        ours = observed.our_stt_fragments()
        assert len(ours) == len(CARRIER_FRAGMENTS)
        assert observed.agent_fragments() == []
        # Y sin comas huérfanas: `", hola"` no lleva espacio delante.
        assert observed.assemble_agent(ours) == WHAT_THE_CARRIER_SAID

    def test_without_the_flag_the_harness_restores_the_separator(self) -> None:
        """La regla la manda el protocolo, no el nombre de la PoC.

        Es el caso de PoC1 —ElevenLabs manda una palabra por frame sin
        espacios— corrido por el reensamblador de PoC3. Que funcione en las dos
        direcciones es lo que permite que el arnés no sepa contra quién habla.
        """
        from poc3_gpt_live.app import media_streams as ms

        observed = ms.ObservedFromPoC3(silent=True)
        for word in ("Hola,", "habla", "Carlos"):
            observed.annotate_bridge_text(
                {"de": FROM_AGENT, "text": word, "has_spaces": False}
            )

        assert (
            observed.assemble_agent(observed.agent_fragments())
            == "Hola, habla Carlos"
        )

    def test_without_hook_the_run_is_marked_as_deaf(self) -> None:
        from poc3_gpt_live.app import media_streams as ms

        observed = ms.ObservedFromPoC3(silent=True)
        harness = ms.MediaStreamsClient(
            "ws://no-se-conecta",
            label="sin-gancho",
            on_receive_message=observed.annotate_bridge_text,
        )

        class _WsToTheHarness:
            async def send_text(self, raw_value: str) -> None:
                harness._dispatch(raw_value)

        _launch(_bridge(_WsToTheHarness(), text_echo=False), AGENT_FRAGMENTS)

        assert harness.text_messages == 0
        assert observed.deaf_to_text is True


class TestTheScriptBranchIsRecorded:
    """Sin texto todos los turnos caen a su línea por defecto, y se ve.

    La ramificación la decide `play_script` con lo que el agente acaba de
    decir, así que antes del gancho una corrida ramificada era
    indistinguible de una que no lo estaba. El bloque lo dice por turno.
    """

    def test_the_block_publishes_by_default_and_deafness(self) -> None:
        from shared.test_scenarios.decision import BranchDecision

        from poc3_gpt_live.app import media_streams as ms

        observed = ms.ObservedFromPoC3(silent=True)
        observed.decisions.append(
            BranchDecision(
                turn="T3",
                rama="accepts",
                linea="va",
                variant="accepts",
                by_default=False,
                motivo="el agente propuso 17,000",
            )
        )
        observed.decisions.append(
            BranchDecision(
                turn="T5",
                rama="base",
                linea="…",
                variant="base",
                by_default=True,
                motivo="sin evidencia en el texto del agente",
            )
        )

        block = observed.branches_block()

        assert block["branched_turns"] == 2
        assert block["by_default"] == 1
        assert block["deaf_to_text"] is True
        assert [d["turn"] for d in block["decisions"]] == ["T3", "T5"]


class TestTheTextEchoCannotCutTheAudio:
    """Mismo guardarraíl que 65eb3ad, aplicado al gancho nuevo.

    Las tres cosas de siempre, en orden: el audio salió, el gancho quedó
    apagado, y la falla quedó escrita en el JSON.
    """

    def test_an_echo_failure_turns_it_off_and_lets_the_audio_out(self) -> None:
        ws = _WebSocketPointingTo(raises_on=TEXT_EVENT)
        live_bridge = _bridge(ws, text_echo=True)

        _launch(
            live_bridge,
            [AGENT_FRAGMENTS[0], _audio_delta()]
            + AGENT_FRAGMENTS[1:]
            + [_audio_delta()],
        )

        # 1. El audio salió, las dos veces, incluida la posterior a la falla.
        assert len(ws.of_kind("mean")) == 2
        # 2. El eco se apagó a la PRIMERA y no se reintentó por fragmento.
        assert ws.crashed_attempts == 1
        assert live_bridge._text_echo is False
        # 3. Y quedó escrita, con sitio, excepción e instante.
        faults = live_bridge.diagnostics_failures
        assert [f["sitio"] for f in faults] == ["text_echo"]
        assert faults[0]["excepcion"] == "RuntimeError"
        assert faults[0]["call_second"] >= 0.0

    def test_the_call_is_left_excluded_in_writing(self) -> None:
        ws = _WebSocketPointingTo(raises_on=TEXT_EVENT)
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, AGENT_FRAGMENTS)

        assert diagnostics_block(live_bridge.diagnostics_failures)[
            "full_capture"
        ] is False

    def test_the_json_record_stays_complete_even_if_the_echo_dies(self) -> None:
        """Lo que se pierde es el cruce con el arnés, no la transcripción."""
        ws = _WebSocketPointingTo(raises_on=TEXT_EVENT)
        live_bridge = _bridge(ws, text_echo=True)

        _launch(live_bridge, AGENT_FRAGMENTS)
        live_bridge._record.normalize()

        assert (
            " ".join(t["text"] for t in live_bridge._record.turns)
            == WHAT_THE_AGENT_SAID
        )


class TestTheEchoIsOnInCampaign:
    """La campaña de PoC3 corre con el arnés por WebSocket local.

    No hay Twilio en ninguna llamada de campaña, así que el otro lado del cable
    es siempre quien espera `arnes.texto`. El default es encendido y lo
    excepcional —una llamada telefónica real— se declara apagándolo.
    """

    def test_the_default_is_on(self) -> None:
        assert telephony.text_echo_enabled({}) is True

    def test_turns_off_with_the_variable(self) -> None:
        assert (
            telephony.text_echo_enabled({telephony.NO_ECHO_VARIABLE: "1"})
            is False
        )

    def test_the_bridge_receives_it_on_without_asking(self, monkeypatch) -> None:
        """El camino que corre en campaña: nadie pasa la bandera a mano."""
        monkeypatch.delenv(telephony.NO_ECHO_VARIABLE, raising=False)
        ws = _WebSocketPointingTo()
        live_bridge = CallBridge(
            websocket=ws,
            client=mock.Mock(),
            negotiation=NegotiationSession(
                trip=TEST_TRIP, call_id="CAdeprueba"
            ),
            record=CallRecord(),
            collector=LatencyCollector(),
            stream_sid="MZdeprueba",
            text_echo=telephony.text_echo_enabled(),
        )

        _launch(live_bridge, AGENT_FRAGMENTS)

        assert len(ws.of_kind(TEXT_EVENT)) == len(AGENT_FRAGMENTS)
        assert live_bridge.emitted_text_fragments == len(AGENT_FRAGMENTS)


class TestADeafRunIsNotFit:
    """Sin texto no hay WER del agente ni ramas del guion: la llamada no sirve.

    Lo que se protege es que eso quede **escrito en el JSON**, no solo en el
    log: una corrida sorda tiene audio y latencias intactas, así que mirando los
    números se ve sana.
    """

    def _save(self, tmp_path, live_bridge):
        return json.loads(
            telephony.save_run(
                live_bridge._negotiation,
                live_bridge._record,
                base_path=tmp_path,
                live_bridge=live_bridge,
            ).read_text(encoding="utf-8")
        )

    def test_without_a_session_it_does_not_blame_a_variable_nobody_set(self) -> None:
        """Si GPT-Live ni arrancó, el motivo lo dice; no "se apagó a mano".

        Sin puente, el bloque no debe caer en la rama del eco apagado, que manda
        a buscar `POC3_SIN_ECO_DE_TEXTO` en un `.env` que no la tiene.
        """
        block = telephony.text_echo_block(
            enabled=False, fragments=0, faults=None, session_initiated=False
        )
        assert block["eligible"] is False
        assert "no llegó a iniciar" in block["motivo"]
        assert telephony.NO_ECHO_VARIABLE not in block["motivo"]

    def test_with_text_it_comes_out_fit(self, tmp_path) -> None:
        live_bridge = _bridge(_WebSocketPointingTo(), text_echo=True)
        _launch(live_bridge, AGENT_FRAGMENTS)

        block = self._save(tmp_path, live_bridge)["text_echo"]

        assert block["eligible"] is True
        assert block["deaf_to_text"] is False
        assert block["emitted_fragments"] == len(AGENT_FRAGMENTS)
        assert block["motivo"] == ""

    def test_the_echo_on_with_not_a_single_fragment_is_not_fit(
        self, tmp_path
    ) -> None:
        """El alpha no mandó `output_transcript.added`: el agente habló solo."""
        live_bridge = _bridge(_WebSocketPointingTo(), text_echo=True)
        _launch(live_bridge, [_audio_delta(), _audio_delta()])

        block = self._save(tmp_path, live_bridge)["text_echo"]

        assert block["deaf_to_text"] is True
        assert block["eligible"] is False
        assert "no sirve" in block["motivo"] or "deaf" in block["motivo"]

    def test_an_echo_that_dies_is_not_fit_either(self, tmp_path) -> None:
        live_bridge = _bridge(
            _WebSocketPointingTo(raises_on=TEXT_EVENT), text_echo=True
        )
        _launch(live_bridge, AGENT_FRAGMENTS)

        block = self._save(tmp_path, live_bridge)["text_echo"]

        # No es "sordo": es que se encendió y se cayó. Se distinguen.
        assert block["deaf_to_text"] is True
        assert block["died_mid_call"] is True
        assert block["eligible"] is False

    def test_manually_disabled_is_declared_with_its_reason(self, tmp_path) -> None:
        live_bridge = _bridge(_WebSocketPointingTo(), text_echo=False)
        _launch(live_bridge, AGENT_FRAGMENTS)

        block = self._save(tmp_path, live_bridge)["text_echo"]

        assert block["enabled"] is False
        assert block["eligible"] is False
        assert telephony.NO_ECHO_VARIABLE in block["motivo"]

    def test_the_key_is_always_there_even_without_a_bridge(self, tmp_path) -> None:
        """Una llamada que ni siquiera corrió no se inventa un eco sano."""
        content = json.loads(
            telephony.save_run(
                NegotiationSession(trip=TEST_TRIP, call_id="CAsinpuente"),
                CallRecord(),
                base_path=tmp_path,
            ).read_text(encoding="utf-8")
        )

        assert content["text_echo"]["eligible"] is False

    def test_the_harness_says_the_same_from_the_other_side(self) -> None:
        from poc3_gpt_live.app import media_streams as ms

        deaf = ms.ObservedFromPoC3(silent=True)
        assert deaf.text_block()["eligible"] is False
        assert deaf.text_block()["deaf_to_text"] is True

        heard_it = ms.ObservedFromPoC3(silent=True)
        heard_it.annotate_bridge_text(
            {"de": FROM_AGENT, "text": " Hola", "has_spaces": True}
        )
        assert heard_it.text_block()["eligible"] is True
        assert heard_it.text_block()["of_agent"] == 1
