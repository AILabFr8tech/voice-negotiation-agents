"""El arnés de PoC1, verificado sin gastar un centavo.

Lo que se prueba aquí es lo que una llamada de prueba **no** delata:

- que el arnés mida el pipeline de producción y no una copia suya (los ganchos
  se insertan donde deben, y sin ellos el pipeline queda idéntico al de antes);
- que el eco del texto distinga las dos direcciones y no invente una tercera;
- que el STT se reensamble con separador, porque pegarlo es el error que le dio
  a PoC3 un WER falso de 84.5%;
- que el aviso de costo no se pueda saltar.

Lo que no se puede probar aquí —que Deepgram, OpenAI y ElevenLabs contesten— es
para lo que existe la corrida de humo con `--turns`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from poc1_pipecat.app import harness, marks
from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw
from shared.carrier_simulator import TextFragment, rms_pcm16

RUTA_MODULO = Path(harness.__file__)


# --------------------------------------------------------------------------
# Lo observado
# --------------------------------------------------------------------------


def test_the_agent_audio_is_decoded_before_measuring_its_energy() -> None:
    """μ-law leído como PCM16 no suena a nada y mide cualquier cosa.

    Es el error que el canal de auditoría de PoC2 tuvo que declarar aparte
    (`as_pcm16`): aquí el mismo descuido haría que el arnés no reconociera la
    voz del agente y avanzara el guion sin esperar respuesta.
    """
    observed = harness.ObservedFromPoC1()
    fuerte = (np.sin(np.arange(160) / 4.0) * 12_000).astype(np.int16)
    observed.annotate_audio_in_ulaw(pcm16_a_ulaw(fuerte))

    assert observed.audio_frames == 1
    assert observed.t_agent_first_speech is not None
    # Contra el nivel conocido y no contra el umbral: los bytes μ-law leídos
    # como PCM16 **también** pasan el umbral (18,608 contra 8,522 reales), así
    # que "suena fuerte" no distingue decodificar de no hacerlo. Lo que sí
    # distingue es que el nivel sea el del audio original.
    expected = rms_pcm16(fuerte.tobytes())
    assert observed.rms[-1] == pytest.approx(expected, rel=0.01)


def test_the_agent_silence_does_not_count_as_speech() -> None:
    observed = harness.ObservedFromPoC1()
    observed.annotate_audio_in_ulaw(pcm16_a_ulaw(np.zeros(160, dtype=np.int16)))
    assert observed.audio_frames == 1
    assert observed.t_agent_first_speech is None


def test_the_stt_is_joined_with_a_space_and_not_glued() -> None:
    """Deepgram manda segmentos, no deltas: concatenarlos pega las palabras."""
    observed = harness.ObservedFromPoC1()
    fragments = [
        TextFragment(t=0.0, sender="carrier", text="Sí, soy Carlos.", segment="", final=True),
        TextFragment(t=1.0, sender="carrier", text="Dígame.", segment="", final=True),
    ]
    assert observed.assemble_stt(fragments) == "Sí, soy Carlos. Dígame."


def test_the_stt_does_not_add_extra_spaces_with_empty_fragments() -> None:
    observed = harness.ObservedFromPoC1()
    fragments = [
        TextFragment(t=0.0, sender="carrier", text="Hola", segment="", final=True),
        TextFragment(t=0.5, sender="carrier", text="   ", segment="", final=True),
        TextFragment(t=1.0, sender="carrier", text="Carlos", segment="", final=True),
    ]
    assert observed.assemble_stt(fragments) == "Hola Carlos"


def _of_agent(text: str, *, has_spaces: bool) -> TextFragment:
    return TextFragment(
        t=0.0,
        sender="agent",
        text=text,
        segment=harness.WITH_SPACES if has_spaces else "",
        final=False,
    )


def test_the_agent_words_are_not_glued_together() -> None:
    """ElevenLabs manda un frame por palabra y SIN el espacio que las separa.

    Concatenarlas da "Carlos,quierohablarsobreunenvío".
    """
    observed = harness.ObservedFromPoC1()
    fragments = [
        _of_agent("Gracias,", has_spaces=False),
        _of_agent("Carlos.", has_spaces=False),
    ]
    assert observed.assemble_agent(fragments) == "Gracias, Carlos."


def test_a_tts_that_does_bring_spaces_does_not_receive_others() -> None:
    """El testigo: si se metiera espacio siempre, saldría doble."""
    observed = harness.ObservedFromPoC1()
    fragments = [
        _of_agent("Bue", has_spaces=True),
        _of_agent("nos días", has_spaces=True),
    ]
    assert observed.assemble_agent(fragments) == "Buenos días"


# --------------------------------------------------------------------------
# El eco del texto
# --------------------------------------------------------------------------


def test_the_incoming_text_is_attributed_to_whoever_said_it() -> None:
    observed = harness.ObservedFromPoC1()
    harness._on_text_received(
        observed,
        {"event": harness.TEXT_EVENT, "de": harness.FROM_CARRIER, "text": "Sí, soy Carlos."},
    )
    harness._on_text_received(
        observed,
        {"event": harness.TEXT_EVENT, "de": harness.FROM_AGENT, "text": "Gracias, Carlos."},
    )

    assert [f.text for f in observed.our_stt_fragments()] == ["Sí, soy Carlos."]
    assert [f.text for f in observed.agent_fragments()] == ["Gracias, Carlos."]


def test_a_message_without_text_leaves_no_fragment() -> None:
    """Un fragmento vacío cuenta como turno contestado sin que nadie hablara."""
    observed = harness.ObservedFromPoC1()
    harness._on_text_received(observed, {"event": harness.TEXT_EVENT, "de": harness.FROM_AGENT})
    harness._on_text_received(
        observed, {"event": harness.TEXT_EVENT, "de": harness.FROM_AGENT, "text": ""}
    )
    assert observed.fragments == []


def test_the_echo_resends_the_original_frame_and_adds_the_message() -> None:
    """El eco NO puede tragarse frames: el audio del TTS pasa por aquí."""
    from pipecat.frames.frames import (
        OutputTransportMessageUrgentFrame,
        TranscriptionFrame,
        TTSTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection

    echo = harness.build_carrier_echo()
    agent = harness.build_agent_echo()
    pushed: list[Any] = []

    async def _capture(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    echo.push_frame = _capture  # type: ignore[method-assign]
    agent.push_frame = _capture  # type: ignore[method-assign]

    # Se parchea la clase, así que `self` viaja como primer argumento.
    async def _nada(_self: Any, frame: Any, direction: Any) -> None:
        return None

    # `super().process_frame` necesita el ciclo de vida entero de pipecat; lo que
    # se prueba es la regla, no la plomería del FrameProcessor.
    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _nada  # type: ignore[assignment]
    try:
        asyncio.run(
            echo.process_frame(
                TranscriptionFrame(text="Sí, soy Carlos.", user_id="u", timestamp="t"),
                FrameDirection.DOWNSTREAM,
            )
        )
        asyncio.run(
            agent.process_frame(
                TTSTextFrame(text="Gracias", aggregated_by="word"),
                FrameDirection.DOWNSTREAM,
            )
        )
    finally:
        fp.FrameProcessor.process_frame = original  # type: ignore[assignment]

    assert isinstance(pushed[0], TranscriptionFrame)
    assert isinstance(pushed[1], OutputTransportMessageUrgentFrame)
    assert pushed[1].message["de"] == harness.FROM_CARRIER
    assert isinstance(pushed[2], TTSTextFrame)
    assert isinstance(pushed[3], OutputTransportMessageUrgentFrame)
    assert pushed[3].message["de"] == harness.FROM_AGENT


def test_the_echo_does_not_resend_the_interim_ones() -> None:
    """Los interinos se reemplazan entre sí: concatenarlos repite la frase."""
    from pipecat.frames.frames import (
        InterimTranscriptionFrame,
        OutputTransportMessageUrgentFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
    import pipecat.processors.frame_processor as fp

    echo = harness.build_carrier_echo()
    pushed: list[Any] = []

    async def _capture(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    echo.push_frame = _capture  # type: ignore[method-assign]

    # Se parchea la clase, así que `self` viaja como primer argumento.
    async def _nada(_self: Any, frame: Any, direction: Any) -> None:
        return None

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _nada  # type: ignore[assignment]
    try:
        asyncio.run(
            echo.process_frame(
                InterimTranscriptionFrame(text="Sí soy", user_id="u", timestamp="t"),
                FrameDirection.DOWNSTREAM,
            )
        )
    finally:
        fp.FrameProcessor.process_frame = original  # type: ignore[assignment]

    assert len(pushed) == 1
    assert not any(isinstance(f, OutputTransportMessageUrgentFrame) for f in pushed)


# --------------------------------------------------------------------------
# Los ganchos: que el arnés mida el pipeline de producción
# --------------------------------------------------------------------------


def test_without_hooks_the_pipeline_does_not_change() -> None:
    """Una llamada de campaña tiene que construir lo mismo que antes."""
    from poc1_pipecat.app.telephony import HarnessHooks, _insert

    by_default = HarnessHooks()
    assert by_default.hang_up_via_rest is True
    assert _insert(by_default.after_turn) == []
    assert _insert(by_default.before_output) == []


def test_the_hooks_insert_what_the_factory_returns() -> None:
    from poc1_pipecat.app.telephony import _insert

    witness = object()
    assert _insert(lambda: (witness,)) == [witness]


def test_the_hook_is_called_once_per_call() -> None:
    """Una sonda compartida entre dos llamadas mezclaría sus ceros."""
    from poc1_pipecat.app.telephony import _insert

    done: list[object] = []

    def _factory() -> tuple[object, ...]:
        new = object()
        done.append(new)
        return (new,)

    first = _insert(_factory)
    second_one = _insert(_factory)
    assert first != second_one
    assert len(done) == 2


def _settings_without_twilio() -> Any:
    """Lo mínimo que `build_serializer` le pide a los ajustes."""

    class _Settings:
        twilio_account_sid = ""
        twilio_auth_token = ""

    return _Settings()


def test_without_hang_up_by_rest_the_serializer_does_not_ask_for_credentials() -> None:
    """Es lo que le permite al arnés correr sin cuenta de Twilio."""
    from poc1_pipecat.app.telephony import HarnessHooks, build_serializer

    serializer = build_serializer(
        _settings_without_twilio(),
        stream_sid="s",
        call_sid="c",
        hooks=HarnessHooks(hang_up_via_rest=False),
    )
    assert serializer is not None


def test_with_hang_up_by_rest_and_no_credentials_it_refuses() -> None:
    """El testigo del de arriba: si no se negara, la prueba anterior no diría nada.

    Y de paso es la razón por la que el arnés tiene que apagarlo: con un
    `callSid` inventado, colgar por REST pegaría contra la API de Twilio.
    """
    from poc1_pipecat.app.telephony import HarnessHooks, build_serializer

    with pytest.raises(ValueError):
        build_serializer(
            _settings_without_twilio(),
            stream_sid="s",
            call_sid="c",
            hooks=HarnessHooks(hang_up_via_rest=True),
        )


def test_the_pipeline_passes_the_serializer_the_hooks_it_received(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pasarle los defaults dejaría el colgado por REST prendido en el arnés.

    Y con credenciales de Twilio en el `.env` eso no falla: intenta colgar una
    llamada inventada al terminar, en silencio. Por eso se verifica el paso del
    argumento y no solo que el arnés lo apague.

    El pipeline entero no se puede construir sin claves de OpenAI, Deepgram y
    ElevenLabs, así que se lo deja reventar **después** del serializer: para
    entonces el espía ya anotó.
    """
    from poc1_pipecat.app import telephony

    seen_items: list[Any] = []

    def _spy(_settings: Any, **kwargs: Any) -> Any:
        seen_items.append(kwargs["hooks"])
        raise RuntimeError("hasta aquí llega la prueba")

    monkeypatch.setattr(telephony, "build_serializer", _spy)
    monkeypatch.setattr(telephony, "verify_for_voice", lambda _a: None)

    hooks = telephony.HarnessHooks(hang_up_via_rest=False)
    with pytest.raises(RuntimeError):
        telephony.build_voice_pipeline(
            _settings_without_twilio(),
            object(),
            object(),
            stream_sid="s",
            call_sid="c",
            hooks=hooks,
        )

    assert seen_items == [hooks]


def test_the_harness_turns_off_hang_up_by_rest() -> None:
    """Con un callSid inventado, colgar por REST pega contra la API de Twilio."""
    assert harness.harness_hooks([]).hang_up_via_rest is False


def test_the_carrier_echo_goes_before_the_aggregator() -> None:
    """El agregador se queda el `TranscriptionFrame`: después ya no pasa.

    Con el eco después del agregador, el arnés recibe la voz del agente y cero
    transcripciones del transportista.
    """
    from poc1_pipecat.app import telephony

    source = Path(telephony.__file__).read_text(encoding="utf-8")
    cuerpo = source[source.index("        [\n            transport.input()") :]
    assert cuerpo.index("_insert(hooks.after_stt)") < cuerpo.index(
        "aggregator_group.user()"
    )


def _emitted(processor: Any, frame: Any) -> list[Any]:
    """Lo que `procesador` empuja al recibir `frame`, sin levantar pipecat."""
    from pipecat.processors.frame_processor import FrameDirection
    import pipecat.processors.frame_processor as fp

    output: list[Any] = []

    async def _capture(f: Any, direction: Any = None) -> None:
        output.append(f)

    processor.push_frame = _capture  # type: ignore[method-assign]

    async def _nada(_self: Any, f: Any, direction: Any) -> None:
        return None

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _nada  # type: ignore[assignment]
    try:
        asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))
    finally:
        fp.FrameProcessor.process_frame = original  # type: ignore[assignment]
    return output


def test_each_pipeline_point_receives_the_echo_meant_for_it() -> None:
    """Uno solo no puede ver las dos direcciones: no coexisten en ningún punto.

    Y montarlos al revés sería silencioso: el del agente colocado tras el STT no
    vería nunca un `TTSTextFrame`, y la bitácora saldría sin la voz del agente.
    """
    from pipecat.frames.frames import (
        OutputTransportMessageUrgentFrame,
        TranscriptionFrame,
        TTSTextFrame,
    )

    probes: list[Any] = []
    hooks = harness.harness_hooks(probes)

    (after_stt,) = hooks.after_stt()
    # El punto de salida monta dos cosas: el eco del texto y la sonda que fecha
    # el audio saliente (la punta izquierda del camino de vuelta).
    before_output, output_probe = hooks.before_output()
    assert isinstance(output_probe, marks.AudioOutputProbe)

    transcript = TranscriptionFrame(text="Sí, soy Carlos.", user_id="u", timestamp="t")
    of_tts = TTSTextFrame(text="Gracias", aggregated_by="word")

    def _messages(processor: Any, frame: Any) -> list[Any]:
        return [
            f
            for f in _emitted(processor, frame)
            if isinstance(f, OutputTransportMessageUrgentFrame)
        ]

    # El de tras el STT reenvía la transcripción y nada más.
    assert len(_messages(after_stt, transcript)) == 1
    assert _messages(harness.build_carrier_echo(), of_tts) == []

    # El de antes de la salida reenvía el TTS y nada más. Si también reenviara
    # la transcripción, el texto del transportista llegaría dos veces.
    assert len(_messages(before_output, of_tts)) == 1
    assert _messages(harness.build_agent_echo(), transcript) == []


def test_the_zero_hook_delivers_a_probe_and_logs_it() -> None:
    """Quien pasó la lista se queda con la sonda: sin eso no hay ceros que leer."""
    probes: list[Any] = []
    hooks = harness.harness_hooks(probes)
    (probe,) = hooks.after_turn()
    assert probes == [probe]
    assert hasattr(probe, "zeros")


def test_the_carrier_echo_ignores_the_tts() -> None:
    """Si reenviara los dos, el texto del agente entraría dos veces."""
    from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    import pipecat.processors.frame_processor as fp

    echo = harness.build_carrier_echo()
    pushed: list[Any] = []

    async def _capture(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    echo.push_frame = _capture  # type: ignore[method-assign]

    async def _nada(_self: Any, frame: Any, direction: Any) -> None:
        return None

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _nada  # type: ignore[assignment]
    try:
        asyncio.run(
            echo.process_frame(
                TTSTextFrame(text="Gracias", aggregated_by="word"),
                FrameDirection.DOWNSTREAM,
            )
        )
    finally:
        fp.FrameProcessor.process_frame = original  # type: ignore[assignment]

    assert len(pushed) == 1
    assert not any(isinstance(f, OutputTransportMessageUrgentFrame) for f in pushed)


def test_the_agent_echo_sends_the_spaces_flag() -> None:
    """Sin ella el arnés no puede saber si reponer el separador."""
    from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    import pipecat.processors.frame_processor as fp

    echo = harness.build_agent_echo()
    pushed: list[Any] = []

    async def _capture(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    echo.push_frame = _capture  # type: ignore[method-assign]

    async def _nada(_self: Any, frame: Any, direction: Any) -> None:
        return None

    # No es parámetro del constructor: pipecat lo fija después de armar el frame.
    trama = TTSTextFrame(text="Gracias", aggregated_by="word")
    trama.includes_inter_frame_spaces = False

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _nada  # type: ignore[assignment]
    try:
        asyncio.run(echo.process_frame(trama, FrameDirection.DOWNSTREAM))
    finally:
        fp.FrameProcessor.process_frame = original  # type: ignore[assignment]

    messages = [
        f for f in pushed if isinstance(f, OutputTransportMessageUrgentFrame)
    ]
    assert messages[0].message["has_spaces"] is False


def test_the_flag_reaches_the_fragment() -> None:
    """El puente entre el mensaje y el reensamblado; si se corta, se pegan."""
    observed = harness.ObservedFromPoC1()
    harness._on_text_received(
        observed,
        {"de": harness.FROM_AGENT, "text": "Gracias,", "has_spaces": False},
    )
    harness._on_text_received(
        observed,
        {"de": harness.FROM_AGENT, "text": "Carlos.", "has_spaces": False},
    )
    assert (
        observed.assemble_agent(observed.agent_fragments())
        == "Gracias, Carlos."
    )


def test_the_zero_probe_goes_after_the_aggregator() -> None:
    """Antes del agregador no existen ni los frames del VAD ni el fin de turno."""
    from poc1_pipecat.app import telephony

    source = Path(telephony.__file__).read_text(encoding="utf-8")
    cuerpo = source[source.index("        [\n            transport.input()") :]
    aggregator_position = cuerpo.index("aggregator_group.user()")
    hook_position = cuerpo.index("_insert(hooks.after_turn)")
    llm_position = cuerpo.index("            llm,")
    assert aggregator_position < hook_position < llm_position


def test_the_echo_goes_before_the_transport_output() -> None:
    """Después de `transporte.output()` el mensaje ya no se serializa."""
    from poc1_pipecat.app import telephony

    source = Path(telephony.__file__).read_text(encoding="utf-8")
    cuerpo = source[source.index("        [\n            transport.input()") :]
    assert cuerpo.index("_insert(hooks.before_output)") < cuerpo.index(
        "transport.output()"
    )


# --------------------------------------------------------------------------
# Cuando el agente cuelga a media corrida
# --------------------------------------------------------------------------
#
# En el escenario 1 el agente cierra la negociación en T7 y cuelga, con T8
# todavía por actuar. Sin el aviso, el arnés sigue empujando contra un WebSocket
# muerto, el bucle de publicación revienta, y con él se van el reporte, la
# bitácora y el WAV de auditoría de una llamada ya pagada.


def _client(**extra: Any) -> Any:
    from poc1_pipecat.app.media_streams import MediaStreamsClient

    return MediaStreamsClient("ws://127.0.0.1:1/ws", label="prueba", **extra)


def test_publishing_after_the_hang_up_does_not_blow_up() -> None:
    """Reventar aquí se lleva la evidencia de una corrida que ya se pagó."""
    from poc1_pipecat.app.media_streams import BYTES_PER_FRAME

    client = _client()
    client._warn_that_closed("el servidor se fue")
    asyncio.run(client.send_frame(b"\xff" * BYTES_PER_FRAME))

    assert client.frames_without_destination == 1
    assert client.sent_frames == 0


def test_a_malformed_frame_is_still_an_error_after_the_hang_up() -> None:
    """El cuelgue no puede volverse una excusa para dejar pasar bugs del arnés."""
    from shared.carrier_simulator import HarnessFailure

    client = _client()
    client._warn_that_closed("el servidor se fue")
    with pytest.raises(HarnessFailure):
        asyncio.run(client.send_frame(b"\xff" * 3))


def test_the_close_from_the_other_side_warns_only_once() -> None:
    warning_list: list[str] = []
    client = _client(on_other_side_closed=warning_list.append)
    client._warn_that_closed("primero")
    client._warn_that_closed("second")
    assert warning_list == ["primero"]
    assert client.remote_closed is True


def test_a_warning_that_fails_does_not_take_down_the_client() -> None:
    def _raises(_motivo: str) -> None:
        raise RuntimeError("boom")

    client = _client(on_other_side_closed=_raises)
    client._warn_that_closed("el servidor se fue")
    assert client.remote_closed is True
    assert any("aviso de cierre" in i for i in client.incidents)


def test_before_the_hang_up_there_is_no_warning() -> None:
    """El testigo: si `remote_closed` arrancara en True, nada de arriba diría nada."""
    client = _client()
    assert client.remote_closed is False
    assert client.frames_without_destination == 0


def test_the_script_is_cut_when_the_room_is_already_closed() -> None:
    """`play_script` respeta `cerrada`: es lo que deja T8 sin actuar."""
    import numpy as np
    from shared.carrier_simulator import ScriptStep, play_script

    observed = harness.ObservedFromPoC1(silent=True)
    closed = asyncio.Event()
    closed.set()

    class _ExplodingPlayer:
        async def play_step(self, _step: Any) -> float:  # pragma: no cover
            raise AssertionError("no debe reproducir con la llamada cerrada")

    script = [
        ScriptStep(turn="T8", text="x", samples=np.zeros(160, dtype=np.int16))
    ]
    asyncio.run(
        play_script(
            script,
            player=_ExplodingPlayer(),
            observed=observed,
            wait_for_agent_s=1.0,
            closed=closed,
        )
    )


def test_the_hang_up_warning_marks_the_observed_and_the_signal() -> None:
    """Sin la señal el guion no se corta; sin la marca, el JSON no lo cuenta."""
    observed = harness.ObservedFromPoC1(silent=True)
    closed = asyncio.Event()
    warn = harness.annotate_hang_up(observed, closed)

    warn("el servidor cerró el Media Stream")
    assert closed.is_set()
    assert observed.was_closed_by_agent is True
    assert observed.close_reason == "el servidor cerró el Media Stream"


def test_the_second_hang_up_does_not_overwrite_the_reason_of_the_first() -> None:
    observed = harness.ObservedFromPoC1(silent=True)
    closed = asyncio.Event()
    warn = harness.annotate_hang_up(observed, closed)
    warn("el agente cerró")
    warn("el socket ya estaba muerto")
    assert observed.close_reason == "el agente cerró"


def test_final_dump_includes_frames_without_destination() -> None:
    """Es lo que distingue "actuó el guion" de "le habló a nadie"."""

    class _Pacing:
        reales = [0.0, 0.02]

        def summary(self) -> dict[str, Any]:
            return {"frames": 2}

    class _Player:
        pacing = _Pacing()

    class _Client:
        frames_without_destination = 37

    observed = harness.ObservedFromPoC1(silent=True)
    harness.close_observed(
        observed, player=_Player(), client=_Client()
    )
    assert observed.frames_without_destination == 37
    assert observed.publishing_pacing == {"frames": 2}
    assert observed.measured_sends == [0.0, 0.02]


def test_the_session_passes_the_closing_signal_to_the_script() -> None:
    """Sin `cerrada=`, el guion sigue actuando contra un pipeline que ya colgó.

    Se verifica sobre el árbol de sintaxis y no con un `in`: `run_session`
    levanta un subproceso con el pipeline entero, así que no hay forma barata de
    llamarla, y un `grep` daría verde con la palabra en un comentario.
    """
    import ast

    tree = ast.parse(RUTA_MODULO.read_text(encoding="utf-8"))
    session = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_session"
    )
    call = next(
        n
        for n in ast.walk(session)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "play_script"
    )
    assert "closed" in {k.arg for k in call.keywords}


def test_the_session_dumps_the_transport_contents_before_returning() -> None:
    """Si no se llama, el ritmo y las tramas sin destino salen vacíos del JSON."""
    import ast

    tree = ast.parse(RUTA_MODULO.read_text(encoding="utf-8"))
    session = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_session"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "close_observed"
        for n in ast.walk(session)
    )


def test_the_report_says_the_agent_hung_up() -> None:
    """Sin esto, una corrida cortada se lee igual que una completa."""
    observed = harness.ObservedFromPoC1(silent=True)
    observed.was_closed_by_agent = True
    observed.close_reason = "el servidor cerró el Media Stream"
    observed.frames_without_destination = 42
    observed.sent_marks = 0

    import io
    import contextlib as _ctx

    output = io.StringIO()
    with _ctx.redirect_stdout(output):
        harness.print_report(observed, "X")
    text = output.getvalue()
    assert "el agente colgó" in text
    assert "42 trama(s) sin destino" in text


# --------------------------------------------------------------------------
# El guion y el costo
# --------------------------------------------------------------------------


def _args(**extra: Any) -> argparse.Namespace:
    base = {
        "scenario": None,
        "clip": None,
        "clips": "shared/test_scenarios/clips",
        "repeat": 1,
        "turns": 0,
    }
    base.update(extra)
    return argparse.Namespace(**base)


def test_turns_trims_the_script(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [object()] * 8
    monkeypatch.setattr(harness, "load_scenario", lambda *a, **k: list(fake))
    assert len(harness.build_script(_args(scenario=1, turns=3))) == 3


def test_without_turns_the_script_goes_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [object()] * 8
    monkeypatch.setattr(harness, "load_scenario", lambda *a, **k: list(fake))
    assert len(harness.build_script(_args(scenario=1))) == 8


def test_turns_at_zero_does_not_leave_the_script_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [object()] * 8
    monkeypatch.setattr(harness, "load_scenario", lambda *a, **k: list(fake))
    assert len(harness.build_script(_args(scenario=1, turns=-2))) == 1


def test_the_cost_grows_with_the_duration() -> None:
    assert harness.estimate_cost(600.0) > harness.estimate_cost(60.0)


def test_the_cost_has_no_telephony_row() -> None:
    """Si apareciera uno, es que alguien conectó Twilio a este camino."""
    source = RUTA_MODULO.read_text(encoding="utf-8")
    assert "verify_for_telephony" not in source.replace(
        "# `verify_for_telephony` NO se llama", ""
    )


def test_without_accept_cost_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El guardarraíl de CLAUDE.md: nada se gasta sin la tabla aprobada."""
    ran = False

    def _should_not(*_a: Any, **_k: Any) -> Any:
        nonlocal ran
        ran = True
        raise AssertionError("se intentó correr la sesión sin aceptar el costo")

    monkeypatch.setattr(harness, "run_session", _should_not)
    monkeypatch.setattr(harness, "verify_for_voice", lambda _a: None)
    monkeypatch.setattr(harness, "load_settings", lambda: object())
    monkeypatch.setattr(harness, "build_script", lambda _a: [])
    monkeypatch.setattr(harness, "output_sample_rate", lambda *a: 8000)

    assert harness.main(["--scenario", "1"]) == 0
    assert not ran
    assert "--accept-cost" in capsys.readouterr().out


def test_must_give_scenario_or_clip(capsys: pytest.CaptureFixture[str]) -> None:
    assert harness.main([]) == 1
    assert "--scenario o --clip" in capsys.readouterr().out


def test_cannot_give_both(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.main(["--scenario", "1", "--clip", "x.ulaw"]) == 1
    assert "y solo uno" in capsys.readouterr().out


def test_serving_without_a_port_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert harness.main(["--run-server"]) == 1
    assert "--port" in capsys.readouterr().out


# --------------------------------------------------------------------------
# La bitácora
# --------------------------------------------------------------------------


def test_the_log_declares_that_poc1_has_no_queue() -> None:
    """Un `source_tail` vacío se leería como "no se midió"; es "no existe"."""
    source = RUTA_MODULO.read_text(encoding="utf-8")
    assert "PoC1 no tiene cola" in source
    assert "METRICS_CONTRACT.md" in source


def test_the_crossing_hangs_off_the_same_call_sid() -> None:
    """Las tres llaves tienen que ser la misma o el cruce es una heurística."""
    from poc1_pipecat.app.marks import zeros_path

    ruta = zeros_path("arnes-call-x", folder=Path("/tmp/x"))
    assert ruta.name == "zeros_arnes-call-x.jsonl"
