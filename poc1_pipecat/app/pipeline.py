"""Building the Pipecat pipeline: services, context and processors.

A single place defines how STT, LLM and TTS are built, so that the smoke test
and the real call cannot diverge. What changes between the two is the end of
the pipeline: in the real call it is the Twilio transport; in the smoke test
it is `OutputCapturer`, which keeps the audio and the text in memory.

No builder in this module opens connections: services connect when the
pipeline starts, not when they are instantiated. That makes it possible to
check that everything assembles without spending a single credit.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService

from poc1_pipecat.app.config import Settings, verify_for_voice
from poc1_pipecat.app.tools import (
    FUNCTION_NAMES,
    build_schemas,
    register_functions,
)
from shared import llm_transport
from shared.negotiation_engine import TripData, NegotiationSession
from shared.system_prompt import render_system_prompt

logger = logging.getLogger(__name__)

# Audio evidence folder. Already gitignored (`recordings/`, `*.wav`): it may
# contain the voice of a real carrier (LFPDPPP).
RECORDINGS_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "recordings"

# 16-bit PCM, which is what both the TTS and the Media Stream produce.
_BYTES_PER_SAMPLE: Final[int] = 2

# Frames that arrive by the dozen every turn and say nothing useful for
# diagnostics: audio chunk by chunk, text word by word, and metrics. They are
# still counted in `frame_count`; what is skipped is the line-by-line trace.
_NOISY_FRAMES: Final[frozenset[str]] = frozenset(
    {
        "TTSAudioRawFrame",
        "TTSTextFrame",
        "AggregatedTextProgressFrame",
        "OutputTransportMessageUrgentFrame",
        "MetricsFrame",
    }
)


# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------


def build_stt(settings: Settings) -> DeepgramSTTService:
    """Deepgram STT.

    `language` comes from `.env`: with `multi`, Nova-3 recognizes Spanish and
    English in the same session, which is what the prompt's rule about
    switching language if the carrier answers in English needs.

    `numerals` and `smart_format` stay at their default (off). Both change how
    amounts are written in the transcript, so they are bake-off knobs, not
    something to pin down here blindly.

    Pipecat logs that `multi` is not a language code it knows and that it
    passes it to the service as is: that is exactly what we want — `multi` is a
    Deepgram code, not one from Pipecat's `Language` enum.
    """
    return DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
        ),
    )


def build_llm(settings: Settings) -> OpenAILLMService:
    """LLM. The model is config, never a literal in the code.

    The timeouts come from `shared/llm_transport/`, which is the same policy
    PoC2 runs: if each PoC picks its own retries, the comparison measures who
    retries better and not the voice framework.

    **pipecat's defaults are dangerous here.** With `retry_on_timeout` off (its
    default) there is no retry **and no cap of its own**: the only bound is the
    OpenAI SDK's, `httpx.Timeout(timeout=600, connect=5)`. A stuck request can
    leave the call silent for up to ten minutes. With `retry_on_timeout=True`, pipecat
    caps the first attempt at `retry_timeout_secs` and retries once with no cap.
    """
    return OpenAILLMService(
        api_key=settings.openai_api_key,
        settings=OpenAILLMService.Settings(model=settings.openai_model),
        retry_on_timeout=True,
        retry_timeout_secs=llm_transport.FIRST_BYTE_TIMEOUT_S,
    )


def build_tts(settings: Settings) -> ElevenLabsTTSService:
    """ElevenLabs Flash TTS.

    The language is deliberately **not** set: Flash v2.5 detects it on its own,
    and pinning it to Spanish would prevent the switch to English the prompt
    asks for.
    """
    verify_for_voice(settings)
    return ElevenLabsTTSService(
        api_key=settings.eleven_api_key,
        settings=ElevenLabsTTSService.Settings(
            voice=settings.eleven_voice_id,
            model=settings.eleven_model,
        ),
    )


def build_vad() -> SileroVADAnalyzer:
    """Silero VAD. Detects that there is speech; it does not decide the end of turn."""
    return SileroVADAnalyzer(params=VADParams(stop_secs=0.2))


def build_turn_analyzer() -> LocalSmartTurnAnalyzerV3:
    """Semantic end of turn, on top of the VAD.

    Runs locally with the ONNX model bundled in pipecat: no network call and no
    per-turn cost.
    """
    return LocalSmartTurnAnalyzerV3()


# --------------------------------------------------------------------------
# LLM context
# --------------------------------------------------------------------------


def build_context(
    trip: TripData, *, compensations_path: Path | str | None = None
) -> LLMContext:
    """Initial context: interpolated v0 system prompt + the 4 schemas.

    The render checks on its own that the maximum price has not slipped in.
    """
    system = render_system_prompt(trip, compensations_path=compensations_path)
    return LLMContext(
        messages=[{"role": "system", "content": system}],
        tools=ToolsSchema(standard_tools=build_schemas(compensations_path)),
    )


# --------------------------------------------------------------------------
# Output processors
# --------------------------------------------------------------------------


def join_tts_fragments(fragments: Sequence[tuple[str, bool]]) -> str:
    """Reassemble a turn's `TTSTextFrame`s into readable text.

    Each fragment comes with its `includes_inter_frame_spaces` flag. Services
    with per-word timestamps —ElevenLabs among them— emit one frame per word
    **without** the spaces that separate them, so they have to be put back
    when concatenating. Joining them with `""` produces
    "Hola,¿habloconCarlos?", which is only a reassembly defect: the text that
    was sent to the TTS never went through here.
    """
    parts: list[str] = []
    for text, has_spaces in fragments:
        if not text:
            continue
        if (
            parts
            and not has_spaces
            and not parts[-1].endswith((" ", "\n"))
            and not text.startswith((" ", "\n"))
        ):
            parts.append(" ")
        parts.append(text)
    return "".join(parts).strip()


class TurnClock:
    """Clock shared by the probes, with zero at the start of each turn."""

    def __init__(self) -> None:
        self._t0: float | None = None

    def restart(self) -> None:
        self._t0 = asyncio.get_running_loop().time()

    def elapsed(self) -> float:
        if self._t0 is None:
            return 0.0
        return asyncio.get_running_loop().time() - self._t0


class TTSTextProbe(FrameProcessor):
    """Sits between the LLM and the TTS: measures when text leaves for ElevenLabs.

    Without this probe there is no way to separate "the LLM was slow" from "the
    TTS was slow": the capturer sits after the TTS and only sees what already
    came back.
    """

    def __init__(self, clock: TurnClock, *, diagnostics: bool = False) -> None:
        super().__init__()
        self._clock = clock
        self._diagnostics = diagnostics
        self.events: list[tuple[float, str, str]] = []
        self.t_first_text: float | None = None

    def restart_turn(self) -> None:
        self.events.clear()
        self.t_first_text = None

    def _annotate(self, tipo: str, detail: str = "") -> None:
        t = self._clock.elapsed()
        self.events.append((t, tipo, detail))
        if self._diagnostics:
            print(f"    [probe  t+{t:6.2f}s] {tipo} {detail}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._annotate("LLM starts responding")
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._annotate("LLM finished the response")
        elif isinstance(frame, AggregatedTextFrame):
            # The full sentence sent to ElevenLabs. It is the useful unit: the
            # individual `TextFrame`s are LLM tokens and arrive two or three
            # letters at a time, so printing them drowns the diagnostics.
            self._annotate("text -> TTS", frame.text.strip())
        elif isinstance(frame, TextFrame) and not isinstance(frame, TTSTextFrame):
            if self.t_first_text is None:
                self.t_first_text = self._clock.elapsed()

        await self.push_frame(frame, direction)


class OutputCapturer(FrameProcessor):
    """Takes the place of the output transport when there is no phone.

    Stores the synthesized audio, each turn's text and the TTFB metrics, and
    decides when the agent finished speaking so the carrier's next turn can be
    sent.

    **It also has to release the TTS pause**, and that is not optional. The
    ElevenLabs service is built with `pause_frame_processing=True` fixed (it
    cannot be turned off by parameter: it is hardcoded in its
    `super().__init__`). With that flag, the TTS **stops processing frames**
    as soon as a response with text ends, and only resumes when a
    `BotStoppedSpeakingFrame` arrives from downstream — which in a normal
    pipeline the output transport emits when it finishes playing.

    Without a transport, that frame never arrives and the TTS stays blocked
    forever: system frames get through, but data frames do not. In particular
    the `FunctionCallResultFrame` gets stuck, the function result never enters
    the context and the LLM never speaks again. Pipecat itself describes the
    scenario in `tts_service.py`: *"BotStopped
    SpeakingFrame never arrives (no audio played), so we must resume here to
    prevent a permanent deadlock"*.
    """

    def __init__(
        self, clock: TurnClock | None = None, *, diagnostics: bool = False
    ) -> None:
        super().__init__()
        self._clock = clock or TurnClock()
        self._diagnostics = diagnostics
        self._audio_chunks: list[bytes] = []
        self._text_fragments: list[tuple[str, bool]] = []
        self.sample_rate: int | None = None
        self.num_channels: int = 1
        self.ttfb: dict[str, float] = {}
        # How many frames of each class arrived. It is what lets us know,
        # without spending again, whether the audio never arrived or arrived
        # and got lost.
        self.frame_count: Counter[str] = Counter()

        # State of the current turn.
        self._t_primer_audio: float | None = None
        self._t_last_speech: float | None = None
        self._turn_bytes: int = 0
        self._tts_stopped: bool = False
        self._llm_response_alive: bool = False
        self._llm_response_ended: bool = False

    # -- What the agent said --

    @property
    def turn_text(self) -> str:
        return join_tts_fragments(self._text_fragments)

    @property
    def raw_audio(self) -> bytes:
        return b"".join(self._audio_chunks)

    @property
    def audio_seconds(self) -> float:
        if not self.sample_rate:
            return 0.0
        samples = len(self.raw_audio) / (_BYTES_PER_SAMPLE * self.num_channels)
        return samples / self.sample_rate

    @property
    def t_primer_audio(self) -> float | None:
        """Seconds from the start of the turn to the first audio chunk."""
        return self._t_primer_audio

    @property
    def turn_bytes(self) -> int:
        return self._turn_bytes

    # -- Turn synchronization --

    def restart_turn(self) -> None:
        self._text_fragments.clear()
        self._t_primer_audio = None
        self._t_last_speech = None
        self._turn_bytes = 0
        self._tts_stopped = False
        self._llm_response_alive = False
        self._llm_response_ended = False

    async def wait_for_end_of_turn(
        self, *, timeout: float = 60.0, silence: float = 2.0
    ) -> str:
        """Wait for the agent to stop speaking and not resume.

        Returns why the wait ended: `"tts_stopped"`, `"silence"` or
        `"no_speech"` (`timeout` ran out).

        **It does not rely only on `TTSStoppedFrame`.** ElevenLabs emits that
        frame when it receives `isFinal` from its WebSocket; in a normal run it
        arrives fine, but the path goes through draining the audio context,
        which can close earlier on its own internal timeout. As a fallback, a
        turn is also considered closed if there was speech, the LLM response
        ended and it has gone `silence` seconds without producing anything new.

        A turn split into two segments by a tool call is not cut short, because
        between segments an `LLMFullResponseStartFrame` arrives that reopens the
        response.
        """
        clock = asyncio.get_running_loop()
        limit = clock.time() + timeout

        while clock.time() < limit:
            await asyncio.sleep(0.1)

            if self._t_last_speech is None:
                continue  # the agent has not said anything yet

            still = clock.time() - self._t_last_speech
            if still < silence or self._llm_response_alive:
                continue
            if self._tts_stopped:
                return "tts_stopped"
            if self._llm_response_ended:
                # There was speech but the TTSStoppedFrame did not arrive: the
                # TTS pause is still on and must be released by hand or the
                # next turn gets stuck.
                await self.release_tts_pause()
                return "silence"

        return "no_speech"

    async def release_tts_pause(self) -> None:
        """Push upstream the `BotStoppedSpeakingFrame` the transport would emit.

        It is the only thing that unblocks the TTS after a response with text
        (see the class docstring). It is sent upstream only: downstream it is
        not needed, and the assistant aggregator already builds the context
        correctly without it.
        """
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        if self._diagnostics:
            print(
                f"    [pause  t+{self._clock.elapsed():6.2f}s] "
                "BotStoppedSpeakingFrame upstream (unblocks the TTS)"
            )

    # -- Evidence --

    def write_wav(self, ruta: Path) -> Path | None:
        """Dump the accumulated audio to a WAV. None if there was no audio."""
        audio = self.raw_audio
        if not audio or not self.sample_rate:
            return None
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(ruta), "wb") as wav:
            wav.setnchannels(self.num_channels)
            wav.setsampwidth(_BYTES_PER_SAMPLE)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio)
        return ruta

    def _record_metrics(self, frame: MetricsFrame) -> None:
        for data_point in frame.data or ():
            if type(data_point).__name__ != "TTFBMetricsData":
                continue
            processor = getattr(data_point, "processor", None)
            valor = getattr(data_point, "value", None)
            if processor and isinstance(valor, (int, float)):
                self.ttfb[processor] = float(valor)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        self.frame_count[type(frame).__name__] += 1
        now = asyncio.get_running_loop().time()

        if isinstance(frame, TTSAudioRawFrame):
            if self._t_primer_audio is None:
                self._t_primer_audio = self._clock.elapsed()
                if self._diagnostics:
                    print(
                        f"    [audio  t+{self._t_primer_audio:6.2f}s] first chunk "
                        f"from ElevenLabs ({len(frame.audio)} bytes, "
                        f"{frame.sample_rate} Hz)"
                    )
            self._audio_chunks.append(frame.audio)
            self._turn_bytes += len(frame.audio)
            self._t_last_speech = now
            self.sample_rate = frame.sample_rate
            self.num_channels = frame.num_channels
        elif isinstance(frame, TTSTextFrame):
            self._text_fragments.append(
                (frame.text, bool(getattr(frame, "includes_inter_frame_spaces", False)))
            )
            self._t_last_speech = now
        elif isinstance(frame, TTSStartedFrame):
            self._tts_stopped = False
            self._t_last_speech = now
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_stopped = True
            self._t_last_speech = now
            if self._diagnostics:
                print(
                    f"    [tts    t+{self._clock.elapsed():6.2f}s] TTSStoppedFrame"
                )
            await self.release_tts_pause()
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_alive = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_response_alive = False
            self._llm_response_ended = True
        elif isinstance(frame, MetricsFrame):
            self._record_metrics(frame)

        name = type(frame).__name__
        if self._diagnostics and name not in _NOISY_FRAMES:
            print(f"    [frame  t+{self._clock.elapsed():6.2f}s] {name}")

        await self.push_frame(frame, direction)


class TranscriptCollector(FrameProcessor):
    """Keeps what the STT transcribes. Only for the smoke test loopback."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self.texts.append(frame.text.strip())
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


@dataclass
class TextPipeline:
    """A pipeline without transport, to negotiate by text with synthesized voice."""

    task: PipelineTask
    capturer: OutputCapturer
    probe: TTSTextProbe
    clock: TurnClock
    context: LLMContext
    llm: OpenAILLMService = field(repr=False)

    def restart_turn(self) -> None:
        """Reset the clock and both probes to zero before a new turn."""
        self.clock.restart()
        self.probe.restart_turn()
        self.capturer.restart_turn()


def build_text_pipeline(
    settings: Settings,
    session: NegotiationSession,
    *,
    diagnostics: bool = False,
    extras_before_tts: Sequence[FrameProcessor] = (),
    extras_after_tts: Sequence[FrameProcessor] = (),
) -> TextPipeline:
    """Level 1 smoke test pipeline: LLM + tool calling + TTS, no input audio.

    It is the same LLM, the same context and the same handlers the real call
    will use; the only things missing are the STT and the transport. The
    carrier's turn comes in as text with `LLMMessagesAppendFrame`.

    With `diagnostics=True` both probes print every frame that crosses them,
    with the time relative to the start of the turn.

    `extras_before_tts` and `extras_after_tts` insert processors at the same
    two positions they occupy in the telephone pipeline. They are received
    already built, instead of being imported here, so that this module does
    not depend on `app/latency.py` — which does depend on this one. That is
    what lets the smoke test exercise the call's real instrumentation without
    duplicating it.
    """
    llm = build_llm(settings)
    register_functions(llm)

    context = build_context(
        session.trip, compensations_path=session.compensations_path
    )
    aggregator_group = LLMContextAggregatorPair(context)

    clock = TurnClock()
    probe = TTSTextProbe(clock, diagnostics=diagnostics)
    capturer = OutputCapturer(clock, diagnostics=diagnostics)

    pipeline = Pipeline(
        [
            aggregator_group.user(),
            llm,
            probe,
            *extras_before_tts,
            build_tts(settings),
            *extras_after_tts,
            capturer,
            aggregator_group.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        app_resources=session,
        # Without a transport there are no audio turns to track, and the script
        # decides when the run ends.
        enable_turn_tracking=False,
        idle_timeout_secs=None,
        # No RTVI client is listening: turning it off removes two
        # `OutputTransportMessageUrgentFrame`s for every pipeline frame, which
        # is half of the diagnostics noise.
        enable_rtvi=False,
    )
    return TextPipeline(
        task=task,
        capturer=capturer,
        probe=probe,
        clock=clock,
        context=context,
        llm=llm,
    )


def verify_registered_functions(llm: OpenAILLMService) -> list[str]:
    """Contract names the LLM does **not** have registered. Empty = all good."""
    return [name for name in FUNCTION_NAMES if not llm.has_function(name)]


async def transcribe_with_deepgram(
    settings: Settings,
    audio: bytes,
    sample_rate: int,
    *,
    max_seconds: float = 20.0,
) -> list[str]:
    """Feed already-synthesized audio to the STT and return what it transcribed.

    It is the smoke test loopback: it checks that Deepgram connects,
    authenticates and returns Spanish text, without needing a phone call. The
    audio is fed in real time because the STT is streaming, so the run takes
    as long as the audio lasts.
    """
    tope = int(max_seconds * sample_rate) * _BYTES_PER_SAMPLE
    audio = audio[:tope]

    collector = TranscriptCollector()
    task = PipelineTask(
        Pipeline([build_stt(settings), collector]),
        params=PipelineParams(audio_in_sample_rate=sample_rate),
        idle_timeout_secs=None,
    )
    runner = PipelineRunner(handle_sigint=False)

    async def feed() -> None:
        seconds_per_chunk = 0.02
        piece = int(sample_rate * seconds_per_chunk) * _BYTES_PER_SAMPLE
        for start in range(0, len(audio), piece):
            await task.queue_frame(
                InputAudioRawFrame(
                    audio=audio[start : start + piece],
                    sample_rate=sample_rate,
                    num_channels=1,
                )
            )
            await asyncio.sleep(seconds_per_chunk)
        # Deepgram closes the utterance a few hundred ms after the last audio.
        await asyncio.sleep(3.0)
        await task.stop_when_done()

    await asyncio.gather(runner.run(task), feed())
    return collector.texts
