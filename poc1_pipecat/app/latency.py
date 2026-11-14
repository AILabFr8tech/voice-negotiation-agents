"""Capture of latency and of the agent's actual speech. The missing layer.

`shared/metrics/` **computes**; this module **measures**. The split is
deliberate: the percentiles must be identical across the 3 PoCs, but capture
depends on the framework and is the only thing rewritten in PoC2 and PoC3.

## The timestamps of a turn

    carrier end of speech             <- zero of the turn
      -> final transcript             (STT)
      -> end of turn                  (the analyzer declares it finished)
      -> first LLM token
      -> first TTS audio byte         <- the sample is closed

`full_turn_ms` runs from zero to the first audio. It is the comparison's
metric; the stages explain where it comes from.

Watch the order: the end of turn arrives **after** the transcript, not before.
`TurnAnalyzerUserTurnStopStrategy` waits until it has text before declaring the
turn closed, so anchoring the STT on the end of turn always comes out negative.

## What counts as a turn

A turn is an exchange: the carrier finishes and the agent answers. The boundary
is the **end of turn**, not every VAD pause.

This matters because with `stop_secs=0.2` the VAD emits an end of speech every
time someone takes a breath mid-sentence: opening a turn for each one would
fill the turn count and the buckets with ghosts that have no measurement.

So ends of speech only update the zero of the turn being formed —the last one
wins, which is when the person really went quiet— and the turn is born when the
analyzer closes it.

### The honest zero: why `stop_secs` is subtracted

The VAD does not signal when you stop talking, it signals once it has seen
`stop_secs` of confirmed silence (0.2 s in PoC1's config). If the frame's
instant were taken as the end of speech, **every** latency would come out
200 ms shorter than what the person perceives, systematically. So the zero is
the frame's instant minus its `stop_secs`, which is when speech really stopped.

The remaining error is the queuing between the VAD deciding and the probe
receiving the frame: same event loop, on the order of microseconds. That is
preferred over mixing `time.time()` (which is what the frame carries) with
`time.monotonic()`: a clock adjustment mid-call would corrupt the subtractions,
and a monotonic clock would not.

## The opening turn

In an outbound call the agent speaks first, so its greeting has no preceding
"end of speech". That turn is measured from when the Media Stream connects
until the first audio: it is the cold start (opening the TTS WebSocket and the
first LLM request with the full prompt). It goes to index 0, and
`shared/metrics/` sends it only to the `apertura` (opening) bucket.

## Why the agent annotator goes AFTER the TTS

`AgentSpeechAnnotator` reconstructs what **was spoken**, not what the LLM meant
to say. If the LLM generates an amount and is interrupted before speaking it,
recording it as said would produce a false hard failure in the financial-limit
audit.

There is a type trap: `TTSTextFrame` inherits from `AggregatedTextFrame`, so
moving the old annotator as is behind the TTS would make it record **one row
per word** instead of one per sentence. That is why `TTSTextFrame`s are
accumulated here and reassembled with `join_tts_fragments`, which already
knows how to put back the spaces ElevenLabs does not send.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Final

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from poc1_pipecat.app.pipeline import join_tts_fragments
from shared.metrics.latency import TurnSample

logger = logging.getLogger(__name__)

# Mark names. They match the parameters of `TurnSample.from_events`, which is
# what does the subtractions.
END_OF_SPEECH_MARK: Final[str] = "t_speech_end"
END_OF_TURN_MARK: Final[str] = "t_end_of_turn"
TRANSCRIPT_MARK: Final[str] = "t_transcript"
FIRST_TOKEN_MARK: Final[str] = "t_primer_token"
FIRST_AUDIO_MARK: Final[str] = "t_primer_audio"


def frame_key(frame: Frame) -> int | None:
    """Key that identifies a frame, the same for both copies of a broadcast.

    Pipecat emits several system frames "as a broadcast": it creates **two**
    instances, one upstream and one downstream, and cross-links them with
    `broadcast_sibling_id` (`downstream.broadcast_sibling_id = upstream.id` and
    vice versa). Each copy has its own `id`, so deduplicating by `id` would not
    be enough: the same end of speech would arrive twice with different
    identities.

    `min(id, sibling)` gives the same number for both copies, which is what is
    needed to count the event only once.
    """
    own = getattr(frame, "id", None)
    if not isinstance(own, int):
        return None
    sibling = getattr(frame, "broadcast_sibling_id", None)
    if isinstance(sibling, int):
        return min(own, sibling)
    return own


class _TurnInProgress:
    """A turn's marks while it fills up. No logic: it is a named bag."""

    __slots__ = ("marks", "verdicts", "interrupted", "is_opening", "sealed")

    def __init__(self, *, is_opening: bool = False) -> None:
        self.marks: dict[str, float] = {}
        self.verdicts: list[float] = []
        self.interrupted: bool = False
        self.is_opening: bool = is_opening
        # A sealed turn already has its end of turn: it stops accepting ends of
        # speech and transcripts, which from then on belong to the next turn.
        self.sealed: bool = False


class LatencyCollector:
    """Accumulates each turn's marks and produces the `TurnSample`s.

    It is not a `FrameProcessor`: it is shared state. The probes feed it from
    different points of the pipeline, because no single processor sees both the
    carrier's end of speech (upstream of the STT) and the TTS's first audio byte
    (downstream of the TTS).

    ## Two turns in flight at once

    There is a turn **being measured** (already closed by the analyzer, waiting
    for the agent's audio) and one **being formed** (accumulating ends of speech
    and transcripts). Both are needed because the carrier can start talking
    before the agent answers — in fact it is the norm during the greeting:
    people answer "¿bueno?" while the agent is still synthesizing.

    With a single turn in flight, that "¿bueno?" would close the opening
    halfway and shift every measurement by one position.

    Marks within a turn are idempotent: the first one wins. That allows putting
    the same probe at several points without keeping track of which frame goes
    through where. The two exceptions take `clave` (key) to deduplicate the same
    frame seen by several probes: the end of speech (the **last** one wins,
    which is when the person really went quiet) and the analyzer's verdicts
    (they accumulate).
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._t_start: float | None = None
        self._samples: list[TurnSample] = []
        self._indice: int = 0
        self._bot_speaking: bool = False
        # The turn waiting for the agent's audio, and the one being formed.
        self._in_measurement: _TurnInProgress | None = None
        self._forming: _TurnInProgress = _TurnInProgress()
        # Frames already counted, so the same event seen by several probes is not
        # counted twice. A few dozen per call: it fits entirely.
        self._seen: set[int] = set()

    # -- Lifecycle --

    def begin(self) -> None:
        """Zero of the call. Called when the Media Stream connects.

        Opens the opening turn right away: the agent's greeting has no preceding
        end of speech, and opening it here also lets us measure the LLM's TTFT
        and the TTS's TTFB for it, which are exactly what make up the cold start.
        """
        self._t_start = self._clock()
        opening = _TurnInProgress(is_opening=True)
        opening.marks[END_OF_SPEECH_MARK] = self._t_start
        self._in_measurement = opening

    @property
    def samples(self) -> list[TurnSample]:
        return list(self._samples)

    def to_json(self) -> list[dict[str, object]]:
        """The samples in the shape `shared/metrics/` expects."""
        return [m.to_dict() for m in self._samples]

    # -- Carrier events --

    def speech_end(self, *, ago: float = 0.0, clave: int | None = None) -> None:
        """The carrier stopped speaking `ago` seconds ago.

        **It does not open a turn**: it moves the zero of the turn being formed.
        The last one wins, because a mid-sentence pause produces an end of speech
        that is not the end of the line.
        """
        if not self._tally(clave):
            return
        if self._forming.sealed:
            return
        self._forming.marks[END_OF_SPEECH_MARK] = self._clock() - max(ago, 0.0)

    def transcript(self) -> None:
        """A final transcript arrived. The last one before the end of turn wins.

        Deepgram emits a final for every phrase its own endpointer closes, not
        one per conversational turn. The one that matters is the last: it is the
        text with which the analyzer decided to close.
        """
        if self._forming.sealed:
            return
        self._forming.marks[TRANSCRIPT_MARK] = self._clock()

    def end_of_turn_verdict(self, probability: float, *, clave: int | None = None) -> None:
        """Probability the end-of-turn analyzer just emitted.

        It is stored as is, without interpreting it. It is the evidence that
        distinguishes "the model said COMPLETE with confidence" from "it closed
        on timeout", which lead to opposite fixes when the agent cuts in
        mid-sentence.
        """
        if not self._tally(clave):
            return
        destination = self._forming if not self._forming.sealed else self._in_measurement
        if destination is not None:
            destination.verdicts.append(float(probability))

    def end_of_turn(self, *, clave: int | None = None) -> None:
        """The analyzer declared the carrier's turn closed.

        It is the turn boundary. It seals the turn being formed and promotes it.

        If the previous turn is still waiting for audio, the agent did not get
        to answer it: the LLM that was running is superseded by the one this end
        of turn triggers, so the audio that comes next answers **this** turn,
        not the previous one.
        """
        if not self._tally(clave):
            return
        self._forming.marks[END_OF_TURN_MARK] = self._clock()
        self._forming.sealed = True

        anterior = self._in_measurement
        if anterior is not None and FIRST_AUDIO_MARK not in anterior.marks:
            if anterior.is_opening:
                # The opening never got to play: the carrier answered "¿bueno?"
                # while the agent was still synthesizing the greeting. No empty
                # sample is emitted —it would be a ghost turn— but the turn that
                # answers does pay the cold start, so it inherits the opening
                # flag so it does not pollute the steady state.
                self._forming.is_opening = True
                self._in_measurement = None
            else:
                # It stays unmeasured, which is the truth: the agent did not answer.
                self._close()

        self._promover()

    def user_started_speaking(self) -> None:
        """If the agent was speaking, the carrier talked over it.

        It is the only real barge-in that can be detected from the frames.
        `InterruptionFrame` is **not** used for this: pipecat emits it at every
        user turn start, whether or not the agent is speaking, so it marked every
        turn as interrupted.
        """
        if self._bot_speaking:
            self._forming.interrupted = True

    # -- Agent events --

    def bot_started_speaking(self) -> None:
        self._bot_speaking = True

    def bot_stopped_speaking(self) -> None:
        self._bot_speaking = False

    def set_mark(self, mark: str) -> None:
        """Record a mark for the turn being measured. The first one wins."""
        if self._in_measurement is None:
            return
        self._in_measurement.marks.setdefault(mark, self._clock())

    def primer_audio(self) -> None:
        """Agent audio arrived. Closes the sample of the turn being measured.

        With no turn being measured, this is the continuation of the same agent
        turn —a second speech segment after a tool call— and it is ignored,
        because `full_turn` runs to the **first** audio.
        """
        if self._in_measurement is None:
            return
        self._in_measurement.marks.setdefault(FIRST_AUDIO_MARK, self._clock())
        self._close()

    def close_pending(self) -> None:
        """On hang-up: closes whatever was left in flight.

        A turn without agent audio comes out with `full_turn_ms` as None, which
        is correct — the agent did not answer. What is **not** closed is a turn
        the analyzer never declared: pipecat did not consider it a turn, and
        inventing it here would inflate the count again.

        `_close` promotes when it finishes, so the loop drains the queue: the
        turn being measured and, if the carrier managed to close another one,
        that one too.
        """
        while self._in_measurement is not None:
            self._close()

    def tally_frame(self, clave: int | None) -> bool:
        """False if this frame was already counted by another probe. Consumes the key.

        Public because there are events —the analyzer's verdicts— where the
        probe needs to deduplicate the frame only once even though it carries
        several data points inside.
        """
        return self._tally(clave)

    # -- Internal --

    def _tally(self, clave: int | None) -> bool:
        """False if this frame was already counted by another probe."""
        if clave is None:
            return True
        if clave in self._seen:
            return False
        self._seen.add(clave)
        return True

    def _promover(self) -> None:
        """Move the sealed turn to measurement, if no other turn holds that slot."""
        if self._in_measurement is not None or not self._forming.sealed:
            return
        self._in_measurement = self._forming
        self._forming = _TurnInProgress()

    def _close(self) -> None:
        turn = self._in_measurement
        if turn is None:
            return
        self._samples.append(
            TurnSample.from_events(
                self._indice,
                t_speech_end=turn.marks.get(END_OF_SPEECH_MARK),
                t_end_of_turn=turn.marks.get(END_OF_TURN_MARK),
                t_transcript=turn.marks.get(TRANSCRIPT_MARK),
                t_primer_token=turn.marks.get(FIRST_TOKEN_MARK),
                t_primer_audio=turn.marks.get(FIRST_AUDIO_MARK),
                interrupted=turn.interrupted,
                is_opening=turn.is_opening,
                end_of_turn_verdicts=turn.verdicts,
            )
        )
        self._indice += 1
        self._in_measurement = None
        # The next turn may have been forming for a while (the carrier spoke
        # while the agent was answering): if it is already sealed, it goes in now.
        self._promover()


class LatencyProbe(FrameProcessor):
    """Reports to the collector the frames that cross this point of the pipeline.

    It lets everything through untouched. It is instantiated several times, at
    the points where events originate: upstream of the STT the TTS audio does
    not exist, and downstream of the TTS the `LLMTextFrame` no longer exists
    (the TTS turns it into `TTSTextFrame`).

    A single class for every point because marks are idempotent: if a frame
    crosses two probes, the second one changes nothing.
    """

    def __init__(self, collector: LatencyCollector, *, label: str = "") -> None:
        super().__init__()
        self._collector = collector
        self._label = label

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        collector = self._collector

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # The honest zero of the turn: when speech really stopped, not when
            # the VAD confirmed it. The key keeps the three probes from counting
            # the same event three times.
            collector.speech_end(
                ago=float(getattr(frame, "stop_secs", 0.0) or 0.0),
                clave=frame_key(frame),
            )
        elif isinstance(frame, UserStoppedSpeakingFrame):
            collector.end_of_turn(clave=frame_key(frame))
        elif isinstance(frame, UserStartedSpeakingFrame):
            collector.user_started_speaking()
        elif isinstance(frame, TranscriptionFrame):
            collector.transcript()
        elif isinstance(frame, MetricsFrame):
            # The end-of-turn analyzer publishes its probability here. It is the
            # only thing that distinguishes a confident COMPLETE from one on
            # timeout. The key is consumed once per frame, not per data point:
            # otherwise a frame with two verdicts would lose the second one.
            verdicts = [d for d in frame.data or () if isinstance(d, TurnMetricsData)]
            if verdicts and collector.tally_frame(frame_key(frame)):
                for data_point in verdicts:
                    collector.end_of_turn_verdict(
                        float(getattr(data_point, "probability", 0.0) or 0.0)
                    )
        elif isinstance(frame, LLMTextFrame):
            collector.set_mark(FIRST_TOKEN_MARK)
        elif isinstance(frame, TTSAudioRawFrame):
            # First byte the TTS produces. Preferred over
            # `BotStartedSpeakingFrame` because it does not depend on the
            # transport's buffer, and the equivalent exists in all 3
            # architectures.
            collector.primer_audio()
        elif isinstance(frame, BotStartedSpeakingFrame):
            collector.bot_started_speaking()
            collector.primer_audio()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            collector.bot_stopped_speaking()

        await self.push_frame(frame, direction)


class AgentSpeechAnnotator(FrameProcessor):
    """Records what the agent **spoke**, reassembled per sentence.

    Sits between the TTS and the transport output. Accumulates the
    `TTSTextFrame`s —which ElevenLabs sends word by word, without the spaces—
    and dumps them as a single line when the speech turn ends.

    The dump is triggered by `TTSStoppedFrame` and, in case that frame does not
    arrive, by `BotStoppedSpeakingFrame`, by an interruption and by the
    `EndFrame`. Dumping twice duplicates nothing: the buffer is emptied on dump.
    """

    def __init__(self, annotate: Callable[[str], None]) -> None:
        super().__init__()
        self._annotate = annotate
        self._fragments: list[tuple[str, bool]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame):
            self._fragments.append(
                (
                    frame.text,
                    bool(getattr(frame, "includes_inter_frame_spaces", False)),
                )
            )
        elif isinstance(
            frame,
            (TTSStoppedFrame, BotStoppedSpeakingFrame, InterruptionFrame, EndFrame),
        ):
            self.dump()

        await self.push_frame(frame, direction)

    def dump(self) -> None:
        """Dump what was accumulated as one line. Empty = does nothing."""
        if not self._fragments:
            return
        text = join_tts_fragments(self._fragments)
        self._fragments.clear()
        if text:
            self._annotate(text)
