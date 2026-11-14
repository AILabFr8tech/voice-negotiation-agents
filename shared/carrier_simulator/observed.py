"""What the harness saw, without knowing through what it saw it.

This is the half of the harness that does not depend on the transport: energy of the
agent's speech, text fragments, turns and script coverage. Each PoC inherits from
`Observed` and adds its own: in PoC2, the room's participants and the
zeros published over the data channel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

from shared.carrier_simulator.audio import rms_pcm16
from shared.carrier_simulator.run import MIN_VALID_TURNS
from shared.carrier_simulator.script import ScriptStep

logger = logging.getLogger(__name__)

# Energy threshold with which the harness decides "the agent is speaking", in
# int16 RMS counts.
#
# **It is not a comparison constant** and it is not compared with PoC3's
# `UMBRAL_DE_VOZ_RMS`: that one goes into the computation of a reported
# latency, this one only decides when it is the harness's turn to speak. The margin is
# generous: the agent's speech is in the thousands (measurement in the appendix of
# `docs/comparativo_resultados.md`). Each PoC can raise or lower it via
# `Observed.speech_threshold_rms` if its track arrives at another level.
RMS_WITH_SPEECH: Final[float] = 200.0

# Identity the harness presents itself with. It serves to separate, within the
# same text stream, the STT of what we publish from the text the
# agent writes about itself. Each PoC overrides it with the one it uses.
DEFAULT_IDENTITY: Final[str] = "carrier"

# Silence that separates two speech streaks of the agent. It is not a measurement
# knob: it only groups contiguous frames so one can ask "how long did this thing
# that just sounded last?". It is short on purpose: over-splitting a long answer
# does no harm (any of its pieces reaches the minimum), whereas
# gluing a stray fragment onto the answer that comes seven seconds later
# would.
STREAK_HANGOVER_S: Final[float] = 0.3

# Segment in which the agent's speech is listened to when it arrives in longer blocks.
# It is the segment size in which it arrives in PoC1 and PoC2; see `Observed.annotate_audio_in_frames`.
LISTENING_SEGMENT_S: Final[float] = 0.020


@dataclass
class SpeechStreak:
    """A contiguous stretch of the agent's speech, with its start and its end.

    It exists because the harness needs to ask the agent's speech something that a
    loose timestamp cannot answer: *how long it lasted*. `t_agent_first_speech` and
    `t_agent_last_speech` say when something was heard, not whether that something was an answer or
    a 160 ms click.
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class TextFragment:
    """A piece of transcript with its local arrival mark.

    `closure` distinguishes the piece of text from the **segment closure**: there are stacks
    where the "this is now final" flag travels in the stream's trailer and can only
    be read after the iteration has ended.
    """

    t: float
    sender: str
    text: str
    segment: str
    final: bool
    closure: bool = False


def text_from_deltas(fragments: list[TextFragment]) -> str:
    """Reassembles a **delta** stream: each piece is appended to the previous one.

    `"".join` and never `strip()` per fragment: the providers send the separating
    space **at the beginning** of the piece (`'Hola,'`, `' llamo'`). Trimming it
    glues the words together, and that is exactly what gave PoC3 a false WER of
    84.5% on a transcript that actually measured 0.0%.
    """
    return "".join(f.text for f in fragments if not f.closure)


@dataclass
class ObservedTurn:
    """What happened around a script step."""

    step: ScriptStep
    t_start: float | None = None
    t_clip_end: float | None = None
    t_agent_first_speech: float | None = None
    advance_reason: str = ""
    stt: str = ""
    response: str = ""
    # Agent speech streaks that sounded in this turn and were **not** taken
    # as its answer. Each one is `{"inicio", "duracion_ms", "motivo",
    # "texto"}`. It goes into the harness JSON: without this, an out-of-sync turn looks
    # the same as a healthy one (its acknowledgement falls in the same range) and is only detected by
    # ear. See `wait_for_agent_turn`.
    discarded_fragments: list[dict[str, Any]] = field(default_factory=list)
    # Whether this turn's real zero was published to the worker.
    # Silence turns have no speech to timestamp and never carry it.
    sent_mark: bool = False
    # Only in the interruption scenario: what happened with this
    # turn's event and its metrics (`interruptions.act_out_event`). `None` if there was none.
    event: dict[str, Any] | None = None

    @property
    def responded(self) -> bool:
        return bool(self.response) or self.t_agent_first_speech is not None

    @property
    def had_agent_audio(self) -> bool:
        """Whether the agent was heard in this turn, not whether the worker says it spoke."""
        return self.t_agent_first_speech is not None

    @property
    def text_without_audio(self) -> bool:
        """The agent wrote and **did not sound**. A technical failure, not a fast turn.

        The run-level `agente_sin_audio` is binary, and that is why it misses
        exactly the most misleading failure: with audio on 6 of 8 turns the run
        passes the whole rule, and those 2 turns go into the percentile as if the
        agent had answered, with the latency the worker computed over
        a TTS nobody heard. Flagged per turn, they are excluded from the percentile and
        counted under reliability, which is where they belong.

        Turns with no answer of any kind do **not** fall here: that is
        "did not answer", which `responded` already counts, and it is something else.
        """
        return bool(self.response.strip()) and self.t_agent_first_speech is None

    @property
    def acknowledgement_ms(self) -> float | None:
        """From the end of the clip to the agent's first audio.

        It is an **externally observed** measure, so it includes the audio's
        travel in both directions. It is not compared head-on against `latencias`.
        """
        if self.t_clip_end is None or self.t_agent_first_speech is None:
            return None
        return round((self.t_agent_first_speech - self.t_clip_end) * 1000.0, 1)

    @property
    def out_of_sync(self) -> bool:
        """Whether something sounded before the answer and was not the answer.

        It does not say the turn is wrong (the discard is precisely what keeps
        it from being wrong). It says it needs looking at: the agent emitted audio that was not
        its answer, and in an earlier run that was enough to
        shift the script by a whole turn.
        """
        return bool(self.discarded_fragments)


@dataclass
class Observed:
    """Everything the harness saw, with the zero at the instant of connection."""

    start: float = field(default_factory=time.time)

    connected: bool = False
    published_frames: int = 0
    clip_frames: int = 0

    # Agent audio
    audio_frames: int = 0
    rms: list[float] = field(default_factory=list)
    t_agent_last_speech: float | None = None
    t_agent_first_speech: float | None = None
    # The agent's speech grouped into contiguous stretches. See `SpeechStreak`.
    speech_streaks: list[SpeechStreak] = field(default_factory=list)

    # The script
    turns: list[ObservedTurn] = field(default_factory=list)
    t_clip_start: float | None = None
    t_clip_end: float | None = None

    # Text
    fragments: list[TextFragment] = field(default_factory=list)

    # The call closed underneath us. It happens when the agent closes the
    # negotiation and hangs up: it is not an error, it is one of the two end
    # conditions. What would be an error is not noticing: on September 15,
    # 2026 the PoC2 harness kept playing for 27 s against a deleted room.
    was_closed_by_agent: bool = False
    close_reason: str = ""

    # The branches taken in the scenarios that are not linear.
    # Empty in scenarios 1, 3 and 5, which are.
    decisions: list[Any] = field(default_factory=list)

    incidents: list[str] = field(default_factory=list)

    # With 20 calls in a row, printing every fragment buries the report.
    silent: bool = False

    # See `RMS_WITH_SPEECH` and `DEFAULT_IDENTITY`.
    speech_threshold_rms: float = RMS_WITH_SPEECH
    own_identity: str = DEFAULT_IDENTITY

    # Audit recording (`--record-audit`). `None` on the normal path:
    # it is diagnostics for listening to the run, it does not go into any metric. See
    # `shared/carrier_simulator/auditoria.py`.
    recorder: Any = None

    # How far the agent's speech has already "sounded", to timestamp the segments of a
    # long block as the phone would play them. Only
    # `annotate_audio_in_frames` uses it; it publishes the most it had to shift.
    listening_playhead: float = field(default=0.0, repr=False)
    max_listening_offset_s: float = 0.0

    def now(self) -> float:
        return round(time.time() - self.start, 3)

    def tally(self, message: str) -> None:
        """Run trace. Silenced in batches of many calls."""
        if not self.silent:
            print(f"  [{self.now():>7.3f}s] {message}")

    def annotate_incident(self, detail: str) -> None:
        self.incidents.append(f"[{self.now():>7.3f}s] {detail}")
        logger.warning("%s", detail)

    def annotate_audio(self, raw_value: bytes, *, moment: float | None = None) -> None:
        """An agent frame, timestamped on arrival or at `moment` (s since `start`)."""
        if self.recorder is not None:
            # Unrounded and before anything else: this timestamps the frame,
            # and `now()` truncates to milliseconds. It is an `append` and nothing more; the
            # file is built on hang-up, not here.
            self.recorder.annotate_agent(
                time.time() - self.start if moment is None else moment, raw_value
            )
        self.audio_frames += 1
        energy = rms_pcm16(raw_value)
        self.rms.append(energy)
        if energy >= self.speech_threshold_rms:
            now = self.now() if moment is None else round(moment, 3)
            if self.t_agent_first_speech is None:
                self.t_agent_first_speech = now
            # The open streak extends; if the last sound is further back
            # than the hangover, this is something else and a new streak starts.
            opened = self.speech_streaks[-1] if self.speech_streaks else None
            if opened is None or now - opened.end > STREAK_HANGOVER_S:
                self.speech_streaks.append(SpeechStreak(start=now, end=now))
            else:
                opened.end = now
            self.t_agent_last_speech = now

    def annotate_audio_in_frames(self, pcm: bytes, *, sample_rate: int) -> None:
        """A long block of the agent's speech, listened to in 20 ms segments.

        PoC1 and PoC2 deliver speech in 20 ms segments; PoC3's bridge, in
        blocks of ~100 ms. Measured block by block, each block is a single
        frame timestamped on arrival: the last speech ends up as much as one block before
        it finishes sounding and up to one block is added to each pause, which
        is enough for a natural pause to count as a stop.

        Here the block is split into segments of `LISTENING_SEGMENT_S` and each segment
        is timestamped where it would sound on a phone: a playhead that advances in real
        time and never lags behind the arrival (if the agent sends faster than
        real time, the phone queues it and hears it later). The most it was shifted
        is kept in `max_listening_offset_s`.
        """
        arrival = time.time() - self.start
        start = max(arrival, self.listening_playhead)
        self.max_listening_offset_s = max(self.max_listening_offset_s, start - arrival)
        step = max(2, int(round(sample_rate * LISTENING_SEGMENT_S)) * 2)
        for k, since in enumerate(range(0, len(pcm), step)):
            self.annotate_audio(pcm[since:since + step], moment=start + k * LISTENING_SEGMENT_S)
        self.listening_playhead = start + (len(pcm) // 2) / sample_rate

    def annotate_text(
        self,
        sender: str,
        text: str,
        *,
        segment: str = "",
        final: bool = False,
        closure: bool = False,
    ) -> TextFragment:
        """Stores a piece of transcript and returns it.

        Each PoC wraps it with whatever it can read from its own protocol: in
        PoC2, the stream's `lk.*` attributes.
        """
        fragment = TextFragment(
            t=self.now(),
            sender=sender,
            text=text,
            segment=segment,
            final=final,
            closure=closure,
        )
        self.fragments.append(fragment)
        if not closure and not self.silent:
            who = "nosotros" if sender == self.own_identity else "agent"
            print(f"  [{fragment.t:>7.3f}s] {who:<9} {text!r}")
        return fragment

    # -- How each stream is reassembled --
    #
    # They are kept as methods and not as standalone functions because **they are not the same
    # in the 3 PoCs**: LiveKit sends the user's STT as replacements (the
    # last fragment contains everything) and the agent's text as deltas.
    # Confusing them breaks WER in both possible directions. The default is
    # delta, which is what most do; whoever has something else overrides it.

    def assemble_stt(self, fragments: list[TextFragment]) -> str:
        """The STT of what we publish."""
        return text_from_deltas(fragments)

    def assemble_agent(self, fragments: list[TextFragment]) -> str:
        """What the agent says about itself."""
        return text_from_deltas(fragments)

    # -- Derived readings --

    def agent_silent_since(self) -> float | None:
        """Seconds the agent has been silent, or `None` if it never spoke."""
        if self.t_agent_last_speech is None:
            return None
        return self.now() - self.t_agent_last_speech

    def agent_fragments(
        self, *, since: float | None = None
    ) -> list[TextFragment]:
        """What the agent says about itself, not the STT of what we send it."""
        return [
            f
            for f in self.fragments
            if f.sender != self.own_identity
            and not f.closure
            and (since is None or f.t >= since)
        ]

    def our_stt_fragments(self) -> list[TextFragment]:
        return [
            f
            for f in self.fragments
            if f.sender == self.own_identity and not f.closure
        ]

    def turns_without_audio(self) -> list[ObservedTurn]:
        """Turns where the agent wrote and did not sound. See `text_without_audio`."""
        return [t for t in self.turns if t.text_without_audio]

    def streaks_after(self, t: float) -> list[SpeechStreak]:
        """The agent's speech streaks that started after `t`.

        `t` is the turn's zero: what the agent had been saying from before is not
        its answer to a clip that had not yet finished sounding.
        """
        return [r for r in self.speech_streaks if r.start > t]

    def agent_text_during(
        self, streak: SpeechStreak, *, cola_seg: float = 3.0
    ) -> str:
        """What the agent said while `streak` was sounding, if the text has arrived yet.

        The window is extended by `cola_seg` at the end because the transcript
        arrives **after** the audio it describes: the fragment is timestamped when
        the provider delivers it, not when it was spoken. It can return an empty
        string and that does not mean the agent was silent: it means the text
        is not there yet, or never will be.
        """
        return self.assemble_agent(
            [
                f
                for f in self.agent_fragments()
                if streak.start <= f.t <= streak.end + cola_seg
            ]
        )

    def had_agent_speech_after(self, t: float) -> bool:
        return (
            self.t_agent_last_speech is not None
            and self.t_agent_last_speech > t
        )

    def turns_reached(self) -> dict[str, Any]:
        """Which script turn the conversation really got to.

        The agent **rushes to close**, and not always on the same script
        turn. A p50 that averages 8-turn calls with 5-turn calls does not
        describe the same thing, and
        the number is lost if it is not counted here: the worker's JSON only sees the
        turns it managed to measure, not the ones the script asked for.

        `reached` counts the turns the agent **answered**. The first one
        it does not answer cuts the count: whatever comes after was played
        against nobody.
        """
        of_script = len(self.turns)
        reached = 0
        for turn in self.turns:
            if not turn.responded:
                break
            reached += 1

        cut_at = ""
        motivo = ""
        if reached < of_script:
            failed = self.turns[reached]
            cut_at = failed.step.turn
            motivo = failed.advance_reason or "sin respuesta del agente"

        if reached == of_script:
            verdict = "el guion se actuó completo"
        elif reached >= MIN_VALID_TURNS:
            verdict = f"incompleto pero utilizable (>= {MIN_VALID_TURNS})"
        else:
            verdict = f"short: menos de {MIN_VALID_TURNS} turnos"

        return {
            "of_script": of_script,
            "reached": reached,
            "full": reached == of_script,
            "cut_at": cut_at,
            "motivo": motivo,
            "verdict": verdict,
        }
