"""The per-turn player: continuous comfort noise and clips paced by the clock.

`Player` does not know where the audio goes out. Each PoC inherits and writes
`publish`: in PoC2 it is `AudioSource.capture_frame` with an `rtc.AudioFrame`; in
PoC1 and PoC3 it is the Media Streams WebSocket. The pacing, the padding between clips and
the instant at which the speech finishes sounding are the same in all three, and that is why
they live here: if each harness timestamped the clip end its own way, the turn's zero
(that is, the origin of all the reported latency) would be different in each PoC.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from shared.carrier_simulator.verbal_backchannel import is_backchannel
from shared.carrier_simulator.audio import (
    MS_PER_FRAME,
    split_into_frames,
    comfort_noise_frame,
)
from shared.carrier_simulator.script import HarnessFailure, ScriptStep
from shared.carrier_simulator.observed import (
    STREAK_HANGOVER_S,
    Observed,
    SpeechStreak,
    ObservedTurn,
)
from shared.carrier_simulator.pacing import FramePacing

# Agent silence the harness waits for before releasing its clip. It is its own
# conversational knob and is **not** compared with the agent's VAD
# `min_silence_duration`: that one decides an end of turn that is measured, this one only keeps
# the harness from talking over the agent. It is set higher on purpose, so as not to cut the agent off
# during one of its own mid-sentence pauses.
#
# The value comes from the measured distribution of the agent's internal pauses, not
# by eye. With less, the harness talks over the agent exactly when it makes a
# long mid-answer pause, that is, in the long answers, which are the ones that
# populate the latency p95: losing those runs would bias the percentile
# downward. The distribution is bimodal, so the criterion is not the p95 but the
# tail: 2.0 s sits above the longest pause observed. It costs a few
# seconds per call (once per turn). Measurements in
# `docs/METRICS_CONTRACT.md` §2.3.2.
AGENT_SILENCE_S: Final[float] = 2.0

# Cap on waiting for the agent to open. If it does not speak within this time, the harness
# releases the clip anyway: a mute agent is also a result, and
# waiting forever would hide it.
MAX_AGENT_WAIT_S: Final[float] = 30.0

# How long to observe after releasing the clip. The agent's turn is around
# 2-4 s according to the 15 calls; 25 s leaves room for tool calling and LLM
# retries without lengthening the run.
WAIT_AFTER_CLIP_S: Final[float] = 25.0

# How long the agent's speech has to last for the harness to take it as its
# answer and advance the turn.
#
# **It is not a comparison constant** and it is not compared with any
# `stop_secs` or `min_silence_duration`: those decide an end of turn that is
# measured, this one only decides when it is the harness's turn to speak. 600 ms sits almost four
# times above the noise fragment that once shifted the script by a turn and
# more than four times below the shortest real answer: no value in
# that gap separates wrongly (measurements in the appendix of
# `docs/comparativo_resultados.md`).
MIN_SUBSTANTIVE_SPEECH_S: Final[float] = 0.6

# Up to this duration the text is asked whether it was a backchannel. Longer than this
# is accepted without asking: no "ajá" lasts two seconds, and requiring text to
# advance would turn a transcription failure into an unanswered turn.
BACKCHANNEL_CAP_S: Final[float] = 2.0


@contextlib.contextmanager
def high_clock_resolution() -> Iterator[bool]:
    """Asks Windows for a 1 ms timer for as long as playback lasts.

    The Windows default is **15.625 ms**, and `asyncio` also wakes a
    timer up to one resolution early (`_clock_resolution`). In
    a 20 ms loop that does not produce accumulated delay (the absolute deadline
    corrects it) but it does produce **bursts**: two or three frames go out back to back with
    less than 1 ms between them and then comes a 30 to 60 ms gap.

    With this, the microsecond minimum between frames disappears, that is, the
    burst (measurement in the appendix of `docs/comparativo_results.md`).

    **It goes here and not in a single PoC on purpose.** The pacing at which the
    audio comes in is part of the stimulus: if PoC1 published evenly and PoC2 in bursts, the
    difference between their latencies would include that, and there would be no way to separate it
    afterwards. It is a condition of the measurement environment, not a knob of the
    comparison, and it is declared as such.

    Sleeping again until the exact deadline was also tried: it is worse
    (max skew 41.4 ms), because each short `sleep` costs a whole turn of the
    event loop. Do not insist.

    Returns whether it succeeded. Outside Windows there is nothing to request and it is a no-op.
    """
    winmm = None
    try:  # pragma: no cover - depends on the operating system
        import ctypes

        winmm = ctypes.WinDLL("winmm")  # type: ignore[attr-defined]
        active = winmm.timeBeginPeriod(1) == 0
    except (ImportError, AttributeError, OSError):
        active = False

    try:
        yield active
    finally:
        if active and winmm is not None:  # pragma: no cover - same
            winmm.timeEndPeriod(1)


class Player:
    """Sole owner of the track: publishes comfort noise and, when asked,
    a clip, always paced by the wall clock and without gaps.

    A single producer and not two competing tasks, for two reasons:

    - two tasks pushing to the same transport would interleave frames and the clip
      would go out with noise wedged in the middle;
    - the pacing has to be set by whoever publishes, with **absolute** deadlines
      against a monotonic clock. The call that pushes the frame claims to return
      once it has been pushed, but the effective pacing depends on the other side, and the
      harness needs to know at what clock instant the clip's speech ended. A
      `sleep(0.02)` in a loop would accumulate drift; against an absolute deadline, it does not.
      PoC2's room smoke test confirmed it: 5,000 frames gave exactly
      100.0 s.
    """

    def __init__(
        self, *, sample_rate: int, observed: Observed, recorder: Any = None
    ) -> None:
        self.sample_rate = sample_rate
        self.per_frame = sample_rate * MS_PER_FRAME // 1000
        self.observed = observed
        # Audit recording (`--record-audit`): `None` on the normal
        # path. See `shared/carrier_simulator/auditoria.py`.
        self.recorder = recorder
        self._pending: list[bytes] | None = None
        self._done: asyncio.Future[float] | None = None
        # The anchor of the time axis: `_onset` is the wall-clock time,
        # `_onset_perf` the same instant on the fine clock. See `pacing.py`.
        self._onset = 0.0
        self._onset_perf = 0.0
        self._sent_ones = 0
        self._current_clip = False
        # Nominal deadline and real instant of each frame. It is always filled, not
        # only with `--record-audit`: it is where the turn's zero comes from, and the
        # skew the metrics contract declares in §1.
        self.pacing = FramePacing()
        # Whether the 1 ms timer was obtained. It goes into the report: a run
        # published in bursts is not comparable with one published evenly.
        self.fine_clock = False

    async def publish(self, raw_value: bytes) -> None:
        """Pushes a mono int16 PCM frame of `per_frame` samples.

        The only thing each PoC has to write. It must return when the frame
        has gone out; the pacing is driven by the loop, not by this call.
        """
        raise NotImplementedError

    async def prepare(self) -> None:
        """Hook for whatever has to be opened before the first frame."""

    # -- The frame domain --
    #
    # The loop below treats frames as opaque bytes: it only counts how many
    # go out and when each one ends. **What is inside is decided by the transport**,
    # and it is not the same in the 3 PoCs: PoC2 publishes PCM16 to the `AudioSource`,
    # while PoC1 and PoC3 send the clip's μ-law bytes as-is over Media
    # Streams. Re-encoding there would change the stimulus, so the slicing and the
    # padding are overridden instead of converted.

    def clip_frames(self, samples: np.ndarray) -> list[bytes]:
        """Slices a clip into transport frames. PCM16 by default."""
        return split_into_frames(samples, self.per_frame)

    def filler_frame(self) -> bytes:
        """What is published between clips. Never digital silence."""
        return comfort_noise_frame(self.per_frame)

    def as_pcm16(self, raw_value: bytes) -> bytes:
        """The frame in PCM16, for the audit recording.

        Identity when the transport already speaks PCM16. A transport that sends
        μ-law **has** to decode here: without this the audit would store
        μ-law interpreted as PCM16, which sounds like nothing and which nobody
        would detect until opening the WAV.
        """
        return raw_value

    async def _enqueue(self, frames: list[bytes]) -> float:
        """Enqueues ready-made frames and waits for the last one to sound."""
        if self._done is not None and not self._done.done():
            raise HarnessFailure("ya hay un clip en el aire")
        self._done = asyncio.get_running_loop().create_future()
        self._pending = list(frames)
        return await self._done

    async def play(self, samples: np.ndarray) -> float:
        """Enqueues the clip and waits for it to go out. Returns the local clip end."""
        return await self._enqueue(self.clip_frames(samples))

    async def play_step(self, step: ScriptStep) -> float:
        """The script step, over the wire. It is what `play_script` calls.

        It exists in addition to `play` because **not all transports start from
        the PCM16 samples**: PoC1 sends the manifest's μ-law bytes as-is,
        and re-encoding them from the PCM16 would give almost the same audio with a different hash,
        that is, it would lose the evidence that the 3 PoCs played the same
        stimulus. Whoever is in that case overrides this and takes from the step
        whatever it needs.
        """
        return await self.play(step.samples)

    def _moment_of(self, trama: int) -> float:
        """**Nominal** instant at which frame `trama` finishes sounding.

        It is the plan, not what happened: it comes from the deadline and not from reading the clock
        after publishing. It serves as a reference axis; to timestamp the turn's
        zero, `_actual_moment_of` is used, which is what really happened.
        """
        elapsed = (trama + 1) * (MS_PER_FRAME / 1000.0)
        return round(self._onset + elapsed - self.observed.start, 6)

    def _now(self) -> float:
        """The current instant, on the same axis as `_moment_of`.

        It is measured with `perf_counter` and translated with the anchor taken at startup.
        `time.time()` would not do: it declares 15.625 ms resolution and is not
        monotonic. See the docstring of `shared/carrier_simulator/pacing.py`.
        """
        return round(
            self._onset
            + (time.perf_counter() - self._onset_perf)
            - self.observed.start,
            6,
        )

    def _actual_moment_of(self, trama: int) -> float:
        """When frame `trama` finished sounding, measured.

        The frame occupies 20 ms from when it was released, so its end is the send
        instant plus the duration. Never `None`: if the frame is not recorded
        it falls back to the nominal, which is the honest behavior: better the plan than
        an invented number.
        """
        if 0 <= trama < len(self.pacing.reales):
            return round(self.pacing.reales[trama] + MS_PER_FRAME / 1000.0, 6)
        return self._moment_of(trama)

    def _release_pending(self) -> None:
        """Releases whoever is waiting for a clip that will no longer finish sounding.

        If the call drops mid-playback, the loop exits and the `await` in
        `play` would hang forever: the whole run would
        block on the turn where the agent hung up, which is exactly when
        getting out is most urgent.
        """
        self._pending = None
        if self._done is not None and not self._done.done():
            self._done.set_result(self._actual_moment_of(self._sent_ones - 1))

    async def launch(self, halt: asyncio.Event) -> None:
        await self.prepare()
        with high_clock_resolution() as fine:
            self.fine_clock = fine
            # The anchor, taken once: `perf_counter` is the working clock
            # and `time()` only translates to the wall axis. See `pacing.py`.
            self._onset = time.time()
            self._onset_perf = time.perf_counter()

            try:
                await self._pump(halt)
            finally:
                self._release_pending()

    async def _pump(self, halt: asyncio.Event) -> None:
        # The deadline is compared against `perf_counter` and **not** against
        # `time.monotonic()`, which on Windows is `GetTickCount64` and has 15.625
        # ms granularity: the subtraction would come out quantized and the loop would publish in
        # bursts (measurement in `docs/METRICS_CONTRACT.md`).
        base = self._onset_perf
        while not halt.is_set():
            first_of_clip = False
            if self._pending:
                raw_value = self._pending.pop(0)
                self.observed.clip_frames += 1
                first_of_clip = not self._current_clip
                self._current_clip = True
                clip_ends = not self._pending
            else:
                raw_value = self.filler_frame()
                clip_ends = False

            nominal = self._moment_of(self._sent_ones) - MS_PER_FRAME / 1000.0

            if self.recorder is not None:
                # It is an `append` and nothing more; the file is built on hang-up.
                self.recorder.annotate_harness(nominal, self.as_pcm16(raw_value))

            # **The send instant.** Here and not after `publish`: what
            # comes after is each transport's own work, and in PoC2
            # `capture_frame` also blocks against its own queue. See `pacing.py`.
            real = self._now()
            self.pacing.annotate(nominal, real)
            if first_of_clip:
                # The clip start is when its first frame was released, not
                # when it finished sounding: `end_of_speech_s` hangs from here.
                self.observed.t_clip_start = real
            await self.publish(raw_value)

            self.observed.published_frames += 1
            self._sent_ones += 1

            if clip_ends:
                self._pending = None
                self._current_clip = False
                end_of_this = self._actual_moment_of(self._sent_ones - 1)
                self.observed.t_clip_end = end_of_this
                if self._done is not None and not self._done.done():
                    self._done.set_result(end_of_this)

            target = base + self._sent_ones * (MS_PER_FRAME / 1000.0)
            lag = target - time.perf_counter()
            if lag > 0:
                await asyncio.sleep(lag)


async def wait_until_agent_is_silent(
    observed: Observed, *, silence_s: float, tope_seg: float
) -> str:
    """Waits for the agent to finish speaking. Returns why it stopped waiting.

    It is what keeps the harness from talking over the greeting: if we released the clip
    blindly, the agent's turn would come out contaminated by barge-in and the test would
    say nothing about whether it heard us.
    """
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        silent = observed.agent_silent_since()
        if silent is not None and silent >= silence_s:
            return "el agente terminó de hablar"
        await asyncio.sleep(0.05)

    if observed.t_agent_first_speech is None:
        return f"el agente no habló en {tope_seg:.0f} s: se suelta el clip igual"
    return f"el agente siguió hablando {tope_seg:.0f} s: se suelta el clip igual"


@dataclass
class AgentWait:
    """How the wait for the agent's answer ended."""

    motivo: str
    t_first_speech: float | None = None
    # Speech streaks that sounded and were not taken as the answer. See
    # `ObservedTurn.discarded_fragments`.
    discarded_items: list[dict[str, Any]] = field(default_factory=list)


# Verdicts of `_streak_verdict`. Anything else is the reason why
# the streak is discarded.
IS_THE_RESPONSE: Final[str] = ""
NOT_KNOWN_YET: Final[str] = "aún suena"


def _streak_verdict(
    streak: SpeechStreak, text: str, *, opened: bool
) -> str:
    """Whether this streak is the agent's answer, is not known yet, or why not.

    Three ranges by duration:

    - **Longer than `BACKCHANNEL_CAP_S`.** It is the answer, and it is accepted
      without asking the text or waiting for it to end. No backchannel
      lasts two seconds, and requiring a transcript here would turn a provider
      failure into an "unanswered" turn, which also cuts the count of
      reached turns.
    - **Between the minimum and that cap.** Here the classifier does have a say, but only
      once the streak has closed: the agent's text arrives together with its audio or
      a little later, so asking as soon as it crosses the minimum is asking
      before there is anything. While it is still sounding, `NOT_KNOWN_YET`.
    - **Shorter than the minimum.** It is the safety net, and it needs no text,
      which is exactly what makes it indispensable: a short fragment may not
      generate a transcript, and then there is nothing to classify. The duration
      does catch it.
    """
    if streak.duration >= BACKCHANNEL_CAP_S:
        return IS_THE_RESPONSE
    if opened:
        # Still growing: it may reach the cap, or close and be judged.
        return NOT_KNOWN_YET
    # The discard reasons stay in Spanish: `validation/phone_tail_poc3.py`
    # filters the logged reasons by their first letter.
    if streak.duration < MIN_SUBSTANTIVE_SPEECH_S:
        return "más corta que el mínimo de voz sustantiva"
    if is_backchannel(text):
        return "apoyo verbal, no una respuesta"
    return IS_THE_RESPONSE


async def wait_for_agent_turn(
    observed: Observed,
    *,
    since: float,
    silence_s: float,
    tope_seg: float,
    closed: asyncio.Event | None = None,
) -> AgentWait:
    """Waits for the agent to answer **and** finish.

    They are two phases and not one, and that is the part that can be done wrong without it being
    noticed: if only silence were waited for, the condition would already be satisfied from
    before the clip was released (the agent has been quiet since it finished its previous
    turn) and the harness would fire the next clip immediately, piling up
    all the scenario's turns.

    `since` is the turn's zero (the end of our clip): only the agent's speech
    **after** that instant counts.

    ## Why phase 1 is not satisfied with energy

    Because anything has energy. In the run
    `spots-poc2-arnes-20260916-113508` the agent released 160 ms of audio at almost
    full scale 2.6 s after the T1 clip; with the old criterion (a
    single frame above the threshold) that satisfied phase 1, the subsequent silence
    satisfied phase 2, and T2 went out at 13.60 s. The real answer
    arrived at 20.32 s, when T2 had already finished sounding. **The
    turn ended up shifted and its latency is not valid.**

    The worst thing about that case is that it does not show in the numbers: T2's acknowledgement came at
    3.42 s, within the healthy range of that same call (3.3–3.4 s). It is only
    detected by listening. That is why what is discarded is returned and written down, instead of
    just being ignored.
    """
    limit = time.monotonic() + tope_seg
    discarded_items: list[dict[str, Any]] = []
    seen_ones: set[int] = set()

    def _is_closed() -> bool:
        return closed is not None and closed.is_set()

    def _find_response() -> SpeechStreak | None:
        """The first streak after the zero that really is the answer.

        It is re-evaluated in full on every pass on purpose: a streak that is
        discarded now for lack of text may qualify on the next pass,
        when the fragment arrives. What has already been recorded as discarded is not
        recorded again (`seen_ones`).

        A streak **still open** that does not reach the minimum is not discarded: it is
        let through silently. It is growing, and judging it now would record a
        discard on every healthy turn: the first frame of any answer
        measures zero.
        """
        for streak in observed.streaks_after(since):
            opened = observed.now() - streak.end <= STREAK_HANGOVER_S
            text = observed.agent_text_during(streak)
            verdict = _streak_verdict(streak, text, opened=opened)
            if verdict == IS_THE_RESPONSE:
                return streak
            if verdict == NOT_KNOWN_YET:
                continue
            if id(streak) not in seen_ones:
                seen_ones.add(id(streak))
                discarded_items.append(
                    {
                        "start": streak.start,
                        "duration_ms": round(streak.duration * 1000.0, 1),
                        "motivo": verdict,
                        "text": text,
                    }
                )
        return None

    # Phase 1: that it really starts answering, after our zero.
    response: SpeechStreak | None = None
    while time.monotonic() < limit:
        if _is_closed():
            # Waiting the full cap against a dead call gives away 25 s per
            # turn. On September 15 it was 27 s in a single run.
            return AgentWait(
                "la sala se cerró mientras esperábamos respuesta",
                discarded_items=discarded_items,
            )
        response = _find_response()
        if response is not None:
            break
        await asyncio.sleep(0.05)
    else:
        # A discarded streak that was left alone is information: the agent sounded
        # and what sounded was not an answer. Telling it apart from a mute agent
        # changes what gets reviewed afterwards.
        _find_response()
        if discarded_items:
            return AgentWait(
                f"el agente solo emitió {len(discarded_items)} fragmento(s) que no "
                "eran su respuesta: venció el tope de seguridad",
                discarded_items=discarded_items,
            )
        return AgentWait(
            "el agente no contestó: venció el tope de seguridad",
            discarded_items=discarded_items,
        )

    first = response.start

    # Second phase: that it finishes.
    while time.monotonic() < limit:
        if _is_closed():
            return AgentWait(
                "la sala se cerró mientras el agente hablaba", first, discarded_items
            )
        silent = observed.agent_silent_since()
        if silent is not None and silent >= silence_s:
            return AgentWait("contestó y terminó", first, discarded_items)
        await asyncio.sleep(0.05)

    return AgentWait(
        "el agente seguía hablando al vencer el tope", first, discarded_items
    )


async def _turn_with_event(
    event: Any,
    *,
    turn: ObservedTurn,
    player: Any,
    observed: Observed,
    wait_for_agent_s: float,
    closed: asyncio.Event | None,
) -> AgentWait:
    """A turn of the interruption scenario. See `interruptions.py`."""
    from shared.carrier_simulator.interruptions import (
        act_out_event,
        wait_for_trigger,
    )

    since = turn.t_clip_end or observed.now()
    trigger = await wait_for_trigger(
        observed, since=since, tope_seg=wait_for_agent_s, closed=closed
    )
    if trigger is None:
        # The agent never spoke for a second continuously: there is nothing to talk
        # over. The turn continues as in the normal scenario.
        turn.event = {
            "tipo": event.tipo,
            "clip_id": event.clip_id,
            "text": event.text,
            "fired": False,
            "motivo": "el agente no habló de corrido lo suficiente para disparar",
        }
        return await wait_for_agent_turn(
            observed,
            since=since,
            silence_s=AGENT_SILENCE_S,
            tope_seg=wait_for_agent_s,
            closed=closed,
        )
    moment, streak = trigger
    observed.tally(
        f"{turn.step.turn} evento {event.tipo}"
        + (f" {event.clip_id} — {event.text!r}" if event.clip_id else "")
        + f" a los {moment - streak.start:.2f} s de su respuesta"
    )
    turn.event = await act_out_event(
        event,
        player=player,
        observed=observed,
        trigger=moment,
        tope_seg=wait_for_agent_s,
        closed=closed,
    )
    return AgentWait(turn.event["advance_reason"], streak.start, [])


async def play_script(
    script: list[ScriptStep],
    *,
    player: Any,
    observed: Observed,
    wait_for_agent_s: float,
    emitter: Any = None,
    closed: asyncio.Event | None = None,
    estado: Any = None,
) -> None:
    """Runs the turns in order, yielding the turn to the agent between one and the next.

    The comfort noise **is never cut**: between turns the same `Player` keeps
    pushing it, so the agent receives a continuous signal just as in a
    real call. A gap of digital silence would break its endpointing.

    With `emitter`, each clip also publishes its real zero to the worker (the
    data channel in PoC2; whatever each PoC uses). It goes **before** starting to
    wait for the agent, so that the packet travels while the turn is still
    in progress; and it is pure diagnostics, so if it fails it stops nothing.

    `closed` is the second end condition: when the agent closes the
    negotiation it hangs up, and from then on playing is talking to nobody. The
    script is cut at that point and the missing turns **are not made up**:
    they are left out, and `evaluate_run` says how far it got.

    With `estado`, the turns that branch choose their line based on what
    the agent has just said. The decision is made **right before**
    playing and with the previous turn's answer already observed: it is the same
    order in which a human acting out the script would decide.
    """
    from shared.test_scenarios.decision import decide

    # What the agent said when opening, before the first clip. It counts: if it dropped an
    # amount in the greeting, the tree has to know.
    if estado is not None:
        opening = observed.assemble_agent(observed.agent_fragments())
        if opening.strip():
            estado.observe("apertura", opening)

    last_response = ""
    for numero, step in enumerate(script, start=1):
        if closed is not None and closed.is_set():
            observed.tally(
                f"se corta el guion en {step.turn}: la sala ya está cerrada "
                f"({len(script) - numero + 1} turno(s) sin actuar)"
            )
            return

        if estado is not None and step.branches_out and step.of_turn is not None:
            decision = decide(
                step.of_turn, estado, agent_text=last_response
            )
            chosen_one = step.variants.get(decision.variant)
            if chosen_one is None:
                # It should not happen: `load_scenario` loads all the branches
                # before connecting. If it did, it continues with the base instead of
                # knocking over a call that is already spending money.
                observed.annotate_incident(
                    f"{step.turn}: la rama {decision.rama!r} pide la variante "
                    f"{decision.variant!r} y no está cargada; se usa la base"
                )
            else:
                step = chosen_one
            observed.decisions.append(decision)
            observed.tally(
                f"{step.turn} rama {decision.rama!r} ({decision.variant}) — "
                f"{decision.motivo}"
            )

        turn = ObservedTurn(step=step)
        observed.turns.append(turn)

        label = f"{step.turn} ({numero}/{len(script)})"
        # Mark in the fragment stream: what belongs to this turn is whatever arrives from
        # here on. Cutting by timestamp would leave out the STT that
        # the provider emits once the clip has already started.
        from_fragment = len(observed.fragments)

        if step.is_silence:
            # Nothing is played: the comfort noise already running IS the
            # script's silence, and it is more faithful than a silence WAV.
            observed.tally(
                f"{label} silencio de {step.silence_s:.0f} s "
                "(el guion pide no contestar)"
            )
            turn.t_start = observed.now()
            await asyncio.sleep(step.silence_s)
            turn.t_clip_end = observed.now()
        else:
            observed.tally(
                f"{label} {step.clip_id} ({step.duration_s:.2f} s) — {step.text!r}"
            )
            turn.t_clip_end = await player.play_step(step)
            turn.t_start = round(turn.t_clip_end - step.duration_s, 3)

            if emitter is not None:
                # Absolute, not relative to the harness's zero: the worker does not have
                # that zero and could not reconstruct them.
                await emitter.send(
                    turn=step.turn,
                    clip_id=step.clip_id,
                    start_of_clip=observed.start + turn.t_start,
                    clip_end=observed.start + turn.t_clip_end,
                    end_of_speech_s=step.end_of_speech_s,
                    drop_db=step.speech_drop_db,
                    speech_threshold_dbfs=step.speech_threshold_dbfs,
                )
                turn.sent_mark = True

        event = getattr(step, "event", None)
        if event is None:
            wait = await wait_for_agent_turn(
                observed,
                since=turn.t_clip_end or observed.now(),
                silence_s=AGENT_SILENCE_S,
                tope_seg=wait_for_agent_s,
                closed=closed,
            )
        else:
            # Interruption scenario. It only runs when the step carries an
            # event; in any other run this branch does not exist.
            wait = await _turn_with_event(
                event,
                turn=turn,
                player=player,
                observed=observed,
                wait_for_agent_s=wait_for_agent_s,
                closed=closed,
            )
        turn.advance_reason = wait.motivo
        turn.t_agent_first_speech = wait.t_first_speech
        turn.discarded_fragments = wait.discarded_items

        for discarded_one in wait.discarded_items:
            # It is recorded as an incident besides staying in the turn: whoever looks at
            # the on-screen report has to find out without opening the JSON.
            observed.annotate_incident(
                f"{step.turn}: se descartó audio del agente en "
                f"{discarded_one['start']:.2f}s "
                f"({discarded_one['duration_ms']:,.0f} ms, {discarded_one['motivo']})"
                + (f" — {discarded_one['text']!r}" if discarded_one["text"] else "")
            )

        # What was heard and said **in this turn**: what arrived since its mark.
        of_turn = observed.fragments[from_fragment:]
        turn.stt = observed.assemble_stt(
            [
                f
                for f in of_turn
                if f.sender == observed.own_identity and not f.closure
            ]
        )
        turn.response = observed.assemble_agent(
            [
                f
                for f in of_turn
                if f.sender != observed.own_identity and not f.closure
            ]
        )
        if turn.event is not None and turn.event.get("t_clip_start") is not None:
            # In a turn with an event, `stt` combines the script line and the clip
            # that sounded over the agent. They are also stored separately,
            # cutting at the onset of the event's clip: the line had already been
            # transcribed by then, because the agent had to answer it before
            # the harness talked over it. The WER of what the STT heard
            # during the agent's speech comes from here, in its own section.
            from_the_clip = turn.event["t_clip_start"]
            own_ones = [
                f
                for f in of_turn
                if f.sender == observed.own_identity and not f.closure
            ]
            turn.event["line_stt"] = observed.assemble_stt(
                [f for f in own_ones if f.t < from_the_clip]
            )
            turn.event["event_stt"] = observed.assemble_stt(
                [f for f in own_ones if f.t >= from_the_clip]
            )
        if turn.event is not None and turn.event.get("t_speech_onset") is not None:
            # What the agent said before and after the event (in the control,
            # the trigger): to read whether on coming back it continued the same sentence or
            # generated a new message. It is split by the text's arrival, which in
            # PoC1 runs a little ahead of the audio; it is for reading, not for measuring.
            reference = turn.event["t_speech_onset"]
            of_agent = [
                f
                for f in of_turn
                if f.sender != observed.own_identity and not f.closure
            ]
            turn.event["agent_before_event"] = observed.assemble_agent(
                [f for f in of_agent if f.t < reference]
            )
            turn.event["agent_after_event"] = observed.assemble_agent(
                [f for f in of_agent if f.t >= reference]
            )

        last_response = turn.response
        if estado is not None and last_response.strip():
            estado.observe(step.turn, last_response)

        acknowledgement = turn.acknowledgement_ms
        observed.tally(
            f"{label} {turn.advance_reason}"
            + (f" · acuse {acknowledgement:,.0f} ms" if acknowledgement is not None else "")
        )
