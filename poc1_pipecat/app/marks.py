"""The real zero of the turn in PoC1: the same packets, out of band.

The packet —the two zeros, their threshold, the `host` and the format version—
is the one from `shared/carrier_simulator/cero.py`, identical to the one PoC2
publishes over the LiveKit data channel. The only thing specific to PoC1 is
**which route it takes**.

## Why to disk and not in band

Twilio Media Streams carries audio and four events (`connected`, `start`,
`media`, `stop`), and pipecat's `TwilioFrameSerializer` discards any event it
does not know: a made-up `event` would never reach the application code. The
only one that does get through is `dtmf`, and using it would mean putting a
tone inside the very conversation being measured. There is no side channel.

So the mark is written to a file and the match against the pipeline JSON is
done afterwards, by wall clock. Both processes run on the same machine, so they
share a clock; the mark carries its `host` just as in PoC2, and if it ever did
not match the worker's, the bias is declared uninterpretable instead of being
corrected.

## What changes and what does not with respect to PoC2

**The data does not change.** The timestamp and the two zeros are computed in
`ZeroEmitter.send`, **before** the packet is delivered, so whatever the route
costs does not move them. That is the condition that makes both biases be
computed on the same instant.

**What the route costs does change**, and that is not a plumbing detail: PoC2
pays for the data channel and, in addition, the audio goes through the
`AudioSource` queue before reaching the agent, whereas in PoC1 the harness
writes to disk and the audio goes straight out through the WebSocket. **PoC1's
bias and PoC2's bias do not measure the same path** even though the zero is
correctly anchored in both. This is declared in `docs/METRICS_CONTRACT.md`.

## Why JSON Lines and one line per turn

It is written live, one envelope per line, as soon as the clip finishes
playing — just as PoC2 publishes live and for the same reason: if the run
crashes mid-call, the turns that did happen are already on disk. A single JSON
assembled at the end would be lost entirely.

There is no `fsync`. A `flush` leaves the line in the OS buffer, which survives
the harness process dying —the realistic scenario— and costs tens of
microseconds. `fsync` would cost milliseconds of blocking the event loop, and
that the player would notice, since it is publishing frames every 20 ms from
another task.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Final

from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from poc1_pipecat.app.latency import frame_key

from shared.carrier_simulator.cero import (
    ZERO_PROTOCOL,
    SUMMARY_KIND,
    PROTOCOL_VERSION,
    ZeroEmitter,
    ZeroMark,
    Envelope,
    decode,
)
from shared.carrier_simulator.cruce import (
    EstimatedZero,
    PairedTurn,
    group,
    pair,
    lines_summary,
)
from shared.carrier_simulator.audio import rms_pcm16
from shared.carrier_simulator.cruce import ms as _ms
from shared.carrier_simulator.observed import RMS_WITH_SPEECH

logger = logging.getLogger(__name__)

# The two possible anchorings of the zero, and the undecidable one. Same names
# as in `poc2_livekit/app/latency.py`: if the two JSONs used different labels
# for the same conclusion, the comparison would have to translate them, and
# that is where nuance gets lost.
ANCHOR_REAL: Final[str] = "anchored_on_real_end_of_speech"
ANCHOR_INCONCLUSIVE: Final[str] = "inconclusive"
ANCHOR_NO_DATA: Final[str] = "no_data"

# Where the harness output lives. It is **not** `recordings/`, on purpose: the
# campaign calls live there and `shared.metrics.cli compute` reads them all.
DIAGNOSTICS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "harness"
)

# The name carries the `callSid`, which is the same key the pipeline uses to
# name its JSON (`recordings/call_{call_sid}.json`) and the one stored inside
# it. That way step 3c's match is a `join` by file name and not a heuristic.
PREFIX: Final[str] = "zeros"
EXTENSION: Final[str] = ".jsonl"


def zeros_path(call_sid: str, *, folder: Path | None = None) -> Path:
    return (folder or DIAGNOSTICS_PATH) / f"{PREFIX}_{call_sid}{EXTENSION}"


class DiskMarkEmitter(ZeroEmitter):
    """Writes the harness envelopes to a `.jsonl`, one per line.

    **It never propagates a write failure.** A diagnostics packet that did not
    go out is no reason to knock over a run that is already spending on STT,
    TTS and LLM; it is recorded in `faults` and the report says so.
    """

    def __init__(
        self,
        ruta: Path | str,
        *,
        publish_sample_rate: int,
        clock: Any = None,
        host: str | None = None,
    ) -> None:
        super().__init__(publish_sample_rate=publish_sample_rate, clock=clock, host=host)
        self.ruta = Path(ruta)
        self._file: Any = None

    def _begin(self) -> Any:
        if self._file is None:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.ruta.open("a", encoding="utf-8", newline="\n")
        return self._file

    async def _deliver(self, carga: bytes) -> None:
        file = self._begin()
        # One line per envelope, with no line breaks inside: `encode_*` already
        # produces compact single-line JSON, so the file can be read piecemeal
        # even if the run crashed halfway.
        file.write(carga.decode("utf-8") + "\n")
        file.flush()

    async def shut(
        self, *, turns_without_audio: Sequence[str] = (), sleep: Any = None
    ) -> bool:
        try:
            return await super().shut(
                turns_without_audio=turns_without_audio, sleep=sleep
            )
        finally:
            self.close_file()

    def close_file(self) -> None:
        """Release the file descriptor. Idempotent: closing twice is not an error."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None


def read_zeros(ruta: Path | str) -> list[Envelope]:
    """The envelopes of a zeros file, in order.

    Tolerant of broken lines and **explicit** about them: a run that crashed
    mid-write leaves the last line incomplete, and that must not prevent
    reading the earlier ones. Whatever could not be read is logged, not
    silently discarded.
    """
    envelopes: list[Envelope] = []
    for numero, linea in enumerate(_lines(Path(ruta)), start=1):
        try:
            envelopes.append(decode(linea))
        except (ValueError, KeyError) as error:
            logger.warning("line %d of %s unreadable: %r", numero, ruta, error)
    return envelopes


def _lines(ruta: Path) -> Iterator[str]:
    if not ruta.is_file():
        return
    with ruta.open("r", encoding="utf-8") as file:
        for linea in file:
            if linea.strip():
                yield linea


def marks_de(ruta: Path | str) -> list[ZeroMark]:
    """The marks of a run, without duplicates and in sequence order.

    The closing summary repeats every mark on purpose —it is the second chance
    for one that got lost— so they must be deduplicated here or every turn
    would appear twice. The summary's copy wins when it exists: it is the one
    the harness confirmed at the end.
    """
    by_sequence: dict[int, ZeroMark] = {}
    for on_top in read_zeros(ruta):
        for mark in on_top.marks:
            if on_top.tipo == SUMMARY_KIND or mark.sequence not in by_sequence:
                by_sequence[mark.sequence] = mark
    return [by_sequence[k] for k in sorted(by_sequence)]


def coverage(ruta: Path | str) -> dict[str, Any]:
    """How many marks were expected and how many there are. For the harness report.

    Without the summary's declared total there is no way to tell "they all
    arrived" from "whatever arrived, arrived", which is exactly the false
    reading this field exists to prevent.
    """
    envelopes = read_zeros(ruta)
    summaries = [s for s in envelopes if s.tipo == SUMMARY_KIND]
    marks = marks_de(ruta)
    declared = summaries[-1].declared_total if summaries else None
    return {
        "file": str(ruta),
        "marks": len(marks),
        "declared_total": declared,
        "had_summary": bool(summaries),
        "complete": declared is not None and declared == len(marks),
        "turns_without_audio": list(summaries[-1].turns_without_audio) if summaries else [],
    }


def as_json(ruta: Path | str) -> dict[str, Any]:
    """The block the harness writes for step 3c to match.

    Same content PoC2 publishes in `livekit_extra.harness`, so that the match
    and the report are written only once.
    """
    data = coverage(ruta)
    data["marks_detail"] = [m.a_dict() for m in marks_de(ruta)]
    data["what_it_is"] = (
        "los dos ceros de cada turno, fechados por el arnés antes de publicar. "
        "Diagnóstico: no entra en `latencias` ni en ningún percentil"
    )
    return data


# --------------------------------------------------------------------------
# Pipeline side: the zero PoC1 estimates
# --------------------------------------------------------------------------


class ZeroProbe(FrameProcessor):
    """Timestamps, on the wall clock, the zero that pipecat's VAD produces.

    It lets everything through untouched, like `LatencyProbe`, and **changes no
    sample**: `LatencyCollector` keeps measuring its stages on `time.monotonic`
    as before. This is harness diagnostics and lives separately for a concrete
    reason: the harness mark comes from another process, and `time.time()` is
    the only clock the two share. Mixing it with the collector's monotonic
    clock would have put a system time adjustment inside the campaign
    latencies.

    ## The zero is computed the same way as in the collector, including which one wins

    Two things, both copied from `LatencyCollector` on purpose:

    1. `VADUserStoppedSpeakingFrame` arrives when the VAD **confirms** the
       silence, i.e. `stop_secs` after the person really went quiet. It is
       subtracted. Without that, every latency would come out 200 ms short,
       systematically — and here the bias would come out 200 ms long.
    2. **The last end of speech of the turn wins, not the first.** With
       `stop_secs=0.2` the VAD fires every time someone takes a breath
       mid-sentence: a three-clip run produced six. The turn is closed by the
       analyzer (`UserStoppedSpeakingFrame`), and the turn's zero is the last
       end of speech before that close. Keeping the first one would measure the
       clip's internal pause and yield a huge bias that looks like a finding.

    ## Fine-grained clock

    `time.time()` on Windows has the granularity of `GetSystemTimeAsFileTime`
    (15.6 ms), which is on the order of the bias we want to measure in PoC1. So
    it is anchored once and advanced with `perf_counter`, exactly as the harness
    player timestamps its send instant (§1 of the contract). Both ends of the
    bias are measured with clocks of the same quality.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.zeros: list[EstimatedZero] = []
        # **All** ends of speech, with their onset, exactly as they came out of
        # the VAD. With `stop_secs=0.2` there are several per turn: the
        # difference against `len(zeros)` is the evidence that the turn is
        # closed by the analyzer and not by every pause. They are kept raw
        # because a diagnostic without STT —and therefore without end of turn—
        # needs to be able to choose the zero by its own, declared criterion.
        self.speech_ends: list[tuple[float | None, float]] = []
        self._indice = 0
        self._start: float | None = None
        self._streak: float | None = None
        self._end: float | None = None
        self._seen: set[int] = set()
        self._anchor_wall = time.time()
        self._anchor_perf = time.perf_counter()

    def _now(self) -> float:
        return self._anchor_wall + (time.perf_counter() - self._anchor_perf)

    def _new(self, frame: Frame) -> bool:
        """The same event can cross the probe twice. It is counted once.

        The key comes from `frame_key`, the same helper `LatencyCollector` uses,
        and not from `id(frame)`. Two reasons, both learned by breaking it:

        - a broadcast creates **two instances** with different `id`s,
          cross-linked by `broadcast_sibling_id`, so `id` does not tie them
          together;
        - CPython **reuses** the `id` of a freed object, so a later frame can
          inherit the id of one already seen and be taken as a repeat. With
          `id` this once swallowed an entire end of turn.
        """
        clave = frame_key(frame)
        if clave is None:
            return True
        if clave in self._seen:
            return False
        self._seen.add(clave)
        return True

    # -- The rule, apart from pipecat's plumbing --
    #
    # Kept separate from `process_frame` on purpose: what follows is what must
    # be testable —what is subtracted and which end of speech wins— and testing
    # it through the `FrameProcessor` would require bringing up the whole
    # lifecycle (StartFrame, TaskManager) to check three subtractions.

    def annotate_start(self, start_secs: float) -> None:
        """A speech streak started, `start_secs` before the VAD said so."""
        self._streak = self._now() - max(start_secs, 0.0)
        if self._start is None:
            # The turn's first onset is the turn's onset.
            self._start = self._streak

    def annotate_end(self, stop_secs: float) -> None:
        """A streak ended, `stop_secs` before the VAD confirmed it.

        The last one wins: a mid-sentence pause also fires here.
        """
        self._end = self._now() - max(stop_secs, 0.0)
        # The onset is cleared per streak, not per turn, so the next one brings
        # its own. Without this every streak in the run keeps the first one's
        # onset and the onset bias comes out in the tens of seconds — which is
        # what came out the first time this was run.
        self.speech_ends.append((self._streak, self._end))
        self._streak = None

    def close_turn(self) -> bool:
        """The analyzer declared the turn closed. Returns whether there was anything to close."""
        if self._end is None:
            return False
        self.zeros.append(
            EstimatedZero(indice=self._indice, start=self._start, end=self._end)
        )
        self._indice += 1
        self._start = None
        self._streak = None
        self._end = None
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._new(frame):
                self.annotate_start(float(getattr(frame, "start_secs", 0.0) or 0.0))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._new(frame):
                self.annotate_end(float(getattr(frame, "stop_secs", 0.0) or 0.0))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._new(frame):
                self.close_turn()

        await self.push_frame(frame, direction)

    def close_pending_turn(self) -> None:
        """Flush the last turn if the analyzer did not get to close it.

        It happens at the end of the run: the harness stops talking and nothing
        triggers the close. Losing that turn would mean losing a sample that
        was actually measured.
        """
        self.close_turn()


# Silence that separates two agent responses. Below this, audio frames continue
# the same response; above it, a new one starts. It is the same criterion the
# harness uses to group the speech it hears (`STREAK_HANGOVER_S`), and it has
# to be: the two ends of the return path are paired by proximity, and if each
# side grouped differently, different segments would be paired.
OUTPUT_HANGOVER_S: Final[float] = 0.3


class AudioOutputProbe(FrameProcessor):
    """Timestamps, on the wall clock, when the pipeline released its first audio frame.

    It is the left end of the **return path**. The right end is measured by the
    harness when it hears the agent, and without this one the return path could
    only be inferred by subtracting medians —`line_latency − full_turn − bias`—,
    which is a bound and not a measurement.

    It goes **after** the TTS and right before `transport.output()`, so it
    marks the instant the audio leaves the pipeline and enters the transport.
    Whatever happens from there on —serialization, the WebSocket, the network,
    the jitter buffer on the other side— is exactly the term the latency
    comparison needs to be able to discount, because it is not the same in the
    two PoCs.

    Same fine-grained clock as `ZeroProbe`, and for the same reason: on Windows
    `time.time()` jumps in 15.6 ms steps, which is on the order of what is being
    measured.
    """

    def __init__(self, *, hangover_seg: float = OUTPUT_HANGOVER_S, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # One instant per agent response, not per frame: there are 50 frames per
        # second and what matters is when each response **started** going out.
        self.outputs: list[float] = []
        # And the same ones, but counting only from when the frame carries real
        # speech. Both are needed because the two ends of the return path do
        # not measure the same event: the harness timestamps when it **heard**
        # the agent (energy above `RMS_WITH_SPEECH`), and the TTS can start
        # delivering frames before anything is audible. With both lists, the
        # TTS onset is separated from the transport instead of contaminating it.
        self.audible_outputs: list[float] = []
        self._hangover = hangover_seg
        self._ultima: float | None = None
        self._ultima_audible: float | None = None
        self._anchor_wall = time.time()
        self._anchor_perf = time.perf_counter()

    def _now(self) -> float:
        return self._anchor_wall + (time.perf_counter() - self._anchor_perf)

    def annotate_frame(self, now: float, *, with_voice: bool = True) -> bool:
        """Record an output frame. Returns whether it opened a new response."""
        opens = self._ultima is None or (now - self._ultima) > self._hangover
        if opens:
            self.outputs.append(now)
        self._ultima = now

        if with_voice:
            if (
                self._ultima_audible is None
                or (now - self._ultima_audible) > self._hangover
            ):
                self.audible_outputs.append(now)
            self._ultima_audible = now
        return opens

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            # Same threshold the harness uses to decide "the agent is speaking".
            # It has to be the same or the two ends would measure different
            # events and the subtraction would not be a transport time.
            self.annotate_frame(
                self._now(),
                with_voice=rms_pcm16(frame.audio) >= RMS_WITH_SPEECH,
            )
        await self.push_frame(frame, direction)


def zero_verdict(
    paired_ones: Sequence[PairedTurn], *, stop_secs: float
) -> dict[str, Any]:
    """Does PoC1's zero fall on the real end of speech or on the VAD's confirmation?

    Unlike PoC2's `zero_verdict()`, which decides **by magnitude** of
    `end_of_turn_ms`, here the decision is made against the real instant the
    harness published. But the discriminant has a limit that must be declared,
    because it is exactly the one PoC2 overlooked for two runs:

    - A bias **below `stop_secs`** is **conclusive**: if the zero were on the
      confirmation, those 200 ms would already have elapsed and the bias could
      not go below that even with zero transport.
    - A bias **above** it concludes nothing by itself. The bias contains the
      audio path and the mismatch between the `end_of_speech` criterion (dB
      drop) and the VAD's (its own probability), and those two together can
      exceed 200 ms without the anchoring being wrong. That is exactly the
      mistake that led to reading PoC2's bias as evidence that LiveKit did not
      discount.
    """
    values = [
        e.speech_bias * 1000.0 for e in paired_ones if e.speech_bias is not None
    ]
    umbral_ms = stop_secs * 1000.0
    if not values:
        return {
            "verdict": ANCHOR_NO_DATA,
            "n_turns": 0,
            "umbral_ms": umbral_ms,
            "detail": (
                "ningún turno emparejó con una marca del arnés que trajera fin "
                "de voz medido: sin eso no hay contra qué decidir"
            ),
        }

    median_value = float(median(values))
    conclusive = median_value < umbral_ms
    return {
        "verdict": ANCHOR_REAL if conclusive else ANCHOR_INCONCLUSIVE,
        "n_turns": len(values),
        "speech_bias_median_ms": round(median_value, 1),
        "speech_bias_min_ms": round(min(values), 1),
        "umbral_ms": umbral_ms,
        "stop_secs": stop_secs,
        "detail": (
            "la mediana del sesgo NO alcanza los stop_secs del VAD: si el cero "
            "estuviera en la confirmación del silencio no podría bajar de ahí. "
            "El cero está en el fin de habla real y no hay que restar nada"
            if conclusive
            else "la mediana del sesgo supera los stop_secs del VAD, y eso solo "
            "NO decide: el sesgo lleva dentro el camino del audio y el "
            "desajuste entre el criterio de end_of_speech y el del VAD. Hay que "
            "acotar esos dos antes de concluir nada sobre el anclaje"
        ),
    }


def lines_block(
    marks: Sequence[ZeroMark],
    zeros: Sequence[EstimatedZero],
    *,
    first_speech: Mapping[str, float] | None = None,
    pipeline_samples: int | None = None,
) -> dict[str, Any]:
    """The run counted per line, not per pipeline turn.

    It is the comparison's unit of analysis (§2.2.5 of the contract) and is
    built the same way in all 3 PoCs, because the grouping lives in
    `shared/carrier_simulator/cruce.py`. What changes between architectures —and
    is exactly what we want to measure— is **how many** turns each pipeline
    closes within the same line.
    """
    lines, loose_ones = group(marks, zeros, first_speech=first_speech)
    rows: list[dict[str, Any]] = []
    for i in lines:
        cero = i.pipeline_zero
        rows.append(
            {
                "indice": i.indice,
                "turn": i.turn,
                "clip_id": i.clip_id,
                "end_of_speech": round(i.end_of_speech, 6),
                "agent_first_speech": (
                    None
                    if i.agent_first_speech is None
                    else round(i.agent_first_speech, 6)
                ),
                "line_latency_ms": _ms(i.latency),
                "pipeline_events": i.fragmentation,
                "pipeline_zero": None if cero is None else cero.end,
                "speech_bias_ms": (
                    None
                    if cero is None or cero.end is None
                    else _ms(cero.end - i.end_of_speech)
                ),
            }
        )

    data = lines_summary(
        lines,
        loose_ones=len(loose_ones),
        pipeline_samples=pipeline_samples,
    )
    data["per_line"] = rows
    return data


def latency_samples(json_path: Path | str) -> int | None:
    """How many latency samples the pipeline published in that call.

    It is read from the JSON already written instead of counted live because
    **what matters is what was saved**: it is the list the percentiles are
    computed over, and counting from another source could give a number nobody
    can verify by opening the file.
    """
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    samples = data.get("latencies")
    if not isinstance(samples, list):
        return None
    # The opening has no line that causes it: the agent greets on its own.
    return len([m for m in samples if not (isinstance(m, dict) and m.get("is_opening"))])


def harness_block(
    ruta: Path | str,
    zeros: Sequence[EstimatedZero],
    *,
    stop_secs: float,
    pipeline_host: str | None = None,
    first_speech: Mapping[str, float] | None = None,
    pipeline_samples: int | None = None,
    audio_outputs: Sequence[float] | None = None,
    audible_outputs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """What the diagnostics write out: the match, the biases and the verdict.

    Same match and same two biases as `livekit_extra.harness` in PoC2, because
    the pairing lives in `shared/carrier_simulator/cruce.py`. What is **not**
    compared head to head are the numbers: PoC2 pays for its `AudioSource`
    queue, the Opus encoder and the SFU, and PoC1 pays for none of that.
    Declared in `docs/METRICS_CONTRACT.md` §2.2.2.
    """
    marks = marks_de(ruta)
    paired_ones = pair(marks, list(zeros))
    with_mark = [e for e in paired_ones if e.mark is not None and e.cero is not None]

    # A **negative** bias would say the end of speech was declared before the
    # speech ended, and that cannot happen: it is not a bad measurement of the
    # zero, it means the end of speech at the end of the clip is missing and
    # the last one in the window belongs to an earlier streak. They are set
    # aside from the summary —inside it they would drag the p50 anywhere— and
    # counted.
    #
    # **The cause is not established, and that matters.** The first time they
    # appeared, the cause was a bug in this module: `_new` deduplicated by
    # `id(frame)`, CPython recycles ids, and it swallowed entire ends of speech.
    # With `frame_key` they disappeared across the 40 measured turns. If they
    # show up again, the hypothesis that the VAD lost the end of the clip has
    # to be raised again from scratch: `speech_ends` keeps every one that
    # arrived, which is the evidence that tells one thing from the other.
    measurable = [e for e in with_mark if (e.speech_bias or 0.0) >= 0.0]
    lost = [e for e in with_mark if (e.speech_bias or 0.0) < 0.0]

    harness_host = next((m.host for m in marks if m.host), None)
    host_pipeline = pipeline_host or socket.gethostname()
    same = None if harness_host is None else harness_host == host_pipeline

    def _summary(values: list[float]) -> dict[str, Any]:
        if same is not True:
            return {"n": 0, "note": "no se resume: los relojes no son el mismo"}
        return _describe(values)

    of_voice = [e.speech_bias * 1000.0 for e in measurable if e.speech_bias is not None]
    of_clip = [
        e.clip_bias * 1000.0 for e in measurable if e.clip_bias is not None
    ]
    criteria = sorted(
        {e.mark.drop_db for e in measurable if e.mark and e.mark.drop_db is not None}
    )

    return {
        "protocolo": ZERO_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "mark_path": "archivo en disco, fuera de banda",
        "what_it_is": (
            "los dos ceros reales del turno, medidos por quien reprodujo el clip, "
            "contra el que el VAD de pipecat estimó. Diagnóstico del arnés: no "
            "entra en `latencias` ni cambia ninguna muestra"
        ),
        "cobertura": {
            **coverage(ruta),
            "vad_turns": len(zeros),
            "paired_turns": len(with_mark),
            "measurable_turns": len(measurable),
            "turns_without_mark": [e.indice for e in paired_ones if e.mark is None],
        },
        "turns_with_lost_end": {
            "n": len(lost),
            "clips": [e.mark.clip_id for e in lost if e.mark],
            "biases_ms": [_ms(e.speech_bias) for e in lost],
            "what_it_means": (
                "el sesgo salió NEGATIVO: el fin de habla estimado cae antes de "
                "la última voz del clip, lo que como latencia es imposible. Falta "
                "el fin de habla del final del clip y el último de la ventana es "
                "de una racha anterior. NO entra al resumen del sesgo — mezclarlo "
                "movería el p50 sin medir nada"
            ),
            "before_concluding": (
                "la causa NO está establecida. La primera vez que aparecieron "
                "fue un bug del arnés (dedup por id(frame), que CPython recicla), "
                "no del VAD, y con el arreglo desaparecieron en 40 turnos. Si "
                "vuelven, contrástalos contra los fines de habla crudos antes de "
                "leerlos como que el VAD pierde finales de clip"
            ),
        },
        "clock": {
            "host_of_harness": harness_host,
            "pipeline_host": host_pipeline,
            "same_host": same,
            "interpretable_bias": same is True,
            "note": (
                "los dos procesos comparten `time.time()`, así que la resta vale"
                if same is True
                else "el arnés y el pipeline NO corrieron en la misma máquina: la "
                "diferencia lleva dentro el desfase entre relojes y no se puede "
                "leer como sesgo"
            ),
        },
        "speech_bias_ms": _summary(of_voice),
        "clip_bias_ms": _summary(of_clip),
        "end_of_speech_criterion": {
            "drop_db": criteria[0] if len(criteria) == 1 else criteria,
            "single": len(criteria) <= 1,
        },
        "zero_verdict": zero_verdict(measurable, stop_secs=stop_secs),
        # The **raw** zeros, all of them, not only the paired ones. Without them
        # it is impossible to regroup by line afterwards: the extra turns the
        # VAD closed are precisely the ones without a mark, and they are the
        # fragmentation data (§2.2.5).
        "zeros_of_pipeline": [
            {"indice": c.indice, "start": c.start, "end": c.end} for c in zeros
        ],
        # The left end of the return path: when the pipeline released the first
        # frame of each response. The right end is measured by the harness when
        # it hears it. See `AudioOutputProbe` and `shared/metrics/return_path.py`.
        "audio_outputs": list(audio_outputs or []),
        # The same, counting only from when the frame carries speech. The
        # difference against the ones above is the TTS onset, which is not
        # transport.
        "audible_outputs": list(audible_outputs or []),
        "lines": lines_block(
            marks,
            list(zeros),
            first_speech=first_speech,
            pipeline_samples=pipeline_samples,
        ),
        "not_directly_comparable_with_poc2": (
            "PoC2 paga la cola de su AudioSource, el encoder Opus y el SFU antes "
            "de que su VAD oiga el audio; PoC1 no paga nada de eso. Los dos "
            "sesgos están bien anclados y aun así no se restan entre sí. Ver "
            "docs/METRICS_CONTRACT.md §2.2.2"
        ),
        "per_turn": [_turn_row(e) for e in paired_ones],
    }


def _turn_row(turn: PairedTurn) -> dict[str, Any]:
    data: dict[str, Any] = {"indice": turn.indice}
    if turn.mark is None or turn.cero is None:
        data["paired"] = False
        return data
    mark = turn.mark
    data.update(
        {
            "paired": True,
            "harness_turn": mark.turn,
            "clip_id": mark.clip_id,
            "estimated_speech_end": turn.cero.end,
            "measured_end_of_speech": mark.measured_end_of_speech,
            "speech_bias_ms": _ms(turn.speech_bias),
            "drop_db": mark.drop_db,
            "speech_threshold_dbfs": mark.speech_threshold_dbfs,
            "clip_end": mark.clip_end,
            "clip_bias_ms": _ms(turn.clip_bias),
            "cola_ms": mark.cola_ms,
            "start_of_clip": mark.start_of_clip,
            "estimated_speech_start": turn.cero.start,
            "onset_bias_ms": _ms(turn.onset_bias),
        }
    )
    return data


def _describe(values: Sequence[float]) -> dict[str, Any]:
    """Five-number summary. Rounded to one decimal on purpose: the noise floor
    of this measurement is the event loop's jitter, not the bit."""
    v = sorted(values)
    n = len(v)
    if not n:
        return {"n": 0}

    def _quartile(p: float) -> float:
        if n == 1:
            return v[0]
        pos = p * (n - 1)
        low = int(pos)
        alto = min(low + 1, n - 1)
        return v[low] + (v[alto] - v[low]) * (pos - low)

    return {
        "n": n,
        "min": round(v[0], 1),
        "q1": round(_quartile(0.25), 1),
        "median_value": round(_quartile(0.5), 1),
        "q3": round(_quartile(0.75), 1),
        "max": round(v[-1], 1),
        "mean": round(sum(v) / n, 1),
    }


__all__ = [
    "ANCHOR_INCONCLUSIVE",
    "ANCHOR_REAL",
    "ANCHOR_NO_DATA",
    "EXTENSION",
    "PREFIX",
    "DIAGNOSTICS_PATH",
    "DiskMarkEmitter",
    "ZeroProbe",
    "harness_block",
    "cobertura",
    "as_json",
    "read_zeros",
    "marks_de",
    "zeros_path",
    "zero_verdict",
]
