"""Cross-match the harness's real zero with the one the worker estimated.

The harness knows **for certain** when it released the turn's last audio with speech: it
pushed it itself, and timestamps it at the send instant (§1 of the contract). The worker
does not know it; it *reconstructs* it from its VAD. This module subtracts one from the
other and calls that difference **bias**.

## Why the pairing lives here and not in each PoC

Pairing is the only part of the cross-match where a different decision produces different
numbers that still look reasonable. If PoC1 matched by index and PoC2 by
proximity, both would publish a `speech_bias_ms` with the same name measured in
ways that do not compare. Like the rest of the harness: the rule lives once.

## Why by proximity and not by turn number

The harness numbers its clips and the worker numbers its samples, and the two counts do
not match: the agent's greeting is a sample without a clip, and a clip that does not
trigger an answer produces no sample. Pairing by index would silently misalign them
as soon as either of those happens, and a bias computed against
the wrong turn looks like data.

## The TWO biases

- **`clip_bias`**: against `clip_end`, the instant the file's last
  sample went out. It is a physical fact and serves as a **control**.
- **`speech_bias`**: against `measured_end_of_speech`, where speech stopped inside
  the clip. It is the one that means something, and it travels with its `drop_db` because the
  same audio measures up to 9 dB differently depending on the threshold.

Publishing only the file one was the first design and it produced impossible
negative biases: a clip drags up to 1.3 s of breathing and room tail after
the last syllable. The difference between the two **is** the tail, not latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shared.carrier_simulator.cero import ZeroMark

# Window within which a mark and a worker zero are considered the same
# turn. Turns are separated by the agent's whole answer (several
# seconds), and the bias being sought is tens or hundreds of milliseconds:
# 2 s lets any real bias through and is not enough to confuse turns.
PAIRING_TOLERANCE_S: float = 2.0

# How long after the last clip a pipeline turn is still attributed to the
# last line. It is generous on purpose: the VAD can close the last
# turn several seconds late if the agent talked over it, and losing that event
# would show less fragmentation than there was. Whatever falls beyond it has no
# line to explain it and is reported as loose.
LAST_LINE_TAIL_S: float = 15.0


@dataclass
class EstimatedZero:
    """What the worker believed about the turn, in wall clock.

    `start` comes from the **first** anchored fragment of the turn and `end` from the
    **last**: a line can arrive split into several pieces, and taking
    the same one for both ends would measure the last piece instead of the turn.

    In wall clock **on purpose**, and not in the monotonic clock with which each PoC
    measures its stages: the harness mark comes from another process, and `time.time()`
    is the only clock both share. Whether sharing it holds is verified
    separately, through `host`.
    """

    indice: int
    start: float | None = None
    end: float | None = None


@dataclass
class PairedTurn:
    """A turn with its mark, or the record that it has none."""

    indice: int
    mark: ZeroMark | None = None
    cero: EstimatedZero | None = None
    distance_s: float | None = None

    @property
    def clip_bias(self) -> float | None:
        """Against the end of the **file**. Control, not the main measurement."""
        if self.mark is None or self.cero is None or self.cero.end is None:
            return None
        return self.cero.end - self.mark.clip_end

    @property
    def speech_bias(self) -> float | None:
        """Against the clip's **last speech**. It is the one that means something."""
        if self.mark is None or self.cero is None or self.cero.end is None:
            return None
        if self.mark.measured_end_of_speech is None:
            return None
        return self.cero.end - self.mark.measured_end_of_speech

    @property
    def onset_bias(self) -> float | None:
        if self.mark is None or self.cero is None or self.cero.start is None:
            return None
        return self.cero.start - self.mark.start_of_clip


def pair(
    marks: Sequence[ZeroMark],
    zeros: Sequence[EstimatedZero],
    *,
    tolerance_s: float = PAIRING_TOLERANCE_S,
) -> list[PairedTurn]:
    """Matches each worker turn with the harness mark that corresponds to it.

    Assignment goes from the closest pair to the farthest, each mark at most
    once, and only within `tolerance_s`. Whatever does not match is left out and
    reported: a turn without a zero is better than an invented zero.
    """
    candidate_ones: list[tuple[float, int, int]] = []
    for i, cero in enumerate(zeros):
        if cero.end is None:
            continue
        for j, mark in enumerate(marks):
            distance = abs(cero.end - mark.clip_end)
            if distance <= tolerance_s:
                candidate_ones.append((distance, i, j))
    # Tie-breaking by index keeps the result deterministic.
    candidate_ones.sort()

    from_zero: dict[int, tuple[int, float]] = {}
    used_marks: set[int] = set()
    for distance, i, j in candidate_ones:
        if i in from_zero or j in used_marks:
            continue
        from_zero[i] = (j, distance)
        used_marks.add(j)

    paired_ones: list[PairedTurn] = []
    for i, cero in enumerate(zeros):
        if i in from_zero:
            j, distance = from_zero[i]
            paired_ones.append(
                PairedTurn(
                    indice=cero.indice,
                    mark=marks[j],
                    cero=cero,
                    distance_s=round(distance, 6),
                )
            )
        else:
            paired_ones.append(PairedTurn(indice=cero.indice, cero=cero))
    return paired_ones


# --------------------------------------------------------------------------
# The line: the unit of analysis
# --------------------------------------------------------------------------
#
# A **pipeline turn** is what the framework's VAD decided was a
# turn. A **line** is a script line released onto the wire. They are not the
# same, and the harness is the only one that knows how many lines there were, because it
# released them.
#
# The difference is neither small nor even across architectures: in PoC1 one
# line can yield two latency samples or more. With `stop_secs = 0.2`
# pipecat's VAD cuts at any mid-sentence breath; with
# `min_silence_duration = 0.55` LiveKit's does not. Counting per
# pipeline turn measures, without saying so, how finely each VAD cuts.
#
# That is why the unit is the line, and the pipeline turns are **grouped**
# inside it instead of being compared one to one. See `docs/METRICS_CONTRACT.md`
# §2.2.5.


@dataclass
class Line:
    """A script line released onto the wire, with what the pipeline made of it.

    `events` are **all** the turns the pipeline closed within its
    window, not the best-matching one: if the VAD split the line in three, all
    three belong to this line and their number is the data.
    """

    indice: int
    mark: ZeroMark
    events: list[EstimatedZero] = field(default_factory=list)
    # When the agent's first audio was heard after this line, in
    # wall clock. The harness measures it on the track it receives, so it
    # exists the same way in the 3 PoCs and does not depend on the worker publishing it.
    agent_first_speech: float | None = None

    @property
    def turn(self) -> str:
        return self.mark.turn

    @property
    def clip_id(self) -> str | None:
        return self.mark.clip_id

    @property
    def end_of_speech(self) -> float:
        """The line's zero: the clip's last real speech.

        Falls back to the end of the file only if the clip did not carry its measurement, which is the
        case of a loose `--clip` outside the manifest.
        """
        if self.mark.measured_end_of_speech is not None:
            return self.mark.measured_end_of_speech
        return self.mark.clip_end

    @property
    def fragmentation(self) -> int:
        """How many turns the pipeline closed within this line."""
        return len(self.events)

    @property
    def pipeline_zero(self) -> EstimatedZero | None:
        """The **last** event: the one that really closed the line.

        The earlier ones are mid-sentence cuts. Taking the first would measure the
        clip's first pause and would give a huge bias that looks like a finding.
        """
        with_end = [e for e in self.events if e.end is not None]
        return with_end[-1] if with_end else None

    @property
    def latency(self) -> float | None:
        """From my end of speech to the agent's first audio, in seconds.

        **It is the comparison's measure**, and it is not built by adding up stages: it goes from the
        instant the harness stopped speaking to the instant the agent was heard,
        regardless of how many turns the VAD split the stretch in
        between into. That is why it survives one framework fragmenting and another not.

        It includes the return path (agent → harness), which is not the same in the
        3 PoCs. That is deliberate: it is the latency that whoever is on the other
        end of the phone would hear, not the one the worker measures internally.
        """
        if self.agent_first_speech is None:
            return None
        return self.agent_first_speech - self.end_of_speech


def group(
    marks: Sequence[ZeroMark],
    zeros: Sequence[EstimatedZero],
    *,
    first_speech: Mapping[str, float] | None = None,
    cola_seg: float = LAST_LINE_TAIL_S,
) -> tuple[list[Line], list[EstimatedZero]]:
    """Distributes the pipeline turns among the lines that caused them.

    Returns `(lines, loose_ones)`. An event belongs to the **last**
    line that started before it: the script's turns are separated
    by the agent's whole answer, so that boundary has no
    ambiguity. The loose ones are those that fall before the first clip (the agent's
    greeting produces some) or more than `cola_seg` after the last one.

    It is the same idea as `pair`, by proximity and not by index, with the
    difference that matters: here an event does not compete with another for its
    line. Several events in the same window **are** the data.
    """
    first_speech = first_speech or {}
    ordered = sorted(marks, key=lambda m: m.start_of_clip)
    lines = [
        Line(
            indice=i,
            mark=mark,
            agent_first_speech=first_speech.get(mark.turn),
        )
        for i, mark in enumerate(ordered)
    ]
    if not lines:
        return [], list(zeros)

    last_end = ordered[-1].clip_end + cola_seg
    loose_ones: list[EstimatedZero] = []
    for cero in zeros:
        if cero.end is None:
            loose_ones.append(cero)
            continue
        chosen: Line | None = None
        for line in lines:
            if cero.end >= line.mark.start_of_clip:
                chosen = line
            else:
                break
        if chosen is None or cero.end > last_end:
            loose_ones.append(cero)
        else:
            chosen.events.append(cero)
    return lines, loose_ones


def lines_summary(
    lines: Sequence[Line],
    *,
    loose_ones: int = 0,
    pipeline_samples: int | None = None,
) -> dict[str, Any]:
    """Per-line latency and the fragmentation rate.

    `pipeline_samples` is how many latency samples the worker published in
    that call. It is kept separate from the VAD events because they are two different
    counts of the same phenomenon (the worker can open a sample without closing a turn
    and vice versa) and the one that goes into the percentiles is the worker's.
    """
    with_latency = [i for i in lines if i.latency is not None]
    latencies = sorted(i.latency * 1000.0 for i in with_latency)  # type: ignore[misc]
    events = sum(i.fragmentation for i in lines)
    n = len(lines)

    # The denominator is the lines the pipeline **heard**, not the ones
    # the harness released. A clip played after the agent hung up does not
    # produce any turn, and counting it would dilute the fragmentation toward 1.0, that
    # is, it would make exactly the framework that cuts the most look more even. It really happens:
    # in scenario 1 the agent closes at T7 and T8 goes out against nobody.
    heard_values = [i for i in lines if i.fragmentation]
    n_heard = len(heard_values)

    data: dict[str, Any] = {
        "what_it_is": (
            "la unidad de análisis es la INTERVENCIÓN —una línea del guion "
            "soltada al cable—, no el turno que el VAD de cada framework "
            "decidió. Ver docs/METRICS_CONTRACT.md §2.2.5"
        ),
        "lines": n,
        "heard_lines": n_heard,
        "with_latency": len(with_latency),
        "pipeline_events": events,
        "events_without_line": loose_ones,
        "fragmentation": round(events / n_heard, 2) if n_heard else None,
        "how_to_read_fragmentation": (
            "turnos que el pipeline cerró por cada intervención que oyó. 1.0 es "
            "un turno por intervención; 1.6 quiere decir que el VAD partió más "
            "de la mitad en dos. NO es un defecto de captura: es cuán finito "
            "corta el VAD de cada framework con su perilla congelada, y por eso "
            "es una métrica comparativa por derecho propio"
        ),
    }
    if pipeline_samples is not None:
        data["pipeline_latency_samples"] = pipeline_samples
        data["sample_fragmentation"] = (
            round(pipeline_samples / n_heard, 2) if n_heard else None
        )
    # The per-line detail, with the **absolute** instants. Without this the
    # block only gives percentiles, and the return path (which is the first audio
    # heard minus the first audio released by the pipeline) cannot be computed
    # afterwards: the call would have to be run again. See
    # `shared/metrics/return_path.py`.
    data["per_line"] = [
        {
            "turn": i.turn,
            "clip_id": i.clip_id,
            "end_of_speech": i.end_of_speech,
            "agent_first_speech": i.agent_first_speech,
            "fragmentation": i.fragmentation,
            "latency_ms": ms(i.latency),
        }
        for i in lines
    ]
    if latencies:
        data["line_latency_ms"] = {
            "n": len(latencies),
            "min": round(latencies[0], 1),
            "p50": round(latencies[len(latencies) // 2], 1),
            "max": round(latencies[-1], 1),
            "what_it_measures": (
                "del fin de voz del arnés al primer audio del agente. Incluye "
                "el camino de vuelta, que no es el mismo en las 3 PoC"
            ),
        }
    return data


def ms(seconds: float | None) -> float | None:
    """Seconds to milliseconds with one decimal, preserving `None`.

    The rounding is not cosmetic: the interpolated quartiles come out with a floating-point
    tail, and `118.0000000003929` in a JSON that someone will read by eye
    suggests a precision the measurement does not have.
    """
    return None if seconds is None else round(seconds * 1000.0, 1)
