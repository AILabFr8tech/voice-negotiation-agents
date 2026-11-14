"""Latency: the per-turn sample contract and automatic percentiles.

This module **does not measure**: it computes. Capture lives in each PoC (it is
the only thing that changes between frameworks); what is fixed here is the shape
of the sample and the arithmetic, so that the percentiles of the 3
architectures mean the same thing.

## What is separated and why

Three buckets, never mixed:

- **opening** (`apertura`) — the first turn of the call. It pays for opening
  the TTS WebSocket and for the first LLM request with the full system prompt.
  Putting it in the same pile as the rest raises the p50 of the whole call by a
  cost that is paid only once.
- **steady state** (`regimen`) — turns 2..n. It is the number that describes
  how the conversation feels and the one that decides the comparison.
- **interrupted** — turns with barge-in. Latency there measures something else
  (how fast the system reacts to being talked over), so it contaminates the
  steady-state p95 if averaged together.

## The per-stage breakdown

In PoC1 and PoC2 the full turn is broken down into end-of-turn detection, STT,
LLM TTFT and TTS TTFB. In PoC3 (GPT-Live) there are no stages to isolate: only
the end-to-end figure exists. Absent stages are omitted from the summary
instead of appearing as nulls, so a PoC3 report does not drag empty columns
along.

### The stages do not happen in the order they are read

Anchoring `stt_ms` at the end of turn would assume this order:

    end of speech -> end of turn -> transcript -> first token -> first audio

In pipecat 1.6 the real order is different, because the end-of-turn analyzer
**waits for the transcript** before declaring the turn closed:

    end of speech -> transcript -> end of turn -> first token -> first audio

With that anchoring, `stt_ms` comes out negative on every turn, and its
magnitude is really the analyzer's inference time. So each stage is anchored at
the event that truly precedes it:

- `stt_ms` hangs off the end of speech — how long the final text takes from
  the moment the person stopped talking.
- `llm_ttft_ms` hangs off the end of turn, which is when the LLM actually
  starts. If there was no end of turn, it falls back to the transcript.

Any delta that still comes out negative is reported as `None` with the reason
in `notes`, never as a number: a negative is an out-of-order mark, and putting
it into a percentile silently corrupts it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Iterable, Sequence

# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------

BUCKET_OPENING: Final[str] = "apertura"
BUCKET_REGIMEN: Final[str] = "regimen"
BUCKET_INTERRUPTED: Final[str] = "interrupted"

BUCKETS: Final[tuple[str, ...]] = (
    BUCKET_OPENING,
    BUCKET_REGIMEN,
    BUCKET_INTERRUPTED,
)

# Main field: end of the carrier's speech -> start of the agent's audio. It is
# the latency metric of the comparison; the stages explain where it comes
# from.
CAMPO_PRINCIPAL: Final[str] = "full_turn_ms"

# Breakdown stages, in the pipeline's real order.
STAGE_FIELDS: Final[tuple[str, ...]] = (
    "end_of_turn_ms",
    "stt_ms",
    "llm_ttft_ms",
    "tts_ttfb_ms",
)

# Percentiles that are always reported. The quartiles are included because they
# describe the shape of the distribution (if p25 and p75 are close together,
# latency is stable; if they are far apart, there are two hidden regimes and the
# p50 just lies less).
PERCENTILES: Final[tuple[float, ...]] = (25.0, 50.0, 75.0, 95.0, 99.0)

# Below these sample sizes, a high percentile is "the maximum in disguise":
# with 8 samples the p99 IS the maximum. The number is not hidden; it is marked
# as unreliable so nobody cites it as if it were a percentile.
MIN_RELIABLE: Final[dict[float, int]] = {95.0: 20, 99.0: 100}


def classify(sample: TurnSample) -> str:
    """Which bucket a sample goes into.

    An interrupted turn goes to `interrupted` **even if it is the first one**:
    the interruption dominates what is being measured. It is an arbitrary but
    fixed decision, and that is why it is documented — with ~8 turns per call,
    moving one sample between buckets visibly changes percentiles.
    """
    if sample.interrupted:
        return BUCKET_INTERRUPTED
    if sample.is_opening:
        return BUCKET_OPENING
    return BUCKET_REGIMEN


@dataclass(frozen=True)
class TurnSample:
    """Latency of one turn. **This is the contract that capture must fill.**

    Everything in milliseconds. `None` means "not measured" (not zero): a silent
    zero would ruin the p50, while a None is excluded from the computation and
    reported as incomplete coverage.

    - `indice`: turn number within the call, starting at 0.
    - `full_turn_ms`: end of the carrier's speech -> first agent audio going out
      to the transport. It is the main metric.
    - `end_of_turn_ms`: how long the analyzer took to declare end of turn.
    - `stt_ms`: until the final transcript of the carrier's turn.
    - `llm_ttft_ms`: until the LLM's first token. **It does not mean the same in
      PoC1 as in PoC2** and the two are not compared without splitting by mode:
      see `docs/METRICS_CONTRACT.md` §2.7.1.
    - `tts_ttfb_ms`: until the TTS's first audio byte.
    - `interrupted`: there was barge-in on this turn — the carrier talked over
      the agent. It is **not** "the agent cut the carrier off": that is a false
      end of turn and is tracked with `end_of_turn_verdicts`.
    - `is_opening`: cold-start turn. If not given, it is derived from
      `indice == 0`.
    - `end_of_turn_verdicts`: probabilities the end-of-turn detector emitted
      during this turn, in order. The last one is the one that closed the turn.
      **Only PoC1 fills it.** LiveKit's audio turn detector does not expose the
      probability to code and PoC3's detection is internal to GPT-Live, so in
      both the tuple stays empty — it is an asymmetry documented in
      `docs/METRICS_CONTRACT.md`, not a gap. It enters no percentile: it is
      evidence for calibration, not a metric.
    - `notes`: why a stage was left as `None` when a number was expected.
    """

    indice: int
    full_turn_ms: float | None = None
    end_of_turn_ms: float | None = None
    stt_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfb_ms: float | None = None
    interrupted: bool = False
    is_opening: bool | None = None
    end_of_turn_verdicts: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.is_opening is None:
            object.__setattr__(self, "is_opening", self.indice == 0)
        # They arrive from the JSON as lists; they are frozen so the sample stays
        # hashable and cannot be mutated by accident when aggregated.
        for campo in ("end_of_turn_verdicts", "notes"):
            valor = getattr(self, campo)
            if not isinstance(valor, tuple):
                object.__setattr__(self, campo, tuple(valor or ()))

    @property
    def bucket(self) -> str:
        return classify(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnSample:
        """Rebuild a sample from the JSON, ignoring unknown keys."""
        fields = {
            "indice",
            CAMPO_PRINCIPAL,
            *STAGE_FIELDS,
            "interrupted",
            "is_opening",
            "end_of_turn_verdicts",
            "notes",
        }
        cleaned = {k: v for k, v in data.items() if k in fields}
        cleaned.setdefault("indice", 0)
        return cls(**cleaned)

    @classmethod
    def from_events(
        cls,
        indice: int,
        *,
        t_speech_end: float | None = None,
        t_end_of_turn: float | None = None,
        t_transcript: float | None = None,
        t_primer_token: float | None = None,
        t_primer_audio: float | None = None,
        interrupted: bool = False,
        is_opening: bool | None = None,
        end_of_turn_verdicts: Sequence[float] = (),
        scale: float = 1000.0,
    ) -> TurnSample:
        """Build the sample from absolute timestamps.

        It exists so that capture does not reimplement the subtractions: it
        passes the instants it already has (in seconds, like `loop.time()`) and
        the millisecond deltas are computed here. Each stage is measured against
        the event that truly precedes it (see the order in the module
        docstring), so if a mark is missing the breakdown degrades instead of
        lying.

        A negative delta is never written: it means the marks arrived out of
        order and the anchoring does not describe that call. `None` is returned
        and the reason is left in `notes`.

        `scale` converts to milliseconds: 1000 if the instants come in seconds,
        1 if they already come in milliseconds.
        """
        notes: list[str] = []

        def delta(since: float | None, until: float | None, stage: str) -> float | None:
            if since is None or until is None:
                return None
            # Rounded to a tenth of a millisecond: subtracting two floats leaves
            # a binary tail ("2797.00000000048") and writing it to the JSON
            # suggests a precision the measurement does not have.
            valor = round((until - since) * scale, 1)
            if valor < 0:
                notes.append(f"{stage}: {valor} ms, marcas fuera de orden")
                return None
            return valor

        # The LLM starts at the end of turn, not at the transcript: between the
        # two lies the analyzer's decision, which is not LLM time.
        anchor_llm = t_end_of_turn if t_end_of_turn is not None else t_transcript
        if anchor_llm is None and (
            is_opening if is_opening is not None else indice == 0
        ):
            # The greeting has no end of turn or transcript before it: the LLM
            # could start as soon as the call connected, which is the zero of
            # the opening. That is exactly where the model's cold start lives,
            # so leaving it empty would lose the most useful data point of that
            # bucket. A steady-state turn deliberately does NOT fall back here:
            # anchoring at the end of speech would add the analyzer's wait to
            # the LLM.
            anchor_llm = t_speech_end

        return cls(
            indice=indice,
            full_turn_ms=delta(t_speech_end, t_primer_audio, "full_turn"),
            end_of_turn_ms=delta(t_speech_end, t_end_of_turn, "end_of_turn"),
            stt_ms=delta(t_speech_end, t_transcript, "stt"),
            llm_ttft_ms=delta(anchor_llm, t_primer_token, "llm_ttft"),
            tts_ttfb_ms=delta(t_primer_token, t_primer_audio, "tts_ttfb"),
            interrupted=interrupted,
            is_opening=is_opening,
            end_of_turn_verdicts=tuple(end_of_turn_verdicts),
            notes=tuple(notes),
        )


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def percentile_label(p: float) -> str:
    """Column name of a percentile: 50.0 -> "p50", 99.9 -> "p99.9"."""
    return f"p{int(p)}" if float(p).is_integer() else f"p{p}"


def percentile(values: Sequence[float], p: float) -> float | None:
    """Percentile `p` (0-100) by linear interpolation between order statistics.

    It is Hyndman-Fan method 7, the default of `numpy.percentile` and of
    `statistics.quantiles(method="inclusive")`. It is implemented by hand for
    two reasons: `shared/` does not depend on numpy (each PoC has its own
    environment), and keeping the method fixed and visible prevents a library
    change from moving the report's numbers without anyone noticing.

    Returns None for an empty list.
    """
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    ordered_items = sorted(float(v) for v in values)
    if len(ordered_items) == 1:
        return ordered_items[0]

    position = (len(ordered_items) - 1) * (p / 100.0)
    inferior = int(position)
    superior = min(inferior + 1, len(ordered_items) - 1)
    fraction = position - inferior
    return ordered_items[inferior] + fraction * (ordered_items[superior] - ordered_items[inferior])


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    """Statistics of a list of latencies, ready for JSON.

    It includes `reliable`, which marks which percentiles have enough sample to
    be cited. The number is reported anyway: hiding it would force recomputing
    it by hand, which is exactly what this module avoids.
    """
    cleaned_items = [float(v) for v in values if v is not None]
    summary: dict[str, Any] = {"n": len(cleaned_items)}
    if not cleaned_items:
        summary["reliable"] = {percentile_label(p): False for p in PERCENTILES}
        return summary

    summary["min"] = round(min(cleaned_items), 1)
    summary["max"] = round(max(cleaned_items), 1)
    summary["mean"] = round(_mean(cleaned_items) or 0.0, 1)
    for p in PERCENTILES:
        valor = percentile(cleaned_items, p)
        summary[percentile_label(p)] = round(valor, 1) if valor is not None else None
    summary["reliable"] = {
        percentile_label(p): len(cleaned_items) >= MIN_RELIABLE.get(p, 1)
        for p in PERCENTILES
    }
    return summary


def _values_of(samples: Iterable[TurnSample], campo: str) -> list[float]:
    return [
        float(getattr(m, campo))
        for m in samples
        if getattr(m, campo, None) is not None
    ]


def summarize_samples(
    samples: Sequence[TurnSample], *, calls: int = 1
) -> dict[str, Any]:
    """Full summary: per bucket, with stage breakdown and warnings.

    It is what gets written as-is into the per-call JSON and what the
    consolidation aggregates across calls.

    `calls` is how many calls were pooled into these samples. Only the warnings
    use it: with one call there must be exactly one opening turn, but when
    consolidating 15 calls there must be 15, and warning about that would be
    noise.
    """
    per_bucket: dict[str, list[TurnSample]] = {b: [] for b in BUCKETS}
    for m in samples:
        per_bucket[classify(m)].append(m)

    output: dict[str, Any] = {
        "n_turns": len(samples),
        "unidad": "ms",
        "buckets": {},
        "advertencias": [],
    }

    for bucket in BUCKETS:
        of_bucket = per_bucket[bucket]
        principal = _values_of(of_bucket, CAMPO_PRINCIPAL)
        block: dict[str, Any] = {
            "n_turns": len(of_bucket),
            "full_turn": summarize_values(principal),
        }

        stages: dict[str, Any] = {}
        for campo in STAGE_FIELDS:
            values = _values_of(of_bucket, campo)
            if values:
                stages[campo.removesuffix("_ms")] = summarize_values(values)
        # PoC3 has no stages: the key is omitted instead of filled with nulls.
        if stages:
            block["stages"] = stages

        absent = len(of_bucket) - len(principal)
        if absent:
            block["turns_without_measurement"] = absent

        output["buckets"][bucket] = block

    output["advertencias"] = _cautions(output, per_bucket, calls)
    return output


def _cautions(
    output: dict[str, Any],
    per_bucket: dict[str, list[TurnSample]],
    calls: int = 1,
) -> list[str]:
    """Warnings to read before citing a percentile."""
    warning_list: list[str] = []

    if output["n_turns"] == 0:
        warning_list.append(
            "sin muestras de latencia: el pipeline no las está registrando "
            "(instrumentación pendiente) o la llamada no tuvo turnos"
        )
        return warning_list

    regimen = output["buckets"][BUCKET_REGIMEN]["full_turn"]
    n_regimen = regimen.get("n", 0)
    if n_regimen == 0:
        warning_list.append("no hay turnos de régimen medidos: solo apertura o interrumpidos")
    else:
        for p, minimum in sorted(MIN_RELIABLE.items()):
            if n_regimen < minimum:
                warning_list.append(
                    f"p{int(p)} de régimen con n={n_regimen} (<{minimum}): es el "
                    "máximo disfrazado, no lo cites como percentil"
                )

    unmeasured = sum(
        output["buckets"][b].get("turns_without_measurement", 0) for b in BUCKETS
    )
    if unmeasured:
        warning_list.append(
            f"{unmeasured} turno(s) sin `{CAMPO_PRINCIPAL}`: cobertura de "
            "medición incompleta"
        )

    n_opening = len(per_bucket[BUCKET_OPENING])
    if n_opening > max(calls, 1):
        warning_list.append(
            f"{n_opening} turnos marcados como apertura para {calls} "
            "llamada(s): debería ser uno por llamada"
        )

    return warning_list


def load_samples(data: Any) -> list[TurnSample]:
    """Read the sample list from the per-call JSON. Tolerates its absence."""
    if not isinstance(data, list):
        return []
    samples: list[TurnSample] = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            continue
        raw = {**raw}
        raw.setdefault("indice", i)
        samples.append(TurnSample.from_dict(raw))
    return samples
