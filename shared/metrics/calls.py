"""Metrics of one call: joins the run's JSON with the reference file.

It takes what the pipeline wrote (transcript and latency samples), adds the
hand-corrected reference, and produces a metrics block that is written **back
into the same JSON**. That way the per-call JSON is self-contained: whoever
opens it sees the percentiles and the WER without recomputing anything.

## What is measured and what is not

Only what distinguishes one voice technology from another: latency per stage,
STT quality and the turn-taking ratings. What the agent says is the vehicle of
the conversation, not the object of study, so no metric about the content of
what was negotiated comes out of here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from shared.metrics import latency as lat
from shared.metrics import reference as ref
from shared.metrics import wer as wer_mod
from shared.metrics.text import CONFIG_DEFAULT, NormalizationConfig

# Key where the pipeline must leave the latency samples, and where this module
# writes the computed block.
SAMPLES_KEY: Final[str] = "latencies"
METRICS_KEY: Final[str] = "metrics"

# Optional key: a PoC can mark that its call must not enter the percentile pool,
# even though the rest of its metrics are usable. The case that motivated it is
# PoC2's turn detector, which can fall back from its remote model to the local
# one **mid-call**: that run mixes two turn-taking architectures and putting it
# in the same pile would contaminate the p50 of the others.
#
# The call is NOT discarded: it is still reported in the detail and its WER
# still counts. The only thing taken away is its vote in the aggregated
# percentiles. No PoC is required to write it; if absent, everything works as
# before.
EXCLUDE_POOLED_KEY: Final[str] = "exclude_from_pooled"

# Round trips to an auxiliary LLM that the PoC was able to isolate, as a list of
# `{"ms": float, "with_function": bool}`. Optional: today only PoC3 fills it,
# where the `gpt-live` model delegates some turns to `gpt-4.1-mini`.
#
# **It is not `llm_ttft_ms`, and it is not mixed with it.** In PoC1 and PoC2 the
# LLM is what produces the answer and its TTFT falls inside `full_turn_ms`. In
# PoC3 the delegate is a secondary brain the voice model consults while it
# **keeps talking**, so its cost is covered by the filler and does NOT appear in
# the turn latency. They are two different magnitudes; putting them in the same
# column would compare apples with oranges across PoCs.
LLM_ROUND_TRIPS_KEY: Final[str] = "llm_round_trips"

# Above this WER, the most likely explanation is not that the STT fails that
# much, but that the reference does not belong to this call: it was generated
# with the wrong scenario, or it was left with the seeded script uncorrected
# while the call went elsewhere. Without this warning, a 97% reads as an STT
# result.
_WER_IMPLAUSIBLE: Final[float] = 0.5


@dataclass
class CallMetrics:
    """Everything computable about a call, ready for JSON and for consolidation."""

    call_sid: str = ""
    poc: int | None = None
    scenario: int | None = None
    repetition: int | None = None
    duration_s: float | None = None
    latency: dict[str, Any] = field(default_factory=dict)
    stt: dict[str, Any] | None = None
    manual_notes: dict[str, Any] | None = None
    warning_list: list[str] = field(default_factory=list)
    # Reason why this call does not enter the percentile pool, or None.
    # See `EXCLUDE_POOLED_KEY`.
    exclude_from_pooled: dict[str, Any] | None = None
    # See `LLM_ROUND_TRIPS_KEY`. Empty in the PoCs that do not publish it.
    round_trips_llm: list[dict[str, Any]] = field(default_factory=list)

    # Rich objects, for consolidation. Not serialized.
    #
    # `_samples` keeps the raw samples and not just the summary because the
    # per-PoC consolidation has to **pool the values** of the 15 calls and take
    # the percentile of the whole pool. Averaging each call's p95 would give a
    # number with no statistical meaning.
    _samples: list[lat.TurnSample] = field(default_factory=list, repr=False)
    _text: wer_mod.TextResult | None = field(default=None, repr=False)
    _amounts: wer_mod.AmountsResult | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """Readable identifier: "2.3" = scenario 2, repetition 3."""
        if self.scenario is None:
            return self.call_sid or "?"
        rep = self.repetition if self.repetition is not None else "?"
        return f"{self.scenario}.{rep}"

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "call_sid": self.call_sid,
            "poc": self.poc,
            "scenario": self.scenario,
            "repetition": self.repetition,
            "duration_s": self.duration_s,
            "latency": self.latency,
            "warning_list": list(self.warning_list),
        }
        if self.stt is not None:
            output["stt"] = self.stt
        if self.manual_notes is not None:
            output["manual_notes"] = self.manual_notes
        if self.exclude_from_pooled is not None:
            output["exclude_from_pooled"] = self.exclude_from_pooled
        if self.round_trips_llm:
            output["round_trips_llm"] = [dict(r) for r in self.round_trips_llm]
        return output


# --------------------------------------------------------------------------
# Pieces of the computation
# --------------------------------------------------------------------------


def _turns_by_role(transcript: Any, role: str) -> list[str]:
    """What one of the two sides said, one element per turn, in order."""
    if not isinstance(transcript, list):
        return []
    return [
        text
        for row in transcript
        if isinstance(row, dict) and row.get("role") == role
        for text in [str(row.get("text", "")).strip()]
        if text
    ]


def _text_by_role(transcript: Any, role: str) -> str:
    """Join what one of the two sides said, in chronological order."""
    return " ".join(_turns_by_role(transcript, role))


def _duration(transcript: Any) -> float | None:
    """Second of the last line. Approximates the call's duration."""
    if not isinstance(transcript, list):
        return None
    timings: list[float] = []
    for row in transcript:
        if not isinstance(row, dict):
            continue
        try:
            timings.append(float(row.get("t")))
        except (TypeError, ValueError):
            continue
    return round(max(timings), 3) if timings else None


def identify(
    data: dict[str, Any], file: ref.ReferenceFile | None
) -> tuple[int | None, int | None, int | None]:
    """PoC, scenario and repetition.

    What the JSON carries is preferred (the machine wrote it), falling back to
    the reference file (the person wrote it). Today the JSON does not carry
    them: making the scenario selectable from the command line is an open
    to-do in PoC1, and when it arrives this order already respects it.
    """

    def of_json(clave: str) -> int | None:
        for container in (data, data.get("summary") or {}):
            if not isinstance(container, dict):
                continue
            if clave in container:
                try:
                    return int(container[clave])
                except (TypeError, ValueError):
                    continue
        return None

    poc = of_json("poc")
    scenario = of_json("scenario")
    repetition = of_json("repetition")

    if file is not None:
        poc = poc if poc is not None else file.integer("poc")
        scenario = scenario if scenario is not None else file.integer("scenario")
        repetition = (
            repetition if repetition is not None else file.integer("repetition")
        )

    return poc, scenario, repetition


def _stt_block(
    file: ref.ReferenceFile,
    carrier_text: str,
    config: NormalizationConfig,
) -> tuple[dict[str, Any], wer_mod.TextResult, wer_mod.AmountsResult]:
    """WER, CER and amount error of the carrier's channel.

    By default the hypothesis is the STT text already in the call's JSON, not
    the one the reference file carries in its `stt:` lines — those are a copy
    for reading, and if the person edited them by mistake they must not
    contaminate the measurement.

    **Unless the file declares `hypothesis: file`.** A file derived from the
    clips (`clips_reference.py`, §2.10) is not edited by hand and its `stt:`
    lines are not a copy: they carry the two exclusions —dead tail and
    pairing— distributed turn by turn. Taking the whole JSON there would bring
    back the text of the excluded turns as **insertions** and the WER would
    come out inflated.
    """
    reference_text = file.reference_text
    if file.metadata.get("hypothesis", "").strip() == ref.FILE_HYPOTHESIS:
        carrier_text = file.hypothesis_text
    text_result = wer_mod.compute_text(reference_text, carrier_text, config)
    amounts_result = wer_mod.compute_amounts(
        reference_text, carrier_text, config
    )

    per_turn = []
    for turn in ref.turns_with_text(file.turns):
        partial = wer_mod.compute_text(turn.reference, turn.hypothesis, config)
        per_turn.append(
            {
                "turn": turn.id,
                "wer": round(partial.wer.sample_rate, 4),
                "words_reference": partial.words_reference,
            }
        )

    block: dict[str, Any] = {
        "normalizacion": config.descripcion(),
        "full_channel": text_result.to_dict(),
        "amounts": amounts_result.to_dict(),
        # Indicative: the grouping by turns is heuristic (see `reference.py`).
        "per_turn_indicative": per_turn,
        "reference": reference_text,
        "hypothesis": carrier_text,
    }
    return block, text_result, amounts_result


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def _read_round_trips(raw_values: Any) -> list[dict[str, Any]]:
    """Normalize `llm_round_trips`, tolerating garbage without crashing.

    It is optional instrumentation: a malformed entry is ignored instead of
    bringing down the computation of a call that did cost money.
    """
    if not isinstance(raw_values, list):
        return []
    output: list[dict[str, Any]] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, dict):
            continue
        ms = raw_value.get("ms")
        if not isinstance(ms, (int, float)) or isinstance(ms, bool):
            continue
        output.append({"ms": float(ms), "with_function": bool(raw_value.get("with_function"))})
    return output


def compute(
    data: dict[str, Any],
    *,
    reference_file: ref.ReferenceFile | None = None,
    config: NormalizationConfig = CONFIG_DEFAULT,
) -> CallMetrics:
    """Compute all the metrics of an already loaded call.

    Nothing here touches disk or the network: it is a pure function over the
    run's JSON plus the reference. That is what allows validating it fully with
    synthetic data, without spending a call.
    """
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    transcript = data.get("transcript")

    poc, scenario, repetition = identify(data, reference_file)

    samples = lat.load_samples(data.get(SAMPLES_KEY))
    latency_block = lat.summarize_samples(samples)

    carrier_turns = _turns_by_role(transcript, ref.CARRIER_ROLE)
    carrier_text = " ".join(carrier_turns)
    agent_text = _text_by_role(transcript, ref.AGENT_ROLE)

    metrics = CallMetrics(
        call_sid=str(data.get("call_sid") or summary.get("call_id") or ""),
        poc=poc,
        scenario=scenario,
        repetition=repetition,
        duration_s=_duration(transcript),
        latency=latency_block,
        _samples=samples,
    )

    metrics.warning_list.extend(latency_block.get("advertencias") or [])

    metrics.round_trips_llm = _read_round_trips(data.get(LLM_ROUND_TRIPS_KEY))

    exclusion = data.get(EXCLUDE_POOLED_KEY)
    if isinstance(exclusion, dict) and exclusion:
        metrics.exclude_from_pooled = dict(exclusion)
        motivo = str(exclusion.get("detail") or exclusion.get("motivo") or "sin motivo")
        metrics.warning_list.append(
            f"fuera de la poza de percentiles: {motivo}. Sus latencias siguen en "
            "el detalle por llamada, pero no votan en los agregados"
        )

    if not agent_text:
        metrics.warning_list.append(
            "la transcripción no trae turnos del agente: sin las dos partes no se "
            "puede revisar el turn-taking"
        )

    if reference_file is None:
        metrics.warning_list.append(
            "sin archivo de referencia: no hay WER/CER (genera uno con "
            "`python -m shared.metrics.cli referencia`)"
        )
    else:
        reference_warnings = ref.validate(reference_file)
        metrics.warning_list.extend(reference_warnings)
        metrics.manual_notes = reference_file.notes.to_dict()
        if reference_file.reference_text.strip():
            block, text_result, amounts_result = _stt_block(
                reference_file, carrier_text, config
            )
            metrics.stt = block
            metrics._text = text_result
            metrics._amounts = amounts_result

            if text_result.wer.sample_rate > _WER_IMPLAUSIBLE:
                metrics.warning_list.append(
                    f"WER de {text_result.wer.sample_rate * 100:.0f}%: revisa que la "
                    "referencia sea de esta llamada y esté corregida — un WER así "
                    "casi siempre es referencia equivocada, no falla del STT"
                )

    return metrics


def load_json(ruta: Path | str) -> dict[str, Any]:
    """Read a call's JSON."""
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{ruta} does not contain a JSON object at the root")
    return data


def compute_from_path(
    ruta_json: Path | str,
    *,
    config: NormalizationConfig = CONFIG_DEFAULT,
    reference_path: Path | str | None = None,
) -> tuple[dict[str, Any], CallMetrics]:
    """Load the JSON and its reference (if any) and compute. Returns both."""
    destination = Path(ruta_json)
    data = load_json(destination)

    call_sid = str(data.get("call_sid") or "")
    if reference_path is not None:
        candidate_one: Path | None = Path(reference_path)
        if candidate_one is not None and not candidate_one.is_file():
            candidate_one = None
    else:
        candidate_one = ref.search(destination.parent, call_sid)

    file = ref.leer(candidate_one) if candidate_one else None
    return data, compute(data, reference_file=file, config=config)


def enrich(
    ruta_json: Path | str,
    *,
    config: NormalizationConfig = CONFIG_DEFAULT,
    reference_path: Path | str | None = None,
) -> CallMetrics:
    """Compute and write the `metrics` block back into the call's JSON.

    Idempotent: running it again recomputes and replaces the block. The rest of
    the JSON is left untouched.
    """
    destination = Path(ruta_json)
    data, metrics = compute_from_path(
        destination, config=config, reference_path=reference_path
    )
    data[METRICS_KEY] = metrics.to_dict()
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def list_calls(folder: Path | str) -> list[Path]:
    """The call JSONs of a folder, in stable order."""
    return sorted(Path(folder).glob("call_*.json"))
