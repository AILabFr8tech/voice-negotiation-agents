"""`progress.jsonl`: one line per finished call, and nothing more.

It is both the log followed live (`Get-Content -Wait progress.jsonl`) and the
memory for resuming. Those two things ask for the same thing: **append only**,
never rewritten, and each line is flushed to disk before the next call starts.

That yields the guarantee that matters: killing the process mid-call corrupts
nothing. The worst that can happen is that the in-flight call did not get to
write its line, and it is run again on resume. One extra call is paid for; the
previous ones are not lost, and no half-written line counts as a good one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shared.lote.plan import BatchItem

# Statuses of a finished call.
OK: str = "ok"
# It ran and the harness marked it red: failed criteria or too few turns.
# That is campaign data, not a batch error.
ROJA: str = "roja"
# It could not be run, or it crashed, or it left no summary.
FAILURE: str = "failure"


def read_progress(ruta: Path) -> tuple[list[dict[str, Any]], int]:
    """The lines already written and how many were discarded as unreadable.

    A broken line can only be the last one, and only if the process died in
    the middle of the `write`. It is discarded silently instead of crashing:
    the call it represents is run again, which is exactly what is wanted.
    """
    if not ruta.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    discarded_ones = 0
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            row = json.loads(linea)
        except ValueError:
            discarded_ones += 1
            continue
        if isinstance(row, dict) and row.get("clave"):
            rows.append(row)
        else:
            discarded_ones += 1
    return rows, discarded_ones


def append_row(ruta: Path, row: dict[str, Any]) -> None:
    """Append a line and make sure it is on disk before returning.

    The `fsync` is not ceremony: without it the line lives in the OS cache, and
    a power cut (or a `Stop-Process` of the wrong process) takes with it the
    record of calls that were actually paid for.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    raw_value = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(ruta, "a", encoding="utf-8", newline="\n") as file:
        file.write(raw_value)
        file.flush()
        os.fsync(file.fileno())


def finished_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    """The calls that do not need to be run again.

    Includes the ones that failed: a failure was already recorded with its
    reason, and retrying it just because the batch resumed would spend money
    repeating what is already known. To rerun one, delete its line by hand.
    """
    return {str(f["clave"]) for f in rows if f.get("clave")}


def cumulative_spend(rows: Iterable[dict[str, Any]]) -> float:
    """What the calls already recorded have cost so far, in USD."""
    return round(sum(float(f.get("cost_usd") or 0.0) for f in rows), 4)


def _p50(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    block = summary.get("line_latency_ms")
    return block.get("p50") if isinstance(block, dict) else None


def progress_row(
    item: BatchItem,
    *,
    estado: str,
    summary: dict[str, Any] | None,
    motivo: str,
    duration_s: float,
    cost_usd: float,
    spent_before: float,
    when: str,
) -> dict[str, Any]:
    """The line written per finished call.

    Flat and short on purpose: this is read with the console open while the
    campaign runs. Whatever does not fit here is in that call's summary, whose
    path goes in `resumen`.
    """
    lines = (summary or {}).get("lines") or {}
    end = (summary or {}).get("end") or {}
    stt = (summary or {}).get("stt") or {}
    round_trips = (summary or {}).get("llm_round_trips") or {}
    return {
        "indice": item.indice,
        "clave": item.clave,
        "when": when,
        "poc": item.poc,
        "scenario": item.scenario,
        "repetition": item.repetition,
        "estado": estado,
        "motivo": motivo,
        "label": (summary or {}).get("label"),
        "duration_s": round(duration_s, 1),
        "turns_reached": end.get("turns_reached"),
        "script_turns": end.get("script_turns"),
        "end": end.get("motivo"),
        "retry": end.get("retry"),
        "lines": lines.get("heard_lines"),
        "fragmentation": lines.get("fragmentation"),
        "latency_p50_ms": _p50(summary),
        # **`indicative_wer`, not `wer`**, and the name is the warning. It is the
        # WER the harness computes live to monitor the run, not the publishable
        # figure: its hypothesis comes from the STT the harness accumulated,
        # which in PoC2 concatenates interim results and repeats text. The
        # report's WER is computed
        # separately, with the reference derived from the clips and the
        # alignment; see `METRICS_CONTRACT.md` §2.10. `progress.jsonl` files from
        # before 21/09/2026 carry the old `wer` key with both defects inside.
        "indicative_wer": stt.get("wer"),
        # Turn-taking, not latency. `None` = no audit was recorded on that
        # call, which is not the same as zero.
        "agent_interruptions": (summary or {}).get("agent_interruptions"),
        "outbound_p50_ms": (summary or {}).get("return_path_p50_ms"),
        # How many times the agent made a tool round trip to the LLM. `None` =
        # that PoC does not fill the key (PoC1 and PoC2), which is not zero.
        "round_trips_llm_n": (round_trips or {}).get("n"),
        "round_trips_llm_p50_ms": (round_trips or {}).get("p50_ms"),
        "cost_usd": round(cost_usd, 4),
        "accumulated_cost_usd": round(spent_before + cost_usd, 4),
    }


def _accumulate(running_count: dict[str, Any], row: dict[str, Any]) -> None:
    """Add a call to a group. The same body for PoC and for scenario."""
    running_count["calls"] += 1
    estado = row.get("estado")
    if estado == OK:
        running_count["ok"] += 1
    elif estado == ROJA:
        running_count["failed_calls"] += 1
    else:
        running_count["failures"] += 1
    running_count["cost_usd"] = round(
        running_count["cost_usd"] + float(row.get("cost_usd") or 0.0), 4
    )
    n = row.get("round_trips_llm_n")
    if isinstance(n, int):
        # The calls that DO carry the value are counted separately: averaging
        # over the total would mix "called no function" with "this PoC does not
        # fill the key", which are different things and both read equally wrong.
        running_count["round_trips_llm"]["calls_with_data"] += 1
        running_count["round_trips_llm"]["total"] += n


def _group() -> dict[str, Any]:
    return {
        "calls": 0,
        "ok": 0,
        "failed_calls": 0,
        "failures": 0,
        "cost_usd": 0.0,
        "round_trips_llm": {"calls_with_data": 0, "total": 0},
    }


def _close(running_count: dict[str, Any]) -> dict[str, Any]:
    """Add the per-call mean, or `None` if nobody filled the key."""
    rt = running_count["round_trips_llm"]
    rt["per_call"] = (
        round(rt["total"] / rt["calls_with_data"], 2)
        if rt["calls_with_data"]
        else None
    )
    return running_count


def summarize_batch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How many calls ended each way. This is what is printed at the end.

    Per PoC **and per scenario**. The per-scenario breakdown exists because the
    tool round trips to the LLM depend on the script, not on the framework: one
    scenario can make twice as many as another, and that number is reported
    (§2.7.1 of the contract). A mean over
    the five scenarios would hide it.
    """
    per_poc: dict[int, dict[str, Any]] = {}
    per_scenario: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        poc = int(row.get("poc") or 0)
        scenario = int(row.get("scenario") or 0)
        _accumulate(per_poc.setdefault(poc, _group()), row)
        _accumulate(per_scenario.setdefault((poc, scenario), _group()), row)
    return {
        "calls": len(rows),
        "cost_usd": cumulative_spend(rows),
        "per_poc": {str(k): _close(v) for k, v in sorted(per_poc.items())},
        "per_scenario": {
            f"poc{poc}-e{scen}": _close(v)
            for (poc, scen), v in sorted(per_scenario.items())
        },
    }
