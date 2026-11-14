"""Which turns carry a tool round trip, in the 3 PoCs.

`llm_round_trips` comes out empty in PoC1 and PoC2, and `CLAUDE.md` took for
granted that filling it required **a new campaign of 200 calls** because *"the
key is written live and is not derived from the existing artifacts"*. **It is
derived**: the function calls are timestamped in artifacts already on disk, in
all three. Details and results in `docs/hallazgo_round_trip.md`.

## Where each one comes from

- **PoC1** — its per-run log (`campana200/calls/<clave>.log`) carries pipecat's
  `… Calling function [name:call_id]` lines with a **local** timestamp.
- **PoC2** — its per-run log is the harness console and knows nothing about the
  worker, which runs in another process. The function calls are in
  `campana200/worker.log`, **a single one for the whole campaign**, in ISO UTC.
  They are attributed by time window: runs are sequential and alternate PoCs,
  so the windows do not overlap.
- **PoC3** — the census (`recordings/censo_<sid>.jsonl`) timestamps each
  delegation event in `t_local_seg`, and `reloj.anchor_wall_s` brings it to
  epoch.

PoC1's clock offset **is not assumed**: `worker.log` publishes the same line in
local time and in ISO UTC, and it comes from there.

## What this reconstruction does NOT give

The exact round-trip duration per `call_id` —the literal content of
`llm_round_trips`—, because PoC1's log does not timestamp the end of the second
`get_chat_completions` and PoC2's does not timestamp the start. The comparison
does not need it: what is compared is how much the round trip **adds** to the
observed latency, and that is measured by splitting the population into turns
with and without one.

## Reading caveat

The split is **not randomized**: whether a turn carries a tool is decided by the
LLM, and the turns with a tool are also the ones with substantive negotiation.
Part of the difference may be that those turns call for longer answers. The
delta is an **upper bound** on the cost of the round trip. Since the bias
applies equally to all three —same model, same prompt, same script— the
comparison across PoCs survives; the absolute value does not.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

# `… | DEBUG | pipecat.services.llm_service:_run_function_call:1335 - …
#    OpenAILLMService#0 Calling function [proponer_precio:call_abc]`
RE_POC1: Final[re.Pattern[str]] = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Calling function \[([a-z_]+):"
)

# The worker's JSON line: `{"message": "tool proponer_precio argumentos=… ->
# …", …, "timestamp": "2026-09-21T03:10:38.653702+00:00"}`
RE_POC2: Final[re.Pattern[str]] = re.compile(
    r'"message": "tool ([a-z_]+) argumentos.*"timestamp": "([0-9T:\-.]+)\+00:00"'
)

# The three alpha events that mark a round trip to the delegate model.
DELEGATION_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "delegation.created",
        "response.function_call_arguments.done",
        "delegation.function_call_output.created",
    }
)

# The event that carries the RESULT. It is the one that decides whether the
# agent was already talking when it found out.
RESULT_EVENT: Final[str] = "delegation.function_call_output.created"

# The functions whose result the response needs: the validated amount and the
# wording of the compensation. `registrar_acuerdo` and `finalizar_sin_acuerdo`
# are called, by instruction, after saying the closing out loud, so the
# response does not wait for their result.
FUNCTIONS_WITH_DATA: Final[frozenset[str]] = frozenset(
    {"proponer_precio", "ofrecer_compensacion_no_monetaria"}
)


@dataclass(frozen=True)
class FunctionCall:
    """A tool call, timestamped in epoch."""

    when: float
    name: str
    is_result: bool = False
    # PoC3: the function the event belongs to, if it can be known (the result
    # is tied to its call by `call_id`).
    function: str | None = None


def time_offset(worker_log: Path) -> float:
    """Seconds to add to local time to get to UTC.

    Measured, not configured: `worker.log` publishes the same line twice, in
    local time and in ISO UTC. A hardcoded time zone would break silently if
    the campaign were rerun on another machine or across a daylight-saving
    change.
    """
    anterior: str | None = None
    for linea in worker_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_POC2.search(linea)
        if m and anterior:
            local = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", anterior)
            if local:
                t_local = dt.datetime.strptime(
                    f"{local.group(1)}.{local.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
                ).replace(tzinfo=dt.timezone.utc).timestamp()
                t_utc = dt.datetime.fromisoformat(m.group(2)).replace(
                    tzinfo=dt.timezone.utc
                ).timestamp()
                return round(t_utc - t_local, 0)
        anterior = linea
    raise ValueError(
        f"{worker_log} has no line in both local time and UTC: without it the "
        "PoC1 log cannot be anchored, and assuming the time zone is exactly what "
        "this module avoids"
    )


def poc1_calls(log: Path, *, offset_s: float) -> list[FunctionCall]:
    """The function calls of a PoC1 run, in epoch."""
    output = []
    for linea in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_POC1.match(linea)
        if not m:
            continue
        when = dt.datetime.strptime(
            m.group(1), "%Y-%m-%d %H:%M:%S.%f"
        ).replace(tzinfo=dt.timezone.utc).timestamp() + offset_s
        output.append(FunctionCall(when=when, name=m.group(2)))
    return output


def worker_calls(worker_log: Path) -> list[FunctionCall]:
    """All of the campaign's PoC2 ones, sorted. They are filtered by window."""
    output = []
    for linea in worker_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_POC2.search(linea)
        if not m:
            continue
        when = dt.datetime.fromisoformat(m.group(2)).replace(
            tzinfo=dt.timezone.utc
        ).timestamp()
        output.append(FunctionCall(when=when, name=m.group(1)))
    output.sort(key=lambda x: x.when)
    return output


def poc3_calls(census: Path, anchor_wall_s: float) -> list[FunctionCall]:
    """The delegation events of a PoC3 call, in epoch.

    `epoch = anchor_wall_s + t_local_seg`. Verified against `t_acuse_pared`
    in the same file, which lives on the epoch axis and comes from the same
    anchor.
    """
    from shared.metrics.census_poc3 import events as read_census

    # With the public API the census carries other names; `read_census`
    # translates them to the alpha's and synthesizes the function-output
    # acknowledgement.
    events = list(read_census(census))
    name_of: dict[str, str] = {}
    for event in events:
        item = (event.get("event") or {}).get("item") or {}
        if item.get("type") == "function_call" and item.get("call_id") and item.get("name"):
            name_of[item["call_id"]] = item["name"]
    output = []
    for event in events:
        tipo = event.get("type")
        if tipo not in DELEGATION_EVENTS:
            continue
        item = (event.get("event") or {}).get("item") or {}
        output.append(
            FunctionCall(
                when=anchor_wall_s + event["t_local_seg"],
                name=tipo,
                is_result=(tipo == RESULT_EVENT),
                function=name_of.get(item.get("call_id", "")),
            )
        )
    return output


def split_by_round_trip(
    windows: Sequence[tuple[float, float, float]],
    calls: Iterable[FunctionCall],
) -> tuple[list[float], list[float], int]:
    """Split the turns into those that carry a round trip and those that do not.

    `windows` are `(end_of_speech, agent_first_speech, latency_ms)` — the
    segment the latency measures, which is exactly where the round trip is paid
    or not. Returns `(with, without, calls_outside_any_window)`.
    """
    dates = [l.when for l in calls]
    having: list[float] = []
    sin: list[float] = []
    for start, end, latency in windows:
        if any(start < t <= end for t in dates):
            having.append(latency)
        else:
            sin.append(latency)
    outside = sum(
        1 for t in dates if not any(i < t <= f for i, f, _ in windows)
    )
    return having, sin, outside


def median_difference(
    having: Sequence[float], sin: Sequence[float], *, replicates: int = 4000, seed: int = 11
) -> tuple[float, tuple[float, float]]:
    """The difference and its bootstrap 95% CI.

    Bootstrap and not a parametric test because the distributions are bimodal
    and heavy-tailed (`CONTRATO_METRICAS.md` §2.9).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    a, b = np.array(having), np.array(sin)
    samples = np.sort(
        [
            np.median(rng.choice(a, a.size, replace=True))
            - np.median(rng.choice(b, b.size, replace=True))
            for _ in range(replicates)
        ]
    )
    punto = float(np.median(a) - np.median(b))
    return (
        round(punto, 1),
        (
            round(float(samples[int(0.025 * replicates)]), 1),
            round(float(samples[int(0.975 * replicates)]), 1),
        ),
    )
