"""Harness validation on PoC2's path: the LiveKit Cloud room, fake agent.

Runs PoC2's harness **as is** (`harness.run_session`, with its connection to the
room, its `AudioSource`, its player, its advance rule and its marks over the data
channel) against `validation.fake_agent_livekit`, which joins the same room and
answers with a known delay. The only thing changed in the harness is that it
**does not dispatch the worker**: the fake agent is there instead. So there is no
STT, LLM or TTS to pay for, and no real agent joins the room.

The reported latency is computed with **the same function as the campaign**:
`poc2_livekit.app.marks.add_lines`, over the marks the fake agent received over
the data channel, just like the worker.

Same six conditions as `validation.validate_harness`. Writes, to `--output`:

- `harness_validation_poc2.csv`: one row per turn, with the same columns as the
  PoC1 validation.
- `harness_validation_poc2_summary.txt`: bias and maximum error per condition.

It needs the LiveKit keys from `.env`. It only spends LiveKit Cloud minutes
(two participants, about 40 minutes with all conditions).

    uv run --project poc2_livekit python -m validation.validate_harness_poc2
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

from validation.validate_harness import (
    CONDITIONS,
    COLUMNS,
    Condition,
    _r,
    choose_voice,
    environment,
    sides_text,
    summary,
    summary_text,
)

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"


async def _wait_for_file(ruta: Path, process: Any, tope_seg: float = 60.0) -> None:
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if process.returncode is not None:
            raise RuntimeError(f"the fake agent died on startup (code {process.returncode})")
        if ruta.is_file():
            return
        await asyncio.sleep(0.2)
    raise RuntimeError("the fake agent did not join the room")


async def run_one(
    condition: Condition,
    scenario: int,
    *,
    clips: Path,
    voice: str,
    work: Path,
    turns: int | None,
) -> list[dict[str, Any]]:
    """One run of PoC2's harness against the fake agent, in a new room."""
    from poc2_livekit.app import harness
    from poc2_livekit.app.config import load_settings
    from poc2_livekit.app.marks import add_lines
    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator import (
        WAIT_AFTER_CLIP_S,
        load_scenario,
        negotiation_state,
    )

    settings = load_settings()
    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    if turns:
        script = script[:turns]

    label = f"validacion-poc2-{condition.name}-e{scenario}"
    room = f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    truth = work / f"truth_{label}.jsonl"
    marks_path = work / f"marks_{label}.json"
    ready = work / f"ready_{label}"
    for ruta in (truth, marks_path, ready):
        ruta.unlink(missing_ok=True)

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_agent_livekit",
        "--room", room, "--clips", str(clips), "--scenario", str(scenario),
        "--voice", voice, "--delay-ms", str(condition.delay_ms),
        "--level-dbfs", str(condition.level_dbfs),
        "--truth", str(truth), "--marks", str(marks_path), "--ready", str(ready),
        *(["--soft-onset"] if condition.soft_onset else []),
        cwd=str(RAIZ),
    )

    # No dispatch: the PoC2 worker does not join this room.
    async def _no_dispatch(*_: Any, **__: Any) -> str:
        return "sin-despacho (validación)"

    # The player anchor translates the harness's wall axis to perf_counter.
    players: list[Any] = []
    original_rep, original_after = harness.Player, harness.dispatch_agent

    class _Recorded(original_rep):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            players.append(self)

    harness.Player = _Recorded  # type: ignore[misc]
    harness.dispatch_agent = _no_dispatch  # type: ignore[assignment]
    try:
        await _wait_for_file(ready, process)
        observed = await harness.run_session(
            settings,
            room=room,
            script=script,
            publish_sample_rate=TELEPHONY_SAMPLE_RATE,
            wait_after_clip=WAIT_AFTER_CLIP_S,
            identification={"poc": "validacion"},
            silent=True,
            estado=negotiation_state(script),
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=30)
    finally:
        harness.Player, harness.dispatch_agent = original_rep, original_after  # type: ignore[misc]
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)

    if not players or not marks_path.is_file():
        raise RuntimeError(f"{label}: the player or the marks are missing")
    rep = players[-1]

    # What the harness reported, with the same function the campaign uses.
    first_speech = {
        t.step.turn: observed.start + t.t_agent_first_speech
        for t in observed.turns
        if t.t_agent_first_speech is not None
    }
    block = add_lines(marks_path, first_speech=first_speech) or {}

    def to_perf(t: float | None) -> float | None:
        return None if t is None else t - rep._onset + rep._onset_perf

    reasons = {t.step.turn: t.advance_reason for t in observed.turns}
    truths = [
        json.loads(linea) for linea in truth.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ] if truth.is_file() else []
    unrecognized = [v for v in truths if v.get("tipo") == "not_recognized"]
    if unrecognized:
        print(f"    [!] {len(unrecognized)} clip(s) the fake agent did not recognize: "
              + ", ".join(f"{v['best']} ({v['correlation']})" for v in unrecognized))
    truths = [v for v in truths if v.get("tipo") == "turn"]

    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for row in block.get("per_line") or []:
        clip_id = row.get("clip_id")
        idx = next(
            (k for k, v in enumerate(truths) if k not in used and v["clip_id"] == clip_id), None
        )
        v = truths[idx] if idx is not None else None
        if idx is not None:
            used.add(idx)
        reported = row.get("line_latency_ms")
        real = v.get("real_latency_ms") if v else None
        harness_end = to_perf(row.get("end_of_speech"))
        harness_voice = to_perf(row.get("agent_first_speech"))
        actual_start = None
        if v:
            actual_start = v["t_mm_start"] if v.get("t_mm_start") is not None else v["t_phrase_start"]
        rows.append({
            "condition": condition.name,
            "target_delay_ms": condition.delay_ms,
            "level_dbfs": condition.level_dbfs,
            "arranque_suave": int(condition.soft_onset),
            "scenario": scenario,
            "turn": row.get("turn"),
            "clip_id": clip_id,
            "advance_reason": reasons.get(row.get("turn"), ""),
            "reported_latency_ms": reported,
            "real_latency_ms": real,
            "error_ms": _r(reported - real) if reported is not None and real is not None else None,
            "end_of_speech_error_ms": (
                _r((harness_end - v["t_end_of_speech"]) * 1000.0) if v and harness_end is not None else None
            ),
            "response_error_ms": (
                _r((harness_voice - actual_start) * 1000.0)
                if v and harness_voice is not None and actual_start is not None else None
            ),
            "real_phrase_latency_ms": v.get("real_phrase_latency_ms") if v else None,
            "error_contra_frase_ms": (
                _r(reported - v["real_phrase_latency_ms"])
                if v and reported is not None and v.get("real_phrase_latency_ms") is not None else None
            ),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "harness_validation" / "poc2"),
                   help="folder for each run's ground truth and marks (not committed)")
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--turns", type=int, default=None, help="truncate each script (for a quick test)")
    p.add_argument("--voice", default=None, help="clip used as the agent's voice (default: the PoC1 one)")
    args = p.parse_args(argv)

    clips = Path(args.clips)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    voice = args.voice or choose_voice(clips)
    print(f"fake agent voice: {voice}")

    rows: list[dict[str, Any]] = []
    run = 0
    total = len(args.condiciones) * len(args.scenarios)
    start = time.monotonic()
    for name in args.condiciones:
        cond = CONDITIONS[name]
        for scen in args.scenarios:
            run += 1
            print(f"[{run}/{total}] {name} · scenario {scen} ...", flush=True)
            new_items = asyncio.run(run_one(
                cond, scen, clips=clips, voice=voice, work=work, turns=args.turns,
            ))
            for f in new_items:
                print(f"    {f['turn']:<4} {f['clip_id']:<12} reported {f['reported_latency_ms']}"
                      f"  real {f['real_latency_ms']}  error {f['error_ms']}")
            rows.extend(new_items)

    with (output / "harness_validation_poc2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary_text(summary(rows), voice=voice, environment=environment()).replace(
        "Validación del arnés contra un agente de retardo conocido",
        "Validación del arnés por el camino de PoC2 (sala de LiveKit Cloud, agente de retardo conocido)",
    ) + sides_text(rows)
    (output / "harness_validation_poc2_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"({(time.monotonic() - start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
