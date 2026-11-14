"""Validation of the interruption measurement over the PoC2 path: LiveKit Cloud room, agent that stops.

Runs the PoC2 harness **as is** (`harness.run_session`, with its room, its
`AudioSource`, its player, its advance rule and the same interruption mode
as the campaign) against `validation.fake_agent_livekit_interruption`, which
joins the same room in place of the worker, stops `S` ms after the
interruption starts playing to it and replies 800 ms after it ends. The only
thing changed in the harness is that it does not dispatch the worker.

It is the twin of `validation.validate_interruptions` (PoC1) and of
`validation.validate_interruptions_poc3`, with the same conditions and
columns, plus the offsets on each side on the same clock. What sets this path
apart is that everything goes through the internet and through Opus: the
interruption reaches the agent late and its last speech reaches the harness
late. That is not a harness error but the transport, which the report
separates just as it does for latency; what we want to see is whether the end
of the speech arrives whole or with a tail.

Writes `interruption_validation_poc2.csv` and
`interruption_validation_poc2_summary.txt` to `--output`. Needs the LiveKit keys
from `.env`; it only spends room minutes (two participants, ~35 minutes in total).

    uv run --project poc2_livekit --with soxr python -m validation.validate_interruptions_poc2

`--with soxr` adds the recognizer's resampler for this run only: PoC2 does not
ship it among its dependencies and its `uv.lock` is left untouched.
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

from validation.validate_harness import choose_voice
from validation.validate_harness_poc2 import _wait_for_file
from validation.validate_interruptions import STOPS_MS, REPLICA_MS, summary
from validation.validate_interruptions_poc3 import COLUMNS_POC3, plan_per_clip, offsets_text

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"


async def run_one(
    stop_ms: float, scenario: int, repetition: int, *,
    clips: Path, interruption_clips: Path, voice: str, work: Path,
) -> list[dict[str, Any]]:
    from poc2_livekit.app import harness
    from poc2_livekit.app.config import load_settings
    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator import WAIT_AFTER_CLIP_S, load_scenario, negotiation_state
    from shared.carrier_simulator.interruptions import attach_events

    settings = load_settings()
    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    plan = attach_events(script, scenario=scenario, repetition=repetition, folder=interruption_clips)

    label = f"interrupciones-poc2-S{int(stop_ms)}-e{scenario}"
    room = f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    truth = work / f"truth_{label}.jsonl"
    ruta_plan = work / f"plan_{label}.json"
    ready = work / f"ready_{label}"
    for ruta in (truth, ready):
        ruta.unlink(missing_ok=True)
    ruta_plan.write_text(json.dumps(plan_per_clip(script, plan), indent=2), encoding="utf-8")

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_agent_livekit_interruption",
        "--room", room, "--clips", str(clips), "--interruption-clips", str(interruption_clips),
        "--scenario", str(scenario), "--plan", str(ruta_plan), "--voice", voice,
        "--stop-ms", str(stop_ms), "--replica-ms", str(REPLICA_MS),
        "--truth", str(truth), "--ready", str(ready),
        cwd=str(RAIZ),
    )

    # No dispatch: the PoC2 worker does not join this room.
    async def _no_dispatch(*_: Any, **__: Any) -> str:
        return "sin-despacho (validación)"

    # The player's anchor converts the harness wall axis to perf_counter.
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

    if not players:
        raise RuntimeError(f"{label}: the harness did not create its player")
    last_player = players[-1]

    def to_perf(t: float | None) -> float | None:
        return None if t is None else observed.start + t - last_player._onset + last_player._onset_perf

    def ms(x: float | None) -> float | None:
        return None if x is None else round(x * 1000.0, 2)

    truths = [
        json.loads(l) for l in truth.read_text(encoding="utf-8").splitlines() if l.strip()
    ] if truth.is_file() else []
    unrecognized = [v for v in truths if v.get("tipo") == "not_recognized"]
    if unrecognized:
        print(f"    [!] {len(unrecognized)} clip(s) the fake agent did not recognize: "
              + ", ".join(f"{v['best']} ({v['correlation']})" for v in unrecognized))
    real_events = {v["clip_id"]: v for v in truths if v.get("tipo") == "event"}
    replicates = {v["clip_id"]: v for v in truths if v.get("tipo") == "replica"}

    rows: list[dict[str, Any]] = []
    for t in observed.turns:
        ev = t.event
        if ev is None:
            continue
        clip = ev.get("clip_id") or ""
        real = real_events.get(clip)
        rep = replicates.get(clip)
        det_real = real.get("actual_stop_ms") if real else None
        det_harness = ev.get("stop_time_ms")
        harness_rep = ev.get("reply_latency_ms")
        real_rep = None
        if rep and rep.get("t_first_segment_with_speech") is not None:
            real_rep = round((rep["t_first_segment_with_speech"] - rep["t_end_of_speech"]) * 1000.0, 3)
        rows.append({
            "target_stop_ms": stop_ms,
            "scenario": scenario,
            "turn": t.step.turn,
            "tipo": ev.get("tipo"),
            "clip_id": clip,
            "fired": int(bool(ev.get("fired"))),
            "harness_stopped": "" if ev.get("stopped") is None else int(ev["stopped"]),
            "harness_stop_time_ms": det_harness,
            "actual_stop_ms": det_real,
            "stop_error_ms": (
                round(det_harness - det_real, 2)
                if ev.get("tipo") == "interrupcion" and det_harness is not None and det_real is not None else None
            ),
            "harness_reply_latency_ms": harness_rep,
            "real_reply_latency_ms": real_rep,
            "error_replica_ms": (
                round(harness_rep - real_rep, 2) if harness_rep is not None and real_rep is not None else None
            ),
            "was_not_speaking": int(bool(real and real.get("was_not_speaking"))),
            "advance_reason": ev.get("advance_reason", ev.get("motivo", "")),
            "speech_onset_offset_ms": (
                ms(real["t_speech_onset"] - to_perf(ev["t_speech_onset"]))
                if real and ev.get("t_speech_onset") is not None else None
            ),
            "last_speech_offset_ms": (
                ms(to_perf(ev["last_speech_before_stopping"]) - real["t_last_speech"])
                if real and real.get("t_last_speech") is not None
                and ev.get("last_speech_before_stopping") is not None else None
            ),
            "replicate_offset_ms": (
                ms(to_perf(ev["reply_start"]) - rep["t_first_segment_with_speech"])
                if rep and rep.get("t_first_segment_with_speech") is not None
                and ev.get("reply_start") is not None else None
            ),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    from shared.carrier_simulator.interruptions import INTERRUPTION_CLIPS_PATH

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--interruption-clips", default=str(INTERRUPTION_CLIPS_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "interruption_validation" / "poc2"),
                   help="folder for the ground truth and the plan of each run (not committed)")
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--stops", nargs="*", type=float, default=list(STOPS_MS))
    p.add_argument("--voice", default=None)
    args = p.parse_args(argv)

    clips = Path(args.clips)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    voice = args.voice or choose_voice(clips)
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for k, s in enumerate(args.stops, start=1):
        for scen in args.scenarios:
            print(f"S = {s:.0f} ms · scenario {scen} ...", flush=True)
            new_items = asyncio.run(run_one(
                s, scen, k, clips=clips, interruption_clips=Path(args.interruption_clips),
                voice=voice, work=work,
            ))
            for f in new_items:
                print(f"    {f['turn']:<4}{f['tipo']:<13}{f['clip_id']:<8}"
                      f" stopped={f['harness_stopped']} harness={f['harness_stop_time_ms']}"
                      f" real={f['actual_stop_ms']} reply={f['harness_reply_latency_ms']}"
                      f"/{f['real_reply_latency_ms']}")
            rows.extend(new_items)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "interruption_validation_poc2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS_POC3)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows).replace(
        "Validación de la medición de interrupciones (agente que se calla a tiempo conocido)",
        "Validación de la medición de interrupciones por el camino de PoC2 "
        "(sala de LiveKit Cloud, agente que se calla a tiempo conocido)",
    ) + offsets_text(rows)
    (output / "interruption_validation_poc2_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"({(time.monotonic() - start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
