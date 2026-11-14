"""Validation of the interruption measurement with an agent that stops at a known time.

Runs the PoC1 harness (the same `play_script` and the same interruption
mode as the campaign) against `validation.fake_agent_interruption`, which
gives long answers and, when an interruption plays over it, stops exactly
`S` ms after the carrier's speech starts and replies 800 ms after it ends.
On a backchannel or in a control it keeps talking.

Three conditions: S = 200, 500 and 1,000 ms, each over the five scenarios
with their event plan (2 interruptions, 2 backchannels and 2 controls per call).
If the harness measures correctly:

- the stop time it reports tracks `S` with millisecond error;
- it says "stopped" on every interruption and "did not stop" on every
  backchannel and control;
- the reply latency it reports tracks the one the agent scheduled.

Writes `interruption_validation.csv` (one row per event) and
`interruption_validation_summary.txt` to `--output`. Uses no external service.

    uv run --project poc1_pipecat python -m validation.validate_interruptions
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Final

from validation.validate_harness import _wait_for_port, _free_port, choose_voice, environment

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"
STOPS_MS: Final[tuple[float, ...]] = (200.0, 500.0, 1000.0)
REPLICA_MS: Final[float] = 800.0

COLUMNS: Final[list[str]] = [
    "target_stop_ms", "scenario", "turn", "tipo", "clip_id", "fired",
    "harness_stopped", "harness_stop_time_ms", "actual_stop_ms", "stop_error_ms",
    "harness_reply_latency_ms", "real_reply_latency_ms", "error_replica_ms",
    "was_not_speaking", "advance_reason",
]


async def run_one(
    stop_ms: float, scenario: int, repetition: int, *,
    clips: Path, interruption_clips: Path, voice: str, work: Path,
) -> list[dict[str, Any]]:
    from poc1_pipecat.app.harness import ObservedFromPoC1, _on_text_received, annotate_hang_up
    from poc1_pipecat.app.media_streams import MediaStreamsClient, MediaStreamsPlayer
    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator import (
        MAX_AGENT_WAIT_S,
        WAIT_AFTER_CLIP_S,
        AGENT_SILENCE_S,
        load_scenario,
        wait_until_agent_is_silent,
        negotiation_state,
        play_script,
    )
    from shared.carrier_simulator.interruptions import attach_events

    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    attach_events(script, scenario=scenario, repetition=repetition, folder=interruption_clips)

    label = f"interrupciones-S{int(stop_ms)}-e{scenario}"
    truth = work / f"truth_{label}.jsonl"
    port = _free_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_agent_interruption",
        "--port", str(port), "--clips", str(clips),
        "--interruption-clips", str(interruption_clips), "--voice", voice,
        "--stop-ms", str(stop_ms), "--replica-ms", str(REPLICA_MS),
        "--truth", str(truth), "--seed", str(scenario),
        cwd=str(RAIZ),
    )
    observed = ObservedFromPoC1(silent=True)
    closed = asyncio.Event()
    try:
        await _wait_for_port(port, process)
        client = MediaStreamsClient(
            f"ws://127.0.0.1:{port}/ws", label=label,
            identification={"poc": "validacion"},
            on_receive_audio=observed.annotate_audio_in_ulaw,
            on_receive_message=lambda m: _on_text_received(observed, m),
            on_other_side_closed=annotate_hang_up(observed, closed),
        )
        player = MediaStreamsPlayer(client, observed=observed)
        halt = asyncio.Event()
        async with client:
            pumping = asyncio.create_task(player.launch(halt))
            try:
                await wait_until_agent_is_silent(
                    observed, silence_s=AGENT_SILENCE_S, tope_seg=MAX_AGENT_WAIT_S
                )
                await play_script(
                    script, player=player, observed=observed,
                    wait_for_agent_s=WAIT_AFTER_CLIP_S, closed=closed,
                    estado=negotiation_state(script),
                )
            finally:
                halt.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await pumping
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=15)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)

    def to_perf(t: float | None) -> float | None:
        """From the harness timeline to the `perf_counter` the agent used to timestamp the ground truth."""
        if t is None:
            return None
        return observed.start + t - player._onset + player._onset_perf

    truths = [json.loads(l) for l in truth.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_events = {v["clip_id"]: v for v in truths if v.get("tipo") == "event"}
    replicates = {v["clip_id"]: v for v in truths if v.get("tipo") == "replica"}

    rows: list[dict[str, Any]] = []
    for t in observed.turns:
        ev = t.event
        if ev is None:
            continue
        real = real_events.get(ev.get("clip_id") or "")
        rep = replicates.get(ev.get("clip_id") or "")
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
            "clip_id": ev.get("clip_id") or "",
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
        })
    # `a_perf` stays available for manual debugging; the comparison uses
    # differences within each clock, which need no conversion.
    del to_perf
    return rows


def summary(rows: list[dict[str, Any]]) -> str:
    text_lines = [
        "Validación de la medición de interrupciones (agente que se calla a tiempo conocido)",
        environment(), "",
        f"{'S (ms)':>7} {'tipo':<13}{'events':>8}{'disparados':>11}{'se detuvo':>10}"
        f"{'err. det. mediana':>18}{'|máx|':>8}{'err. réplica med.':>18}",
    ]
    for s in STOPS_MS:
        for tipo in ("interrupcion", "apoyo", "control"):
            fs = [f for f in rows if f["target_stop_ms"] == s and f["tipo"] == tipo]
            if not fs:
                continue
            fired_list = [f for f in fs if f["fired"]]
            det = [f for f in fired_list if f["harness_stopped"] == 1]
            err = [float(f["stop_error_ms"]) for f in fired_list if f["stop_error_ms"] not in (None, "")]
            rep = [float(f["error_replica_ms"]) for f in fired_list if f["error_replica_ms"] not in (None, "")]
            med = f"{statistics.median(err):.1f}" if err else "—"
            mx = f"{max(abs(e) for e in err):.1f}" if err else "—"
            mr = f"{statistics.median(rep):.1f}" if rep else "—"
            text_lines.append(f"{s:>7.0f} {tipo:<13}{len(fs):>8}{len(fired_list):>11}{len(det):>10}{med:>18}{mx:>8}{mr:>18}")
    return "\n".join(text_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from shared.carrier_simulator.interruptions import INTERRUPTION_CLIPS_PATH

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--interruption-clips", default=str(INTERRUPTION_CLIPS_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "interruption_validation"))
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--stops", nargs="*", type=float, default=list(STOPS_MS))
    p.add_argument("--voice", default=None)
    args = p.parse_args(argv)

    clips = Path(args.clips)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    voice = args.voice or choose_voice(clips)
    rows: list[dict[str, Any]] = []
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
    with (output / "interruption_validation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows)
    (output / "interruption_validation_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
