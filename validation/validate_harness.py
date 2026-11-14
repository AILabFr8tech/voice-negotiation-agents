"""Harness validation: is the latency it reports the latency that actually occurred?

Runs PoC1's harness **as is** (the same Media Streams client, the same player,
the same advance rule, the same mark emitter and the same per-line latency
computation) against `validation.fake_agent`, which answers with a known delay.
The only thing that changes relative to a campaign run is who is on the other
side of the WebSocket.

Conditions (each one over the 5 scenarios):

- `retardo_300`, `retardo_1000`, `retardo_2500`: speech at −20 dBFS, delay D.
- `nivel_-6`, `nivel_-35`: D = 1,000 ms, speech at −6 and at −35 dBFS.
- `arranque_suave`: D = 1,000 ms, −20 dBFS, with an "mm" 12 dB quieter before
  the phrase.

`retardo_1000` is also the −20 dBFS condition of the level test.

Writes, to `--output`:

- `harness_validation.csv`: one row per turn, with the latency reported by the
  harness, the real one and the difference, split into the end-of-speech side
  and the response side.
- `harness_validation_summary.txt`: bias and maximum error per condition.

It costs nothing: there is no STT, LLM, TTS or telephony. It takes about 40
minutes with all conditions, because the harness runs in real time.

    uv run --project poc1_pipecat python -m validation.validate_harness
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import math
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"


@dataclass(frozen=True)
class Condition:
    name: str
    delay_ms: float
    level_dbfs: float
    soft_onset: bool = False


CONDITIONS: Final[dict[str, Condition]] = {
    c.name: c
    for c in (
        Condition("retardo_300", 300.0, -20.0),
        Condition("retardo_1000", 1000.0, -20.0),
        Condition("retardo_2500", 2500.0, -20.0),
        Condition("nivel_-6", 1000.0, -6.0),
        Condition("nivel_-35", 1000.0, -35.0),
        Condition("arranque_suave", 1000.0, -20.0, soft_onset=True),
    )
}

COLUMNS: Final[list[str]] = [
    "condition", "target_delay_ms", "level_dbfs", "arranque_suave",
    "scenario", "turn", "clip_id", "advance_reason",
    "reported_latency_ms", "real_latency_ms", "error_ms",
    "end_of_speech_error_ms", "response_error_ms",
    "real_phrase_latency_ms", "error_contra_frase_ms",
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait_for_port(port: int, process: Any, tope_seg: float = 60.0) -> None:
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if process.returncode is not None:
            raise RuntimeError(f"the fake agent died on startup (code {process.returncode})")
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"the fake agent did not open port {port}")


# --------------------------------------------------------------------------
# The fake agent's voice
# --------------------------------------------------------------------------


def reproducible_ones(script_path: Path | None = None) -> list[str]:
    """The clips some script can play: the base ones and the branches.

    An alternate that appears in no branch was recorded, but the harness never
    picks it.
    """
    import yaml

    ruta = script_path or RAIZ / "shared" / "test_scenarios" / "scenarios.yaml"
    data = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    ids: list[str] = []
    for scen in data["scenarios"]:
        for turn in scen["turns"]:
            base = f"e{scen['id']}_{str(turn['id']).lower()}"
            ids.append(base)
            alternates = turn.get("alternates") or []
            for rama in turn.get("branches") or []:
                linea = rama.get("linea")
                if linea in alternates:
                    alt = f"{base}_alt{alternates.index(linea) + 1}"
                    if alt not in ids:
                        ids.append(alt)
    return ids


def choose_voice(folder: Path) -> str:
    """The clip used as the agent's voice: the one with the longest continuous streak.

    It is chosen with the same rule the harness uses to group the agent's speech
    (20 ms frames with RMS ≥ 200, streaks separated by more than 0.3 s of
    silence), at −20 dBFS. That way the base condition is not contaminated by a
    response that the harness itself would split in two.
    """
    import numpy as np

    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator.script import load_clip
    from shared.carrier_simulator.observed import STREAK_HANGOVER_S, RMS_WITH_SPEECH
    from validation.fake_agent import RESPONSE_DURATION_S, to_level

    best, best_duration = "", -1.0
    for clip_id in sorted(reproducible_ones()):
        ruta = folder / f"{clip_id}.ulaw"
        if not ruta.is_file():
            continue
        pcm, _ = load_clip(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        if len(pcm) < RESPONSE_DURATION_S * TELEPHONY_SAMPLE_RATE:
            continue
        voice = to_level(pcm[: int(RESPONSE_DURATION_S * TELEPHONY_SAMPLE_RATE)], -20.0)
        frames = voice[: len(voice) // 160 * 160].reshape(-1, 160).astype(np.float64)
        active = np.sqrt(np.mean(frames**2, axis=1)) >= RMS_WITH_SPEECH
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue
        start = end = idx[0]
        for i in idx[1:]:
            if (i - end) * 0.02 > STREAK_HANGOVER_S:
                break
            end = i
        dur = (end - start) * 0.02
        if dur > best_duration:
            best, best_duration = clip_id, dur
    if not best:
        raise SystemExit(f"no usable clips in {folder}")
    return best


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


async def run_one(
    condition: Condition,
    scenario: int,
    *,
    clips: Path,
    voice: str,
    work: Path,
    seed: int,
    turns: int | None,
) -> list[dict[str, Any]]:
    """One run of PoC1's harness against the fake agent. Returns rows."""
    from poc1_pipecat.app.harness import (
        ObservedFromPoC1,
        _on_text_received,
        annotate_hang_up,
        first_speech_per_turn,
    )
    from poc1_pipecat.app.marks import (
        DiskMarkEmitter,
        lines_block,
        marks_de,
        zeros_path,
    )
    from poc1_pipecat.app.media_streams import (
        MediaStreamsClient,
        MediaStreamsPlayer,
        verify_stimulus,
    )
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

    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    if turns:
        script = script[:turns]
    verify_stimulus(script, clips)

    label = f"validacion-{condition.name}-e{scenario}"
    truth = work / f"truth_{label}.jsonl"
    port = _free_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_agent",
        "--port", str(port), "--clips", str(clips), "--voice", voice,
        "--delay-ms", str(condition.delay_ms),
        "--level-dbfs", str(condition.level_dbfs),
        "--truth", str(truth), "--seed", str(seed),
        *(["--soft-onset"] if condition.soft_onset else []),
        cwd=str(RAIZ),
    )
    observed = ObservedFromPoC1(silent=True)
    closed = asyncio.Event()
    try:
        await _wait_for_port(port, process)
        client = MediaStreamsClient(
            f"ws://127.0.0.1:{port}/ws",
            label=label,
            identification={"poc": "validacion"},
            on_receive_audio=observed.annotate_audio_in_ulaw,
            on_receive_message=lambda m: _on_text_received(observed, m),
            on_other_side_closed=annotate_hang_up(observed, closed),
        )
        player = MediaStreamsPlayer(client, observed=observed)
        marks_path = zeros_path(client.call_sid, folder=work)
        marks_path.unlink(missing_ok=True)
        emitter = DiskMarkEmitter(marks_path, publish_sample_rate=player.sample_rate)
        halt = asyncio.Event()
        async with client:
            pumping = asyncio.create_task(player.launch(halt))
            try:
                await wait_until_agent_is_silent(
                    observed, silence_s=AGENT_SILENCE_S, tope_seg=MAX_AGENT_WAIT_S
                )
                await play_script(
                    script,
                    player=player,
                    observed=observed,
                    wait_for_agent_s=WAIT_AFTER_CLIP_S,
                    emitter=emitter,
                    closed=closed,
                    estado=negotiation_state(script),
                )
            finally:
                halt.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await pumping
                await emitter.shut()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=15)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)

    # What the harness reported, with the same function the campaign uses.
    block = lines_block(
        marks_de(marks_path), [], first_speech=first_speech_per_turn(observed)
    )
    # Player anchor: translates the harness's wall axis to perf_counter,
    # which is the clock in which the fake agent timestamped the ground truth.
    def to_perf(t: float | None) -> float | None:
        if t is None:
            return None
        return t - player._onset + player._onset_perf

    reasons = {t.step.turn: t.advance_reason for t in observed.turns}
    truths = [
        json.loads(linea) for linea in truth.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    truths = [v for v in truths if v.get("tipo") == "turn"]

    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for row in block.get("per_line") or []:
        clip_id = row.get("clip_id")
        # The ground truth for the same clip, in order: a scenario does not repeat a clip.
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


def _r(x: float) -> float:
    return round(float(x), 2)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bias and error per condition. The bias is the mean of (reported − real)."""
    output = []
    for name in CONDITIONS:
        of_case = [f for f in rows if f["condition"] == name]
        if not of_case:
            continue
        err = [float(f["error_ms"]) for f in of_case if f["error_ms"] not in (None, "")]
        lost = sum(1 for f in of_case if f["reported_latency_ms"] in (None, ""))
        abs_err = sorted(abs(e) for e in err)
        output.append({
            "condition": name,
            "turns": len(of_case),
            "with_measurement": len(err),
            "no_harness_measurement": lost,
            "mean_bias_ms": _r(statistics.fmean(err)) if err else None,
            "median_bias_ms": _r(statistics.median(err)) if err else None,
            "deviation_ms": _r(statistics.stdev(err)) if len(err) > 1 else None,
            "min_error_ms": _r(min(err)) if err else None,
            "max_error_ms": _r(max(err)) if err else None,
            "max_absolute_error_ms": _r(abs_err[-1]) if err else None,
            "p95_absolute_error_ms": (
                _r(abs_err[min(len(abs_err) - 1, math.ceil(0.95 * len(abs_err)) - 1)]) if err else None
            ),
        })
    return output


def summary_text(res: list[dict[str, Any]], *, voice: str, environment: str) -> str:
    text_lines = [
        "Validación del arnés contra un agente de retardo conocido",
        f"voz del agente falso: {voice}",
        environment,
        "",
        "error = latencia reportada por el arnés − latencia real (ms)",
        f"{'condición':<16}{'turns':>7}{'measured_items':>8}{'sin med.':>9}{'bias':>8}"
        f"{'median_value':>9}{'desv.':>7}{'mín':>8}{'máx':>8}{'|máx|':>8}{'|p95|':>8}",
    ]
    for r in res:
        def f(x: Any) -> str:
            return "—" if x is None else f"{x:.1f}"
        text_lines.append(
            f"{r['condition']:<16}{r['turns']:>7}{r['with_measurement']:>8}{r['no_harness_measurement']:>9}"
            f"{f(r['mean_bias_ms']):>8}{f(r['median_bias_ms']):>9}{f(r['deviation_ms']):>7}"
            f"{f(r['min_error_ms']):>8}{f(r['max_error_ms']):>8}"
            f"{f(r['max_absolute_error_ms']):>8}{f(r['p95_absolute_error_ms']):>8}"
        )
    return "\n".join(text_lines) + "\n"


def sides_text(rows: list[dict[str, Any]]) -> str:
    """How the error splits between the end-of-speech side and the response side.

    On the PoC2 and PoC3 paths the audio crosses a real transport (the LiveKit
    Cloud room, the bridge), and that trip stays inside the error: the
    end-of-speech side is the inbound leg (it comes out negative: the harness
    marked before the agent heard) and the response side is the outbound leg.
    Error = response − end of speech.
    """
    text_lines = ["", "por lado (medias, ms): fin de voz = −ida · respuesta = vuelta",
              f"{'condición':<16}{'fin de voz':>12}{'response':>12}"]
    for name in CONDITIONS:
        of_case = [f for f in rows if f["condition"] == name]
        eos = [float(f["end_of_speech_error_ms"]) for f in of_case if f["end_of_speech_error_ms"] not in (None, "")]
        resp = [float(f["response_error_ms"]) for f in of_case if f["response_error_ms"] not in (None, "")]
        if not of_case:
            continue
        def m(v: list[float]) -> str:
            return f"{statistics.fmean(v):.1f}" if v else "—"
        text_lines.append(f"{name:<16}{m(eos):>12}{m(resp):>12}")
    return "\n".join(text_lines) + "\n"


def environment() -> str:
    import platform

    return f"system: {platform.platform()} · python {platform.python_version()}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "harness_validation"),
                   help="folder for each run's marks and ground truth (not committed)")
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--turns", type=int, default=None, help="truncate each script (for a quick test)")
    p.add_argument("--voice", default=None, help="clip used as the agent's voice (chosen automatically by default)")
    p.add_argument("--seed", type=int, default=20260924)
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
    for name in args.condiciones:
        cond = CONDITIONS[name]
        for scen in args.scenarios:
            run += 1
            print(f"[{run}/{total}] {name} · scenario {scen} ...", flush=True)
            new_items = asyncio.run(run_one(
                cond, scen, clips=clips, voice=voice, work=work,
                seed=args.seed + run, turns=args.turns,
            ))
            for f in new_items:
                print(f"    {f['turn']:<4} {f['clip_id']:<12} reported {f['reported_latency_ms']}"
                      f"  real {f['real_latency_ms']}  error {f['error_ms']}")
            rows.extend(new_items)

    with (output / "harness_validation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary_text(summary(rows), voice=voice, environment=environment())
    (output / "harness_validation_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
