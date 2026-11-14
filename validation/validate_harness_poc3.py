"""Harness validation on PoC3's path: production bridge, fake model.

Runs PoC3's harness **as is** (`harness.act_out_script`, with its Media Streams
client, its player, its advance rule, its marks to disk and its audit recorder)
against PoC3's production bridge (`telephony.create_app`), in its own process,
with the model replaced by `validation.fake_model_poc3.FakeModel`, which answers
with a known delay.

The reported latency is computed with **the same function as the campaign**:
`shared.metrics.external_latency_poc3.call_turns`, which combines the harness
marks, the audit WAV and the bridge JSON. So the validation also covers that
reconstruction, which is where the report's PoC3 figures come from.

Same six conditions as `validation.validate_harness`. Writes, to `--output`:

- `harness_validation_poc3.csv`: one row per turn, with the same columns as the
  PoC1 validation.
- `harness_validation_poc3_summary.txt`: bias and maximum error per condition.

It costs nothing: there is no OpenAI or telephony. It takes about 40 minutes
with all conditions, because the harness runs in real time.

    uv run --project poc3_gpt_live python -m validation.validate_harness_poc3
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
    _wait_for_port,
    _free_port,
    _r,
    choose_voice,
    environment,
    sides_text,
    summary,
    summary_text,
)

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"


async def run_one(
    condition: Condition,
    scenario: int,
    *,
    clips: Path,
    voice: str,
    work: Path,
    seed: int,
    turns: int | None,
    format_kind: str = "pcm24",
) -> list[dict[str, Any]]:
    """One run of PoC3's harness against the bridge with the fake model."""
    from poc3_gpt_live.app import harness
    from poc3_gpt_live.app.marks import zeros_path
    from poc3_gpt_live.app.telephony import RUTA_WS
    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator import (
        WAIT_AFTER_CLIP_S,
        AuditRecorder,
        event_log_de_la_corrida,
        load_scenario,
    )
    from shared.metrics import external_latency_poc3 as ext3

    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    if turns:
        script = script[:turns]

    label = f"validacion-poc3-{condition.name}-e{scenario}"
    truth = work / f"truth_{label}.jsonl"
    recordings = work / "recordings_poc3"
    port = _free_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_model_poc3",
        "--port", str(port), "--clips", str(clips), "--scenario", str(scenario),
        "--voice", voice, "--delay-ms", str(condition.delay_ms),
        "--level-dbfs", str(condition.level_dbfs),
        "--truth", str(truth), "--recordings", str(recordings),
        "--seed", str(seed), "--format-kind", format_kind,
        *(["--soft-onset"] if condition.soft_onset else []),
        cwd=str(RAIZ),
    )

    # The player anchor translates the harness's wall axis to perf_counter,
    # which is the clock the fake model timestamped the ground truth with. The
    # harness creates its player internally, so it is recorded on construction.
    players: list[Any] = []
    original = harness.MediaStreamsPlayer

    class _Recorded(original):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            players.append(self)

    recorder = AuditRecorder(harness_sample_rate=TELEPHONY_SAMPLE_RATE, agent_sample_rate=TELEPHONY_SAMPLE_RATE)
    harness.MediaStreamsPlayer = _Recorded  # type: ignore[misc]
    try:
        await _wait_for_port(port, process)
        observed, call_sid = await harness.act_out_script(
            f"ws://127.0.0.1:{port}/{RUTA_WS}",
            script=script,
            folder=work,
            label=label,
            wait_after_clip=WAIT_AFTER_CLIP_S,
            recorder=recorder,
        )
        # Hanging up triggers the bridge's dump; the server shuts itself down.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=60)
    finally:
        harness.MediaStreamsPlayer = original  # type: ignore[misc]
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)

    # The same artifacts a campaign run leaves behind.
    log_path = work / f"event_log_{call_sid}.json"
    log_path.write_text(
        json.dumps(event_log_de_la_corrida(observed, call_sid, scenario=scenario),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ruta_wav = harness._dump_audit(recorder, call_sid, folder=work)
    bridge_path = recordings / f"call_{call_sid}.json"
    if ruta_wav is None or not bridge_path.is_file():
        raise RuntimeError(f"{label}: missing artifacts (wav={ruta_wav}, bridge={bridge_path})")

    # What the harness reported, with the same function the campaign uses.
    measured_items, _, report = ext3.call_turns(
        call_sid=call_sid,
        scenario=scenario,
        zeros_path=zeros_path(call_sid, folder=work),
        bridge_json_path=bridge_path,
        wav_path=ruta_wav,
        path_of_log=log_path,
        clips_path=clips,
    )
    if not players:
        raise RuntimeError(f"{label}: the harness did not create its player")
    rep = players[-1]

    def to_perf(t: float | None) -> float | None:
        return None if t is None else t - rep._onset + rep._onset_perf

    marks = ext3.marks_de(zeros_path(call_sid, folder=work))
    per_turn = {t.turn: t for t in measured_items}
    reasons = {t.step.turn: t.advance_reason for t in observed.turns}
    truths = [
        json.loads(linea) for linea in truth.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    unrecognized = [v for v in truths if v.get("tipo") == "not_recognized"]
    if unrecognized:
        print(f"    [!] {len(unrecognized)} clip(s) the fake model did not recognize")
    truths = [v for v in truths if v.get("tipo") == "turn"]
    if report.get("motivo"):
        print(f"    [!] the reconstruction produced no turns: {report['motivo']}")

    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for mark in marks:
        turn, clip_id = mark.get("turn"), mark.get("clip_id")
        idx = next(
            (k for k, v in enumerate(truths) if k not in used and v["clip_id"] == clip_id), None
        )
        v = truths[idx] if idx is not None else None
        if idx is not None:
            used.add(idx)
        measured = per_turn.get(turn)
        reported = measured.line_latency_ms if measured else None
        real = v.get("real_latency_ms") if v else None
        harness_end = to_perf(measured.end_of_speech) if measured else None
        harness_voice = to_perf(measured.first_speech_heard) if measured else None
        actual_start = None
        if v:
            actual_start = v["t_mm_start"] if v.get("t_mm_start") is not None else v["t_phrase_start"]
        rows.append({
            "condition": condition.name,
            "target_delay_ms": condition.delay_ms,
            "level_dbfs": condition.level_dbfs,
            "arranque_suave": int(condition.soft_onset),
            "scenario": scenario,
            "turn": turn,
            "clip_id": clip_id,
            "advance_reason": reasons.get(turn, ""),
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
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "harness_validation" / "poc3"),
                   help="folder for the marks, the ground truth and the bridge's files (not committed)")
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--turns", type=int, default=None, help="truncate each script (for a quick test)")
    p.add_argument("--voice", default=None, help="clip used as the agent's voice (default: the PoC1 one)")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--format-kind", choices=("pcm24", "pcmu"), default="pcm24",
                   help="bridge session format: pcm24 (September) or pcmu (October)")
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
                cond, scen, clips=clips, voice=voice, work=work,
                seed=args.seed + run, turns=args.turns, format_kind=args.format_kind,
            ))
            for f in new_items:
                print(f"    {f['turn']:<4} {f['clip_id']:<12} reported {f['reported_latency_ms']}"
                      f"  real {f['real_latency_ms']}  error {f['error_ms']}")
            rows.extend(new_items)

    with (output / "harness_validation_poc3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary_text(summary(rows), voice=voice, environment=environment()).replace(
        "Validación del arnés contra un agente de retardo conocido",
        "Validación del arnés por el camino de PoC3 (puente real, modelo de retardo conocido)",
    ) + sides_text(rows)
    (output / "harness_validation_poc3_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"({(time.monotonic() - start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
