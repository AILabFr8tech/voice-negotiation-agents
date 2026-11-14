"""Validation of the interruption measurement over the PoC3 path: real bridge, model that stops.

Runs the PoC3 harness **as is** (`harness.act_out_script`, with its Media
Streams client, its player, its advance rule and the same interruption mode as
the campaign) against the PoC3 production bridge (`telephony.create_app`) with
the model replaced by `validation.fake_model_poc3_interruption`, which stops
`S` ms after the interruption starts playing to it and replies 800 ms after it
ends.

It is the twin of `validation.validate_interruptions` (PoC1 path), with the
same three conditions (S = 200, 500 and 1,000 ms over the five scenarios) and
the same columns. What sets this path apart, and what we want to see, is how
the bridge delivers the agent's speech: in 100 ms blocks (the model's cadence
in the general-availability version) that the harness timestamps on arrival.
The onset of a block arrives when it starts playing; its end plays ~100 ms
after it arrived. The fake model's ground truth is in playback time, so the
difference shows up in `stop_error_ms`.

**The audio measurement** (`validation.interruption_pauses.measure_in_audio`),
which the report uses to count in how many events the recording and the
harness agree, is validated in the same run: the harness records its audit
WAV, it is measured with 20 ms windows, and the `*_audio*` columns give its
detection, its stop error and its reply error against the same ground truth.

Writes `interruption_validation_poc3.csv` and
`interruption_validation_poc3_summary.txt` to `--output`. Uses no external service.

    uv run --project poc3_gpt_live python -m validation.validate_interruptions_poc3
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

from validation.validate_harness import _wait_for_port, _free_port, choose_voice, environment
from validation.validate_interruptions import COLUMNS, STOPS_MS, REPLICA_MS, summary

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
# Extra columns over the PoC1 validation, to split the stop error into its
# two sides, on the same clock (`perf_counter`):
# - `speech_onset_offset_ms`: how long after the harness released the
#   interruption's speech the model had it (bridge, upstream resampling);
# - `last_speech_offset_ms`: how long after (positive) or before (negative)
#   the model's last segment with speech played the harness timestamped it.
# - `replicate_offset_ms`: how long after (positive) or before (negative) the
#   reply's first segment with speech played the harness timestamped it.
# The stop error is, up to rounding, the second minus the first.
# Plus the audio measurement on the harness audit WAV, against the same ground truth.
COLUMNS_POC3: Final[list[str]] = [
    *COLUMNS, "speech_onset_offset_ms", "last_speech_offset_ms", "replicate_offset_ms",
    "audio_stopped", "audio_stop_time_ms", "audio_stop_error_ms",
    "audio_reply_latency_ms", "error_replica_audio_ms",
]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"


def plan_per_clip(script: list[Any], plan: dict[str, tuple[str, str | None]]) -> dict[str, str]:
    """Script clip → event type of its turn, covering every branch of the turn."""
    output: dict[str, str] = {}
    for step in script:
        if step.turn not in plan or step.is_silence:
            continue
        tipo = plan[step.turn][0]
        for clip in [step.clip_id, *(v.clip_id for v in step.variants.values())]:
            if clip:
                output[clip] = tipo
    return output


async def run_one(
    stop_ms: float, scenario: int, repetition: int, *,
    clips: Path, interruption_clips: Path, voice: str, work: Path, seed: int,
    format_kind: str = "pcm24",
) -> list[dict[str, Any]]:
    from poc3_gpt_live.app import harness
    from poc3_gpt_live.app.telephony import RUTA_WS
    from shared.audio import TELEPHONY_SAMPLE_RATE, read_wav_stereo
    from shared.carrier_simulator import WAIT_AFTER_CLIP_S, load_scenario
    from shared.carrier_simulator.auditoria import AuditRecorder
    from shared.carrier_simulator.interruptions import NO_REPLY_WAIT_S, attach_events
    from validation.interruption_pauses import measure_in_audio, frames_with_speech

    script = load_scenario(scenario, clips, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    plan = attach_events(script, scenario=scenario, repetition=repetition, folder=interruption_clips)

    label = f"interrupciones-poc3-S{int(stop_ms)}-e{scenario}"
    truth = work / f"truth_{label}.jsonl"
    ruta_plan = work / f"plan_{label}.json"
    ruta_plan.write_text(json.dumps(plan_per_clip(script, plan), indent=2), encoding="utf-8")
    port = _free_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "validation.fake_model_poc3_interruption",
        "--port", str(port), "--clips", str(clips),
        "--interruption-clips", str(interruption_clips), "--scenario", str(scenario),
        "--plan", str(ruta_plan), "--voice", voice,
        "--stop-ms", str(stop_ms), "--replica-ms", str(REPLICA_MS),
        "--truth", str(truth), "--recordings", str(work / "recordings_poc3"),
        "--seed", str(seed), "--format-kind", format_kind,
        cwd=str(RAIZ),
    )
    # The player's anchor converts the harness timeline to `perf_counter`, the
    # clock the fake model used to timestamp the ground truth. The harness
    # creates its player internally, so it is recorded on construction.
    players: list[Any] = []
    original = harness.MediaStreamsPlayer

    class _Recorded(original):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            players.append(self)

    harness.MediaStreamsPlayer = _Recorded  # type: ignore[misc]
    recorder = AuditRecorder(harness_sample_rate=TELEPHONY_SAMPLE_RATE, agent_sample_rate=harness.LISTEN_SAMPLE_RATE)
    try:
        await _wait_for_port(port, process)
        observed, _ = await harness.act_out_script(
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

    if not players:
        raise RuntimeError(f"{label}: the harness did not create its player")
    last_player = players[-1]

    def to_perf(t: float | None) -> float | None:
        return None if t is None else observed.start + t - last_player._onset + last_player._onset_perf

    def ms(x: float | None) -> float | None:
        return None if x is None else round(x * 1000.0, 2)

    # The audit WAV is on the harness timeline (`observed.start`), like the
    # event instants.
    ruta_wav = work / f"auditoria_{label}.wav"
    recorder.dump(ruta_wav, room=label)
    _, agent_channel, wav_sample_rate = read_wav_stereo(ruta_wav)
    with_voice = frames_with_speech(agent_channel, wav_sample_rate)

    truths = [json.loads(l) for l in truth.read_text(encoding="utf-8").splitlines() if l.strip()]
    unrecognized = [v for v in truths if v.get("tipo") == "not_recognized"]
    if unrecognized:
        print(f"    [!] {len(unrecognized)} clip(s) the fake model did not recognize")
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
        audio: dict[str, Any] = {}
        if ev.get("fired") and ev.get("t_speech_onset") is not None:
            end_of_speech = float(ev.get("t_end_of_speech") or ev["t_speech_onset"])
            audio = measure_in_audio(
                with_voice, reference=float(ev["t_speech_onset"]), end_of_speech=end_of_speech,
                until=end_of_speech + NO_REPLY_WAIT_S,
            )
        det_audio = audio.get("stop_ms")
        rep_audio = audio.get("resumes_ms") if ev.get("tipo") == "interrupcion" else None
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
            "audio_stopped": "" if not audio else int(audio["pause_ms"] is not None),
            "audio_stop_time_ms": det_audio,
            "audio_stop_error_ms": (
                round(det_audio - det_real, 2)
                if ev.get("tipo") == "interrupcion" and det_audio is not None and det_real is not None else None
            ),
            "audio_reply_latency_ms": rep_audio,
            "error_replica_audio_ms": (
                round(rep_audio - real_rep, 2) if rep_audio is not None and real_rep is not None else None
            ),
        })
    return rows


def offsets_text(rows: list[dict[str, Any]]) -> str:
    """Median and range of the two sides of the stop error, per condition."""
    import statistics

    text_lines = ["", "Los dos lados, en el mismo reloj (interrupciones, mediana [mín, máx] en ms)",
              f"{'S (ms)':>7}  {'llegada de la interrupción al modelo':<38}"
              f"{'última voz: arnés − reproducción':<34}{'réplica: arnés − reproducción':<30}"]
    for s in sorted({f["target_stop_ms"] for f in rows}):
        fs = [f for f in rows if f["target_stop_ms"] == s and f["tipo"] == "interrupcion"]
        parts = []
        for col in ("speech_onset_offset_ms", "last_speech_offset_ms", "replicate_offset_ms"):
            v = [float(f[col]) for f in fs if f[col] not in (None, "")]
            parts.append(f"{statistics.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]" if v else "—")
        text_lines.append(f"{s:>7.0f}  {parts[0]:<38}{parts[1]:<34}{parts[2]:<30}")
    return "\n".join(text_lines) + "\n"


def audio_text(rows: list[dict[str, Any]]) -> str:
    """The audio measurement against the ground truth: detection, false alarms and errors, per condition."""
    import statistics

    def span(v: list[float]) -> str:
        return f"{statistics.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]" if v else "—"

    text_lines = ["", "Medida del audio (WAV de auditoría, ventanas de 20 ms; la de detención de PoC3 en el reporte)",
              f"{'S (ms)':>7}  {'detectadas':<12}{'false_alarms':<10}{'err. detención':<30}err. réplica"]
    for s in sorted({f["target_stop_ms"] for f in rows}):
        fs = [f for f in rows if f["target_stop_ms"] == s and f["fired"]]
        inter = [f for f in fs if f["tipo"] == "interrupcion"]
        others = [f for f in fs if f["tipo"] != "interrupcion"]
        det = sum(1 for f in inter if f["audio_stopped"] == 1)
        false_alarms = sum(1 for f in others if f["audio_stopped"] == 1)
        e = [float(f["audio_stop_error_ms"]) for f in inter if f["audio_stop_error_ms"] not in (None, "")]
        r = [float(f["error_replica_audio_ms"]) for f in inter if f["error_replica_audio_ms"] not in (None, "")]
        text_lines.append(f"{s:>7.0f}  {f'{det}/{len(inter)}':<12}{f'{false_alarms}/{len(others)}':<10}{span(e):<30}{span(r)}")
    return "\n".join(text_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from shared.carrier_simulator.interruptions import INTERRUPTION_CLIPS_PATH

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", default=str(RAIZ / "shared" / "test_scenarios" / "clips"))
    p.add_argument("--interruption-clips", default=str(INTERRUPTION_CLIPS_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--work", default=str(RAIZ / "diagnostics" / "interruption_validation" / "poc3"),
                   help="folder for the ground truth, the plan and the bridge files (not committed)")
    p.add_argument("--scenarios", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--stops", nargs="*", type=float, default=list(STOPS_MS))
    p.add_argument("--voice", default=None)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--format-kind", choices=("pcm24", "pcmu"), default="pcm24",
                   help="bridge session format: pcm24 (October 5) or pcmu (October 9)")
    args = p.parse_args(argv)

    clips = Path(args.clips)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    voice = args.voice or choose_voice(clips)
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    run = 0
    for k, s in enumerate(args.stops, start=1):
        for scen in args.scenarios:
            run += 1
            print(f"S = {s:.0f} ms · scenario {scen} ...", flush=True)
            new_items = asyncio.run(run_one(
                s, scen, k, clips=clips, interruption_clips=Path(args.interruption_clips),
                voice=voice, work=work, seed=args.seed + run, format_kind=args.format_kind,
            ))
            for f in new_items:
                print(f"    {f['turn']:<4}{f['tipo']:<13}{f['clip_id']:<8}"
                      f" stopped={f['harness_stopped']} harness={f['harness_stop_time_ms']}"
                      f" real={f['actual_stop_ms']} reply={f['harness_reply_latency_ms']}"
                      f"/{f['real_reply_latency_ms']}")
            rows.extend(new_items)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "interruption_validation_poc3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS_POC3)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows).replace(
        "Validación de la medición de interrupciones (agente que se calla a tiempo conocido)",
        "Validación de la medición de interrupciones por el camino de PoC3 "
        "(puente real, modelo que se calla a tiempo conocido)",
    )
    text += offsets_text(rows)
    text += audio_text(rows)
    (output / "interruption_validation_poc3_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"({(time.monotonic() - start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
