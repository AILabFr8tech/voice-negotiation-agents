"""PoC3 harness: acts out the script against the full bridge, without a phone.

It is the third sibling of `poc1_pipecat/app/harness.py` and `poc2_livekit/app/harness.py`,
and shares with them everything that decides what a measurable turn is: the player,
the advance rule, the zero cross-check, the log and the report live in
`shared/carrier_simulator/`. What PoC3 contributes is its transport —the Media Streams
WebSocket, same as PoC1— and its bridge to GPT-Live.

## What runs and what does not

The **production** bridge runs: `telefonia.create_app`, with the voice model,
the delegation to `gpt-4.1-mini`, tool calling and the negotiation engine. Not
a copy of it. The only thing passed in from here is `on_finish`, which does not
touch the audio path: it signals when writing has finished so that the server
shuts itself down.

**The text echo is on**, which is what lets the harness learn
what the agent said. In a campaign the other side is this harness and not Twilio, so
the echo is the default and is only turned off with `POC3_SIN_ECO_DE_TEXTO`. A
deaf run is not usable and says so in writing: see `text_block`.

Twilio does not run. Nobody is dialed and no number is touched: the `callSid` is
invented by `ClienteDeMediaStreams` with the prefix `ARNES`, which no real SID uses.

## Two processes, and it is not a detail

The server runs in a subprocess, for the same reason as in PoC1: sharing an event
loop between the harness and the bridge injects each one's pauses into the publishing
pacing of the other, and those pauses go straight into what is being measured. It matters even more
here, because the bridge's 8k-to-24k resampling runs on that same loop.

## The zero cross-check

Three files, a single key. The bridge writes
`recordings/call_{call_sid}.json` as in any call and
`recordings/census_{call_sid}.jsonl` with the model's events; the harness
writes `diagnostics/harness/zeros_{call_sid}.jsonl` with the two real zeros of
each turn. The `call_sid` is the same in all three, so joining them is a
`join` by file name and not a heuristic.

## What this harness does NOT do

It does not compute the two-stage discriminant. It leaves the inputs —the marks, the bridge's
wall axis and `end_ms` in the census— in files that are joined by
`call_sid`, and the analysis lives elsewhere. Mixing it in here would mean a run
had to be redone every time a threshold changed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Any, Final

from poc3_gpt_live.app.config import (
    RECORDINGS_PATH,
    load_settings,
    force_utf8_output,
    verify_for_voice,
)
from poc3_gpt_live.app.marks import DIAGNOSTICS_PATH, DiskMarkEmitter
from poc3_gpt_live.app.media_streams import (
    MediaStreamsClient,
    ObservedFromPoC3,
    MediaStreamsPlayer,
)
from poc3_gpt_live.diagnostics.verification import (
    display as print_verification,
    review,
)
from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.carrier_simulator import (
    MAX_AGENT_WAIT_S,
    WAIT_AFTER_CLIP_S,
    MIN_VALID_TURNS,
    AUDIT_PREFIX,
    RUTA_CLIPS_DEFAULT,
    AGENT_SILENCE_S,
    CostContext,
    ReportContext,
    HarnessFailure,
    AuditRecorder,
    ScriptStep,
    Tariffs,
    UnauditableSampleRate,
    event_log_de_la_corrida,
    load_clip,
    load_scenario,
    write_summary,
    negotiation_state,
    wait_until_agent_is_silent,
    evaluate_run,
    print_cost_warning,
    print_report as print_simulator_report,
    audio_frame_line,
    published_frame_line,
    measure_overlap,
    play_script,
    quote_summary,
    call_summary,
    output_sample_rate,
)

# The agent answers over the same wire and at the same rate: Media Streams is
# 8 kHz μ-law in both directions, same as in PoC1. The bridge has already
# undone the 24 k resampling before returning the frame.
LISTEN_SAMPLE_RATE: Final[int] = TELEPHONY_SAMPLE_RATE

# Rates from `docs/costos_poc3.md`. No telephony line item: there is
# none here. `por_minuto` is the `gpt-live-1` voice session, billed by
# session time and per second; `por_llamada` is the delegated model at its measured
# mean, which is a lower bound and is declared as such in the document.
VOICE_COST_PER_MINUTE: Final[float] = 0.05
DELEGATED_COST_PER_CALL: Final[float] = 0.0025

HARNESS_TARIFFS: Final[Tariffs] = Tariffs(
    per_minute=VOICE_COST_PER_MINUTE,
    per_call=DELEGATED_COST_PER_CALL,
)


def estimate_cost(seconds: float) -> float:
    """Approximate cost of a run. No telephony: there is no trunk here.

    In seconds and not rounded minutes: the `gpt-live-1` rate is
    billed per second, and rounding to the minute would overestimate by ~19% with this
    script's durations. The derivation is in `docs/costos_poc3.md`.
    """
    return HARNESS_TARIFFS.estimate(seconds)


# --------------------------------------------------------------------------
# The server: the production bridge, in its own process
# --------------------------------------------------------------------------


async def _serve(port: int) -> None:
    """Start `telefonia.create_app` and wait for **one** call.

    When the harness hangs up, the bridge saves its JSON as in any call
    and signals via `on_finish`; then the server shuts itself down. It is single-call
    by design: leaving it alive would force the harness to kill it, which is
    how the dump gets lost.

    `text_echo` is not forced here. The environment is left to decide it, which is
    what the campaign does, so that the harness does not run with a configuration
    the campaign calls would not have.
    """
    import uvicorn

    from poc3_gpt_live.app.telephony import create_app

    servidor: Any = None

    def _on_finish(call_sid: str, destination: Path) -> None:
        print(f"  bridge json at       {destination}")
        if servidor is not None:
            servidor.should_exit = True

    settings = load_settings()
    app = create_app(settings, on_finish=_on_finish)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    servidor = uvicorn.Server(config)
    await servidor.serve()


# --------------------------------------------------------------------------
# The observed record: what only this side of the wire knows
# --------------------------------------------------------------------------


def annotate_hang_up(observed: ObservedFromPoC3, closed: asyncio.Event) -> Any:
    """The "agent hung up" notice, ready to hand to the client.

    It lives outside `act_out_script` so it can be tested without starting anything: it is
    the signal the script depends on to stop when the agent closes the
    negotiation, and without it the harness keeps playing clips to a bridge
    that no longer exists.
    """

    def _hung_up(motivo: str) -> None:
        if closed.is_set():
            return
        closed.set()
        observed.was_closed_by_agent = True
        observed.close_reason = motivo
        observed.tally(f"el agente colgó: {motivo}")

    return _hung_up


def close_observed(
    observed: ObservedFromPoC3, *, player: Any, client: Any
) -> None:
    """Dump into `observado` what only the transport knows.

    Separate for the same reason as `annotate_hang_up`: it runs in the `finally` of an
    already-paid run, which is where nobody looks until a number is missing.
    """
    observed.publishing_pacing = player.pacing.summary()
    observed.measured_sends = list(player.pacing.reales)
    observed.frames_without_destination = client.frames_without_destination


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait_for_server(
    port: int, process: Any, tope_seg: float = 120.0
) -> None:
    """Wait until `/ws` accepts connections.

    PoC3 starts faster than PoC1 —it loads neither Silero nor Smart Turn: its
    turn detection lives on OpenAI's server and there is no knob— but
    importing FastAPI and the websockets client is not free.
    """
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if process.returncode is not None:
            raise HarnessFailure(
                f"the harness server died on startup "
                f"(code {process.returncode})"
            )
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.25)
    raise HarnessFailure(f"the server did not open port {port} within {tope_seg:.0f} s")


async def act_out_script(
    url: str,
    *,
    script: list[ScriptStep],
    folder: Path,
    label: str,
    wait_after_clip: float,
    recorder: Any = None,
) -> tuple[ObservedFromPoC3, str]:
    """Act out the script against a bridge that is ALREADY listening on `url`.

    Separate from `run_session` on purpose: no process is started here
    and no credential is read, so this function can be run in full
    against a test bridge and **without spending a cent**. That is what
    `TestLaRutaCompletaContraElAlphaFalso` does, which is the only way to know that
    the pieces fit without paying for a call to find out.
    """
    folder.mkdir(parents=True, exist_ok=True)

    observed = ObservedFromPoC3(recorder=recorder)
    # The agent closed and hung up. It is one of the two end conditions, not an
    # error: in scenario 1 the agent closes at T7 and there are script turns
    # still to act out.
    closed = asyncio.Event()
    _agent_hung_up = annotate_hang_up(observed, closed)

    client = MediaStreamsClient(
        url,
        label=label,
        # No `escenario`/`repeticion` on purpose: labeling this run would
        # make it indistinguishable from a campaign run in the bridge JSON.
        identification={"poc": "3"},
        on_receive_audio=observed.annotate_audio_in_ulaw,
        on_receive_message=observed.annotate_bridge_text,
        on_other_side_closed=_agent_hung_up,
    )

    player = MediaStreamsPlayer(
        client, observed=observed, recorder=recorder
    )
    emitter = DiskMarkEmitter(
        call_id=client.call_sid,
        publish_sample_rate=player.sample_rate,
        folder=folder,
    )

    halt = asyncio.Event()
    async with client:
        pumping = asyncio.create_task(player.launch(halt))
        try:
            # The agent speaks first: its greeting starts when the bridge sees the
            # call come in. It is allowed to finish before the first clip is
            # released, or the first turn measures an overlap and not a response.
            motivo = await wait_until_agent_is_silent(
                observed,
                silence_s=AGENT_SILENCE_S,
                tope_seg=MAX_AGENT_WAIT_S,
            )
            print(f"  [{observed.now():>7.3f}s] {motivo}\n")

            await play_script(
                script,
                player=player,
                observed=observed,
                wait_for_agent_s=wait_after_clip,
                emitter=emitter,
                closed=closed,
                estado=negotiation_state(script),
            )
        except KeyboardInterrupt:  # pragma: no cover - depends on the terminal
            print("\n  cut by hand.")
        finally:
            halt.set()
            # Broad on purpose: whatever comes from here belongs to a run that
            # **has already been paid for**, and letting a pumping failure take down the
            # report, the log and the WAV would throw away all the evidence.
            try:
                await pumping
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 - report and continue
                observed.annotate_incident(
                    f"the publishing loop ended badly: {error!r}"
                )

            try:
                observed.marks_summary_sent = await emitter.shut(
                    turns_without_audio=[
                        t.step.turn for t in observed.turns_without_audio()
                    ]
                )
            except Exception as error:  # noqa: BLE001 - report and continue
                observed.annotate_incident(
                    f"the zeros summary could not be closed: {error!r}"
                )
            observed.sent_marks = emitter.sent
            observed.failed_marks = list(emitter.faults)
            close_observed(observed, player=player, client=client)

    return observed, client.call_sid


async def run_session(
    *,
    script: list[ScriptStep],
    folder: Path,
    label: str,
    wait_after_clip: float,
    recorder: Any = None,
) -> tuple[ObservedFromPoC3, str]:
    """Start the bridge in its own process, act out the script and hang up."""
    from poc3_gpt_live.app.telephony import RUTA_WS

    port = _free_port()
    root = Path(__file__).resolve().parents[2]

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "poc3_gpt_live.app.harness",
        "--run-server",
        "--port",
        str(port),
        cwd=str(root),
    )

    try:
        print("  starting the bridge against GPT-Live...")
        await _wait_for_server(port, process)

        observed, call_sid = await act_out_script(
            f"ws://127.0.0.1:{port}/{RUTA_WS}",
            script=script,
            folder=folder,
            label=label,
            wait_after_clip=wait_after_clip,
            recorder=recorder,
        )

        # Hanging up the WebSocket is what triggers the bridge's dump: the
        # call JSON, the closed census and the bridge audit.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=60)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)

    return observed, call_sid


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def print_report(observed: ObservedFromPoC3, call_sid: str) -> int:
    """The shared report, plus what can only be said about Media Streams."""
    header = [
        f"  published at          {TELEPHONY_SAMPLE_RATE} Hz, μ-law over Media Streams",
        published_frame_line(observed),
        audio_frame_line(observed),
    ]
    if observed.sent_marks:
        summary = "yes" if observed.marks_summary_sent else "NO"
        header.append(
            f"  zeros written         {observed.sent_marks} "
            f"(summary on close: {summary}"
            + (
                f", {len(observed.failed_marks)} failed"
                if observed.failed_marks
                else ""
            )
            + ")"
        )
    if observed.was_closed_by_agent:
        header.append(
            f"  agent hung up         yes — {observed.close_reason}"
            + (
                f" · {observed.frames_without_destination} frame(s) without destination"
                if observed.frames_without_destination
                else ""
            )
        )
    if observed.deaf_to_text:
        header.append(
            "  agent text            NONE — deaf run, NOT usable: "
            "check POC3_SIN_ECO_DE_TEXTO and `text_echo` in the JSON"
        )

    return print_simulator_report(
        observed,
        ReportContext(
            title=f"PoC3 harness — {call_sid}",
            # The alpha exposes no discrete STT: the transcript of what it
            # heard arrives via `input_transcript.added`, from the same model that
            # answers. Calling it "Deepgram" or "the STT" would suggest a stage that
            # does not exist in this architecture.
            stt_name="GPT-Live (native, no discrete STT)",
            header=header,
            where_the_mutes_are_marked="the zero cross-check",
            footer=(
                f"\n  The bridge leaves its JSON in poc3_gpt_live/grabaciones/llamada_{call_sid}.json"
                f"\n  and the event census in censo_{call_sid}.jsonl, with the same key."
                "\n  This is a harness run, NOT a campaign run: do not leave it in that folder if"
                "\n  you are going to run `shared.metrics.cli calcular` on it.",
            ),
        ),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_script(args: argparse.Namespace) -> list[ScriptStep]:
    """Translate the flags into the list of turns to be acted out."""
    folder = Path(args.clips)
    if args.scenario is not None:
        script = load_scenario(
            args.scenario, folder, publish_sample_rate=TELEPHONY_SAMPLE_RATE
        )
    else:
        step = load_clip(Path(args.clip), publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        script = [step] * max(1, int(args.repeat))

    if args.turns:
        # For the smoke test: half a dozen fewer turns is half a dozen turns
        # not paid for. The cut shows in the log through the remaining
        # script, so a short run is not mistaken for a full one.
        script = script[: max(1, int(args.turns))]
    return script


def _print_cost_warning(
    script: list[ScriptStep], seconds: float, *, call_sid: str
) -> None:
    print_cost_warning(
        CostContext(
            title="PoC3 harness run, WITHOUT telephony.",
            tariffs=HARNESS_TARIFFS,
            is_spent=(
                (
                    "gpt-live-1",
                    "the whole voice session, per second, from when it opens "
                    "until it closes — not only while someone is speaking",
                ),
                (
                    "gpt-4.1-mini",
                    "the delegated model: every tool call in the negotiation",
                ),
            ),
            is_not_spent=(
                ("Twilio", "nothing is dialed: the WebSocket is local and the callSid invented"),
                ("ngrok", "no tunnel needed: the harness and the bridge are local"),
                ("STT/TTS", "none: the model has both built in, and that is part of what is measured"),
            ),
            fields=(
                ("callSid", call_sid),
                ("transport", "ws://127.0.0.1 — local Media Streams"),
                ("cost table", "docs/costos_poc3.md — batch cap $25.00"),
            ),
        ),
        script,
        seconds,
    )


def _dump_audit(recorder: Any, call_sid: str, *, folder: Path) -> Path | None:
    """Write the harness audit WAV and its JSON.

    It is not the same file as the bridge's. This one is recorded by **this** side of the
    wire, with channel 0 timestamped by the player, which does know its deadline;
    the bridge's is timestamped when the frame arrives. For PoC3 the difference decides
    what can be asked: see `docs/costos_poc3.md` and the channel 0 finding.
    """
    ruta_wav = folder / f"{AUDIT_PREFIX}_{call_sid}.wav"
    try:
        descriptive = recorder.dump(ruta_wav, room=call_sid)
    except Exception as error:  # noqa: BLE001 - diagnostics: report and continue
        print(f"\n  [!] could not write the audit recording: {error!r}")
        return None

    ruta_json = ruta_wav.with_suffix(".json")
    ruta_json.write_text(
        json.dumps(descriptive, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  Audit (NOT campaign evidence) at {ruta_wav}\n"
        f"    {descriptive['duration_s']:.1f} s at {descriptive['sample_rate_hz']} Hz · "
        f"channel 0 the harness, channel 1 the agent · what it is, in {ruta_json.name}"
    )
    return ruta_wav


def round_trips_of(call_sid: str) -> list[dict[str, Any]] | None:
    """The round trips to the delegated model, read from the JSON the bridge just left.

    The harness does not see them go by: the delegation lives inside `telefonia.create_app`,
    on the other side of the WebSocket. They are read from the file instead of being carried
    through a key of `Observado`, which would be a new path just for this.

    `None` if the file is missing or lacks the key, which is what must
    happen in a run where the agent called no function.
    """
    ruta = RECORDINGS_PATH / f"call_{call_sid}.json"
    if not ruta.is_file():
        return None
    try:
        raw_values = json.loads(ruta.read_text(encoding="utf-8")).get(
            "llm_round_trips"
        )
    except (json.JSONDecodeError, OSError):
        return None
    return raw_values if isinstance(raw_values, list) else None


def unverified(revision: dict[str, Any]) -> int:
    """How many instrument reviews failed. Adds to the criteria.

    They go together on purpose: `evaluate_run` says whether the script was acted out and
    `print_report` whether the run is usable, but neither checks whether the
    INSTRUMENT held up — a call can be acted out in full and still have a
    broken reconciliation. Adding them here makes the batch see as failed the call whose
    clock was not interpretable, which is exactly the one that must not enter the cross-check.
    """
    return sum(1 for r in revision.get("reviews", []) if not r["ok"])


def build_analyzer() -> argparse.ArgumentParser:
    """The CLI, separate so it can be queried without running anything.

    The batch invokes it with `--scenario`, `--repetition`, `--summary` and
    `--accept-cost` (`shared/lote/executor.call_command`): those four
    flags are a contract with it, not a detail of this file.
    """
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.app.harness",
        description=(
            "PoC3 harness: acts out the script against the GPT-Live bridge over a "
            "local WebSocket. It does not use Twilio and dials nobody."
        ),
    )
    arg_parser.add_argument(
        "--run-server",
        action="store_true",
        help="internal mode: starts the bridge and waits. Used by the harness itself",
    )
    arg_parser.add_argument("--port", type=int, help="with --run-server, the port")
    arg_parser.add_argument(
        "--folder",
        default=str(DIAGNOSTICS_PATH),
        help="where the harness marks, log and audit go",
    )
    arg_parser.add_argument(
        "--scenario", type=int, help="script scenario number (1-5)"
    )
    arg_parser.add_argument(
        "--clip", help="a single artifact from shared/audio/ instead of a scenario"
    )
    arg_parser.add_argument(
        "--clips",
        default=str(RUTA_CLIPS_DEFAULT),
        help="folder with the prepared clips",
    )
    arg_parser.add_argument("--repeat", type=int, default=1, metavar="N")
    arg_parser.add_argument(
        "--turns",
        type=int,
        default=0,
        metavar="N",
        help="cut the script to the first N turns. For the smoke test: costs less",
    )
    arg_parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_AFTER_CLIP_S,
        help=(
            f"seconds of observation after the clip "
            f"(default {WAIT_AFTER_CLIP_S:.0f})"
        ),
    )
    arg_parser.add_argument(
        "--record-audit",
        action="store_true",
        help=(
            "write a two-channel WAV of the run (channel 0 the harness, "
            "channel 1 the agent). It is for LISTENING to the run: it does not enter latencies, "
            "nor WER, nor any percentile"
        ),
    )
    arg_parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        metavar="N",
        help="which repetition of the scenario this is. Only labels the summary",
    )
    arg_parser.add_argument(
        "--summary",
        default=None,
        metavar="RUTA",
        help=(
            "write a small JSON with how the call ended, its cost and the "
            "WER. Without --accept-cost it writes the estimate instead and does not "
            "spend. Used by batch mode"
        ),
    )
    arg_parser.add_argument(
        "--accept-cost", action="store_true", help="required to execute"
    )
    arg_parser.add_argument(
        "--interruptions",
        action="store_true",
        help=(
            "interruption scenario: on some turns the harness talks "
            "over the agent (interruption, backchannel or control). Requires "
            "--scenario and --repetition and the clips from "
            "`python -m shared.audio.interrupciones preparar`"
        ),
    )
    arg_parser.add_argument(
        "--interruption-clips",
        default=None,
        help="folder with the prepared interruption clips",
    )
    return arg_parser


def main(argv: list[str] | None = None) -> int:
    args = build_analyzer().parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    folder = Path(args.folder)

    if args.run_server:
        if args.port is None:
            print("[FAILURE] --run-server requires --port")
            return 1
        asyncio.run(_serve(args.port))
        return 0

    if (args.scenario is None) == (args.clip is None):
        print("[FAILURE] give --scenario or --clip, and only one")
        return 1

    settings = load_settings()
    verify_for_voice(settings)
    # `verificar_para_telefonia` is NOT called: this path does not need Twilio.

    try:
        script = build_script(args)
        if args.interruptions:
            from shared.carrier_simulator.interruptions import (
                INTERRUPTION_CLIPS_PATH,
                activate,
            )

            activate(
                script,
                scenario=args.scenario,
                repetition=args.repetition,
                folder=args.interruption_clips or INTERRUPTION_CLIPS_PATH,
            )
        if args.record_audit:
            # Before the cost warning and before spending anything: finding out that the WAV
            # cannot be built after the call would be the worst moment.
            output_sample_rate(TELEPHONY_SAMPLE_RATE, LISTEN_SAMPLE_RATE)
    except (HarnessFailure, UnauditableSampleRate) as error:
        print(f"[FAILURE] {error}")
        return 1

    label = time.strftime("%Y%m%d-%H%M%S")
    call_sid = f"arnes-call-{label}"
    total = MAX_AGENT_WAIT_S + sum(
        step.max_duration_s() + args.wait for step in script
    )

    if not args.accept_cost:
        print()
        _print_cost_warning(script, total, call_sid=call_sid)
        if args.summary:
            write_summary(
                Path(args.summary),
                quote_summary(
                    poc=3,
                    scenario=args.scenario,
                    script=script,
                    seconds=total,
                    cost_usd=estimate_cost(total),
                ),
            )
        return 0

    recorder = (
        AuditRecorder(harness_sample_rate=TELEPHONY_SAMPLE_RATE, agent_sample_rate=LISTEN_SAMPLE_RATE)
        if args.record_audit
        else None
    )

    onset = time.monotonic()
    observed, call_sid = asyncio.run(
        run_session(
            script=script,
            folder=folder,
            label=label,
            wait_after_clip=args.wait,
            recorder=recorder,
        )
    )
    duration = time.monotonic() - onset
    faults = print_report(observed, call_sid)

    log_path = folder / f"event_log_{call_sid}.json"
    event_log = event_log_de_la_corrida(observed, call_sid, scenario=args.scenario)
    event_log["audio_delivery"] = {
        "pacing": observed.publishing_pacing,
        "source_tail": {
            "n": 0,
            "note": (
                "PoC3 has no source queue, same as PoC1: the WebSocket "
                "exposes no occupancy and there is no `AudioSource` between the send and the "
                "wire. See docs/METRICS_CONTRACT.md §2.2.2"
            ),
        },
        "series": {
            "what_they_are": (
                "one entry per published frame: `instante` is the measured send "
                "(wall clock). PoC3 has no `cola_ms` to go with it"
            ),
            "moment": [round(t, 6) for t in observed.measured_sends],
        },
    }
    event_log["text"] = observed.text_block()
    event_log["branches"] = observed.branches_block()
    event_log["text_left_tail"] = observed.left_tail_block()
    log_path.write_text(
        json.dumps(event_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  What was played is in {log_path}")

    ruta_auditoria = (
        _dump_audit(recorder, call_sid, folder=folder)
        if recorder is not None
        else None
    )

    # The seven reviews, over the files this run just left.
    # It goes after writing the log because three of them read it, and before
    # the run verdict because they are different things: `evaluar_corrida`
    # says whether the script was acted out, and this says whether the INSTRUMENT held up.
    revision = review(call_sid, folder=folder, recordings=RECORDINGS_PATH)
    print_verification(revision)
    verification_path = folder / f"verification_{call_sid}.json"
    verification_path.write_text(
        json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if observed.decisions:
        print(
            "  The run branched: correct the WER reference with those\n"
            "  lines, not the script's, or the WER is measured against a different sentence."
        )

    resultado = evaluate_run(observed)

    if args.summary:
        # Before any `return`: the batch needs the summary **also**
        # for the calls that came out failed. An unusable run is a
        # data point of the campaign —and of its cost—, not a gap.
        write_summary(
            Path(args.summary),
            call_summary(
                observed,
                resultado,
                poc=3,
                scenario=args.scenario,
                repetition=args.repetition,
                label=call_sid,
                duration_s=duration,
                cost_usd=estimate_cost(duration),
                failed_criteria=faults + unverified(revision),
                # PoC1 and PoC2 fill `intervenciones` and `camino_de_vuelta` with
                # the cross-check their own pipeline builds. PoC3 **does not have them**, and
                # it goes empty instead of zero: its per-turn latency lives in
                # `latencias` of the bridge JSON, and the cross-check against the real
                # zeros is the two-stage discriminant, which runs outside the
                # harness. A zero here would read as "it was measured and came out zero".
                lines=None,
                return_path=None,
                llm_round_trips=round_trips_of(call_sid),
                overlap=measure_overlap(ruta_auditoria),
                artifacts={
                    "event_log": str(log_path),
                    "verification": str(verification_path),
                    "bridge_json": str(
                        RECORDINGS_PATH / f"call_{call_sid}.json"
                    ),
                    "census": str(RECORDINGS_PATH / f"census_{call_sid}.jsonl"),
                    **(
                        {"auditoria_wav": str(ruta_auditoria)}
                        if ruta_auditoria
                        else {}
                    ),
                },
            ),
        )

    print()
    if resultado.retry:
        print(
            f"Unusable run ({resultado.motivo}): {resultado.reached} of "
            f"{resultado.of_script} turns, minimum {MIN_VALID_TURNS}."
        )
        return 1
    if faults:
        print(f"{faults} criterion/criteria failed.")
        return 1
    print(
        f"Green: {resultado.reached} turn(s) acted out, heard and answered "
        f"({resultado.motivo})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
