"""`python -m shared.lote`: the full campaign, unattended.

Run it with the interpreter of either PoC: this package is standard library
only, and what imports pipecat or LiveKit is each call's subprocess, each with
its own.

    poc1_pipecat/.venv/Scripts/python.exe -m shared.lote \
        --label prueba --scenarios 1 --repetitions 1 --cap-usd 0.60

Without `--accept-cost` it quotes and spends nothing: it prints the table of
what the whole campaign would cost, using the estimate it asked each harness
for, and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from shared.lote.executor import (
    PoCDescriptor,
    run_call,
    get_quote,
    default_descriptors,
    time_cap,
)
from shared.lote.plan import BatchItem, build_plan, plan_fingerprint, span
from shared.lote.progress import (
    FAILURE,
    OK,
    ROJA,
    append_row,
    finished_keys,
    cumulative_spend,
    read_progress,
    progress_row,
    summarize_batch,
)
from shared.lote.worker import PoC2Worker

RAIZ: Path = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Plan and folder
# --------------------------------------------------------------------------


def campaign_folder(label: str, *, base: Path | None = None) -> Path:
    return (base or (RAIZ / "diagnostics" / "lote")) / label


def write_or_verify_plan(
    folder: Path, items: list[BatchItem], *, parsed_args: dict[str, Any]
) -> str:
    """Save the plan the first time and compare against it on later runs.

    Returns "" if everything matches, or the reason resuming is not possible.
    Resuming with a different plan under the same label is the expensive
    mistake: the keys carry the index, so one extra scenario shifts every index
    and the batch would skip calls that were never made.
    """
    ruta = folder / "plan.json"
    fingerprint = plan_fingerprint(items)
    content = {
        "fingerprint": fingerprint,
        "parsed_args": parsed_args,
        "calls": [i.as_json() for i in items],
    }
    if not ruta.is_file():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ""

    try:
        anterior = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"could not read {ruta}: {error!r}"
    if anterior.get("fingerprint") != fingerprint:
        return (
            f"the label already has a different plan in {ruta}\n"
            f"      saved:     {anterior.get('parsed_args')}\n"
            f"      requested: {parsed_args}\n"
            "      Use another --label, or delete that folder if the previous "
            "campaign no longer matters."
        )
    return ""


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------


def quote_the_plan(
    items: list[BatchItem],
    descriptors: dict[int, PoCDescriptor],
    *,
    folder: Path,
    interruptions: bool = False,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Ask each harness what each scenario costs. Spends nothing.

    One quote per (PoC, scenario) pair rather than per call: the cost of a call
    does not depend on which repetition it is, and each quote costs starting an
    interpreter with its framework on top.
    """
    pairs = sorted({(i.poc, i.scenario) for i in items})
    quotes: dict[tuple[int, int], dict[str, Any]] = {}
    for poc, scenario in pairs:
        descriptor = descriptors[poc]
        sample = next(i for i in items if i.poc == poc and i.scenario == scenario)
        print(f"  quoting PoC{poc} scenario {scenario}...", flush=True)
        quote = get_quote(
            descriptor, sample, root=RAIZ, folder=folder, interruptions=interruptions
        )
        if quote is None:
            print(
                f"  [!] PoC{poc} scenario {scenario} returned no quote: check "
                f"{folder / 'quotes'}"
            )
            continue
        quotes[(poc, scenario)] = quote
    return quotes


def print_cost_table(
    items: list[BatchItem],
    quotes: dict[tuple[int, int], dict[str, Any]],
    *,
    already_spent: float,
    cap_usd: float,
) -> float:
    """The table CLAUDE.md requires before spending. Returns the total."""
    print("\n" + "=" * 70)
    print("Harness batch. NO telephony: no phone rings.")
    print("=" * 70)
    print("\n  pending calls per scenario\n")
    print(f"    {'PoC':<5}{'scen':<5}{'n':<4}{'cap s':>8}{'each':>9}{'subtotal':>11}")
    total = 0.0
    for (poc, scenario), quote in sorted(quotes.items()):
        how_many = sum(1 for i in items if i.poc == poc and i.scenario == scenario)
        per_unit = float(quote.get("estimated_usd") or 0.0)
        subtotal = per_unit * how_many
        total += subtotal
        print(
            f"    {poc:<5}{scenario:<5}{how_many:<4}"
            f"{float(quote.get('seconds_cap') or 0.0):>8.0f}"
            f"{per_unit:>9.3f}{subtotal:>11.2f}"
        )
    unquoted = [
        i for i in items if (i.poc, i.scenario) not in quotes
    ]
    if unquoted:
        print(f"\n  [!] {len(unquoted)} call(s) without a quote: they will not be run")

    print(f"\n  estimated        ~${total:.2f} USD (upper bound)")
    if already_spent:
        print(f"  already spent    ${already_spent:.2f} on the calls already recorded")
        print(f"  campaign total   ~${already_spent + total:.2f}")
    print(f"  cap              ${cap_usd:.2f}: the batch stops on its own when it gets there")
    if already_spent + total > cap_usd:
        print(
            "\n  [!] the full plan does not fit under the cap: the batch will run "
            "what fits\n      and stop with the rest pending."
        )
    return total


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _state_of(resultado: Any) -> tuple[str, str]:
    """Map the subprocess result to the row's status and reason."""
    if resultado.motivo:
        return FAILURE, resultado.motivo
    summary = resultado.summary or {}
    end = summary.get("end") or {}
    if resultado.code_value == 0:
        return OK, str(end.get("motivo") or "")
    # Exit code 1 with a summary: the harness ran and marked the call red. That is
    # campaign data, not a batch failure.
    return ROJA, str(end.get("detail") or end.get("motivo") or "criterios en rojo")


def print_row(row: dict[str, Any], *, total: int) -> None:
    """One line per call. The same as what goes into `progress.jsonl`, readable."""
    mark = {OK: "ok ", ROJA: "RED", FAILURE: "FAIL"}.get(str(row["estado"]), "?")
    p50 = row.get("latency_p50_ms")
    frag = row.get("fragmentation")
    # The old `wer` key is still read for batches from before 21/09/2026; the
    # `~` in the row flags that it is indicative in both cases.
    wer = row.get("indicative_wer", row.get("wer"))
    print(
        f"  [{row['indice']:>3}/{total}] [{mark:<4}] PoC{row['poc']} "
        f"e{row['scenario']}r{row['repetition']} · "
        f"{row.get('turns_reached') if row.get('turns_reached') is not None else '—'}"
        f"/{row.get('script_turns') if row.get('script_turns') is not None else '—'} turns · "
        f"p50 {f'{p50:,.0f} ms' if p50 is not None else '—':>10} · "
        f"frag {f'{frag:.2f}' if frag is not None else '—':>5} · "
        f"WER~{f'{wer * 100:.1f}%' if wer is not None else '—':>6} · "
        f"${row['accumulated_cost_usd']:.2f} · {row.get('end') or row.get('motivo') or ''}"
    )


def print_round_trips(summary: dict[str, Any]) -> None:
    """Tool round trips to the LLM, per PoC and per scenario.

    Printed separately from the status table because it is not a count of
    calls: it is the term from §2.7.1 of the contract, the one that makes the
    p50s of the three comparable. It depends on the script, so the breakdown by
    scenario is the one that matters.

    A PoC that does not fill the key prints no row, instead of printing zero.
    """
    with_data_point = {
        poc: c for poc, c in summary["per_poc"].items()
        if c["round_trips_llm"]["calls_with_data"]
    }
    if not with_data_point:
        return
    print("")
    print("  tool round trips to the LLM (§2.7.1):")
    for poc, running_count in with_data_point.items():
        rt = running_count["round_trips_llm"]
        print(
            f"    PoC{poc}  {rt['total']} in {rt['calls_with_data']} call(s) · "
            f"{rt['per_call']} per call"
        )
        for clave, group in summary["per_scenario"].items():
            if not clave.startswith(f"poc{poc}-"):
                continue
            its_own = group["round_trips_llm"]
            if not its_own["calls_with_data"]:
                continue
            print(
                f"      {clave.split('-', 1)[1]}  {its_own['total']:>3} in "
                f"{its_own['calls_with_data']:>3} · {its_own['per_call']} per call"
            )


def run_the_batch(
    pending_items: list[BatchItem],
    descriptors: dict[int, PoCDescriptor],
    *,
    folder: Path,
    quotes: dict[tuple[int, int], dict[str, Any]],
    cap_usd: float,
    spent: float,
    plan_total: int,
    record_audit: bool,
    worker: PoC2Worker | None,
    interruptions: bool = False,
) -> int:
    """The loop. Returns how many calls were left unrun."""
    progress_path = folder / "progress.jsonl"
    # The campaign folder is built by the batch, not by the subprocess: the
    # harness writes wherever it is told and has no need to know the layout.
    for name in ("summaries", "calls"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    remained = 0
    parado = False

    for item in pending_items:
        if parado:
            remained += 1
            continue
        quote = quotes.get((item.poc, item.scenario))
        if quote is None:
            remained += 1
            continue

        estimated = float(quote.get("estimated_usd") or 0.0)
        # The cap is checked **before** launching, using the estimate of the
        # upcoming call. Checking it afterwards would turn it into a warning.
        if spent + estimated > cap_usd:
            print(
                f"\n  [cap] ${spent:.2f} spent and the next call costs "
                f"~${estimated:.2f}: it would exceed ${cap_usd:.2f}. The batch stops."
            )
            parado = True
            remained += 1
            continue

        if worker is not None and descriptors[item.poc].needs_worker:
            if not worker.ensure():
                row = progress_row(
                    item,
                    estado=FAILURE,
                    summary=None,
                    motivo=f"the PoC{item.poc} worker did not register",
                    duration_s=0.0,
                    cost_usd=0.0,
                    spent_before=spent,
                    when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                append_row(progress_path, row)
                print_row(row, total=plan_total)
                continue

        resultado = run_call(
            descriptors[item.poc],
            item,
            root=RAIZ,
            summary_path=folder / "summaries" / f"{item.clave}.json",
            log_file_path=folder / "calls" / f"{item.clave}.log",
            tope_seg=time_cap(quote.get("seconds_cap")),
            accept_cost=True,
            record_audit=record_audit,
            interruptions=interruptions,
        )
        estado, motivo = _state_of(resultado)
        # A call that crashed may still have spent money, and left no summary to
        # read how much from. It is charged the estimate, which is an upper
        # bound: underestimating spend is the only way a cap stops being a cap.
        cost = (
            float((resultado.summary or {}).get("cost_usd") or 0.0)
            if resultado.had_summary
            else estimated
        )
        row = progress_row(
            item,
            estado=estado,
            summary=resultado.summary,
            motivo=motivo,
            duration_s=resultado.duration_s,
            cost_usd=cost,
            spent_before=spent,
            when=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        # Written **before** moving on to the next one: if the batch dies now,
        # this call is already on disk.
        append_row(progress_path, row)
        spent = float(row["accumulated_cost_usd"])
        print_row(row, total=plan_total)
        if estado == FAILURE:
            print(f"          log at {resultado.log_file_path}")

    return remained


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        prog="shared.lote",
        description=(
            "Runs many harness calls, alternating PoCs. Resumable, with a "
            "spend cap, and a failing call does not bring down the batch."
        ),
    )
    arg_parser.add_argument(
        "--label",
        required=True,
        help="campaign name. Determines the folder and is the key for resuming",
    )
    arg_parser.add_argument(
        "--pocs", default="1,2", help="which PoCs run, interleaved (default 1,2)"
    )
    arg_parser.add_argument(
        "--scenarios", default="1-5", help="script scenarios (default 1-5)"
    )
    arg_parser.add_argument(
        "--repetitions", type=int, default=3,
        help="per scenario and PoC (default 3); with --from-repetition, the last one",
    )
    arg_parser.add_argument(
        "--from-repetition", type=int, default=1,
        help=(
            "first repetition of the plan (default 1). To extend a campaign "
            "already run, under another label, without repeating the existing ones"
        ),
    )
    arg_parser.add_argument(
        "--only",
        default="",
        help=(
            "only these scenario:repetition combinations, comma-separated "
            "(e.g. 5:1,1:2). They keep their index from the full plan, to "
            "rerun failed calls under a new label"
        ),
    )
    arg_parser.add_argument(
        "--cap-usd",
        type=float,
        required=True,
        help="the batch stops on its own before exceeding this",
    )
    arg_parser.add_argument(
        "--record-audit",
        action="store_true",
        help=(
            "one two-channel WAV per call. It is for LISTENING to them: it does "
            "not feed latencies, WER or any percentile. Large files"
        ),
    )
    arg_parser.add_argument(
        "--interruptions",
        action="store_true",
        help=(
            "interruption scenario: each call carries events over the agent's "
            "speech (see shared/carrier_simulator/interruptions.py). Use its "
            "own --label: it does not mix with a normal campaign"
        ),
    )
    arg_parser.add_argument(
        "--no-worker",
        action="store_true",
        help=(
            "do not start the PoC2 worker: one is already up in another terminal. "
            "With this the batch also does not restart it if it dies"
        ),
    )
    arg_parser.add_argument(
        "--base",
        default=None,
        help="where the campaign folders go (default diagnostics/lote/)",
    )
    arg_parser.add_argument(
        "--accept-cost", action="store_true", help="required to execute"
    )
    return arg_parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        pocs = span(args.pocs)
        scenarios = span(args.scenarios)
        items = build_plan(
            pocs=pocs, scenarios=scenarios, repetitions=args.repetitions,
            from_repetition=args.from_repetition,
        )
        if args.only:
            chosen_ones = {
                tuple(int(x) for x in pair.split(":"))
                for pair in args.only.split(",") if pair.strip()
            }
            items = [i for i in items if (i.scenario, i.repetition) in chosen_ones]
            if not items:
                raise ValueError(f"--only matches no call in the plan: {args.only}")
    except ValueError as error:
        print(f"[FALLA] {error}")
        return 1

    descriptors = default_descriptors(RAIZ)
    missing_items = [p for p in pocs if p not in descriptors or not descriptors[p].exists]
    if missing_items:
        for poc in missing_items:
            where = descriptors[poc].interpreter if poc in descriptors else "?"
            print(f"[FAIL] PoC{poc} has no interpreter at {where}")
        return 1

    base = Path(args.base) if args.base else None
    folder = campaign_folder(args.label, base=base)
    problem = write_or_verify_plan(
        folder,
        items,
        parsed_args={
            "pocs": pocs,
            "scenarios": scenarios,
            "repetitions": args.repetitions,
            # Only when it is not the default: that way the plan of existing
            # campaigns is still written the same and they resume without complaint.
            **({"from_repetition": args.from_repetition} if args.from_repetition != 1 else {}),
            **({"solo": args.only} if args.only else {}),
        },
    )
    if problem:
        print(f"[FALLA] {problem}")
        return 1

    rows, discarded_ones = read_progress(folder / "progress.jsonl")
    done = finished_keys(rows)
    spent = cumulative_spend(rows)
    pending_items = [i for i in items if i.clave not in done]

    print(f"\nCampaign {args.label} — {folder}")
    print(f"  plan             {len(items)} calls ({'+'.join(f'PoC{p}' for p in pocs)}, "
          f"scenarios {args.scenarios}, "
          + (f"repetitions {args.from_repetition}-{args.repetitions})"
             if args.from_repetition != 1 else f"{args.repetitions} repetition(s))"))
    if done:
        print(f"  already recorded {len(done)} · ${spent:.2f} spent")
    if discarded_ones:
        print(
            f"  [!] {discarded_ones} unreadable line(s) in the progress file: they are "
            "ignored and those calls are run again"
        )
    print(f"  pending          {len(pending_items)}")
    if not pending_items:
        print("\nNothing left to run.")
        print(json.dumps(summarize_batch(rows), ensure_ascii=False, indent=2))
        return 0

    print()
    quotes = quote_the_plan(
        pending_items, descriptors, folder=folder, interruptions=args.interruptions
    )
    print_cost_table(
        pending_items, quotes, already_spent=spent, cap_usd=args.cap_usd
    )
    if not args.accept_cost:
        print("\nRun it again with --accept-cost to execute it.")
        return 0

    needs_worker = any(descriptors[i.poc].needs_worker for i in pending_items)
    worker: PoC2Worker | None = None
    if needs_worker and not args.no_worker:
        worker = PoC2Worker(
            interpreter=descriptors[2].interpreter,
            root=RAIZ,
            log_file_path=folder / "worker.log",
        )
        print("\n  starting the PoC2 worker...", flush=True)
        if not worker.start():
            print(
                f"  [!] the worker did not register. Log at {folder / 'worker.log'}.\n"
                "      PoC2 calls will be recorded as failures; PoC1 calls run."
            )
    elif needs_worker:
        print("\n  --no-worker: assuming the PoC2 worker is already up.")

    print(f"\n  live progress: Get-Content -Wait {folder / 'progress.jsonl'}\n")
    interrupted = False
    try:
        remained = run_the_batch(
            pending_items,
            descriptors,
            folder=folder,
            quotes=quotes,
            cap_usd=args.cap_usd,
            spent=spent,
            plan_total=len(items),
            record_audit=args.record_audit,
            worker=worker,
            interruptions=args.interruptions,
        )
    except KeyboardInterrupt:
        interrupted = True
        remained = 0
        print(
            "\n\n  [Ctrl+C] the in-flight call was killed and left NO line: "
            "relaunching with the\n           same --label reruns it "
            "from there."
        )
    finally:
        if worker is not None:
            worker.halt()

    rows, _ = read_progress(folder / "progress.jsonl")
    summary = summarize_batch(rows)
    print("\n" + "=" * 70)
    print(f"Campaign {args.label}: {summary['calls']} call(s) recorded, "
          f"${summary['cost_usd']:.2f}")
    for poc, running_count in summary["per_poc"].items():
        print(
            f"  PoC{poc}  {running_count['calls']:>3} calls · {running_count['ok']} ok · "
            f"{running_count['failed_calls']} red · {running_count['failures']} failures · "
            f"${running_count['cost_usd']:.2f}"
        )
    print_round_trips(summary)
    if remained:
        print(f"  {remained} left unrun: relaunch with the same --label")
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
