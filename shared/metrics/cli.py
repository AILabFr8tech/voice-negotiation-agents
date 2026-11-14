"""Command line for the metrics. It is the whole workflow, in four verbs.

    # 1. After each call: generate the pre-filled reference file.
    python -m shared.metrics.cli reference --json <call.json> --scenario 2 \
        --repetition 1 --poc 1

    # 2. Correct the `yo:` lines by hand and enter the two 1-5 ratings (2-3 min).

    # 3. Compute the metrics and write them back into the call's JSON.
    python -m shared.metrics.cli compute --folder poc1_pipecat/grabaciones

    # 4. Consolidate the PoC into Markdown + CSV.
    python -m shared.metrics.cli report --folder poc1_pipecat/grabaciones \
        --poc "PoC1 — Pipecat" --output docs/metrics

    # And to check that the whole computation works without spending a call:
    python -m shared.metrics.cli demo

Only what distinguishes one voice technology from another is measured: latency,
STT and turn-taking. What the agent negotiates is the vehicle of the
conversation, not the object of study.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from shared.metrics import latency as lat
from shared.metrics import calls as mod_calls
from shared.metrics import reference as ref
from shared.metrics import clips_reference as clips
from shared.metrics import report as mod_report
from shared.metrics.scenarios import Script, load_script
from shared.metrics.text import CONFIG_DEFAULT

logger = logging.getLogger(__name__)

MARKDOWN_NAME = "metricas_{poc}.md"
CSV_NAME = "metricas_{poc}.csv"


def _slug(text: str) -> str:
    cleaned = [c if c.isalnum() else "_" for c in text.strip().lower()]
    return "".join(cleaned).strip("_") or "poc"


def _batch_jsons(
    folder: Path, tope: int | None, poc: int | None, *, con_reference: bool = False
) -> list[Path]:
    """The calls of a batch, and **only** those.

    It exists because `--folder grabaciones/` mixes pools: the harness calls
    live there alongside the August **telephony** calls, which are a different
    transport and a different reference method (`CONTRATO_METRICAS.md` §2.10).
    Consolidating both together adds up two different things and the result
    means nothing. The batch's `progress.jsonl` is the exact list of what goes in.
    """
    import json

    progress = folder / "progress.jsonl"
    if not progress.is_file():
        raise SystemExit(f"cannot find {progress}")
    paths: list[Path] = []
    for linea in progress.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        row = json.loads(linea)
        if row.get("estado") != "ok":
            continue
        if tope is not None and int(row.get("indice", 0)) > tope:
            continue
        if poc is not None and int(row.get("poc", 0)) != poc:
            continue
        summary = folder / "summaries" / f"{row['clave']}.json"
        if not summary.is_file():
            continue
        art = json.loads(summary.read_text(encoding="utf-8")).get("artifacts") or {}
        destination = art.get("pipeline_json") or art.get("worker_json")
        if not destination or not Path(destination).is_file():
            continue
        if con_reference:
            # Unpaired runs have no derived reference (§2.10), and letting
            # them in would give a different `n` for latency than for WER
            # within the same report.
            sid = str(json.loads(Path(destination).read_text(encoding="utf-8")).get("call_sid") or "")
            if ref.search(Path(destination).parent, sid) is None:
                continue
        paths.append(Path(destination))
    if not paths:
        raise SystemExit(f"no `ok` calls with JSON in {folder}")
    return paths


def _jsons_to_process(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "lote", None):
        return _batch_jsons(
            Path(args.lote),
            getattr(args, "until", None),
            getattr(args, "only_poc", None),
            con_reference=getattr(args, "con_reference", False),
        )
    if getattr(args, "json", None):
        destination = Path(args.json)
        if not destination.is_file():
            raise SystemExit(f"cannot find {destination}")
        return [destination]
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"cannot find the folder {folder}")
    found_ones = mod_calls.list_calls(folder)
    if not found_ones:
        raise SystemExit(f"no `call_*.json` files in {folder}")
    return found_ones


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------


def _batch_runs(folder: Path, tope: int | None) -> list[tuple[clips.ClipRun, Path]]:
    """Read the batch and return each run together with its call's JSON."""
    import json

    progress = folder / "progress.jsonl"
    rows = [
        json.loads(l)
        for l in progress.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    output: list[tuple[clips.ClipRun, Path]] = []
    for row in rows:
        if row.get("estado") != "ok":
            continue
        if tope is not None and int(row.get("indice", 0)) > tope:
            continue
        summary = folder / "summaries" / f"{row['clave']}.json"
        if not summary.is_file():
            continue
        art = json.loads(summary.read_text(encoding="utf-8")).get("artifacts") or {}
        event_log = art.get("event_log")
        call = art.get("pipeline_json") or art.get("worker_json")
        if not event_log or not call or not Path(call).is_file():
            continue
        data = json.loads(Path(event_log).read_text(encoding="utf-8"))
        output.append(
            (
                clips.read_run(
                    data,
                    clave=str(row["clave"]),
                    poc=int(row["poc"]),
                    repetition=str(row.get("repetition", "")),
                    call_sid=str(json.loads(Path(call).read_text(encoding="utf-8")).get("call_sid") or ""),
                ),
                Path(call),
            )
        )
    return output


def cmd_clip_reference(args: argparse.Namespace) -> int:
    """Derive the WER reference from the played clips, with no hand correction.

    Valid only for harness runs: the carrier is a `.ulaw`, so what was heard is
    exactly the clip's text. See `CONTRATO_METRICAS.md` §2.10.
    """
    import json

    folder = Path(args.campana)
    runs = _batch_runs(folder, args.until)
    if not runs:
        raise SystemExit(f"no `ok` runs with artifacts in {folder}")

    manifest = clips.clips_manifest(args.manifest)
    problems = [
        f"{c.clave}: {p}" for c, _ in runs for p in clips.verify_against_manifest(c, manifest)
    ]
    if problems:
        for p in problems[:20]:
            print(f"  MISMATCH  {p}")
        raise SystemExit(
            f"{len(problems)} turns where the log and the manifest do not "
            "match: the derived reference is not reliable"
        )
    print(f"cross-check against the manifest: {sum(len(c.turns) for c, _ in runs)} turns, 0 mismatches")

    if args.pair_up:
        pairs = clips.match_pairs([c for c, _ in runs])
        trimmed_items: list[tuple[clips.ClipRun, Path]] = []
        loose_files = 0
        for run, ruta in runs:
            group = pairs[run.pair]
            if len(group) < 2:
                loose_files += 1
                continue
            trimmed_items.append((clips.trim_to_pair(run, clips.common_clips(group)), ruta))
        print(f"pairing by (scenario, repetition): {len(trimmed_items)} runs, "
              f"{loose_files} unpaired and discarded")
        runs = trimmed_items

    written_list = skipped = 0
    for run, ruta in runs:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        content = clips.generate(
            run,
            transcript=data.get("transcript") or [],
            script_version=args.script_version,
        )
        destination = ref.path_to_reference(ruta.parent, run.call_sid)
        try:
            ref.write(content, destination, overwrite=args.overwrite)
            written_list += 1
        except FileExistsError:
            skipped += 1

    print(f"references written: {written_list}   skipped (already existed): {skipped}")
    _coverage_table(runs)
    return 0


def _coverage_table(runs: Sequence[tuple[clips.ClipRun, Path]]) -> None:
    """Which script turn each PoC reached, per scenario.

    It belongs in the method, not as a finding: it says whether the two pools
    contain the same stimulus, which is sample validity. It does not interpret
    why they differ.
    """
    from collections import Counter, defaultdict

    cob: dict[tuple[int, int], Counter] = defaultdict(Counter)
    n: Counter = Counter()
    for run, _ in runs:
        n[(run.poc, run.scenario)] += 1
        for t in run.turns:
            cob[(run.poc, run.scenario)][t.turn] += 1

    print("\nturn coverage (% of calls in that bucket that played the turn)")
    for scenario in sorted({e for _, e in cob}):
        print(f"  scenario {scenario}:")
        for poc in sorted({p for p, e in cob if e == scenario}):
            total = n[(poc, scenario)]
            cells = "  ".join(
                f"{t}:{cob[(poc, scenario)][t] * 100 // total:3d}%"
                for t in sorted(cob[(poc, scenario)], key=lambda x: int(x[1:] or 0))
            )
            print(f"    PoC{poc} (n={total:2d})  {cells}")


def cmd_reference(args: argparse.Namespace) -> int:
    """Generate the reference file pre-filled with the script."""
    script = load_script(args.script)
    paths = _jsons_to_process(args)

    if len(paths) > 1 and args.scenario is not None:
        raise SystemExit(
            "--scenario applies to a single call; with --folder the scenario "
            "comes from the JSON or is left blank for you to fill in"
        )

    generated_items = 0
    for ruta_json in paths:
        data = mod_calls.load_json(ruta_json)
        call_sid = str(data.get("call_sid") or "")

        poc, scenario_id, repetition = mod_calls.identify(data, None)
        if args.scenario is not None:
            scenario_id = args.scenario
        if args.repetition is not None:
            repetition = args.repetition
        if args.poc is not None:
            poc = args.poc

        scenario = None
        if scenario_id is not None:
            try:
                scenario = script.scenario(int(scenario_id))
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc

        content = ref.generate(
            scenario=scenario,
            transcript=data.get("transcript") or [],
            poc=poc if poc is not None else "",
            repetition=repetition if repetition is not None else "",
            call_sid=call_sid,
            script_version=script.version,
        )

        destination = ref.path_to_reference(ruta_json.parent, call_sid)
        try:
            written = ref.write(content, destination, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"  skipped    {destination.name} — already exists (use --overwrite)")
            logger.debug("%s", exc)
            continue

        generated_items += 1
        if scenario is None:
            print(
                f"  generated  {written.name} — NO scenario: fill in `escenario:` "
                "by hand and transcribe from scratch"
            )
        else:
            print(f"  generated  {written.name} — seeded with scenario {scenario.label}")

    print()
    print(f"{generated_items} reference file(s) generated.")
    if generated_items:
        print(
            "Now, for each one: correct only the `yo:` lines where you "
            "deviated from the script, and enter `naturalness` and `turn_taking` (1-5). "
            "That is all the manual work."
        )
    return 0


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------


def cmd_compute(args: argparse.Namespace) -> int:
    """Compute the metrics and write them into each call's JSON."""
    paths = _jsons_to_process(args)

    with_warnings = 0
    for ruta in paths:
        metrics = mod_calls.enrich(ruta, config=CONFIG_DEFAULT)
        stt = metrics.stt or {}
        wer = ((stt.get("full_channel") or {}).get("wer") or {}).get("sample_rate")
        p50 = mod_report.bucket_value(
            metrics.latency, "regimen", "full_turn", "p50"
        )
        print(
            f"  {ruta.name}\n"
            f"      scenario {metrics.label}  "
            f"p50 steady state {'—' if p50 is None else f'{p50:,.0f}'} ms  "
            f"WER {'—' if wer is None else f'{wer * 100:.1f}%'}"
        )
        for aviso in metrics.warning_list:
            print(f"      · {aviso}")
        if metrics.warning_list:
            with_warnings += 1

    print()
    print(
        f"{len(paths)} call(s) computed; `{mod_calls.METRICS_KEY}` block "
        f"written into each JSON. {with_warnings} with warnings."
    )
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """Check that a call was captured correctly. Computes nothing.

    It is the check for the **warm-up call**, the one that does not count: it
    confirms that the JSON carries the latency stages, both sides of the
    dialogue and the run label, before spending the 15 formal ones.

    It matters because two stages —`end_of_turn` and `stt`— are only filled in
    a real call: the smoke test has no end-of-turn analyzer and no inbound STT,
    so it cannot exercise them. If they come out empty here, stop.
    """
    paths = _jsons_to_process(args)
    problems = 0

    for ruta in paths:
        data = mod_calls.load_json(ruta)
        samples = data.get(mod_calls.SAMPLES_KEY) or []
        transcript = data.get("transcript") or []
        roles = sorted({str(t.get("role", "")) for t in transcript if t.get("role")})

        print(f"\n{ruta.name}")
        label = " ".join(
            f"{c}={data.get(c)}" for c in ("poc", "scenario", "repetition")
        )
        print(f"  run label                 {label}")
        if data.get("scenario") is None:
            problems += 1
            print(
                "  [X] no scenario: tag it with --scenario/--repetition or "
                "label it by hand in the reference"
            )

        print(f"  roles in the transcript   {roles}")
        if roles != ["agent", "carrier"]:
            problems += 1
            print("  [X] one of the two sides of the dialogue is missing")

        print(f"  latency samples           {len(samples)}")
        if not samples:
            problems += 1
            print("  [X] no samples: the pipeline is not instrumented")
            continue

        for campo in (lat.CAMPO_PRINCIPAL, *lat.STAGE_FIELDS):
            with_data_point = sum(1 for m in samples if m.get(campo) is not None)
            mark = "   " if with_data_point else "[X]"
            if not with_data_point:
                problems += 1
            print(f"  {mark} {campo:<20} {with_data_point}/{len(samples)} turns")

    print()
    if problems:
        print(
            f"{problems} problem(s). In a REAL call all five stages must "
            "carry data; if `end_of_turn` or `stt` come out empty, stop before "
            "running the 15."
        )
        return 1
    print("Capture complete. Ready for the 15 formal calls.")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """Consolidate all the calls of a PoC into Markdown and CSV."""
    paths = _jsons_to_process(args)

    metrics: list[mod_calls.CallMetrics] = []
    for ruta in paths:
        _, una = mod_calls.compute_from_path(ruta, config=CONFIG_DEFAULT)
        metrics.append(una)

    consolidated = mod_report.consolidate(metrics, poc=args.poc)

    output = Path(args.output)
    slug = _slug(args.poc)
    md_path = mod_report.write_markdown(
        consolidated, output / MARKDOWN_NAME.format(poc=slug)
    )
    ruta_csv = mod_report.write_csv(
        consolidated, output / CSV_NAME.format(poc=slug)
    )

    print(f"  Markdown  {md_path}")
    print(f"  CSV       {ruta_csv}")
    print()
    if consolidated.warning_list:
        print("Warnings:")
        for aviso in consolidated.warning_list:
            print(f"  · {aviso}")
    return 0


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole workflow with synthetic data. Spends no calls."""
    from shared.metrics.synthetic import synthetic_corpus, write_corpus

    if args.output:
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        temporal = None
    else:
        temporal = tempfile.TemporaryDirectory(prefix="demo_metrics_")
        folder = Path(temporal.name)

    try:
        calls = synthetic_corpus(poc=1)
        write_corpus(folder, calls)
        print(
            f"Synthetic corpus: {len(calls)} calls "
            f"(5 scenarios × 3 repetitions) in {folder}\n"
        )

        computed: list[mod_calls.CallMetrics] = []
        for ruta in mod_calls.list_calls(folder):
            computed.append(mod_calls.enrich(ruta))

        consolidated = mod_report.consolidate(computed, poc="PoC1 — Pipecat (demo)")
        markdown = mod_report.generate_markdown(consolidated)

        md_path = mod_report.write_markdown(consolidated, folder / "consolidated.md")
        ruta_csv = mod_report.write_csv(consolidated, folder / "consolidated.csv")

        print(markdown)
        print("=" * 74)
        print(f"Markdown  {md_path}")
        print(f"CSV       {ruta_csv}")
        if temporal is not None:
            print(
                "\n(Temporary folder: deleted on exit. Use --output <dir> to "
                "keep it.)"
            )
        return 0
    finally:
        if temporal is not None:
            temporal.cleanup()


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def _add_source(sub: argparse.ArgumentParser, *, require: bool = True) -> None:
    group = sub.add_mutually_exclusive_group(required=require)
    group.add_argument("--json", help="a single call JSON")
    group.add_argument("--folder", help="folder with `call_*.json` files")
    group.add_argument(
        "--lote",
        help=(
            "a batch folder: takes EXACTLY its `ok` calls and leaves out "
            "the telephony calls that share `grabaciones/` (§2.10)"
        ),
    )
    sub.add_argument("--until", type=int, default=None, help="with --lote: last index")
    sub.add_argument(
        "--only-poc", type=int, default=None, dest="only_poc",
        help="with --lote: only the calls of that PoC",
    )
    sub.add_argument(
        "--with-reference", action="store_true", dest="con_reference",
        help=(
            "with --lote: only those with a derived reference, i.e. the paired "
            "pool — so `n` is the same for latency and for WER"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        prog="python -m shared.metrics.cli",
        description=(
            "Automatic computation of the voice comparison metrics: "
            "latency percentiles, WER/CER with Spanish normalization, and "
            "per-PoC consolidation."
        ),
    )
    arg_parser.add_argument(
        "--script",
        default=None,
        help="alternative path to escenarios.yaml (default: the repo's)",
    )
    subs = arg_parser.add_subparsers(dest="command", required=True)

    p_ref = subs.add_parser(
        "reference",
        help="generate the reference file pre-filled with the script",
    )
    _add_source(p_ref)
    p_ref.add_argument("--scenario", type=int, help="scenario number (1-5)")
    p_ref.add_argument("--repetition", type=int, help="repetition number (1-3)")
    p_ref.add_argument("--poc", type=int, help="PoC number (1-3)")
    p_ref.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing file (LOSES the hand corrections)",
    )
    p_ref.set_defaults(function=cmd_reference)

    p_clips = subs.add_parser(
        "referencia-de-clips",
        help=(
            "derive the WER reference from the played clips (harness runs "
            "only; there is nothing to correct by hand)"
        ),
    )
    p_clips.add_argument(
        "--campana", required=True, help="batch folder, with progress.jsonl and summaries/"
    )
    p_clips.add_argument(
        "--until", type=int, default=None, help="last index to consider (pool cutoff)"
    )
    p_clips.add_argument(
        "--manifest",
        default="shared/test_scenarios/clips/manifest.json",
        help="clip manifest, for the cross-check",
    )
    p_clips.add_argument(
        "--no-pair",
        dest="pair_up",
        action="store_false",
        help="do not trim to the pair's common subset (the pools stop being comparable)",
    )
    p_clips.add_argument("--script-version", default="", help="script version label")
    p_clips.add_argument(
        "--overwrite", action="store_true", help="replace existing references"
    )
    p_clips.set_defaults(function=cmd_clip_reference)

    p_calc = subs.add_parser(
        "compute", help="compute and write the `metrics` block into each JSON"
    )
    _add_source(p_calc)
    p_calc.set_defaults(function=cmd_compute)

    p_ver = subs.add_parser(
        "verify",
        help="check that the call was captured correctly (use it on the warm-up call)",
    )
    _add_source(p_ver)
    p_ver.set_defaults(function=cmd_verify)

    p_rep = subs.add_parser("report", help="per-PoC consolidation in Markdown and CSV")
    _add_source(p_rep)
    p_rep.add_argument(
        "--poc", default="PoC sin nombre", help='label, e.g. "PoC1 — Pipecat"'
    )
    p_rep.add_argument(
        "--output", default="docs/metrics", help="folder to write the files into"
    )
    p_rep.set_defaults(function=cmd_report)

    p_demo = subs.add_parser(
        "demo",
        help="validate the whole computation with synthetic data, without spending calls",
    )
    p_demo.add_argument(
        "--output", default=None, help="keep the synthetic corpus in this folder"
    )
    p_demo.set_defaults(function=cmd_demo)

    return arg_parser


def _configure_output() -> None:
    """Force UTF-8 on the output.

    The Windows console arrives in cp1252, which fits neither the accents nor
    the arrows in the tables: without this, the report crashed with
    `UnicodeEncodeError` when printed. `errors="replace"` is the fallback in case
    the terminal cannot handle UTF-8 either — better an odd character on screen
    than losing the whole report.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure_stream = getattr(stream, "reconfigure", None)
        if reconfigure_stream is None:
            continue
        try:
            reconfigure_stream(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic terminal
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    function: Any = args.function
    return int(function(args))


if __name__ == "__main__":
    sys.exit(main())
