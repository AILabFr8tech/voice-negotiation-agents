"""What the report needs that exists only on the campaign machine.

Run once on that machine; it writes to `docs/report/data/`:

- `populations.csv`: one row per call run (both campaigns): status, how many
  criteria failed and which ones, reconstructed from the summary and the log.
- `poblaciones_summary.txt`: calls run, excluded by reason, and analyzed, per
  PoC and per scenario.
- `delegaciones_sample.jsonl`: what GPT-Live sent to gpt-4.1-mini in some PoC3
  calls (the text request, the function it asked for and with which
  arguments), and which fields the delegate's response carried.
- `environment.txt`: operating system, CPU, RAM, versions of Python and of the
  analysis libraries, the **non-secret** values from `.env`, and the model and
  voice the GPT-Live server confirmed.

**It never writes keys or secrets.** From `.env` it reads only the closed list
`PUBLIC_ENV_KEYS`; any other key is ignored without reading its value.

Usage, from the repo root:

    uv run --project poc3_gpt_live --with scipy python -m validation.local_inventory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Iterable

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE_DEFAULT: Final[Path] = RAIZ / "diagnostics" / "lote"
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"

# The campana200 pool runs up to call 180 (`export_for_report`).
TOPE_CAMPANA200: Final[int] = 180

# The only `.env` keys that are read. None of them is secret.
PUBLIC_ENV_KEYS: Final[tuple[str, ...]] = (
    "OPENAI_MODEL",
    "OPENAI_LIVE_MODEL",
    "OPENAI_LIVE_VOICE",
    "DEEPGRAM_MODEL",
    "DEEPGRAM_LANGUAGE",
    "ELEVENLABS_MODEL",
    "ELEVENLABS_VOICE_ID",
    "LIVEKIT_TURN_DETECTOR_VERSION",
)

# The harness's five criteria (`shared/carrier_simulator/reporte.py`), in order.
CRITERIA: Final[tuple[str, ...]] = (
    "clips_not_played",
    "stt_missed_some_clip",
    "insufficient_turns",
    "turn_without_response",
    "response_without_audio",
)

COLUMNS: Final[list[str]] = [
    "campana", "indice", "clave", "poc", "scenario", "repetition", "estado",
    "in_the_pool", "failed_criteria", *CRITERIA, "failed_poc3_reviews",
    "reconstruction_adds_up", "batch_reason", "end",
]


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------


def leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    rows = []
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            rows.append(json.loads(linea))
        except ValueError:
            continue
    return rows


def leer_json(ruta: str | Path | None) -> dict[str, Any] | None:
    if not ruta:
        return None
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except ValueError:
        return None


def reconstructed_criteria(
    summary: dict[str, Any] | None, event_log: dict[str, Any] | None
) -> dict[str, int | None]:
    """Which of the five criteria failed, from what was written down.

    The harness only stores the **number** of failed criteria; the names are
    reconstructed here with the same rules as `print_report`, over the turns
    reached. `None` means there is no data to decide.
    """
    end = (summary or {}).get("end") or {}
    output: dict[str, int | None] = dict.fromkeys(CRITERIA)
    if "valid" in end:
        output["insufficient_turns"] = int(not end["valid"])
    if "turns_without_agent_audio" in end:
        output["response_without_audio"] = int(bool(end["turns_without_agent_audio"]))
    turns = (event_log or {}).get("turns")
    reached = end.get("turns_reached")
    if isinstance(turns, list) and isinstance(reached, int):
        acted_out = turns[:reached]
        spoken_items = [t for t in acted_out if t.get("played_line")]
        # A turn that appears in the log with its line is a clip that was played.
        output["clips_not_played"] = int(not spoken_items)
        output["stt_missed_some_clip"] = int(
            not spoken_items or any(not (t.get("stt") or "").strip() for t in spoken_items)
        )
        output["turn_without_response"] = int(
            not acted_out
            or any(
                not (t.get("agent") or "").strip() and not t.get("discarded_fragments")
                for t in acted_out
            )
        )
    return output


def failed_reviews(verification: dict[str, Any] | None) -> list[str]:
    """The PoC3 instrument reviews that did not pass, by name."""
    return [
        str(r.get("que", "?"))
        for r in (verification or {}).get("reviews", [])
        if not r.get("ok")
    ]


def population_row(
    campana: str, row: dict[str, Any], folder: Path, in_the_pool: bool
) -> dict[str, Any]:
    clave = str(row.get("clave"))
    summary = leer_json(folder / "summaries" / f"{clave}.json")
    art = (summary or {}).get("artifacts") or {}
    criteria = reconstructed_criteria(summary, leer_json(art.get("event_log")))
    reviews = failed_reviews(leer_json(art.get("verification")))
    failed_list = (summary or {}).get("failed_criteria")
    known = [v for v in criteria.values() if v is not None]
    cuadra = (
        ""
        if failed_list is None or len(known) < len(CRITERIA)
        else int(sum(known) + len(reviews) == int(failed_list))
    )
    return {
        "campana": campana,
        "indice": row.get("indice"),
        "clave": clave,
        "poc": row.get("poc"),
        "scenario": row.get("scenario"),
        "repetition": row.get("repetition"),
        "estado": row.get("estado"),
        "in_the_pool": int(in_the_pool),
        "failed_criteria": failed_list,
        **criteria,
        "failed_poc3_reviews": ";".join(reviews),
        "reconstruction_adds_up": cuadra,
        "batch_reason": row.get("motivo") or "",
        "end": row.get("end") or "",
    }


def pairs_ok(rows: Iterable[dict[str, Any]]) -> set[str]:
    """The keys of the PoC1/PoC2 pairs with both calls `ok`."""
    cells: dict[tuple[Any, Any], dict[int, dict[str, Any]]] = defaultdict(dict)
    for f in rows:
        cells[(f["scenario"], f["repetition"])][int(f["poc"])] = f
    keys: set[str] = set()
    for pair in cells.values():
        if all(p in pair and pair[p].get("estado") == "ok" for p in (1, 2)):
            keys |= {str(pair[1]["clave"]), str(pair[2]["clave"])}
    return keys


def populations_summary(rows: list[dict[str, Any]]) -> list[str]:
    text_lines = []
    for campana in sorted({f["campana"] for f in rows}):
        fs = [f for f in rows if f["campana"] == campana]
        text_lines.append(f"== {campana}: {len(fs)} llamadas corridas")
        for poc in sorted({str(f["poc"]) for f in fs}):
            of_poc = [f for f in fs if str(f["poc"]) == poc]
            analyzed = [f for f in of_poc if f["in_the_pool"]]
            text_lines.append(
                f"  PoC{poc}: corridas {len(of_poc)} · analizadas {len(analyzed)} · "
                f"excluidas {len(of_poc) - len(analyzed)}"
            )
            reasons = Counter(exclusion_reason(f) for f in of_poc if not f["in_the_pool"])
            for motivo, n in reasons.most_common():
                text_lines.append(f"      {n:>3}  {motivo}")
            per_scen = Counter(str(f["scenario"]) for f in analyzed)
            runs_esc = Counter(str(f["scenario"]) for f in of_poc)
            text_lines.append(
                "      por escenario (analyzed/runs): "
                + ", ".join(f"E{e} {per_scen[e]}/{runs_esc[e]}" for e in sorted(runs_esc))
            )
    return text_lines


def exclusion_reason(f: dict[str, Any]) -> str:
    if f["campana"] == "campana_poc3_pcmu_reintentos":
        return "repetición de una llamada fallida de PoC3: no entra, el análisis usa solo primeros intentos"
    if f["campana"] == "campana200" and f["indice"] is not None and int(f["indice"]) > TOPE_CAMPANA200:
        return f"después de la llamada {TOPE_CAMPANA200} (fuera de la poza)"
    if f["estado"] != "ok":
        names = [c for c in CRITERIA if f.get(c) == 1]
        if f.get("failed_poc3_reviews"):
            names.append("revisión del instrumento: " + f["failed_poc3_reviews"])
        return f"{f['estado']}: " + (", ".join(names) or "criterio no reconstruible")
    return "otra PoC falló en ese escenario y repetición (regla de las tres PoC)"


def populations(lote: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    from shared.metrics.campaigns import CAMPAIGN_POC3, valid_combinations

    c200 = lote / "campana200"
    c3 = lote / CAMPAIGN_POC3
    if not ((c200 / "progress.jsonl").is_file() and (c3 / "progress.jsonl").is_file()):
        return output
    # Three-PoC rule (fixed before running the PoC3 campaign in μ-law): a
    # scenario and repetition enters if it passed in all three.
    valid_ones = valid_combinations(lote)

    def enters(f: dict[str, Any], campana: str) -> bool:
        if campana == "campana200" and int(f.get("indice") or 0) > TOPE_CAMPANA200:
            return False
        return f.get("estado") == "ok" and (int(f["scenario"]), int(f["repetition"])) in valid_ones

    output += [population_row("campana200", f, c200, enters(f, "campana200"))
               for f in leer_jsonl(c200 / "progress.jsonl")]
    output += [population_row(CAMPAIGN_POC3, f, c3, False) for f in leer_jsonl(c3 / "progress.jsonl")]
    # What enters from PoC3: the first attempt, or its repeat if it failed
    # (October 9 amendment). Repeats get their own row, with key `-bis`; all
    # first attempts are kept, for the failure rate.
    from shared.metrics.campaigns import (
        CAMPAIGN_POC3_RETRIES, RETRY_SUFFIX, USE_RETRIES, poc3_for_analysis, retries_of_poc3,
    )

    by_key = {str(f["clave"]): f for f in output if f["campana"] == CAMPAIGN_POC3}
    for folder, f in poc3_for_analysis(lote):
        if folder.name == CAMPAIGN_POC3:
            by_key[str(f["clave"])]["in_the_pool"] = int(enters(f, CAMPAIGN_POC3))
        else:
            row = population_row(CAMPAIGN_POC3_RETRIES, {**f, "clave": f["batch_key"]},
                                     folder, enters(f, CAMPAIGN_POC3))
            row["clave"] = f["clave"]
            output.append(row)
    if not USE_RETRIES:
        # The repeats were run but do not enter: they stay as their own row,
        # outside the pool, so they can be counted.
        for f in retries_of_poc3(lote):
            row = population_row(CAMPAIGN_POC3_RETRIES, f, lote / CAMPAIGN_POC3_RETRIES, False)
            row["clave"] = str(f["clave"]) + RETRY_SUFFIX
            output.append(row)
    return output


# --------------------------------------------------------------------------
# PoC3 delegation
# --------------------------------------------------------------------------


def texts_of(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("text")
    )


def census_delegations(census: Path, clave: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Requests to the delegate and its function, plus what the server confirmed.

    Returns `(records, session)`. `session` is the `session` object of the first
    `session.started` (confirmed model and voice), with no audio at all.
    """
    records: list[dict[str, Any]] = []
    session: dict[str, Any] = {}
    from shared.metrics.census_poc3 import events as read_census

    for ev in read_census(census):
        tipo = ev.get("type")
        event = ev.get("event") if isinstance(ev.get("event"), dict) else ev
        t = ev.get("t_local_seg")
        if tipo == "session.started" and not session:
            s = event.get("session") or {}
            session = {k: s.get(k) for k in ("model", "id", "status") if k in s}
            voice = ((s.get("audio") or {}).get("output") or {}).get("voice")
            if voice:
                session["voice"] = voice
        elif tipo == "delegation.created":
            item = event.get("item") or {}
            records.append({
                "clave": clave, "t_local_seg": t, "tipo": tipo,
                "target": item.get("target"), "text": texts_of(item.get("content")),
            })
        elif tipo == "response.created":
            r = event.get("response") or {}
            instructions = r.get("instructions")
            records.append({
                "clave": clave, "t_local_seg": t, "tipo": tipo,
                "response_fields": sorted(r.keys()),
                "model_name": r.get("model"),
                "temperature": r.get("temperature"),
                "top_p": r.get("top_p"),
                "instructions": instructions if isinstance(instructions, str) else None,
                "n_tools": len(r.get("tools") or []),
            })
        elif tipo == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                records.append({
                    "clave": clave, "t_local_seg": t, "tipo": tipo,
                    "function": item.get("name"), "parsed_args": item.get("arguments"),
                })
            elif item.get("type") == "message":
                records.append({
                    "clave": clave, "t_local_seg": t, "tipo": tipo,
                    "delegate_text": texts_of(item.get("content")),
                })
    return records, session


def delegation_sample(lote: Path, n_calls: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    c3 = lote / "campana_poc3_pcmu"
    if not (c3 / "progress.jsonl").is_file():
        return [], []
    records: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    taken = 0
    for f in leer_jsonl(c3 / "progress.jsonl"):
        if f.get("estado") != "ok" or taken >= n_calls:
            continue
        summary = leer_json(c3 / "summaries" / f"{f['clave']}.json")
        census = ((summary or {}).get("artifacts") or {}).get("census")
        if not census or not Path(census).is_file():
            continue
        recs, session = census_delegations(Path(census), str(f["clave"]))
        records += recs
        sessions.append({"clave": f["clave"], **session})
        taken += 1
    return records, sessions


def mark_instructions(records: list[dict[str, Any]]) -> None:
    """Records whether the delegate received the same system prompt as the other PoCs.

    Compares against `v0.md` rendered with the campaign's trip. The full text
    is not repeated in every row: the verdict and its length are enough.
    """
    from shared.negotiation_engine.trip import TEST_TRIP
    from shared.system_prompt import render_system_prompt

    v0 = render_system_prompt(TEST_TRIP).strip()
    for r in records:
        if r.get("tipo") != "response.created":
            continue
        text = r.pop("instructions", None)
        r["received_instructions"] = bool(text)
        r["instructions_length"] = len(text) if text else 0
        r["instructions_equal_to_v0"] = bool(text) and text.strip() == v0
        r["instructions_start"] = (text or "")[:300]


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def public_env_values(ruta: Path) -> dict[str, str]:
    """Only the keys on the closed list. The others are not even read."""
    if not ruta.is_file():
        return {}
    output = {}
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in linea or linea.lstrip().startswith("#"):
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        if clave in PUBLIC_ENV_KEYS:
            output[clave] = valor.strip().strip('"').strip("'")
    return output


def total_memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class Estado(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            estado = Estado()
            estado.dwLength = ctypes.sizeof(Estado)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
            return round(estado.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (OSError, ValueError, AttributeError):
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def version_of(packet: str) -> str:
    try:
        from importlib.metadata import version

        return version(packet)
    except Exception:  # noqa: BLE001 - a missing package must not bring down the inventory
        return "no instalado"


def environment(env: Path, sessions: list[dict[str, Any]]) -> list[str]:
    text_lines = [
        f"system: {platform.system()} {platform.release()} ({platform.version()})",
        f"cpu: {cpu_name()} · núcleos lógicos: {os.cpu_count()}",
        f"ram_gb: {total_memory_gb()}",
        f"python: {platform.python_version()}",
    ]
    text_lines += [f"{p}: {version_of(p)}" for p in ("numpy", "scipy")]
    text_lines.append("")
    text_lines.append("valores públicos del .env:")
    text_lines += [f"  {k}={v}" for k, v in public_env_values(env).items()]
    text_lines.append("")
    text_lines.append("sesión de GPT-Live confirmada por el servidor (primer session.started):")
    confirmed_ones = Counter(json.dumps({k: v for k, v in s.items() if k in ("model", "voice")}, sort_keys=True) for s in sessions)
    text_lines += [f"  {n} calls: {v}" for v, n in confirmed_ones.items()]
    return text_lines


# --------------------------------------------------------------------------


def write_csv(ruta: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--env", type=Path, default=RAIZ / ".env")
    p.add_argument("--sample-calls", type=int, default=5,
                   help="how many PoC3 calls to inspect for the delegation sample")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = populations(args.lote)
    write_csv(args.output / "populations.csv", COLUMNS, rows)
    summary = populations_summary(rows)
    (args.output / "poblaciones_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    records, sessions = delegation_sample(args.lote, args.sample_calls)
    mark_instructions(records)
    with (args.output / "delegaciones_sample.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    environment_lines = environment(args.env, sessions)
    (args.output / "environment.txt").write_text("\n".join(environment_lines) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("")
    print("\n".join(environment_lines))
    requests = sum(1 for r in records if r["tipo"] == "delegation.created")
    print(f"\n{requests} delegation requests in {len(sessions)} sample calls")
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
