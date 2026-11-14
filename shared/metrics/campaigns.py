"""Which campaign is each PoC's in the report, and which calls enter the comparison.

PoC3 was rerun on 9 October 2026 with GPT-Live's public API and 8 kHz μ-law
(`campana_poc3_pcmu`), because the alpha access of the 22 September campaign
(`campana_poc3`) was withdrawn. The alpha one stays archived and does not enter
the results. See `docs/report/data/regla_campana_poc3_pcmu.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

CAMPAIGN_POC1_POC2: Final[str] = "campana200"
CAMPAIGN_POC3: Final[str] = "campana_poc3_pcmu"
CAMPAIGN_POC3_ALPHA: Final[str] = "campaign_poc3"
# A single retry of each PoC3 call that failed its criteria (amendment to the
# rule, 9 October). The failure rate uses the first attempts.
CAMPAIGN_POC3_RETRIES: Final[str] = "campana_poc3_pcmu_reintentos"
# The retries do not enter the analysis, so as not to give PoC3 a second chance
# that PoC1 and PoC2 did not have. They were run and remain recorded
# (`retries_of_poc3`); the analysis uses first attempts only.
USE_RETRIES: Final[bool] = False
LATENCY_CAMPAIGNS: Final[tuple[str, ...]] = (CAMPAIGN_POC1_POC2, CAMPAIGN_POC3)

# PoC1 and PoC2 are analyzed on their first 90 calls each (alternating): from
# call 181 on, ElevenLabs rejected the key and none had a voice.
TOPE_CAMPANA200: Final[int] = 180
# The repetitions that all three PoCs ran.
REPETICIONES: Final[int] = 18

# PoC3 in the interruption campaigns: the 5 October one ran all three PoCs at
# 24 kHz; PoC3's was rerun in μ-law on 9 October.
INTERRUPTIONS_POC1_POC2: Final[str] = "interruptions_full_v2"
INTERRUPTIONS_POC3: Final[str] = "interruptions_poc3_pcmu"


def _progress(ruta: Path) -> list[dict]:
    if not ruta.is_file():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def retries_of_poc3(lote: Path) -> list[dict]:
    """The retries that were run, whether or not they enter the analysis."""
    return _progress(lote / CAMPAIGN_POC3_RETRIES / "progress.jsonl")


def poc3_runs(lote: Path) -> list[tuple[str, dict]]:
    """`(campaign, row)` of each PoC3 scenario and repetition that enters the analysis.

    The first attempt, or its retry if the first one failed and was retried.
    The call that failed and was retried is not returned: it counts only in the
    failure rate, which is computed from the first attempts (`CAMPAIGN_POC3`).
    """
    retries = {
        (int(r["scenario"]), int(r["repetition"])): r
        for r in retries_of_poc3(lote)
    } if USE_RETRIES else {}
    output: list[tuple[str, dict]] = []
    for r in _progress(lote / CAMPAIGN_POC3 / "progress.jsonl"):
        cell = (int(r["scenario"]), int(r["repetition"]))
        if r.get("estado") != "ok" and cell in retries:
            output.append((CAMPAIGN_POC3_RETRIES, retries[cell]))
        else:
            output.append((CAMPAIGN_POC3, r))
    return output


def valid_combinations(lote: Path) -> set[tuple[int, int]]:
    """(scenario, repetition) pairs that passed the criteria in all three PoCs.

    It is the rule fixed before running the PoC3 μ-law campaign: it replaces
    the PoC1/PoC2 pairing rule, and makes all three compare the same set of
    scenarios and repetitions.
    """
    ok: dict[int, set[tuple[int, int]]] = {1: set(), 2: set(), 3: set()}
    for r in _progress(lote / CAMPAIGN_POC1_POC2 / "progress.jsonl"):
        if int(r.get("indice") or 0) > TOPE_CAMPANA200:
            continue
        if r.get("estado") == "ok" and int(r["repetition"]) <= REPETICIONES:
            ok[int(r["poc"])].add((int(r["scenario"]), int(r["repetition"])))
    for _, r in poc3_runs(lote):
        if r.get("estado") == "ok" and int(r["repetition"]) <= REPETICIONES:
            ok[3].add((int(r["scenario"]), int(r["repetition"])))
    return ok[1] & ok[2] & ok[3]


RETRY_SUFFIX: Final[str] = "-bis"


def poc3_for_analysis(lote: Path) -> list[tuple[Path, dict]]:
    """`(folder, row)` of the PoC3 calls that enter the analysis.

    The retry keeps the plan's key (`0005-poc3-e5-r1`) and is distinguished by
    `RETRY_SUFFIX` in `clave`; `batch_key` is the key of its files.
    """
    output = []
    for campana, row in poc3_runs(lote):
        f = {**row, "batch_key": row["clave"]}
        if campana == CAMPAIGN_POC3_RETRIES:
            f["clave"] = row["clave"] + RETRY_SUFFIX
        output.append((lote / campana, f))
    return output
