"""Sensitivity of PoC1 and PoC2 WER to the Deepgram parameters that differ.

PoC1 and PoC2 use Deepgram nova-3 with each framework's default values, and those
values do not match: Pipecat leaves the profanity filter on and filler words off;
LiveKit does the opposite. This script measures, from the records that already
exist and without running any calls:

a) how many words the profanity filter masked (tokens with `*` in what each PoC
   transcribed);
b) PoC1 and PoC2 WER and CER **with filler words removed** from the reference and
   from the transcript, and the PoC1 vs PoC2 test repeated on that;
c) whether either of the two affects the amount error.

The reference and the hypothesis (`export_wer.call_texts`, with the last-line
rule), the population (the PoC1 and PoC2 pairs under the three-PoC rule) and the
alignment are those of `shared/metrics/export_wer.py`, unchanged. The only
difference in (b) is the normalization with `remove_fillers=True`, which already
existed in `shared/metrics/text.py`. That module's filler list is extended **only
within this process** with the ones Deepgram writes in English (`uh`, `um`,
`mhm`, `hmm`, `hm`); the module file is not touched.

Writes `deepgram_sensitivity.csv` (one row per call) and
`deepgram_sensitivity.txt` (the summary) to `--output`.

    uv run --project poc3_gpt_live --with scipy python -m validation.deepgram_sensitivity

Takes on the order of 15 minutes: the character-level alignment of a full call
takes about 3 s and is done twice per call.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

from shared.metrics import text as module_text
from shared.metrics.campaigns import valid_combinations
from shared.metrics.export_for_report import (
    LOTE_DEFAULT,
    DEFAULT_OUTPUT,
    read_progress,
    read_summary,
    campaign200_pairs,
)
from shared.metrics.export_wer import call_slack, call_texts, json_transcript
from shared.metrics.text import CONFIG_DEFAULT, NormalizationConfig, tokenize_text
from shared.metrics.wer import compute_amounts, compute_text

ENGLISH_FILLERS: Final[frozenset[str]] = frozenset({"uh", "um", "mhm", "hmm", "hm"})
FILLERS: Final[frozenset[str]] = frozenset(module_text.FILLERS) | ENGLISH_FILLERS
NO_FILLERS: Final[NormalizationConfig] = NormalizationConfig(remove_fillers=True)
MASKED: Final[re.Pattern[str]] = re.compile(r"\S*\*\S*")

COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition",
    "masked", "masked_text",
    "muletillas_reference", "fillers_hypothesis",
    "words_reference", "words_errors", "characters_reference", "characters_errors",
    "amounts_reference", "amounts_errors",
    "sm_words_reference", "sm_words_errors", "sm_characters_reference",
    "sm_characters_errors", "sm_amounts_reference", "sm_amounts_errors",
]


def counts(reference: str, hypothesis: str, config: NormalizationConfig) -> dict[str, int]:
    t = compute_text(reference, hypothesis, config)
    m = compute_amounts(reference, hypothesis, config).alignment
    return {
        "words_reference": t.words_reference,
        "words_errors": t.wer.substitutions + t.wer.deleted + t.wer.insertions,
        "characters_reference": t.cer.hits + t.cer.substitutions + t.cer.deleted,
        "characters_errors": t.cer.substitutions + t.cer.deleted + t.cer.insertions,
        "amounts_reference": m.hits + m.substitutions + m.deleted,
        "amounts_errors": m.substitutions + m.deleted + m.insertions,
    }


def fillers_in(text: str) -> Counter[str]:
    return Counter(t for t in tokenize_text(text, CONFIG_DEFAULT) if t in FILLERS)


def row_of_the_call(
    row: dict[str, Any], reference: str, hypothesis: str, transcript: str
) -> dict[str, Any]:
    """`transcript`: the raw text from the JSON, where the masked words are searched for
    (the already normalized hypothesis loses the `*` if its last line was trimmed)."""
    masked = MASKED.findall(transcript)
    original = counts(reference, hypothesis, CONFIG_DEFAULT)
    sin = counts(reference, hypothesis, NO_FILLERS)
    return {
        "poc": int(row["poc"]), "clave": row["clave"],
        "scenario": row["scenario"], "repetition": row["repetition"],
        "masked": len(masked),
        "masked_text": " | ".join(masked),
        "muletillas_reference": json.dumps(dict(fillers_in(reference)), ensure_ascii=False),
        "fillers_hypothesis": json.dumps(dict(fillers_in(hypothesis)), ensure_ascii=False),
        **original,
        **{f"sm_{k}": v for k, v in sin.items()},
    }


def sample_rate(rows: list[dict[str, Any]], prefix: str) -> float:
    n = sum(int(f[f"{prefix}_reference"]) for f in rows)
    e = sum(int(f[f"{prefix}_errors"]) for f in rows)
    return e / n if n else float("nan")


def summary(rows: list[dict[str, Any]], contexts: list[str]) -> str:
    from scipy import stats

    pairs = sum(1 for f in rows if f["poc"] == 1)
    text_lines = [f"Sensibilidad a la configuración de Deepgram ({pairs} pares de PoC1 y PoC2, "
              "regla de las tres PoC)", ""]
    for poc in (1, 2):
        fs = [f for f in rows if f["poc"] == poc]
        tot = Counter()
        tot_ref = Counter()
        for f in fs:
            tot.update(json.loads(f["fillers_hypothesis"]))
            tot_ref.update(json.loads(f["muletillas_reference"]))
        text_lines += [
            f"PoC{poc}: {len(fs)} llamadas",
            f"  (a) palabras enmascaradas por el filtro de groserías: "
            f"{sum(f['masked'] for f in fs)} en {sum(1 for f in fs if f['masked'])} llamadas",
            f"  muletillas en la transcripción: {dict(tot)}",
            f"  muletillas en la referencia:    {dict(tot_ref)}",
            f"  (b) WER  original {100 * sample_rate(fs, 'words'):.2f} %   sin muletillas "
            f"{100 * sample_rate(fs, 'sm_words'):.2f} %",
            f"      CER  original {100 * sample_rate(fs, 'characters'):.2f} %   sin muletillas "
            f"{100 * sample_rate(fs, 'sm_characters'):.2f} %",
            f"  (c) montos original {100 * sample_rate(fs, 'amounts'):.2f} %   sin muletillas "
            f"{100 * sample_rate(fs, 'sm_amounts'):.2f} %",
            "",
        ]

    def per_call(prefix: str, poc: int) -> dict[tuple[str, str], float]:
        return {
            (str(f["scenario"]), str(f["repetition"])):
                int(f[f"{prefix}_errors"]) / max(1, int(f[f"{prefix}_reference"]))
            for f in rows if f["poc"] == poc
        }

    for label, prefix in (("original", "words"), ("sin muletillas", "sm_words")):
        a, b = per_call(prefix, 1), per_call(prefix, 2)
        commons = sorted(set(a) & set(b))
        diff = [a[k] - b[k] for k in commons]
        p = stats.wilcoxon(diff).pvalue
        text_lines.append(
            f"Wilcoxon pareado PoC1 vs PoC2, WER por llamada ({label}): {len(commons)} pares, "
            f"p = {p:.4g}, con Bonferroni ×3 = {min(1.0, 3 * p):.4g}"
        )
    if contexts:
        text_lines += ["", "Palabras enmascaradas, con su contexto:", *contexts]
    return "\n".join(text_lines) + "\n"


def context(hypothesis: str, window: int = 4) -> list[str]:
    words = hypothesis.split()
    return [
        " ".join(words[max(0, i - window): i + window + 1])
        for i, p in enumerate(words) if "*" in p
    ]


def check_against_exporter(rows: list[dict[str, Any]], ruta: Path) -> str:
    """The original computation must give the same result as `wer_calls.csv`."""
    if not ruta.is_file():
        return "\ncotejo: no hay wer_llamadas.csv con qué comparar\n"
    with ruta.open(encoding="utf-8") as f:
        published_ones = {r["clave"]: r for r in csv.DictReader(f)}
    different_ones = []
    for row in rows:
        r = published_ones.get(row["clave"])
        if r is None or r["reason_if_omitted"]:
            continue
        errors = int(r["words_s"]) + int(r["words_d"]) + int(r["words_i"])
        if (errors, int(r["words_reference"])) != (row["words_errors"], row["words_reference"]):
            different_ones.append(row["clave"])
    return (
        f"\ncotejo con wer_llamadas.csv: {len(rows) - len(different_ones)} de {len(rows)} llamadas "
        f"dan los mismos conteos de palabras" + (f"; distintas: {different_ones}" if different_ones else "") + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    # The extended list applies only in this process: `tokenize_text` reads the
    # module global on every call.
    module_text.FILLERS = FILLERS  # type: ignore[misc]

    c200 = args.lote / "campana200"
    runs = campaign200_pairs(read_progress(c200), valid_combinations(args.lote))
    rows: list[dict[str, Any]] = []
    contexts: list[str] = []
    warning_list: list[str] = []
    for k, row in enumerate(runs, start=1):
        if k % 10 == 0 or k == len(runs):
            print(f"  {k}/{len(runs)} calls", flush=True)
        clave = str(row["clave"])
        art = (read_summary(c200, clave) or {}).get("artifacts") or {}
        event_log = Path(art["event_log"]) if art.get("event_log") else None
        destination = next(
            (Path(art[k]) for k in ("pipeline_json", "worker_json") if art.get(k)), None
        )
        if event_log is None or destination is None or not event_log.is_file() or not destination.is_file():
            warning_list.append(f"{clave}: falta la bitácora o el JSON de la llamada")
            continue
        log_data = json.loads(event_log.read_text(encoding="utf-8"))
        call = json.loads(destination.read_text(encoding="utf-8"))
        t = call_texts(log_data, call, call_slack(log_data, art))
        transcript = json_transcript(call)
        new_item = row_of_the_call(row, t.reference, t.hypothesis, transcript)
        rows.append(new_item)
        contexts += [f"  PoC{new_item['poc']} {clave}: …{c}…" for c in context(transcript)]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "deepgram_sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows, contexts) + check_against_exporter(rows, args.output / "wer_calls.csv")
    if warning_list:
        text += "\nwarning_list:\n" + "\n".join(warning_list) + "\n"
    (args.output / "deepgram_sensitivity.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
