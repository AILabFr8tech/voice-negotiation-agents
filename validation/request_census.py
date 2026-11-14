"""Census of the requests GPT-Live passed to `gpt-4.1-mini` in the PoC3 campaign.

The report stated, from a sample of five calls, that each request was the
carrier's last sentence, transcribed. Here the requests of all analyzed calls are
checked, from the event census saved by the bridge
(`poc3_gpt_live/recordings/census_<sid>.jsonl`):

- **The request text** (`delegation.created`) against the transcript of the
  carrier turn that triggered it (`user_bidi_turn_id` → `turn.done`): equal, only
  a part, or with something more.
- **What the delegate receives**: the input tokens of its first response for each
  request (`response.completed`). The request is one sentence; if the tokens
  exceed what PoC1's full prompt measures, the delegate also receives
  instructions and conversation, even though their text is not visible.

For comparison, it also reads how many input tokens the first model call carries
in each PoC1 call (the full prompt with the four tools), from the Pipecat log.

Costs nothing. Writes to `docs/report/data/`: `requests.csv` (one row per
request, without the text), `prompt_poc1.csv`, `delegate_responses.csv` and
`requests_summary.txt`.

    uv run --project poc3_gpt_live python -m validation.request_census
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote" / "campana_poc3_pcmu"
BATCH_POC1: Final[Path] = RAIZ / "diagnostics" / "lote" / "campana200"
RE_PROMPT: Final[re.Pattern[str]] = re.compile(r"prompt tokens: (\d+)")
RECORDINGS: Final[Path] = RAIZ / "poc3_gpt_live" / "recordings"
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"

COLUMNS: Final[list[str]] = [
    "clave", "scenario", "call_second", "request_words",
    "turn_words", "ratio", "input_tokens",
]


def normalize(text: str) -> list[str]:
    """Lowercase, without accents or punctuation: the words."""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]", " ", s).split()


def ratio(request: list[str], turn: list[str]) -> str:
    """`equal`, `parte` (a contiguous stretch of the turn) or `otra` (contains something the turn did not say).

    Compared without spaces: the turn transcript sometimes splits a word in two
    ("Mánd eme", "veint iocho") that the request carries whole.
    """
    a, b = "".join(request), "".join(turn)
    if a == b:
        return "equal"
    if a and a in b:
        return "parte"
    return "otra"


def call_requests(census: Path) -> list[dict]:
    from shared.metrics.census_poc3 import events as read_census

    # In the public API `delegation.created` no longer carries the request text
    # and there is no `turn.done`: only the count and the delegate's input tokens remain.
    events = list(read_census(census))
    turns = {}
    for e in events:
        if e["type"] == "turn.done":
            t = e["event"]["turn"]
            turns[t["id"]] = t.get("transcript") or ""
    # The delegate's first response after each request carries its tokens.
    output = []
    for k, e in enumerate(events):
        if e["type"] != "delegation.created":
            continue
        item = e["event"].get("item") or {}
        text = " ".join(c.get("text", "") for c in item.get("content", []) if isinstance(c, dict))
        turn = turns.get(item.get("user_bidi_turn_id"), "")
        tokens = next(
            (((s["event"].get("response") or {}).get("usage") or {}).get("input_tokens")
             for s in events[k + 1:] if s["type"] == "response.completed"),
            None,
        )
        req, tur = normalize(text), normalize(turn)
        output.append({
            "call_second": round(e["t_local_seg"], 1),
            "request_words": len(req), "turn_words": len(tur),
            "ratio": ("sin_texto" if not req else ratio(req, tur) if tur else "no_turn"),
            "input_tokens": tokens,
        })
    return output


def said_fair_rate(census: Path) -> bool:
    """The delegate answered with the sentence the instructions dictate when asked about the cap."""
    from shared.metrics.census_poc3 import events as read_census

    for e in read_census(census):
        if e["type"] != "response.output_item.done":
            continue
        item = e["event"].get("item") or {}
        if item.get("type") == "message":
            text = " ".join(c.get("text", "") for c in item.get("content", []) if isinstance(c, dict))
            if "tarifa justa" in text.lower():
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    with (DATA / "populations.csv").open(encoding="utf-8") as fh:
        valid_ones = {r["clave"] for r in csv.DictReader(fh)
                   if r["campana"].startswith("campana_poc3_pcmu") and r["estado"] == "ok" and r["in_the_pool"] == "1"}
    from shared.metrics.campaigns import poc3_for_analysis

    # The first attempt, or its repetition if it failed (key `-bis`).
    progress = [f for _, f in poc3_for_analysis(LOTE.parent)]
    rows = []
    delegate = []
    for r in progress:
        if r["clave"] not in valid_ones:
            continue
        census = RECORDINGS / f"census_{r['label']}.jsonl"
        if not census.is_file():
            raise SystemExit(f"{r['clave']}: {census.name} missing")
        for f in call_requests(census):
            rows.append({"clave": r["clave"], "scenario": r["scenario"], **f})
        delegate.append({"clave": r["clave"], "scenario": r["scenario"],
                         "fair_tariff": int(said_fair_rate(census))})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "delegate_responses.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clave", "scenario", "fair_tariff"])
        w.writeheader()
        w.writerows(delegate)
    with (args.output / "requests.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in rows:
            w.writerow({k: ("" if f.get(k) is None else f[k]) for k in COLUMNS})
    with (DATA / "populations.csv").open(encoding="utf-8") as fh:
        poc1 = [r["clave"] for r in csv.DictReader(fh)
                if r["campana"] == "campana200" and r["poc"] == "1" and r["estado"] == "ok" and r["in_the_pool"] == "1"]
    prompts = []
    for clave in poc1:
        m = RE_PROMPT.search((BATCH_POC1 / "calls" / f"{clave}.log").read_text(encoding="utf-8", errors="replace"))
        if m:
            prompts.append({"clave": clave, "first_call_tokens": int(m.group(1))})
    with (args.output / "prompt_poc1.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clave", "first_call_tokens"])
        w.writeheader()
        w.writerows(prompts)
    tokens = [f["input_tokens"] for f in rows if f["input_tokens"] is not None]
    text_lines = [
        f"calls: {len({f['clave'] for f in rows})}  requests: {len(rows)}",
        "relación con el turno del transportista: "
        + ", ".join(f"{k} {sum(f['ratio'] == k for f in rows)}" for k in ("equal", "parte", "otra", "no_turn")),
        f"tokens de entrada del delegado: mediana {statistics.median(tokens):.0f}, "
        f"de {min(tokens)} a {max(tokens)} (n={len(tokens)})" if tokens else "sin tokens",
        f"prompt de PoC1 en su primera llamada al modelo: mediana "
        f"{statistics.median([x['first_call_tokens'] for x in prompts]):.0f} tokens (n={len(prompts)})"
        if prompts else "sin prompt de PoC1",
    ]
    (args.output / "requests_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
