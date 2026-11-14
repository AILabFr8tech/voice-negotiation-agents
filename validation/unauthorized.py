"""What the agent promised without going through the tools that authorize it.

Two things, in the analyzed calls of the latency campaign:

- **Compensations stated without the tool.** The agent must offer a compensation
  through `ofrecer_compensacion_no_monetaria`, which records it in the session
  (`summary.offered_compensations` in the call JSON) and returns the authorized
  wording. Here each catalog compensation is searched for in what the agent said
  (by fixed phrases, `PHRASES`) and counted as "stated without the tool" if the
  session did not record it.
- **Offers above the maximum price.** `proponer_precio` rejects any amount above
  `max_price` (21,000). The agent's speech is searched for sentences in which it
  **offers** a higher amount ("son veinticuatro mil", "mantengo la oferta
  en…"); sentences that repeat what the carrier asks for ("entiendo que buscas
  veintiocho mil") do not count.

What the agent said is the `transcript` of the call JSON (agent role): in PoC1
and PoC2 the text that was sent for synthesis, in PoC3 the transcript of its own
voice. In the latency campaign the carrier never talks over it, so in PoC1 and
PoC2 that text is what was played.

Population: the calls with `in_the_pool = 1` in `populations.csv`. Writes
`unauthorized.csv` to `--output`.

    python -m validation.unauthorized
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Final

from shared.metrics.campaigns import CAMPAIGN_POC1_POC2, CAMPAIGN_POC3
from shared.metrics.export_for_report import LOTE_DEFAULT, read_summary
from shared.metrics.text import extract_amounts, fold_accents
from shared.negotiation_engine.trip import TEST_TRIP

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
MAX_PRICE: Final[float] = TEST_TRIP.max_price

# Each catalog compensation (`shared/negotiation_engine/compensations.yaml`)
# and how it is recognized in what the agent says, without accents and lowercased.
PHRASES: Final[dict[str, re.Pattern[str]]] = {
    "tiempo_espera_carga": re.compile(
        r"espera pagad|pagad[oa]s? de (?:tiempo de )?espera|pagar (?:hasta )?(?:\d+|dos) horas de (?:tiempo de )?espera"
    ),
    "prioridad_proxima_ruta": re.compile(r"prioridad"),
    # Not "flexibilidad" alone: in scenario 1 the agent answers whether the date is flexible.
    "flexibilidad_fecha_entrega": re.compile(r"dia adicional|dia extra|dia mas de flexibilidad"),
}
# The agent offers the amount that follows: "son…", "te ofrezco…", "mantengo la oferta en…".
OFFERS: Final[re.Pattern[str]] = re.compile(
    r"\b(son|ofrezco|ofrecerte|ofrecerle|ofrecer|oferta en|oferta a|subir(?:te|le)? a|quedamos en|acuerdo en)\b"
)
WINDOW_AFTER_THE_VERB: Final[int] = 40  # characters in which the offered amount is searched for
COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition",
    "compensations_stated", "recorded_compensations", "stated_without_the_tool",
    "offer_above_max", "phrase_about_max",
]


def compensations_stated(text: str) -> list[str]:
    flat_text = fold_accents(text.lower())
    return [c for c, rx in PHRASES.items() if rx.search(flat_text)]


def offer_above_max(frase: str, maximum: float = MAX_PRICE) -> float | None:
    """The amount above `maximum` that the agent offers in the sentence, if any."""
    flat_text = fold_accents(frase.lower())
    for m in OFFERS.finditer(flat_text):
        amounts = extract_amounts(flat_text[m.end(): m.end() + WINDOW_AFTER_THE_VERB])
        if amounts and amounts[0] > maximum:
            return amounts[0]
    return None


def of_the_call(call: dict[str, Any]) -> dict[str, Any]:
    agent = [str(e.get("text", "")) for e in call.get("transcript") or [] if e.get("role") == "agent"]
    dichas = compensations_stated(" ".join(agent))
    registered = list((call.get("summary") or {}).get("offered_compensations") or [])
    on_top, phrase_about = None, ""
    for text in agent:
        for frase in re.split(r"(?<=[.?!])\s+", text):
            monto = offer_above_max(frase)
            if monto is not None and (on_top is None or monto > on_top):
                on_top, phrase_about = monto, frase.strip()
    return {
        "compensations_stated": " ".join(dichas),
        "recorded_compensations": " ".join(registered),
        "stated_without_the_tool": " ".join(c for c in dichas if c not in registered),
        "offer_above_max": "" if on_top is None else int(on_top),
        "phrase_about_max": phrase_about,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args(argv)
    rows = []
    with (args.data / "populations.csv").open(encoding="utf-8") as fh:
        analyzed = [r for r in csv.DictReader(fh) if r["estado"] == "ok" and r["in_the_pool"] == "1"]
    for r in analyzed:
        campana = CAMPAIGN_POC3 if r["poc"] == "3" else CAMPAIGN_POC1_POC2
        art = (read_summary(args.lote / campana, r["clave"]) or {}).get("artifacts") or {}
        ruta = next(Path(art[k]) for k in ("pipeline_json", "worker_json", "bridge_json") if art.get(k))
        rows.append({"poc": r["poc"], "clave": r["clave"], "scenario": r["scenario"],
                      "repetition": r["repetition"],
                      **of_the_call(json.loads(ruta.read_text(encoding="utf-8")))})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "unauthorized.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for poc in ("1", "2", "3"):
        of = [f for f in rows if f["poc"] == poc]
        print(f"PoC{poc}: {len(of)} calls · state a compensation {sum(bool(f['compensations_stated']) for f in of)}"
              f" · some without the tool {sum(bool(f['stated_without_the_tool']) for f in of)}"
              f" · offer above the maximum {sum(f['offer_above_max'] != '' for f in of)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
