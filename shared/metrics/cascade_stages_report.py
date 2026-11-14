"""The segments of a typical PoC1 and PoC2 response, in order, as LaTeX macros.

The figure of GPT-Live's segments (`gpt_live_clock_report`) has its counterpart
here for the two cascades. It joins two tables from `docs/report/data/`:

- `stages.csv`: what the framework publishes for each response (end of turn,
  first LLM chunk, first TTS audio and their total, `full_turn_ms`);
- `lines.csv`: what the harness measured (observed latency and the two
  transports).

`stages.csv` does not carry the script turn, so rows are paired by order within
each call, and only in the calls with the same number of responses in both
tables. The pairing checks itself: in those calls, the framework's total is the
harness's latency without transport, within a few milliseconds.

The segments go in order and add up to the observed latency: inbound transport,
end of turn, language model, synthesis and outbound transport. The model's
segment is the remainder: in LiveKit the LLM starts before the end of turn is
confirmed (preemptive generation, enabled by default in LiveKit Agents 1.6) and
synthesis waits for the confirmation, so the part of the LLM that runs during
the end-of-turn wait does not lengthen the response. In Pipecat the stages do
not overlap and that remainder is the measured LLM.

Speech recognition (STT) is not one more segment: it runs from the same end of
speech as the end-of-turn wait and almost always finishes earlier, so it does
not lengthen the response. It is published separately: the typical response's
one (`RTramoEjStt`), to draw it over its end of turn, and the fraction of
responses in which it finishes earlier (`RTramoSttAntesPct`).

Writes `docs/report/latex/generated/cascade_stages.tex`.

    python -m shared.metrics.cascade_stages_report
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np

from shared.metrics.analysis_report import LETTER, calls_ok, macro, ms, outbound_to_first_sound

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "latex" / "generated" / "cascade_stages.tex"
CASCADES: Final[tuple[str, ...]] = ("1", "2")
# An end of turn over 1.5 s is the detector's maximum wait, not the normal one: both
# cascades split into a mode at ~0.55 s and another at 2.5 s or more, with nothing between.
LONG_END_MS: Final[float] = 1500.0
# The segments of a response, in the order they happen.
SEGMENTS: Final[dict[str, str]] = {
    "Ida": "inbound", "Fin": "end", "Llm": "llm", "Tts": "tts", "Vuelta": "outbound",
}


def _leer(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pair(stages: list[dict[str, str]], lines: list[dict[str, str]], poc: str
              ) -> tuple[list[dict[str, float | str]], int]:
    """The tool-free responses with their segments, and how many calls were paired."""
    per_call: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in stages:
        if e["poc"] == poc:
            per_call[e["clave"]].append(e)
    turns: dict[str, list[dict[str, str]]] = defaultdict(list)
    for i in lines:
        if i["poc"] == poc and i["observed_latency_ms"]:
            turns[i["clave"]].append(i)
    rows: list[dict[str, float | str]] = []
    calls = 0
    for clave, ivs in turns.items():
        es = sorted(per_call.get(clave, []), key=lambda e: int(e["indice"]))
        if len(es) != len(ivs):
            continue
        calls += 1
        for e, i in zip(es, sorted(ivs, key=lambda i: int(i["turn"][1:]))):
            if i["tool_in_response"] != "0":
                continue
            total = float(i["observed_latency_ms"])
            # Only responses without a short sound before the content: in them the
            # outbound is measured directly and the segments add up to the latency
            # with no other gap.
            primer = float(i["first_sound_latency_ms"]) if i["first_sound_latency_ms"] else None
            outbound, estado = outbound_to_first_sound(total, primer, float(i["outbound_ms"]), i["origin"] or None)
            if estado != "directa" or outbound is None:
                continue
            inbound = float(i["ida_ms"])
            end, tts = float(e["end_of_turn_ms"]), float(e["tts_ttfb_ms"])
            rows.append({
                "clave": clave, "turn": i["turn"],
                "inbound": inbound, "end": end, "llm": total - inbound - end - tts - outbound, "tts": tts,
                "outbound": outbound, "total": total,
                # Runs in parallel with the end of turn, from the same end of speech.
                "stt": float(e["stt_ms"]) if e["stt_ms"] else float("nan"),
                # What the LLM ran during the end-of-turn wait.
                "traslape": end + float(e["llm_ttft_ms"]) + tts - float(e["full_turn_ms"]),
            })
    return rows, calls


def typical_response(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    """The response closest to the median in each segment and in the total."""
    columns = [*SEGMENTS.values(), "total"]
    v = {c: np.array([float(f[c]) for f in rows]) for c in columns}
    center = {c: np.median(v[c]) for c in columns}
    width = {c: np.subtract(*np.percentile(v[c], [75, 25])) or 1.0 for c in columns}
    return min(rows, key=lambda f: sum(abs(float(f[c]) - center[c]) / width[c] for c in columns))


def iqr(v: list[float]) -> float:
    return float(np.subtract(*np.percentile(v, [75, 25])))


def dispersion_macros(stages: list[dict[str, str]], lines: list[dict[str, str]]) -> list[str]:
    """Where the width of each observed-latency box comes from.

    The interquartile range with and without tool, for all three; and in the
    cascades, what share of the responses waited for the long end of turn:
    LiveKit's detector waits its `max_delay` (2.5 s) when it estimates the
    carrier has not finished, and Pipecat's falls back to its maximum wait for
    the same reason.
    """
    tex: list[str] = []
    for poc, L in LETTER.items():
        rows = [i for i in lines if i["poc"] == poc and i["observed_latency_ms"]]
        all_items = [float(i["observed_latency_ms"]) for i in rows]
        sin = [float(i["observed_latency_ms"]) for i in rows if i["tool_in_response"] == "0"]
        tex += [macro(f"RDispIqr{L}", ms(iqr(all_items))), macro(f"RDispIqrSinHerr{L}", ms(iqr(sin)))]
    for poc in CASCADES:
        L = LETTER[poc]
        end = np.array([float(e["end_of_turn_ms"]) for e in stages if e["poc"] == poc and e["end_of_turn_ms"]])
        largo = end >= LONG_END_MS
        # The responses with both marks: how often the final transcript arrived
        # before the detector declared the turn finished.
        pairs = [(float(e["stt_ms"]), float(e["end_of_turn_ms"])) for e in stages
                 if e["poc"] == poc and e["stt_ms"] and e["end_of_turn_ms"]]
        tex += [
            macro(f"RTramoFinLargoPct{L}", f"{100 * largo.mean():.1f}"),
            macro(f"RTramoFinLargoMed{L}", ms(float(np.median(end[largo])))),
            macro(f"RTramoFinCortoMed{L}", ms(float(np.median(end[~largo])))),
            macro(f"RTramoSttAntesPct{L}", f"{100 * np.mean([s < f for s, f in pairs]):.1f}"),
        ]
    return tex


def generate(data: Path = DATA, output: Path = OUTPUT) -> list[str]:
    # The same population as the rest of the paper: the calls that passed in all three.
    valid_ones = calls_ok(data / "calls.csv")
    stages = [e for e in _leer(data / "stages.csv") if e["clave"] in valid_ones]
    lines = [i for i in _leer(data / "lines.csv") if i["clave"] in valid_ones]
    tex = ["% Generado por shared/metrics/cascade_stages_report.py; no editar a mano."]
    tex += dispersion_macros(stages, lines)
    for poc in CASCADES:
        L = LETTER[poc]
        rows, calls = pair(stages, lines, poc)
        tex += [macro(f"RTramoN{L}", str(len(rows))), macro(f"RTramoLlamadas{L}", str(calls))]
        for name, column in [*SEGMENTS.items(), ("Total", "total")]:
            tex.append(macro(f"RTramo{name}Med{L}", ms(float(np.median([float(f[column]) for f in rows])))))
        tex.append(macro(f"RTramoTraslapeMed{L}", ms(float(np.median([float(f["traslape"]) for f in rows])))))
        example = typical_response(rows)
        for name, column in [*SEGMENTS.items(), ("Total", "total")]:
            x = float(example[column])
            tex += [macro(f"RTramoEj{name}{L}", f"{x / 1000:.3f}"), macro(f"RTramoEj{name}Ms{L}", ms(x))]
        # What the framework counts in its own figure: from end of turn to first audio.
        self_reported = sum(float(example[c]) for c in ("end", "llm", "tts"))
        tex.append(macro(f"RTramoEjPropiaMs{L}", ms(self_reported)))
        stt = float(example["stt"])
        tex += [macro(f"RTramoEjStt{L}", f"{stt / 1000:.3f}"), macro(f"RTramoEjSttMs{L}", ms(stt))]
    output.write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
    return tex


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args(argv)
    print(f"{len(generate(args.data, args.output)) - 1} macros in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
