"""GPT-Live's latency on the model's own clock and per route, as LaTeX macros.

It summarizes two tables already in `docs/report/data/`:

- `model_clock_poc3.csv` (`validation.model_clock_poc3`): per campaign
  response, how long the model took on its own clock and what happened outside
  it, split into parts that add up to the latency; and
  `model_clock_poc3_drift.csv`, how far ahead that clock ran in each call.
- `transport_gpt_live.csv` (`validation.transport_gpt_live`): the same lines
  over WebSocket and over WebRTC, in separate sessions.
- `network_gpt_live.csv` (`validation.network_gpt_live`): the network round
  trip to the API's ingress server.

It also picks the tool-free response whose parts are closest to the median of
each one, to draw it as an example.

Writes `docs/report/latex/generated/gpt_live_clock.tex`. It does not touch the
call artifacts: only medians and quartiles of those two tables.

    python -m shared.metrics.gpt_live_clock_report
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Final

import numpy as np

from shared.metrics.analysis_report import macro, ms

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "latex" / "generated" / "gpt_live_clock.tex"
ROUTES: Final[dict[str, str]] = {"ws": "Ws", "rtc": "Rtc"}
# The parts each response's latency is split into, in table order.
PARTS: Final[dict[str, str]] = {
    "Ultima": "last_word_ms", "Modelo": "model_ms", "Salida": "output_ms", "Entrega": "delivery_ms",
}


def _values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.array([float(f[column]) for f in rows if f.get(column) not in ("", None)])


def _service(rows: list[dict[str, str]]) -> np.ndarray:
    """What the service took outside the model's clock: up to the last word plus the output."""
    return _values(rows, "last_word_ms") + _values(rows, "output_ms")


def _quartiles(name: str, v: np.ndarray) -> list[str]:
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return [macro(f"{name}Med", ms(q2)), macro(f"{name}Qa", ms(q1)), macro(f"{name}Qb", ms(q3))]


def typical_response(rows: list[dict[str, str]]) -> dict[str, str]:
    """The tool-free response closest to the median in each part and in the total."""
    columns = [*PARTS.values(), "latency_ms"]
    center = {c: np.median(_values(rows, c)) for c in columns}
    width = {c: np.subtract(*np.percentile(_values(rows, c), [75, 25])) or 1.0 for c in columns}
    sin = [f for f in rows if f["tool"] == "0"]
    return min(sin, key=lambda f: sum(abs(float(f[c]) - center[c]) / width[c] for c in columns))


def clock_macros(rows: list[dict[str, str]]) -> list[str]:
    sin = [f for f in rows if f["tool"] == "0"]
    having = [f for f in rows if f["tool"] == "1"]
    tex = [macro("RRelojN", str(len(rows)))]
    tex += _quartiles("RRelojLat", _values(rows, "latency_ms"))
    tex += _quartiles("RRelojPrimera", _values(rows, "model_first_ms"))
    tex += _quartiles("RRelojResto", _values(rows, "rest_ms"))
    tex += _quartiles("RRelojCola", _values(rows, "cola_ms"))
    tex += _quartiles("RRelojServicio", _service(rows))
    for name, column in PARTS.items():
        tex += _quartiles(f"RReloj{name}", _values(rows, column))
    # To show that the medians of the parts do not add up to the median of the wait.
    tex.append(macro("RRelojSumaMedMs", ms(sum(float(np.median(_values(rows, c))) for c in PARTS.values()))))
    # No thousands comma: these are coordinates for the per-layer figure.
    tex += [macro(f"RReloj{name}Valor", f"{np.median(_values(rows, column)):.1f}")
            for name, column in (("Modelo", "model_ms"), ("Resto", "rest_ms"))]
    for suffix, sub in (("SinHerr", sin), ("ConHerr", having)):
        tex += [
            macro(f"RRelojN{suffix}", str(len(sub))),
            macro(f"RRelojLat{suffix}Med", ms(float(np.median(_values(sub, "latency_ms"))))),
            macro(f"RRelojResto{suffix}Med", ms(float(np.median(_values(sub, "rest_ms"))))),
            macro(f"RRelojServicio{suffix}Med", ms(float(np.median(_service(sub))))),
        ]
        tex += [macro(f"RReloj{name}{suffix}Med", ms(float(np.median(_values(sub, column)))))
                for name, column in PARTS.items()]
    example = typical_response(rows)
    for name, column in [*PARTS.items(), ("Total", "latency_ms")]:
        x = float(example[column])
        tex += [macro(f"RRelojEj{name}", f"{x / 1000:.3f}"), macro(f"RRelojEj{name}Ms", ms(x))]
    return tex


def drift_macros(rows: list[dict[str, str]]) -> list[str]:
    return [
        macro("RRelojDerivaLlamadas", str(len(rows))),
        macro("RRelojDerivaMed", ms(float(np.median(_values(rows, "model_ms"))))),
        macro("RRelojDerivaLapsoMed", ms(float(np.median(_values(rows, "span_s"))))),
        macro("RRelojDerivaUsoMed", ms(float(np.nanmedian(_values(rows, "usage_ms"))))),
    ]


def route_macros(rows: list[dict[str, str]]) -> list[str]:
    tex = [
        macro("RViaLineas", str(len({f["clip"] for f in rows}))),
        macro("RViaSesiones", str(max(int(f["session"]) for f in rows))),
    ]
    medians = {}
    for via, name in ROUTES.items():
        sub = [f for f in rows if f["via"] == via]
        audio = _values(sub, "primer_audio_ms")
        medians[via] = float(np.median(audio))
        tex += [
            macro(f"RVia{name}N", str(audio.size)),
            macro(f"RVia{name}Med", ms(medians[via])),
            macro(f"RVia{name}UltimaMed", ms(float(np.median(_values(sub, "last_word_ms"))))),
            macro(f"RVia{name}ModeloMed", ms(float(np.median(_values(sub, "model_ms"))))),
            macro(f"RVia{name}FueraMed", ms(float(np.median(audio - _values(sub, "model_ms"))))),
        ]
    tex.append(macro("RViaDif", ms(medians["rtc"] - medians["ws"])))
    return tex


def network_macros(rows: list[dict[str, str]]) -> list[str]:
    """The most recent measurement of the round trip to the ingress server."""
    ultima = max(rows, key=lambda f: f["date"])
    return [macro("RRedMed", ms(float(ultima["p50_ms"]))), macro("RRedIntentos", ultima["attempts"])]


def _leer(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise SystemExit(f"missing {ruta}: run it first (see the docstring)")
    with ruta.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def generate(data: Path = DATA, output: Path = OUTPUT) -> list[str]:
    clock = _leer(data / "model_clock_poc3.csv")
    drift = _leer(data / "model_clock_poc3_drift.csv")
    routes = _leer(data / "transport_gpt_live.csv")
    network = _leer(data / "network_gpt_live.csv")
    tex = ["% GENERADO por shared/metrics/gpt_live_clock_report.py — no editar a mano."]
    tex += clock_macros(clock) + drift_macros(drift) + route_macros(routes) + network_macros(network)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return tex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args(argv)
    print("\n".join(generate(args.data, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
