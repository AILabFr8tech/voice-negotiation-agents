"""The timeline of the ``ok'' that closed a deal, as LaTeX macros.

Reads `docs/report/data/backchannel_timeline.csv` (`validation.backchannel_timeline`)
and writes `docs/report/latex/generated/backchannel_timeline.tex`: the start and
end of each piece, in seconds since the line before the backchannel began (to
draw it), and the intervals the text cites.

    python -m shared.metrics.timeline_report
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Final

from shared.metrics.analysis_report import macro, ms

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data" / "backchannel_timeline.csv"
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "latex" / "generated" / "backchannel_timeline.tex"
# CSV piece and the name of its macro.
PIECES: Final[dict[str, str]] = {
    "linea": "Linea", "transcripcion_de_la_linea": "TransLinea", "agente_antes": "Antes",
    "apoyo": "Apoyo", "transcripcion_del_apoyo": "TransApoyo", "registrar_acuerdo": "Registro",
    "agente_despues": "Despues", "linea_siguiente": "Sig", "cuelga": "Cuelga",
}


def _seg(x: float) -> str:
    return f"{x:.1f}"


def generate(data: Path = DATA, output: Path = OUTPUT) -> list[str]:
    if not data.is_file():
        raise SystemExit(f"missing {data}: run it first (see the docstring)")
    with data.open(encoding="utf-8") as fh:
        rows = {f["que"]: f for f in csv.DictReader(fh)}
    missing_items = sorted(set(PIECES) - set(rows))
    if missing_items:
        raise SystemExit(f"missing pieces in {data.name}: {', '.join(missing_items)}")
    beg = {q: float(rows[q]["start_s"]) for q in PIECES}
    end = {q: float(rows[q]["end_s"]) for q in PIECES if rows[q]["end_s"]}
    tex = ["% GENERADO por shared/metrics/timeline_report.py — no editar a mano."]
    for q, name in PIECES.items():
        tex.append(macro(f"RLt{name}Ini", f"{beg[q]:.2f}"))
        if q in end:
            tex.append(macro(f"RLt{name}Fin", f"{end[q]:.2f}"))
    tex += [
        macro("RLtEje", f"{5 * (int(max(end.values()) // 5) + 1)}"),
        macro("RLtCallaSeg", _seg(end["agente_antes"] - beg["apoyo"])),
        macro("RLtRegistroSeg", _seg(beg["registrar_acuerdo"] - beg["apoyo"])),
        macro("RLtVuelveSeg", _seg(beg["agente_despues"] - beg["registrar_acuerdo"])),
        macro("RLtCuelgaSeg", _seg(beg["cuelga"] - beg["linea_siguiente"])),
        macro("RLtMonto", ms(float(rows["registrar_acuerdo"]["text"].split("=")[1]))),
    ]
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
