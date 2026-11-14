"""Incrusta en sn-article.tex las macros de cifras y las referencias que usa el artículo.

La plantilla de Springer Nature pide un solo .tex, sin \\input. Este script reescribe
el bloque entre "% --- Cifras generadas" y "% --- fin de las cifras generadas ---"
con la definición de cada macro \\R... que aparece en el resto del archivo, tomada de
los mismos archivos de docs/reporte/latex/generado/ que lee el artículo en español.
También reescribe el bloque de "Referencias generadas" con la bibliografía que arma
BibTeX desde el sn-bibliography.bib de la misma carpeta, con el estilo de Springer
(docs/paper/sn-mathphys-num.bst), en el orden de las citas: así Overleaf no corre
BibTeX ni necesita el .bst. Correrlo cada vez que cambien las cifras, las citas o el
texto. Sin argumentos, trabaja sobre las dos versiones (en/ y es/, la que exista).

    python docs/paper/en/incrustar_cifras.py [archivo.tex ...]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

HERE: Final[Path] = Path(__file__).resolve().parent
GENERATED: Final[Path] = HERE.parents[1] / "report" / "latex" / "generated"
SOURCES: Final[tuple[str, ...]] = (
    "results.tex", "interruptions.tex", "gpt_live_clock.tex", "backchannel_timeline.tex",
    "cascade_stages.tex",
)
TEX: Final[tuple[Path, ...]] = (HERE / "sn-article.tex", HERE.parent / "es" / "sn-article.tex")
BST: Final[Path] = HERE.parent / "sn-mathphys-num.bst"
START: Final[str] = "% --- Cifras generadas"
END: Final[str] = "% --- fin de las cifras generadas ---"
START_REF: Final[str] = "% --- Referencias generadas"
END_REF: Final[str] = "% --- fin de las referencias generadas ---"
RE_MACRO: Final[re.Pattern[str]] = re.compile(r"\\(R[A-Z][A-Za-z]*)")
# \cite, \citep y \citet; no \citeauthoryear, que aparece dentro de las referencias.
RE_CITATION: Final[re.Pattern[str]] = re.compile(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def definitions() -> dict[str, str]:
    d: dict[str, str] = {}
    for name in SOURCES:
        for linea in (GENERATED / name).read_text(encoding="utf-8").splitlines():
            if (m := re.match(r"\\newcommand\{\\(R[A-Za-z]+)\}", linea)):
                d.setdefault(m.group(1), linea)
    return d


def limits(text: str, start: str, end: str) -> tuple[int, int]:
    """Dónde empieza y termina el contenido de un bloque (sin sus dos líneas de marca)."""
    return text.index("\n", text.index(start)) + 1, text.index(end)


def embed(text: str, defs: dict[str, str]) -> tuple[str, int]:
    a, b = limits(text, START, END)
    cuerpo = text[:a] + text[b:]
    used = set(RE_MACRO.findall(cuerpo))
    # Una macro puede usar otra dentro de su definición.
    pending_items = list(used)
    while pending_items:
        for other in RE_MACRO.findall(defs.get(pending_items.pop(), "")):
            if other not in used:
                used.add(other)
                pending_items.append(other)
    missing_items = sorted(u for u in used if u not in defs)
    if missing_items:
        raise SystemExit("sin definición en generado/: " + ", ".join(missing_items))
    block = "".join(defs[u] + "\n" for u in sorted(used))
    return text[:a] + block + text[b:], len(used)


def references(text: str, bib: Path) -> tuple[str, int]:
    a, b = limits(text, START_REF, END_REF)
    citations = RE_CITATION.findall(text[:a] + text[b:])
    keys = list(dict.fromkeys(c.strip() for group in citations for c in group.split(",")))
    aux = "".join(f"\\citation{{{c}}}\n" for c in keys) + "\\bibstyle{sn-mathphys-num}\n\\bibdata{refs}\n"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(BST, d)
        shutil.copy(bib, Path(d) / "refs.bib")
        (Path(d) / "refs.aux").write_text(aux, encoding="utf-8")
        if subprocess.run(["bibtex", "refs"], cwd=d, capture_output=True).returncode != 0:
            raise SystemExit("BibTeX dio avisos o errores:\n" + (Path(d) / "refs.blg").read_text(encoding="utf-8", errors="replace"))
        bbl = (Path(d) / "refs.bbl").read_text(encoding="utf-8")
    return text[:a] + bbl + text[b:], len(keys)


def main(argv: list[str]) -> int:
    defs = definitions()
    for tex in [Path(a) for a in argv] or [t for t in TEX if t.exists()]:
        text, r = references(tex.read_text(encoding="utf-8"), tex.parent / "sn-bibliography.bib")
        text, n = embed(text, defs)
        tex.write_text(text, encoding="utf-8", newline="\n")
        print(f"{n} macros y {r} referencias incrustadas en {tex.parent.name}/{tex.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
