"""Arma docs/paper/en/supplementary.tex, el material suplementario del artículo en inglés.

    python docs/paper/en/armar_suplementario.py

El texto está en `supplementary_plantilla.tex`. Este script le incrusta lo que sale del
análisis, para que el archivo compile solo (en Overleaf no hay carpeta `generado/`):

- las tablas de líneas de cada escenario (`guion_en_e<N>.tex`, `guion_en_interrupciones.tex`);
- las líneas en que el audio difiere del guion escrito (`guiones_notas.tex`);
- las macros de cifras que usa el texto (`\\R...`), con "sí"/"no" en inglés.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1] / "report" / "latex" / "generated"
TEMPLATE = HERE / "supplementary_template.tex"
OUTPUT = HERE / "supplementary.tex"
SOURCES = ("results.tex", "interruptions.tex")
RE_MACRO = re.compile(r"\\(R[A-Z][A-Za-z]*)")
RE_DEF = re.compile(r"^\\newcommand\{\\(R[A-Z][A-Za-z]*)\}", re.M)


def rows(name: str) -> str:
    """Los renglones de un archivo generado, sin su línea de comentario."""
    text_lines = (GEN / name).read_text(encoding="utf-8").splitlines()
    return "\n".join(s for s in text_lines if not s.startswith("%"))


def notes_in() -> str:
    """`guiones_notas.tex` con su frase fija en inglés; las líneas quedan en español."""
    t = rows("script_notes.tex")
    assert "el guion decía" in t
    return t.replace("el guion decía", "the written script said")


def main() -> int:
    text = TEMPLATE.read_text(encoding="utf-8")
    for n in range(1, 6):
        text = text.replace(f"%%GUION:e{n}%%", rows(f"script_en_e{n}.tex"))
    text = text.replace("%%GUION:interrupciones%%", rows("script_en_interruptions.tex"))
    text = text.replace("%%NOTAS%%", notes_in())
    assert "%%GUION" not in text and "%%NOTAS" not in text

    defs: dict[str, str] = {}
    for f in SOURCES:
        for linea in (GEN / f).read_text(encoding="utf-8").splitlines():
            if (m := RE_DEF.match(linea)):
                defs[m.group(1)] = linea
    used = sorted(set(RE_MACRO.findall(text)))
    missing_items = [u for u in used if u not in defs]
    assert not missing_items, f"macros sin definir: {missing_items}"
    block = "\n".join(defs[u].replace("{sí}", "{yes}") for u in used)
    assert text.count("%%MACROS%%") == 1
    text = text.replace("%%MACROS%%", block)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(used)} macros incrustadas en {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
