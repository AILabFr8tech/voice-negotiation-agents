import csv

import pytest

from shared.metrics import timeline_report as lt


def _csv(ruta, rows):
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clave", "que", "start_s", "end_s", "text"])
        w.writeheader()
        w.writerows({"clave": "x", "que": q, "start_s": a, "end_s": b, "text": t} for q, a, b, t in rows)


ROWS = [
    ("linea", 0.0, 5.45, "línea"), ("transcripcion_de_la_linea", 5.315, "", "stt"),
    ("agente_antes", 8.851, 11.158, "antes"), ("apoyo", 10.366, 10.696, "Ok."),
    ("transcripcion_del_apoyo", 11.159, "", "Ok."), ("registrar_acuerdo", 12.622, "", "monto_final=16500"),
    ("agente_despues", 13.412, 22.992, "después"), ("linea_siguiente", 25.015, 27.175, "following"),
    ("cuelga", 25.228, "", ""),
]


def test_timeline_macros(tmp_path):
    _csv(tmp_path / "lt.csv", ROWS)
    tex = "\n".join(lt.generate(tmp_path / "lt.csv", tmp_path / "lt.tex"))
    assert "\\newcommand{\\RLtApoyoIni}{10.37}" in tex
    assert "\\newcommand{\\RLtDespuesFin}{22.99}" in tex
    assert "\\RLtRegistroFin" not in tex
    assert "\\newcommand{\\RLtEje}{30}" in tex
    assert "\\newcommand{\\RLtCallaSeg}{0.8}" in tex
    assert "\\newcommand{\\RLtRegistroSeg}{2.3}" in tex
    assert "\\newcommand{\\RLtMonto}{16{,}500}" in tex


def test_a_piece_is_missing(tmp_path):
    _csv(tmp_path / "lt.csv", ROWS[:-1])
    with pytest.raises(SystemExit, match="cuelga"):
        lt.generate(tmp_path / "lt.csv", tmp_path / "lt.tex")
