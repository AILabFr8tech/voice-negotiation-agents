import csv

import pytest

from shared.metrics import gpt_live_clock_report as rg


def _csv(ruta, columns, rows):
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _data(tmp_path):
    clock = [
        {"tool": h, "latency_ms": lat, "model_ms": mod, "model_first_ms": mod - 200,
         "rest_ms": lat - mod, "last_word_ms": 1000, "cola_ms": 100, "output_ms": 250,
         "delivery_ms": lat - 1000 - mod - 250}
        for h, lat, mod in (("0", 2000, 600), ("0", 2200, 600), ("1", 3000, 1400), ("1", 3200, 1600))
    ]
    _csv(tmp_path / "model_clock_poc3.csv", list(clock[0]), clock)
    drift = [{"clave": "a", "span_s": 120, "model_ms": 800, "usage_ms": 0},
              {"clave": "b", "span_s": 100, "model_ms": 600, "usage_ms": "nan"}]
    _csv(tmp_path / "model_clock_poc3_drift.csv", list(drift[0]), drift)
    routes = [
        {"date": "2026-10-21", "via": via, "session": s, "clip": c, "primer_audio_ms": base + 10 * s,
         "last_word_ms": 950, "model_ms": 0}
        for via, base in (("ws", 1200), ("rtc", 1300)) for s in (1, 2) for c in ("e1_t2", "e1_t3")
    ]
    _csv(tmp_path / "transport_gpt_live.csv", list(routes[0]), routes)
    network = [{"date": "2026-10-20", "host": "h", "ip": "1", "attempts": 15, "p50_ms": 80.0, "min_ms": 70, "max_ms": 90},
           {"date": "2026-10-21", "host": "h", "ip": "1", "attempts": 15, "p50_ms": 41.7, "min_ms": 39, "max_ms": 66}]
    _csv(tmp_path / "network_gpt_live.csv", list(network[0]), network)
    return tmp_path


def test_summarizes_the_two_tables(tmp_path):
    tex = "\n".join(rg.generate(_data(tmp_path), tmp_path / "gpt_live_clock.tex"))
    assert "\\newcommand{\\RRelojN}{4}" in tex
    assert "\\newcommand{\\RRelojModeloSinHerrMed}{600}" in tex
    assert "\\newcommand{\\RRelojModeloConHerrMed}{1{,}500}" in tex
    assert "\\newcommand{\\RRelojRestoMed}{1{,}600}" in tex
    assert "\\newcommand{\\RRelojModeloValor}{1000.0}" in tex
    assert "\\newcommand{\\RRelojRestoValor}{1600.0}" in tex
    assert "\\newcommand{\\RViaSesiones}{2}" in tex
    assert "\\newcommand{\\RViaLineas}{2}" in tex
    assert "\\newcommand{\\RViaWsMed}{1{,}215}" in tex
    assert "\\newcommand{\\RViaWsFueraMed}{1{,}215}" in tex
    assert "\\newcommand{\\RViaDif}{100}" in tex


def test_distributes_the_wait_and_picks_an_example(tmp_path):
    tex = "\n".join(rg.generate(_data(tmp_path), tmp_path / "gpt_live_clock.tex"))
    assert "\\newcommand{\\RRelojServicioMed}{1{,}250}" in tex
    assert "\\newcommand{\\RRelojEntregaMed}{350}" in tex
    assert "\\newcommand{\\RRelojSumaMedMs}{2{,}600}" in tex
    assert "\\newcommand{\\RRelojSalidaSinHerrMed}{250}" in tex
    # The no-tool line closest to the medians is the 2,200 ms one.
    assert "\\newcommand{\\RRelojEjTotal}{2.200}" in tex
    assert "\\newcommand{\\RRelojEjEntregaMs}{350}" in tex
    assert "\\newcommand{\\RRelojDerivaMed}{700}" in tex
    assert "\\newcommand{\\RRelojDerivaUsoMed}{0}" in tex
    # For the network, the most recent measurement is the one that counts.
    assert "\\newcommand{\\RRedMed}{42}" in tex


def test_a_table_is_missing(tmp_path):
    (tmp_path / "model_clock_poc3.csv").write_text("herramienta\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="drift"):
        rg.generate(tmp_path, tmp_path / "x.tex")
