"""The macros of the interruptions section, on synthetic data."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from shared.metrics import interruptions_report as ir

pytest.importorskip("scipy")


def _csv(ruta: Path, rows: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _event(poc, clave, turn, tipo, det, *, estado="ok", final="0", t="", rep="", pause="",
            no_end="0", text="", stt="", glued_line="0"):
    # The key ends in the repetition: c<poc><k> is pair k of the three PoCs.
    return {"poc": poc, "clave": clave, "scenario": "1", "repetition": clave[-1],
            "turn": turn, "tipo": tipo, "fired": "1", "stopped": det,
            "stop_time_ms": t, "reply_latency_ms": rep,
            "pause_duration_ms": pause, "endless_pause": no_end if det == "1" else "",
            "text": text, "event_stt": stt, "corrected_event_stt": stt, "glued_line": glued_line,
            "event_wer": "0" if text else "", "final_turn": final, "estado": estado}


def _validation(offsets: bool) -> list[dict]:
    rows = []
    for s, err in (("200.0", "10"), ("500.0", "20")):
        row = {"target_stop_ms": s, "tipo": "interrupcion", "fired": "1", "harness_stopped": "1",
                "stop_error_ms": err}
        if offsets:
            row.update(speech_onset_offset_ms="200", last_speech_offset_ms="250")
        rows.append(row)
        rows.append({**row, "tipo": "control", "harness_stopped": "0", "stop_error_ms": ""})
    return rows


def _macros(tex: list[str]) -> dict[str, str]:
    out = {}
    for linea in tex:
        if linea.startswith("\\newcommand"):
            name, valor = linea[len("\\newcommand{\\"):].split("}{", 1)
            out[name] = valor[:-1]
    return out


@pytest.fixture
def data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    c = d / ir.CAMPANA
    rows = []
    for poc in "123":
        for k in range(3):
            clave = f"c{poc}{k}"
            rows += [
                _event(poc, clave, "T2", "interrupcion", "1", t=str(300 * int(poc) + k), rep="1600", pause="3000",
                        text="Oiga, espere.", stt="oiga espere"),
                _event(poc, clave, "T3", "apoyo", "1", pause=str(1000 * int(poc) + 10 * k), text="Ajá.", stt="ajá"),
                _event(poc, clave, "T4", "control", "1" if poc == "3" else "0", pause="900"),
            ]
    # The farewell does not count, and a failed call does.
    rows.append(_event("2", "c20", "T7", "interrupcion", "1", final="1", t="9999", text="Espéreme.", stt=""))
    for f in rows:
        if f["clave"] == "c21":
            f["estado"] = "roja"
    _csv(c / "interruptions_events.csv", rows)
    _csv(c / "interruption_pauses.csv", [
        {"clave": f["clave"], "turn": f["turn"], "pause_ms": f["pause_duration_ms"] if f["stopped"] == "1" else ""}
        for f in rows])
    sheet, clave = [], []
    for n, f in enumerate(x for x in rows if x["tipo"] == "apoyo"):
        sheet.append({"id": f"R{n}", "classification": "C" if f["poc"] == "3" else "R", "note": ""})
        clave.append({"id": f"R{n}", "poc": f["poc"], "clave": f["clave"], "turn": f["turn"]})
    _csv(c / "resumption_sheet_author.csv", sheet)
    _csv(c / "resumption_sheet_key.csv", clave)
    _csv(d / "interruptions_with_failed" / "interruptions_events.csv",
         [{"poc": "1", "clave": "p1"}, {"poc": "2", "clave": "p2"}])
    _csv(d / ir.VALIDATION["A"], _validation(False))
    _csv(d / ir.VALIDATION["B"], _validation(True))
    _csv(d / ir.VALIDATION["C"], _validation(True))
    for folder in ir.TRANSPORT.values():
        _csv(d / folder / "interruption_validation_poc2.csv", _validation(True))
    return d


def test_population_without_farewell_and_with_failed(data, tmp_path):
    m = _macros(ir.generate(data, tmp_path / "output")[0])
    assert m["RIntDespedida"] == "1" and m["RIntDespedidaB"] == "1"
    assert m["RIntIntFracB"] == "3/3" and m["RIntDetMedB"] == "601"  # without the farewell's 9,999
    assert m["RIntRojasB"] == "1" and m["RIntLlamadasB"] == "3"
    assert m["RIntIntWerB"] == "0.0" and m["RIntIntSinTextoB"] == "0/3"
    assert m["RIntCtlFracC"] == "3/3" and m["RIntCtlFracA"] == "0/3" and m["RIntCtlPauMedA"] == "---"
    assert m["RIntApoRetomaC"] == "3/3" and m["RIntApoPauMedC"] == "3{,}010"
    assert m["RIntCoincide"] == "27/27"


def test_resumptions_tests_and_background(data, tmp_path):
    m = _macros(ir.generate(data, tmp_path / "output")[0])
    assert m["RIntRetCC"] == "3" and m["RIntRetRA"] == "3" and m["RIntRetTotalB"] == "3"
    assert m["RIntPruebaFamilia"] == str(len(ir.COMPARISONS)) and m["RIntPruebaAlfa"] == "0.00625"
    # Paired: one pair per repetition, in all three comparisons.
    assert m["RIntPruebaPausaTresUnoN"] == "3" and m["RIntPruebaApoyoControlTresN"] == "3"
    # With three calls per PoC no test can survive the correction.
    assert all(m[f"RIntPrueba{c.name}Pasa"] == "no" for c in ir.COMPARISONS)
    assert m["RIntPilotoLlamadasA"] == "1" and m["RIntPilotoLlamadas"] == "2" and m["RIntPrimeraLlamadas"] == "0"


def test_validation_and_transport(data, tmp_path):
    m = _macros(ir.generate(data, tmp_path / "output")[0])
    assert m["RValIntDetC"] == "2/2" and m["RValIntFalsasC"] == "0/2"
    assert m["RValIntErrMinC"] == "10" and m["RValIntErrMaxC"] == "20" and m["RValIntIdaA"] == "---"
    assert m["RIntTransAntesTotal"] == "450" and m["RIntTransErrProm"] == "15"
    assert m["RIntDetSinTransB"] == "586" and m["RIntDetSinPuenteC"] == "886"
    assert m["RIntRedLatTotal"] == "---"  # without resultados.json nothing is made up


def test_an_invalid_classification_stops_the_generator(data, tmp_path):
    ruta = data / ir.CAMPANA / "resumption_sheet_author.csv"
    text = ruta.read_text(encoding="utf-8").replace(",C,", ",X,", 1)
    ruta.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        ir.generate(data, tmp_path / "output")
