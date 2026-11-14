"""La hoja de retomes: sin PoC, en orden aleatorio, y el conteo por PoC al final."""

from __future__ import annotations

import csv

import pytest

from validation import resumption_sheet as hr


def _events(tmp_path):
    ruta = tmp_path / "events.csv"
    rows = [
        {"poc": "1", "clave": "c1", "turn": "T2", "tipo": "apoyo", "fired": "1", "stopped": "1",
         "endless_pause": "0", "text": "Ajá.", "agent_before_event": "Le ofrezco",
         "agent_after_event": "Le ofrezco 16 mil.", "estado": "ok"},
        {"poc": "3", "clave": "c3", "turn": "T4", "tipo": "apoyo", "fired": "1", "stopped": "1",
         "endless_pause": "0", "text": "Ok.", "agent_before_event": "La oferta son",
         "agent_after_event": "diecisiete mil pesos.", "estado": "ok"},
        {"poc": "2", "clave": "c2", "turn": "T3", "tipo": "apoyo", "fired": "1", "stopped": "1",
         "endless_pause": "1", "text": "Sí.", "agent_before_event": "Carlos,",
         "agent_after_event": "", "estado": "ok"},  # no retomó: no entra
        {"poc": "3", "clave": "c3", "turn": "T5", "tipo": "control", "fired": "1", "stopped": "1",
         "endless_pause": "0", "text": "", "agent_before_event": "a",
         "agent_after_event": "b", "estado": "ok"},  # control: solo si se pide
    ]
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return ruta


def test_the_sheet_does_not_say_the_poc_and_the_key_does(tmp_path):
    sheet = tmp_path / "sheet.csv"
    assert hr.prepare(_events(tmp_path), sheet, kinds=("apoyo",)) == 2
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert set(rows[0]) == set(hr.SHEET_COLUMNS) and "poc" not in rows[0]
    clave = list(csv.DictReader((tmp_path / "sheet_clave.csv").open(encoding="utf-8-sig")))
    assert sorted(c["poc"] for c in clave) == ["1", "3"]


def test_count_by_poc_and_does_not_count_empty_rows(tmp_path):
    sheet = tmp_path / "sheet.csv"
    hr.prepare(_events(tmp_path), sheet, kinds=("apoyo",))
    with pytest.raises(SystemExit):
        hr.tally(sheet)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    clave = {c["id"]: c["poc"] for c in csv.DictReader((tmp_path / "sheet_clave.csv").open(encoding="utf-8-sig"))}
    for f in rows:
        f["classification"] = "r" if clave[f["id"]] == "1" else "C"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hr.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text, count = hr.tally(sheet)
    assert {(c["poc"], c["repitió"], c["continuó"]) for c in count} == {("1", 1, 0), ("3", 0, 1)}
    assert "PoC1" in text and "PoC3" in text


def test_counting_leaves_out_the_farewell(tmp_path):
    sheet = tmp_path / "sheet.csv"
    events = _events(tmp_path)
    hr.prepare(events, sheet, kinds=("apoyo",))
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    for f in rows:
        f["classification"] = "C"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hr.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    marked = tmp_path / "events_con_final.csv"
    with events.open(encoding="utf-8") as fh:
        ev = list(csv.DictReader(fh))
    for e in ev:
        e["final_turn"] = "1" if e["clave"] == "c1" else "0"
    with marked.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev[0]))
        w.writeheader()
        w.writerows(ev)
    _, count = hr.tally(sheet, events_csv=marked)
    assert [c["poc"] for c in count] == ["3"]
