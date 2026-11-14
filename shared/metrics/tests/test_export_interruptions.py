"""The interruption-campaign exporter, on two synthetic calls."""

from __future__ import annotations

import csv
import json

from shared.metrics import export_interruptions as ex


def test_exports_events_of_ok_calls_and_summarizes_per_poc(tmp_path):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    progress = []
    for poc, estado in ((1, "ok"), (3, "ok"), (1, "failure")):
        clave = f"000{len(progress) + 1}-poc{poc}-e1-r1"
        progress.append({"clave": clave, "poc": poc, "scenario": 1, "repetition": 1, "estado": estado})
        events = [
            {"tipo": "interrupcion", "clip_id": "int_01", "fired": True, "stopped": True,
             "stop_time_ms": 400.0, "reply_latency_ms": 900.0, "advance_reason": "x"},
            {"tipo": "apoyo", "clip_id": "apo_01", "fired": True, "stopped": False,
             "stop_time_ms": 5000.0},
            {"tipo": "control", "clip_id": None, "fired": False, "motivo": "no habló"},
        ]
        (c / "summaries" / f"{clave}.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    (c / "progress.jsonl").write_text("\n".join(json.dumps(p) for p in progress), encoding="utf-8")

    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader((tmp_path / "out" / "interruptions_events.csv").open(encoding="utf-8")))
    # The failed call is not included.
    assert len(rows) == 6
    assert {f["poc"] for f in rows} == {"1", "3"}
    text = (tmp_path / "out" / "interruptions_summary.txt").read_text(encoding="utf-8")
    assert "se calló (esperado): 1/1 (100%)" in text
    assert "se calló (falsa detención): 0/1 (0%)" in text
    assert "disparados   0" in text


def test_what_the_stt_heard_of_the_event_goes_in_its_own_section(tmp_path):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    events = [
        {"tipo": "interrupcion", "clip_id": "int_03", "text": "Oiga, ¿eso incluye las casetas?",
         "event_stt": "oiga eso incluye las casetas", "fired": True, "stopped": True,
         "stop_time_ms": 400.0},
        {"tipo": "interrupcion", "clip_id": "int_04", "text": "Ya, ya le entendí. ¿Y el anticipo?",
         "event_stt": "ya le entendí", "fired": True, "stopped": True,
         "stop_time_ms": 500.0},
        {"tipo": "apoyo", "clip_id": "apo_02", "text": "Mjm.", "event_stt": "",
         "fired": True, "stopped": False, "stop_time_ms": 5000.0},
    ]
    (c / "summaries" / "0001-poc1-e1-r1.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    (c / "progress.jsonl").write_text(
        json.dumps({"clave": "0001-poc1-e1-r1", "poc": 1, "scenario": 1, "repetition": 1, "estado": "ok"}),
        encoding="utf-8",
    )
    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader((tmp_path / "out" / "interruptions_events.csv").open(encoding="utf-8")))
    assert float(rows[0]["event_wer"]) == 0.0
    assert float(rows[1]["event_wer"]) > 0.0
    assert float(rows[2]["event_wer"]) == 1.0
    text = (tmp_path / "out" / "interruptions_summary.txt").read_text(encoding="utf-8")
    assert "Lo que el STT oyó del evento" in text
    # Summed errors over summed words: 4 deleted out of 12 in the interruptions.
    linea = next(l for l in text.splitlines() if "interrupcion" in l and "%" in l and l.startswith("PoC1"))
    assert "33.3%" in linea and "0 (0%)" in linea
    linea = next(l for l in text.splitlines() if "apoyo" in l and "%" in l and l.startswith("PoC1"))
    assert "100.0%" in linea and "1 (100%)" in linea


def test_the_failed_enter_only_if_requested_and_are_marked(tmp_path):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    progress = []
    for i, estado in enumerate(("ok", "roja", "failure"), start=1):
        clave = f"000{i}-poc2-e{i}-r1"
        progress.append({"clave": clave, "poc": 2, "scenario": i, "repetition": 1, "estado": estado})
        events = [
            {"tipo": "interrupcion", "clip_id": "int_01", "fired": True, "stopped": True,
             "stop_time_ms": 900.0, "reply_latency_ms": 3000.0},
            # What the failed call did not get to fire before it closed.
            {"tipo": "control", "clip_id": None, "fired": False},
        ]
        (c / "summaries" / f"{clave}.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    (c / "progress.jsonl").write_text("\n".join(json.dumps(p) for p in progress), encoding="utf-8")

    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "only_ok")]) == 0
    only_ok = list(csv.DictReader((tmp_path / "only_ok" / "interruptions_events.csv").open(encoding="utf-8")))
    assert {f["estado"] for f in only_ok} == {"ok"}

    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "con"), "--include-failed"]) == 0
    having = list(csv.DictReader((tmp_path / "con" / "interruptions_events.csv").open(encoding="utf-8")))
    assert sorted({f["estado"] for f in having}) == ["ok", "roja"]
    text = (tmp_path / "con" / "interruptions_summary.txt").read_text(encoding="utf-8")
    assert "PoC2 (2 llamadas, 1 roja(s) incluida(s))" in text
    assert "se calló (esperado): 2/2 (100%)" in text


def test_the_pause_and_the_resumption_appear_in_the_csv_and_the_summary(tmp_path):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    events = [
        {"tipo": "apoyo", "clip_id": "apo_01", "fired": True, "stopped": True,
         "stop_time_ms": 300.0, "pause_duration_ms": 900.0, "endless_pause": False,
         "speaks_again_ms": 650.0},
        {"tipo": "apoyo", "clip_id": "apo_02", "fired": True, "stopped": True,
         "stop_time_ms": 200.0, "pause_duration_ms": 2300.0, "endless_pause": True},
    ]
    (c / "summaries" / "0001-poc3-e1-r1.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    (c / "progress.jsonl").write_text(json.dumps(
        {"clave": "0001-poc3-e1-r1", "poc": 3, "scenario": 1, "repetition": 1, "estado": "ok"}), encoding="utf-8")
    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader((tmp_path / "out" / "interruptions_events.csv").open(encoding="utf-8")))
    assert [(f["pause_duration_ms"], f["endless_pause"], f["speaks_again_ms"]) for f in rows] == [
        ("900.0", "0", "650.0"), ("2300.0", "1", "")]
    text = (tmp_path / "out" / "interruptions_summary.txt").read_text(encoding="utf-8")
    assert "retomó 1/2 · pausa p50 900 ms · vuelve a hablar p50 650 ms · no volvió antes de la línea siguiente 1" in text


def test_removes_the_attached_previous_line_and_nothing_else():
    from shared.metrics.export_interruptions import remove_glued_line

    linea = "El problema es el tiempo de carga. Siempre me tienen horas esperando."
    text, glued_line = remove_glued_line(
        linea, "El problema es el tiempo de carga. Siempre me tienen horas esperando Momento Eso no fue lo que quedamos")
    assert glued_line and text == "Momento Eso no fue lo que quedamos"
    # With a recognition error in the line, it is still recognized as the line.
    text, glued_line = remove_glued_line(
        "Bueno. Con las condiciones que me da, ¿en cuánto queda el flete al final?",
        "Bueno, con las condiciones que me da, en cuánto queda el plete al final Sí, sí")
    assert glued_line and text == "Sí, sí"


def test_without_the_attached_line_it_touches_nothing():
    from shared.metrics.export_interruptions import remove_glued_line

    assert remove_glued_line("¿A cómo andan pagando esa ruta?", "Espéreme, espéreme. ¿Cuánto dijo?") == (
        "Espéreme, espéreme. ¿Cuánto dijo?", False)
    assert remove_glued_line("¿A cómo andan pagando esa ruta?", "Ajá") == ("Ajá", False)
    assert remove_glued_line("", "Ajá Ajá Ajá") == ("Ajá Ajá Ajá", False)


def test_the_exporter_corrects_with_the_log_line(tmp_path):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    event_log = tmp_path / "event_log.json"
    event_log.write_text(json.dumps({"turns": [
        {"turn": "T4", "played_line": "No le pedí más dinero. Le pregunté por el tiempo de espera."}]}),
        encoding="utf-8")
    events = [{"turn": "T4", "tipo": "apoyo", "clip_id": "apo_05", "text": "Ándale.", "fired": True,
                "stopped": False, "event_stt": "No le pedí más dinero. Le pregunté por el tiempo de espera Ándale"}]
    (c / "summaries" / "0001-poc3-e4-r1.json").write_text(
        json.dumps({"events": events, "artifacts": {"event_log": str(event_log)}}), encoding="utf-8")
    (c / "progress.jsonl").write_text(json.dumps(
        {"clave": "0001-poc3-e4-r1", "poc": 3, "scenario": 4, "repetition": 1, "estado": "ok"}), encoding="utf-8")
    assert ex.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    row = next(csv.DictReader((tmp_path / "out" / "interruptions_events.csv").open(encoding="utf-8")))
    assert row["glued_line"] == "1" and row["corrected_event_stt"] == "Ándale"
    assert row["event_wer"] == "0.0"
    assert row["final_turn"] == "1"  # T4 is the last turn in this log
    summary = (tmp_path / "out" / "interruptions_summary.txt").read_text(encoding="utf-8")
    assert "1 con la línea anterior pegada, quitada" in summary
