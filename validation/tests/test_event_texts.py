"""El listado de texto por evento sale de la bitácora de cada llamada."""

from __future__ import annotations

import json

from validation import event_texts as tx


def _campana(tmp_path, estado="ok"):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    event_log = tmp_path / "event_log.json"
    event_log.write_text(json.dumps({"turns": [
        {"turn": "T2", "played_line": "¿Cuánto paga?", "agent": "Le ofrezco 17 mil. Sí, le ofrezco 17 mil.",
         "event": {"tipo": "apoyo", "text": "Ajá.", "fired": True, "stopped": True,
                    "stop_time_ms": 300.0, "pause_duration_ms": 1100.0,
                    "endless_pause": False, "speaks_again_ms": 900.0}},
        {"turn": "T3", "played_line": "Ok", "agent": "Perfecto.",
         "event": {"tipo": "control", "fired": True, "stopped": False}},
        {"turn": "T4", "played_line": "Va", "agent": "Bien."},
    ]}), encoding="utf-8")
    (c / "summaries" / "0001-poc2-e1-r1.json").write_text(
        json.dumps({"artifacts": {"event_log": str(event_log)}}), encoding="utf-8")
    (c / "progress.jsonl").write_text(json.dumps(
        {"clave": "0001-poc2-e1-r1", "poc": 2, "estado": estado}), encoding="utf-8")
    return c


def test_lists_the_events_with_what_the_agent_said(tmp_path):
    c = _campana(tmp_path)
    output = tmp_path / "texts.md"
    assert tx.main(["--campana", str(c), "--output", str(output)]) == 0
    md = output.read_text(encoding="utf-8")
    assert "## PoC2 · apoyo (1)" in md
    assert "`poc2/0001-poc2-e1-r1_T2_apoyo.wav`" in md
    assert "pausa de 1,100 ms, volvió a hablar 900 ms después del clip" in md
    assert "Agente: Le ofrezco 17 mil. Sí, le ofrezco 17 mil." in md
    assert "control" not in md.split("## PoC2 · apoyo")[1]  # los controles solo si se piden


def test_the_failed_and_the_controls_only_if_requested(tmp_path):
    c = _campana(tmp_path, estado="roja")
    assert tx.entries_of(c, include_failed=False, kinds=("apoyo",)) == []
    all_items = tx.entries_of(c, include_failed=True, kinds=("apoyo", "control"))
    assert [e["tipo"] for e in all_items] == ["apoyo", "control"]
    assert all_items[1]["what_it_did"] == "siguió hablando (no se calló)"


def test_with_split_text_shows_before_and_after(tmp_path):
    c = _campana(tmp_path)
    event_log = tmp_path / "event_log.json"
    data = json.loads(event_log.read_text(encoding="utf-8"))
    data["turns"][0]["event"].update(
        agent_before_event="Le ofrezco 17", agent_after_event="Sí, le ofrezco 17 mil.")
    event_log.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "texts.md"
    tx.main(["--campana", str(c), "--output", str(output)])
    md = output.read_text(encoding="utf-8")
    assert "- Agente antes del evento: Le ofrezco 17" in md
    assert "- Agente después del evento: Sí, le ofrezco 17 mil." in md
