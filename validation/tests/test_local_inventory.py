"""El inventario local, sobre una campaña sintética."""

from __future__ import annotations

import json
from pathlib import Path

from validation import local_inventory as inv


def _write(ruta: Path, data) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _call(folder: Path, clave: str, *, valid=True, no_audio=(), stt="hola",
             failed_list=0, reviews=()) -> None:
    event_log = folder / f"event_log_{clave}.json"
    verification = folder / f"verification_{clave}.json"
    _write(event_log, {"turns": [
        {"turn": "T1", "played_line": "Sí, soy Carlos.", "stt": stt, "agent": "Buenas"},
    ]})
    _write(verification, {"reviews": [{"que": q, "ok": False} for q in reviews]})
    _write(folder / "summaries" / f"{clave}.json", {
        "failed_criteria": failed_list,
        "end": {"valid": valid, "turns_reached": 1,
                "turns_without_agent_audio": list(no_audio)},
        "artifacts": {"event_log": str(event_log), "verification": str(verification)},
    })


def test_populations_rebuilds_criteria_and_pairs(tmp_path: Path):
    c200 = tmp_path / "campana200"
    rows = [
        {"indice": 1, "clave": "a1", "poc": 1, "scenario": 1, "repetition": 1, "estado": "ok"},
        {"indice": 2, "clave": "a2", "poc": 2, "scenario": 1, "repetition": 1, "estado": "ok"},
        {"indice": 3, "clave": "b1", "poc": 1, "scenario": 1, "repetition": 2, "estado": "ok"},
        {"indice": 4, "clave": "b2", "poc": 2, "scenario": 1, "repetition": 2, "estado": "roja"},
        {"indice": 181, "clave": "c1", "poc": 1, "scenario": 1, "repetition": 19, "estado": "ok"},
    ]
    (c200).mkdir(parents=True)
    (c200 / "progress.jsonl").write_text("\n".join(json.dumps(f) for f in rows), encoding="utf-8")
    for f in rows:
        _call(c200, f["clave"])
    _call(c200, "b2", no_audio=["T1"], failed_list=1)
    c3 = tmp_path / "campana_poc3_pcmu"
    of_poc3 = [
        {"indice": 1, "clave": "p1", "poc": 3, "scenario": 1, "repetition": 1, "estado": "ok"},
        {"indice": 2, "clave": "p2", "poc": 3, "scenario": 1, "repetition": 2, "estado": "ok"},
    ]
    c3.mkdir(parents=True)
    (c3 / "progress.jsonl").write_text(chr(10).join(json.dumps(f) for f in of_poc3), encoding="utf-8")
    for f in of_poc3:
        _call(c3, f["clave"])

    output = {f["clave"]: f for f in inv.populations(tmp_path)}
    assert output["a1"]["in_the_pool"] == 1 and output["a2"]["in_the_pool"] == 1
    assert output["b1"]["in_the_pool"] == 0 and output["c1"]["in_the_pool"] == 0
    # Regla de las tres PoC: PoC3 pasó en (1, 2), pero PoC2 no.
    assert output["p1"]["in_the_pool"] == 1 and output["p2"]["in_the_pool"] == 0
    assert output["b2"]["response_without_audio"] == 1
    assert output["b2"]["reconstruction_adds_up"] == 1
    assert "sin audio" not in inv.exclusion_reason(output["b2"])  # nombres, no texto libre
    assert inv.exclusion_reason(output["b2"]).startswith("roja: response_without_audio")
    assert "regla de las tres PoC" in inv.exclusion_reason(output["b1"])
    assert "después de la llamada 180" in inv.exclusion_reason(output["c1"])


def test_the_env_only_delivers_the_public_keys(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-secreto\nOPENAI_LIVE_VOICE=marin\n# OPENAI_MODEL=x\n",
                   encoding="utf-8")
    assert inv.public_env_values(env) == {"OPENAI_LIVE_VOICE": "marin"}


def test_the_census_gives_the_request_the_function_and_the_session(tmp_path: Path):
    census = tmp_path / "census.jsonl"
    events = [
        {"type": "session.started", "t_local_seg": 0.1,
         "event": {"session": {"model": "gpt-live-1", "audio": {"output": {"voice": "marin"}}}}},
        {"type": "delegation.created", "t_local_seg": 1.0,
         "event": {"item": {"target": "responses", "content": [{"type": "input_text", "text": "Valida 16000"}]}}},
        {"type": "response.created", "t_local_seg": 1.1,
         "event": {"response": {"model": "gpt-4.1-mini", "instructions": None, "tools": [1, 2, 3, 4]}}},
        {"type": "response.output_item.done", "t_local_seg": 1.5,
         "event": {"item": {"type": "function_call", "name": "proponer_precio", "arguments": "{\"monto\":16000}"}}},
    ]
    census.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    records, session = inv.census_delegations(census, "x")
    assert session == {"model": "gpt-live-1", "voice": "marin"}
    assert [r["tipo"] for r in records] == [
        "delegation.created", "response.created", "response.output_item.done"]
    assert records[0]["text"] == "Valida 16000"
    assert records[2]["function"] == "proponer_precio"
    inv.mark_instructions(records)
    assert records[1]["received_instructions"] is False
