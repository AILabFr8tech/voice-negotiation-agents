import json
from pathlib import Path

from shared.metrics import census_poc3


def _census(tmp_path: Path, events: list[dict]) -> Path:
    ruta = tmp_path / "census.jsonl"
    ruta.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return ruta


def _response(t: float, internal: dict) -> dict:
    return {"t_local_seg": t, "type": "response.event",
            "event": {"type": "response.event", "delegation_id": "d1", "event": internal}}


def test_the_public_api_comes_out_with_the_alpha_names(tmp_path):
    census = _census(tmp_path, [
        {"t_local_seg": 0.0, "type": "session.started",
         "event": {"session": {"audio": {"format": {"type": "audio/pcmu", "rate": 8000}}}}},
        {"t_local_seg": 1.0, "type": "session.output_audio.delta",
         "event": {"type": "session.output_audio.delta", "delta": "<audio 801 bytes>"}},
        {"t_local_seg": 2.0, "type": "session.delegation.created", "event": {"delegation": {"id": "d1"}}},
        _response(2.0, {"type": "response.created"}),
        _response(3.0, {"type": "response.output_item.done",
                         "item": {"type": "function_call", "call_id": "c1", "name": "proponer_precio"}}),
        _response(3.2, {"type": "response.completed"}),
        _response(3.5, {"type": "response.created"}),
    ])
    output = list(census_poc3.events(census))
    kinds = [e["type"] for e in output]
    assert kinds == [
        "session.started", "output_audio.delta", "delegation.created", "response.created",
        "response.output_item.done", "response.completed",
        "delegation.function_call_output.created", "response.created",
    ]
    # 801 bytes of μ-law at 8 kHz are one 100 ms block.
    assert output[1]["event"]["audio_ms"] == 100.0
    # The function output is timestamped when the delegate starts again.
    acknowledgement = output[6]
    assert acknowledgement["t_local_seg"] == 3.5 and acknowledgement["event"]["item"]["call_id"] == "c1"


def test_the_alpha_census_passes_the_same(tmp_path):
    raw_value = [{"t_local_seg": 1.0, "type": "delegation.created", "event": {"item": {"content": []}}},
             {"t_local_seg": 2.0, "type": "output_audio.delta", "event": {"start_ms": 0, "end_ms": 100}}]
    assert list(census_poc3.events(_census(tmp_path, raw_value))) == raw_value
