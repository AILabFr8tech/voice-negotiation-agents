"""Tests for the per-call metrics.

Two things matter more than the rest:

- `TestCumplimiento`: automates two of the three binary checks that the script
  template asked for by hand. If this gets it wrong, a run in which the agent
  said an amount above the ceiling gets signed off as good.
- `TestJSONDeLaLlamadaReal`: the JSON that telephony produces today has **no**
  latency samples. The computation has to degrade with a warning, not blow up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shared.metrics import reference as ref
from shared.metrics.calls import (
    METRICS_KEY,
    compute,
    compute_from_path,
    load_json,
    enrich,
    identify,
    list_calls,
)


def _call_json(
    *,
    call_sid: str = "CA1",
    historial: list[float] | None = None,
    agent_text: str = "Le ofrezco dieciséis mil pesos.",
    with_latencies: bool = True,
    resultado: str = "sin_acuerdo",
    monto_final: float | None = None,
) -> dict[str, Any]:
    """A JSON with the exact shape that `telefonia.save_run` writes."""
    historial = historial if historial is not None else [16_000.0, 17_000.0]
    data: dict[str, Any] = {
        "call_sid": call_sid,
        "summary": {
            "call_id": call_sid,
            "company": "Freight Technologies",
            "origin": "Monterrey, Nuevo León",
            "destination": "Laredo, Texas",
            "cargo_type": "caja seca de 53 pies con carga general",
            "target_price": 18_000.0,
            "offer_history": historial,
            "offered_compensations": [],
            "resultado": {
                "resultado": resultado,
                "timestamp": "2026-08-05T12:00:00+00:00",
                "monto_final": monto_final,
                "condiciones": [],
                "offered_compensations": [],
                "offer_history": historial,
                "counteroffers": max(len(historial) - 1, 0),
                "motivo": None if resultado == "acuerdo" else "rechazo_definitivo",
                "reason_category": (
                    None if resultado == "acuerdo" else "rechazo_definitivo"
                ),
                "validated_final_amount": (
                    None
                    if monto_final is None
                    else any(abs(monto_final - o) <= 0.01 for o in historial)
                ),
                "call_id": call_sid,
                "metadata": {},
            },
        },
        "transcript": [
            {"t": 2.0, "role": ref.AGENT_ROLE, "text": agent_text},
            {"t": 8.0, "role": ref.CARRIER_ROLE, "text": "está muy bajo"},
            {"t": 14.5, "role": ref.AGENT_ROLE, "text": "Entiendo su punto."},
        ],
    }
    if with_latencies:
        data["latencies"] = [
            {"indice": 0, "full_turn_ms": 2100.0, "llm_ttft_ms": 800.0},
            {"indice": 1, "full_turn_ms": 1100.0, "llm_ttft_ms": 400.0},
            {"indice": 2, "full_turn_ms": 1200.0, "llm_ttft_ms": 450.0},
        ]
    return data


class TestDuration:
    def test_transcript_duration(self) -> None:
        metrics = compute(_call_json())
        assert metrics.duration_s == pytest.approx(14.5)

    def test_without_transcript_there_is_no_duration(self) -> None:
        data = _call_json()
        data["transcript"] = []
        assert compute(data).duration_s is None


class TestLatency:
    def test_percentiles_in_the_block(self) -> None:
        metrics = compute(_call_json())
        regimen = metrics.latency["buckets"]["regimen"]["full_turn"]
        assert regimen["n"] == 2
        assert regimen["p50"] == pytest.approx(1150.0)

    def test_opening_separately(self) -> None:
        metrics = compute(_call_json())
        opening = metrics.latency["buckets"]["apertura"]["full_turn"]
        assert opening["n"] == 1
        assert opening["p50"] == pytest.approx(2100.0)


class TestRealCallJSON:
    """The shape that telephony produces today: no latencies, with or without the agent."""

    def test_without_latencies_it_warns_and_does_not_blow_up(self) -> None:
        data = _call_json(with_latencies=False)
        metrics = compute(data)
        assert metrics.latency["n_turns"] == 0
        assert any("sin muestras de latencia" in a for a in metrics.warning_list)

    def test_without_agent_turns_it_warns(self) -> None:
        """This is the case of the August 3 call: there is no written audit."""
        data = _call_json()
        data["transcript"] = [
            {"t": 14.6, "role": ref.CARRIER_ROLE, "text": "Sí, claro que sí."}
        ]
        metrics = compute(data)
        assert any("no trae turnos del agente" in a for a in metrics.warning_list)

    def test_without_reference_there_is_no_wer(self) -> None:
        metrics = compute(_call_json())
        assert metrics.stt is None
        assert any("sin archivo de referencia" in a for a in metrics.warning_list)


class TestWithReference:
    def _file(self, reference: str, naturalness: int = 4) -> Any:
        content = (
            f"poc: 1\nscenario: 2\nrepetition: 1\n"
            f"## T1\nyo: {reference}\n"
            f"{ref.NOTES_HEADER}\n"
            f"naturalness: {naturalness}\nturn_taking: 3\n"
        )
        return ref.parse(content)

    def test_computes_wer(self) -> None:
        data = _call_json()
        # The carrier's transcript says "está muy bajo".
        file = self._file("está muy bajo")
        metrics = compute(data, reference_file=file)
        assert metrics.stt is not None
        assert metrics.stt["full_channel"]["wer"]["sample_rate"] == 0.0

    def test_wer_with_real_error(self) -> None:
        file = self._file("está muy caro")
        metrics = compute(
            _call_json(), reference_file=file
        )
        assert metrics.stt["full_channel"]["wer"]["substitutions"] == 1

    def test_takes_the_manual_notes(self) -> None:
        file = self._file("está muy bajo", naturalness=5)
        metrics = compute(
            _call_json(), reference_file=file
        )
        assert metrics.manual_notes["naturalness"] == 5
        assert metrics.manual_notes["completed"] is True

    def test_implausible_wer_warns(self) -> None:
        """A very high WER is almost always a wrong reference, not bad STT.

        It happened when running the module on the real August 3 call seeded
        with scenario 2: it gave 97% because that call was improvised and does
        not follow that script. Without the warning, the number reads as an STT
        result.
        """
        file = self._file("un texto que no tiene nada que ver con la llamada")
        metrics = compute(
            _call_json(), reference_file=file
        )
        assert metrics.stt["full_channel"]["wer"]["sample_rate"] > 0.5
        assert any("referencia sea de esta llamada" in a for a in metrics.warning_list)

    def test_normal_wer_does_not_warn(self) -> None:
        file = self._file("está muy bajo")
        metrics = compute(
            _call_json(), reference_file=file
        )
        assert not any("referencia sea de esta llamada" in a for a in metrics.warning_list)

    def test_identification_from_the_reference(self) -> None:
        file = self._file("está muy bajo")
        metrics = compute(
            _call_json(), reference_file=file
        )
        assert (metrics.poc, metrics.scenario, metrics.repetition) == (1, 2, 1)
        assert metrics.label == "2.1"


class TestIdentify:
    def test_the_json_takes_precedence_over_the_reference(self) -> None:
        """The JSON was written by the machine; the reference, by a person."""
        data = _call_json()
        data["scenario"] = 4
        file = ref.parse("scenario: 2\n")
        _, scenario, _ = identify(data, file)
        assert scenario == 4

    def test_without_anything_it_stays_none(self) -> None:
        assert identify(_call_json(), None) == (None, None, None)


class TestPersistence:
    def test_enrich_writes_the_block(self, tmp_path: Path) -> None:
        destination = tmp_path / "call_CA1.json"
        destination.write_text(
            json.dumps(_call_json(), ensure_ascii=False), encoding="utf-8"
        )
        enrich(destination)

        returned = load_json(destination)
        assert METRICS_KEY in returned
        assert returned[METRICS_KEY]["latency"]["n_turns"] == 3
        # What was already there is left untouched.
        assert returned["summary"]["target_price"] == 18_000.0
        assert len(returned["transcript"]) == 3

    def test_the_block_only_carries_technology_metrics(self, tmp_path: Path) -> None:
        """Latency, STT and manual notes. Nothing from the conversation content."""
        destination = tmp_path / "call_CA1.json"
        destination.write_text(json.dumps(_call_json()), encoding="utf-8")
        enrich(destination)
        block = json.loads(destination.read_text(encoding="utf-8"))["metrics"]
        assert set(block) <= {
            "call_sid",
            "poc",
            "scenario",
            "repetition",
            "duration_s",
            "latency",
            "stt",
            "manual_notes",
            "warning_list",
        }

    def test_is_idempotent(self, tmp_path: Path) -> None:
        destination = tmp_path / "call_CA1.json"
        destination.write_text(json.dumps(_call_json()), encoding="utf-8")
        first = enrich(destination)
        second_one = enrich(destination)
        assert first.to_dict() == second_one.to_dict()

    def test_finds_the_reference_alongside(self, tmp_path: Path) -> None:
        destination = tmp_path / "call_CA1.json"
        destination.write_text(json.dumps(_call_json()), encoding="utf-8")
        ref.write(
            "## T1\nyo: está muy bajo\n",
            ref.path_to_reference(tmp_path, "CA1"),
        )
        _, metrics = compute_from_path(destination)
        assert metrics.stt is not None

    def test_list_calls_sorts(self, tmp_path: Path) -> None:
        for sid in ("CA3", "CA1", "CA2"):
            (tmp_path / f"call_{sid}.json").write_text("{}", encoding="utf-8")
        (tmp_path / "other_thing.json").write_text("{}", encoding="utf-8")
        names = [p.name for p in list_calls(tmp_path)]
        assert names == ["call_CA1.json", "call_CA2.json", "call_CA3.json"]
