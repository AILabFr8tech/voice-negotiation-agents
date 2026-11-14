"""La latencia de inicio: que mida el camino y no otra cosa.

Lo que se prueba es lo que la haría mentir en silencio: que se le olvide restar
la cabeza del clip (y entonces mida el silencio del WAV en vez del transporte),
que use un umbral distinto del que midió el fin de voz (y entonces la cabeza no
sea comparable con nada), y que no se dé cuenta de que los clips en disco no son
los que se reprodujeron.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from shared.carrier_simulator import speech_onset as mod

SAMPLE_RATE = 8000


def _clip(folder: Path, name: str, *, head_s: float, speech_s: float) -> Path:
    """Un WAV con `head_s` de silencio y después voz fuerte."""
    head = np.zeros(int(head_s * SAMPLE_RATE), dtype=np.int16)
    t = np.arange(int(speech_s * SAMPLE_RATE))
    voice = (np.sin(t / 5.0) * 12_000).astype(np.int16)
    samples = np.concatenate([head, voice])

    ruta = folder / f"{name}.wav"
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return ruta


def _row(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "paired": True,
        "harness_turn": "T1",
        "clip_id": "c1",
        "start_of_clip": 1000.0,
        "started_speaking_at": 1000.0,
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# La resta
# --------------------------------------------------------------------------


def test_the_clip_head_is_discounted(tmp_path: Path) -> None:
    """Sin restarla, un clip con 1 s de silencio mediría 1 s de latencia falsa."""
    _clip(tmp_path, "c1", head_s=1.0, speech_s=1.0)
    # El VAD marcó el arranque 200 ms después de que empezó a sonar la voz.
    row = _row(started_speaking_at=1000.0 + 1.0 + 0.2)

    [turn] = mod.compute([row], clips=tmp_path)
    assert turn.head_ms == pytest.approx(1000.0, abs=15.0)
    assert turn.latency_ms == pytest.approx(200.0, abs=15.0)


def test_two_clips_with_different_heads_give_the_same_latency(tmp_path: Path) -> None:
    """Es la propiedad que hace comparable la medida entre turnos."""
    _clip(tmp_path, "short", head_s=0.05, speech_s=1.0)
    _clip(tmp_path, "largo", head_s=1.20, speech_s=1.0)

    rows = [
        _row(clip_id="short", started_speaking_at=1000.0 + 0.05 + 0.3),
        _row(
            clip_id="largo",
            start_of_clip=2000.0,
            started_speaking_at=2000.0 + 1.20 + 0.3,
        ),
    ]
    a, b = mod.compute(rows, clips=tmp_path)
    assert a.latency_ms == pytest.approx(b.latency_ms, abs=20.0)


def test_published_threshold_wins_over_the_derived_one(tmp_path: Path) -> None:
    """Si se re-derivara del pico, la cabeza dejaría de casar con el fin de voz."""
    _clip(tmp_path, "c1", head_s=0.5, speech_s=1.0)

    no_threshold = mod.compute([_row()], clips=tmp_path)[0]
    # Un umbral absurdamente alto deja fuera toda la voz: la cabeza se va al
    # final del clip. Que cambie es la prueba de que el umbral se usa.
    with_threshold = mod.compute([_row(speech_threshold_dbfs=-1.0)], clips=tmp_path)[0]
    assert with_threshold.head_ms != no_threshold.head_ms
    assert with_threshold.umbral_dbfs == pytest.approx(-1.0)


def test_call_time_hangs_off_the_first_turn(tmp_path: Path) -> None:
    _clip(tmp_path, "c1", head_s=0.1, speech_s=0.5)
    rows = [
        _row(start_of_clip=1000.0, started_speaking_at=1000.4),
        _row(start_of_clip=1030.0, started_speaking_at=1030.4),
    ]
    a, b = mod.compute(rows, clips=tmp_path)
    assert a.t_in_call == pytest.approx(0.0)
    assert b.t_in_call == pytest.approx(30.0)


# --------------------------------------------------------------------------
# El testigo: que los clips en disco sean los que se reprodujeron
# --------------------------------------------------------------------------


def test_the_recomputed_end_matches_the_published_one(tmp_path: Path) -> None:
    _clip(tmp_path, "c1", head_s=0.2, speech_s=1.0)
    row = _row(measured_end_of_speech=1000.0 + 1.2, started_speaking_at=1000.5)
    [turn] = mod.compute([row], clips=tmp_path)
    assert turn.end_disagreement_ms == pytest.approx(0.0, abs=15.0)


def test_a_changed_clip_is_detected(tmp_path: Path) -> None:
    """Es la red que impide leer números calculados sobre otro audio."""
    _clip(tmp_path, "c1", head_s=0.2, speech_s=1.0)
    # La corrida dice que la voz terminó medio segundo después de donde termina
    # en el archivo que hay en disco.
    row = _row(measured_end_of_speech=1000.0 + 1.7, started_speaking_at=1000.5)
    [turn] = mod.compute([row], clips=tmp_path)
    assert turn.end_disagreement_ms is not None
    assert abs(turn.end_disagreement_ms) > 400.0

    data = mod.summary([turn])
    assert data["end_disagreement_max_ms"] > 400.0


# --------------------------------------------------------------------------
# Las dos formas del JSON
# --------------------------------------------------------------------------


def test_understands_the_poc1_onset(tmp_path: Path) -> None:
    _clip(tmp_path, "c1", head_s=0.1, speech_s=0.5)
    row = _row(started_speaking_at=None, estimated_speech_start=1000.4)
    del row["started_speaking_at"]
    [turn] = mod.compute([row], clips=tmp_path)
    assert turn.latency_ms == pytest.approx(300.0, abs=15.0)


def test_the_per_turn_is_found_nested() -> None:
    """PoC2 lo cuelga de `livekit_extra.arnes`; PoC1, de la raíz."""
    nested = {"livekit_extra": {"harness": {"per_turn": [{"clip_id": "x"}]}}}
    assert mod.rows_per_turn(nested) == [{"clip_id": "x"}]
    flat_text = {"per_turn": [{"clip_id": "y"}]}
    assert mod.rows_per_turn(flat_text) == [{"clip_id": "y"}]


def test_without_per_turn_it_invents_nothing() -> None:
    assert mod.rows_per_turn({"other_thing": [1, 2, 3]}) == []


# --------------------------------------------------------------------------
# Lo que se salta
# --------------------------------------------------------------------------


def test_an_unpaired_turn_does_not_enter(tmp_path: Path) -> None:
    _clip(tmp_path, "c1", head_s=0.1, speech_s=0.5)
    assert mod.compute([_row(paired=False)], clips=tmp_path) == []


def test_a_turn_without_onset_does_not_enter(tmp_path: Path) -> None:
    """Sin el instante del VAD no hay latencia; inventarla sería peor que nada."""
    _clip(tmp_path, "c1", head_s=0.1, speech_s=0.5)
    row = _row()
    del row["started_speaking_at"]
    assert mod.compute([row], clips=tmp_path) == []


def test_a_clip_that_is_not_on_disk_does_not_enter(tmp_path: Path) -> None:
    assert mod.compute([_row()], clips=tmp_path) == []


# --------------------------------------------------------------------------
# El resumen
# --------------------------------------------------------------------------


def test_the_slope_is_positive_when_the_latency_grows() -> None:
    turns = [
        mod.TurnStart("T1", "c", 0.0, 0.0, 100.0, -30.0, 0.0),
        mod.TurnStart("T2", "c", 10.0, 0.0, 200.0, -30.0, 0.0),
        mod.TurnStart("T3", "c", 20.0, 0.0, 300.0, -30.0, 0.0),
    ]
    assert mod.summary(turns)["slope_ms_per_s"] == pytest.approx(10.0)


def test_a_flat_latency_has_no_slope() -> None:
    turns = [
        mod.TurnStart(f"T{i}", "c", float(i * 10), 0.0, 200.0, -30.0, 0.0)
        for i in range(3)
    ]
    assert mod.summary(turns)["slope_ms_per_s"] == pytest.approx(0.0)


def test_with_two_turns_no_pending_is_published() -> None:
    """Dos puntos siempre dan una recta perfecta: no significaría nada."""
    turns = [
        mod.TurnStart("T1", "c", 0.0, 0.0, 100.0, -30.0, 0.0),
        mod.TurnStart("T2", "c", 10.0, 0.0, 900.0, -30.0, 0.0),
    ]
    assert "slope_ms_per_s" not in mod.summary(turns)


def test_without_turns_the_summary_does_not_fall_over() -> None:
    assert mod.summary([])["n"] == 0


# --------------------------------------------------------------------------
# Arranques imposibles
# --------------------------------------------------------------------------
#
# Una latencia negativa diría que el VAD marcó el habla antes de que sonara la
# voz del clip. Pasa cuando el VAD viene disparado desde antes del clip.


def test_an_impossible_onset_does_not_enter_the_summary() -> None:
    turns = [
        mod.TurnStart("T1", "c", 0.0, 0.0, 200.0, -30.0, 0.0),
        mod.TurnStart("T2", "c", 10.0, 0.0, -975.0, -30.0, 0.0),
        mod.TurnStart("T3", "c", 20.0, 0.0, 300.0, -30.0, 0.0),
    ]
    data = mod.summary(turns)
    assert data["n"] == 2
    assert data["impossible_onsets"] == 1
    assert data["min_ms"] == pytest.approx(200.0)
    assert data["p50_ms"] == pytest.approx(250.0)


def test_usable_splits_the_two_groups() -> None:
    measurable, impossible_cases = mod.usable(
        [
            mod.TurnStart("T1", "c", 0.0, 0.0, 10.0, -30.0, 0.0),
            mod.TurnStart("T2", "c", 1.0, 0.0, -1.0, -30.0, 0.0),
        ]
    )
    assert [t.turn for t in measurable] == ["T1"]
    assert [t.turn for t in impossible_cases] == ["T2"]


def test_a_latency_of_zero_is_measurable() -> None:
    """El borde: cero es raro pero no imposible; negativo sí lo es."""
    measurable, impossible_cases = mod.usable(
        [mod.TurnStart("T1", "c", 0.0, 0.0, 0.0, -30.0, 0.0)]
    )
    assert len(measurable) == 1 and not impossible_cases


def test_if_all_are_impossible_the_cli_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Publicar un p50 sobre puros imposibles sería peor que no publicar nada."""
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "c1", head_s=1.0, speech_s=1.0)
    run = tmp_path / "run.json"
    # El VAD marcó el arranque medio segundo antes de que el clip empezara.
    run.write_text(
        json.dumps({"per_turn": [_row(started_speaking_at=999.5)]}), encoding="utf-8"
    )

    assert mod.main([str(run), "--clips", str(clips)]) == 1
    assert "IMPOSIBLE" in capsys.readouterr().out


def test_with_some_impossible_the_cli_warns_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "c1", head_s=1.0, speech_s=1.0)
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "per_turn": [
                    _row(started_speaking_at=999.5),
                    _row(start_of_clip=2000.0, started_speaking_at=2001.3),
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mod.main([str(run), "--clips", str(clips)]) == 0
    output = capsys.readouterr().out
    assert "1 turno(s) con arranque IMPOSIBLE" in output


# --------------------------------------------------------------------------
# La CLI
# --------------------------------------------------------------------------


def test_the_cli_fails_if_the_clips_are_not_the_ones_that_played(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un desacuerdo grande no puede salir como una tabla más."""
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "c1", head_s=0.2, speech_s=1.0)
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {"per_turn": [_row(measured_end_of_speech=1000.0 + 1.7, started_speaking_at=1000.5)]}
        ),
        encoding="utf-8",
    )

    assert mod.main([str(run), "--clips", str(clips)]) == 1
    assert "no son los que se reprodujeron" in capsys.readouterr().out


def test_the_cli_comes_out_green_with_a_healthy_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "c1", head_s=0.2, speech_s=1.0)
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {"per_turn": [_row(measured_end_of_speech=1000.0 + 1.2, started_speaking_at=1000.5)]}
        ),
        encoding="utf-8",
    )

    assert mod.main([str(run), "--clips", str(clips)]) == 0
    assert "LATENCIA ms" in capsys.readouterr().out


def test_the_cli_warns_if_there_is_no_per_turn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"nada": 1}), encoding="utf-8")
    assert mod.main([str(run)]) == 1
    assert "per_turn" in capsys.readouterr().out
