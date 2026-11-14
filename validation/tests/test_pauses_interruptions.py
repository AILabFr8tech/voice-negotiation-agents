"""La pausa del agente alrededor de cada evento, medida del canal del agente."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from shared.audio.wav import write_wav_stereo
from validation import interruption_pauses as pz

SAMPLE_RATE = 8_000


def _tone(seg: float) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seg)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 300 * t) * 6_000).astype(np.int16)


def _voice(*segments: tuple[float, float], dur: float = 30.0) -> np.ndarray:
    """Booleanos de 20 ms con voz en los tramos dados (segundos)."""
    v = np.zeros(int(dur / pz.WINDOW_S), dtype=bool)
    for a, b in segments:
        v[int(a / pz.WINDOW_S):int(b / pz.WINDOW_S)] = True
    return v


def test_brief_pause_and_resume():
    # Habla hasta 6.0, calla 6.0-7.2, retoma. El "mjm" fue de 5.0 a 5.5.
    pause, no_end, returns = pz.measure_pause(
        _voice((0, 6.0), (7.2, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert pause == pytest.approx(1200, abs=20)
    assert no_end is False
    assert returns == pytest.approx(1700, abs=20)


def test_the_gaps_between_words_are_not_a_pause():
    pause, no_end, returns = pz.measure_pause(
        _voice((0, 6.0), (6.4, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert pause is None and returns is None and no_end is False


def test_a_pause_that_starts_late_does_not_count():
    """Fuera de los 3 s de la detención del arnés, la pausa es de otra cosa."""
    pause, _, _ = pz.measure_pause(
        _voice((0, 9.0), (10, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert pause is None


def test_if_it_does_not_come_back_the_pause_is_a_lower_bound():
    pause, no_end, returns = pz.measure_pause(
        _voice((0, 6.0)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert no_end is True and returns is None
    assert pause == pytest.approx(7500, abs=20)


def _campana(tmp_path, *, estado="ok"):
    c = tmp_path / "campana"
    (c / "summaries").mkdir(parents=True)
    dur = 20.0
    agent = np.zeros(int(SAMPLE_RATE * dur), dtype=np.int16)
    agent[: int(SAMPLE_RATE * 6.0)] = _tone(6.0)          # apoyo a los 5 s: pausa 6.0-7.2
    agent[int(SAMPLE_RATE * 7.2):int(SAMPLE_RATE * 14.0)] = _tone(6.8)
    wav = c / "auditoria.wav"
    write_wav_stereo(wav, np.zeros_like(agent), agent, SAMPLE_RATE)
    events = [
        {"turn": "T4", "tipo": "apoyo", "clip_id": "apo_02", "fired": True, "stopped": True,
         "t_speech_onset": 5.0, "t_end_of_speech": 5.5},
        {"turn": "T5", "tipo": "control", "fired": True, "stopped": False,
         "t_speech_onset": 10.0},
        {"turn": "T6", "tipo": "control", "fired": False},
    ]
    (c / "summaries" / "0001-poc3-e1-r1.json").write_text(
        json.dumps({"events": events, "artifacts": {"auditoria_wav": str(wav)}}), encoding="utf-8")
    (c / "progress.jsonl").write_text(json.dumps(
        {"clave": "0001-poc3-e1-r1", "poc": 3, "scenario": 1, "repetition": 1, "estado": estado}),
        encoding="utf-8")
    return c


def test_exports_per_event_and_summarizes_per_poc(tmp_path):
    c = _campana(tmp_path)
    assert pz.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    with (tmp_path / "out" / "interruption_pauses.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["tipo"], r["pause_ms"]) for r in rows] == [("apoyo", "1200.0"), ("control", "")]
    text = (tmp_path / "out" / "interruption_pauses_summary.txt").read_text(encoding="utf-8")
    assert "PoC3  apoyo           1  1/1       1,200 [1,200, 1,200]" in text
    assert "PoC3  control         1  0/1" in text


def test_the_failed_only_if_requested(tmp_path):
    c = _campana(tmp_path, estado="roja")
    assert pz.rows_of(c, include_failed=False)[0] == []
    assert len(pz.rows_of(c, include_failed=True)[0]) == 2


def test_the_stop_is_the_last_segment_with_speech():
    m = pz.measure_in_audio(_voice((0, 6.0), (7.2, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert m["last_speech"] == pytest.approx(5.98) and m["stop_ms"] == pytest.approx(980, abs=1)


def test_if_it_was_already_quiet_the_stop_comes_out_negative():
    m = pz.measure_in_audio(_voice((0, 4.9), (7.0, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert m["pause_ms"] is not None and m["stop_ms"] == pytest.approx(-120, abs=1)


def test_the_csv_carries_the_stop(tmp_path):
    c = _campana(tmp_path)
    pz.main(["--campana", str(c), "--output", str(tmp_path / "out")])
    with (tmp_path / "out" / "interruption_pauses.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["audio_stop_ms"]) == pytest.approx(980, abs=1)
    assert rows[1]["audio_stop_ms"] == ""


def test_without_a_pause_the_longest_silence_is_reported():
    m = pz.measure_in_audio(_voice((0, 6.0), (6.6, 12)), reference=5.0, end_of_speech=5.5, until=13.5)
    assert m["pause_ms"] is None and m["max_silence_ms"] == pytest.approx(600, abs=20)
