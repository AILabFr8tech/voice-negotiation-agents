"""Pruebas de la validación del arnés, las figuras de los clips y la sensibilidad de Deepgram.

Lo que corre en tiempo real contra el arnés no se prueba aquí: tarda minutos. Se
prueban las piezas que deciden los números.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from shared.audio import TELEPHONY_SAMPLE_RATE
from validation import fake_agent as fa
from validation import validate_harness as va


def test_the_playable_are_47_and_leave_out_the_two_alternatives_without_branch():
    ids = va.reproducible_ones()
    assert len(ids) == 47
    assert len(set(ids)) == 47
    assert "e5_t6_alt1" not in ids and "e5_t7_alt1" not in ids
    # Las alternativas que sí tienen rama entran.
    for alt in ("e2_t5_alt1", "e2_t5_alt2", "e2_t7_alt1", "e2_t8_alt1",
                "e4_t4_alt1", "e4_t6_alt1", "e4_t8_alt1"):
        assert alt in ids


def _voice(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * TELEPHONY_SAMPLE_RATE)) / TELEPHONY_SAMPLE_RATE
    sig = np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    sig[: int(0.05 * TELEPHONY_SAMPLE_RATE)] = 0.0
    return (sig * 12000).astype(np.int16)


def test_at_level_puts_the_speech_rms_where_requested():
    from shared.audio.level import rms_of_speech

    for level in (-35.0, -20.0):
        assert rms_of_speech(fa.to_level(_voice(), level), TELEPHONY_SAMPLE_RATE) == pytest.approx(level, abs=0.1)


def test_the_response_without_soft_onset_starts_at_its_speech_onset():
    r = fa.build_response(_voice(), level_dbfs=-20.0, soft_onset=False)
    assert r.start_mm is None
    # La voz sintética arranca a los 50 ms.
    assert r.phrase_start == pytest.approx(0.05 * TELEPHONY_SAMPLE_RATE, abs=80)


def test_the_soft_onset_puts_the_mm_12_db_down_and_the_phrase_after_the_pause():
    r = fa.build_response(_voice(), level_dbfs=-20.0, soft_onset=True)
    assert r.start_mm == 0
    n_mm = int(fa.MM_DURATION_S * TELEPHONY_SAMPLE_RATE)
    assert r.phrase_start == n_mm + int(fa.MM_PAUSE_S * TELEPHONY_SAMPLE_RATE)
    mm = r.pcm[:n_mm].astype(np.float64)
    rms = 20 * np.log10(np.sqrt(np.mean(mm**2)) / 32768)
    assert rms == pytest.approx(-20.0 - fa.MM_BELOW_SPEECH_DB, abs=0.2)


def test_the_fingerprints_identify_each_clip_by_its_first_frame(tmp_path):
    from validation.synthetic_clips import generate

    generate(tmp_path)
    fingerprints = fa.clip_fingerprints(tmp_path)
    assert len(fingerprints) == 49
    assert all(len(k) == fa.SAMPLES_PER_FRAME for k in fingerprints)


def test_the_summary_computes_bias_and_max_error():
    rows = [
        {"condition": "retardo_300", "error_ms": e, "reported_latency_ms": 1.0}
        for e in (-10.0, -20.0, 5.0)
    ] + [{"condition": "retardo_300", "error_ms": None, "reported_latency_ms": None}]
    (r,) = va.summary(rows)
    assert r["turns"] == 4 and r["with_measurement"] == 3 and r["no_harness_measurement"] == 1
    assert r["mean_bias_ms"] == pytest.approx(-8.33, abs=0.01)
    assert r["max_absolute_error_ms"] == 20.0
    assert r["min_error_ms"] == -20.0 and r["max_error_ms"] == 5.0


def test_whisper_is_read_per_word():
    from validation.clip_figures import whisper_words

    word = SimpleNamespace(word=" hola", start=0.1, end=0.4)
    segment = SimpleNamespace(text=" hola", words=[word])

    class Fake:
        def transcribe(self, audio, **kw):
            assert kw["word_timestamps"] and kw["language"] == "es"
            assert audio.dtype == np.float32 and len(audio) == 2 * 8000
            return iter([segment]), None

    text, words = whisper_words(Fake(), np.zeros(8000, dtype=np.int16))
    assert text == "hola"
    assert words == [("hola", 0.1, 0.4)]


def test_the_sensitivity_counts_masked_and_removes_fillers(monkeypatch):
    from shared.metrics import text as module_text
    from validation import deepgram_sensitivity as sd

    monkeypatch.setattr(module_text, "FILLERS", sd.FILLERS)
    row = {"poc": "1", "clave": "0001-poc1-e1-r1", "scenario": "1", "repetition": "1"}
    reference = "Mmm. Está un poquito bajo, pero puede funcionar. Dieciocho mil."
    hypothesis = "está un poquito bajo pero puede f****** um 18,000"
    r = sd.row_of_the_call(row, reference, hypothesis, hypothesis)
    assert r["masked"] == 1 and r["masked_text"] == "f******"
    # Las tapadas se cuentan en el texto crudo: la hipótesis con la última línea
    # recortada ya viene normalizada, sin el `*`.
    assert sd.row_of_the_call(row, reference, "esta un poquito bajo", hypothesis)["masked"] == 1
    assert json.loads(r["muletillas_reference"]) == {"mmm": 1}
    assert json.loads(r["fillers_hypothesis"]) == {"um": 1}
    # Original: falta "mmm" (borrado), "funcionar"→"f" (sustitución), "um" sobra (inserción).
    assert r["words_errors"] == 3
    # Sin muletillas solo queda la palabra enmascarada.
    assert r["sm_words_errors"] == 1
    assert r["sm_words_reference"] == r["words_reference"] - 1
    assert r["amounts_errors"] == 0 and r["sm_amounts_errors"] == 0


def test_the_distance_to_full_speech_is_measured_from_the_first_audible_segment(tmp_path):
    from shared.audio import write_wav_stereo  # noqa: F401  (solo confirma que existe)
    from validation import speech_onset as av

    sample_rate = 8000
    silence = np.zeros(sample_rate, dtype=np.int16)
    # 100 ms a −40 dBFS (audible para el arnés, lejos de la voz plena) y luego
    # 1 s de voz a −10 dBFS: la voz plena empieza 100 ms después del primer tramo.
    suave = np.full(int(0.1 * sample_rate), int(32768 * 10 ** (-40 / 20)), dtype=np.int16)
    suave[::2] *= -1
    t = np.arange(sample_rate) / sample_rate
    fuerte = (np.sin(2 * np.pi * 200 * t) * 32768 * 10 ** (-7 / 20)).astype(np.int16)
    agent = np.concatenate([fuerte, silence, suave, fuerte, silence])
    streaks = av.harness_streaks(agent, sample_rate)
    assert len(streaks) == 2
    a, b = streaks[1]
    distance, pico = av.distance_to_full_speech(agent[a:b], sample_rate)
    assert distance == pytest.approx(100.0, abs=10.0)
    assert pico == pytest.approx(-7.0, abs=0.1)
