"""Pruebas de las piezas de la validación de interrupciones por los caminos de PoC3 y PoC2.

La corrida completa va en tiempo real y no se prueba aquí. Se prueba lo que
decide la verdad: el plan que recibe el modelo falso, la posición de cada
muestra que reporta el reconocedor, y dónde cae el último tramo con voz.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation import recognizer as rc
from validation.fake_model_poc3_interruption import SAMPLES_PER_SEGMENT, SAMPLE_RATE, ModelThatStops
from validation.validate_interruptions_poc3 import plan_per_clip


def _step(turn: str, clip_id: str, variants: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        turn=turn, clip_id=clip_id, is_silence=False,
        variants={v: SimpleNamespace(clip_id=v) for v in variants},
    )


def test_the_plan_covers_all_the_branches_of_the_turn():
    script = [_step("T1", "e1_t1"), _step("T2", "e1_t2"), _step("T3", "e1_t3", ("e1_t3a", "e1_t3b"))]
    plan = {"T2": ("apoyo", "apo_01"), "T3": ("interrupcion", "int_01")}
    assert plan_per_clip(script, plan) == {
        "e1_t2": "apoyo", "e1_t3": "interrupcion", "e1_t3a": "interrupcion", "e1_t3b": "interrupcion",
    }


def test_the_recognizer_reports_the_sample_where_the_clip_started():
    chance = np.random.default_rng(3)
    clip = (chance.normal(0, 1, SAMPLE_RATE) * 4000).astype(np.int16)
    template = rc.Template(
        clip_id="c", text="c", start=clip[: int(rc.TEMPLATE_LENGTH_S * SAMPLE_RATE)].astype(np.float64),
        end_of_speech=int(0.8 * SAMPLE_RATE), largo=len(clip),
    )
    ear = rc.Recognizer([template], SAMPLE_RATE, min_correlation=0.9, min_advantage=0.0)
    antes = 7_321
    sig = np.concatenate([np.zeros(antes, np.int16), clip, np.zeros(SAMPLE_RATE, np.int16)])
    end = None
    for k, i in enumerate(range(0, len(sig), 480)):
        r = ear.receive(sig[i:i + 480], arrival=k * 0.02)
        if isinstance(r, rc.EndOfSpeech):
            end = r
    assert end is not None and end.start_sample == antes


def test_last_segment_with_speech_is_dated_in_playback_time():
    # Dos deltas de 100 ms: el segundo trae voz solo en sus primeros 40 ms.
    pcm = np.zeros(2 * 2400, np.int16)
    pcm[:2400] = 3000
    pcm[2400:2400 + 2 * SAMPLES_PER_SEGMENT] = 3000
    sends = [10.0, 10.1]
    segments = ModelThatStops._speech_segments(pcm, sends)
    assert segments[0] == 10.0
    # El último tramo con voz empieza 20 ms después de que salió el segundo delta.
    assert abs(segments[-1] - 10.12) < 1e-9


def test_for_livekit_the_cut_trims_the_response_and_dates_its_last_segment():
    from validation.fake_agent_livekit import SAMPLES_PER_FRAME as SEGMENT, Scheduled
    from validation.fake_agent_livekit_interruption import cut, speech_segments

    r = Scheduled(np.full(10 * SEGMENT, 3000, np.int16), 1000, 1000, None, {})
    cut(r, 1000 + 3 * SEGMENT)
    assert len(r.pcm) == 3 * SEGMENT
    # Tres tramos con voz; el último empieza dos tramos después del inicio.
    assert speech_segments(r) == [1000, 1000 + SEGMENT, 1000 + 2 * SEGMENT]
    # Un corte anterior al inicio deja la respuesta vacía, sin tramos.
    cut(r, 500)
    assert speech_segments(r) == []


def test_the_audio_summary_counts_detections_and_false_ones():
    from validation.validate_interruptions_poc3 import audio_text

    rows = [
        {"target_stop_ms": 200.0, "fired": 1, "tipo": "interrupcion", "audio_stopped": 1,
         "audio_stop_error_ms": 10.0, "error_replica_audio_ms": -5.0},
        {"target_stop_ms": 200.0, "fired": 1, "tipo": "control", "audio_stopped": 0,
         "audio_stop_error_ms": None, "error_replica_audio_ms": None},
    ]
    text = audio_text(rows)
    assert "1/1" in text and "0/1" in text and "10.0 [10.0, 10.0]" in text
