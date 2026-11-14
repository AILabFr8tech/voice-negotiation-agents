"""Prueba de `validation.turns_without_latency_poc3` sobre una llamada sintética.

Se arma una llamada con el esquema de la campaña: los clips en el canal 0 del
WAV, el agente en el canal 1, las marcas en época y un puente que da muestra
solo para los turnos lentos. El script tiene que encontrar los rápidos, medirlos
con la regla del arnés y cortar su audio con el fin de voz donde dice.
"""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

pytest.importorskip("scipy")

from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.metrics import external_latency_poc3 as ext3
from validation import turns_without_latency_poc3 as tsl
from validation.synthetic_clips import generate

EPOCH = 1_700_000_000.0
SAMPLE_RATE = TELEPHONY_SAMPLE_RATE


def _call(tmp_path, *, resumes=None, trims_at_end=None):
    clips = tmp_path / "clips"
    manifest = json.loads(generate(clips).read_text(encoding="utf-8"))
    entries = [c for c in manifest["clips"] if c["scenario"] == 1 and c["variant"] == "base"]
    entries = [c for c in entries if c["duration_s"] >= 0.8][:6]
    # Latencia del agente por turno; los de 0.4 s no tienen muestra del puente.
    latencies = [1.5, 0.4, 1.8, 0.4, 1.6, 2.0][: len(entries)]

    chance = np.random.default_rng(3)
    t, marks, samples, blocks = 1.0, [], [], []
    for entry, lat in zip(entries, latencies):
        clip = ext3.ulaw_a_pcm16((clips / f"{entry['id']}.ulaw").read_bytes())
        start = t
        end_of_speech = start + float(entry["end_of_speech_s"])
        onset = end_of_speech + lat
        blocks.append((0, start, clip))
        blocks.append((1, onset, (chance.normal(0, 3000, int(1.5 * SAMPLE_RATE))).astype(np.int16)))
        if resumes is not None and entry is entries[1]:
            # El agente retoma la palabra justo cuando arranca la línea, tras una
            # pausa de `retoma` s desde el final de su respuesta anterior.
            blocks.append((1, start - 0.1, (chance.normal(0, 3000, int(0.8 * SAMPLE_RATE))).astype(np.int16)))
        marks.append({
            "turn": entry["turn"], "clip_id": entry["id"],
            "start_of_clip": start + EPOCH, "measured_end_of_speech": end_of_speech + EPOCH,
        })
        if lat > 1.0:
            samples.append({
                "t_zero_wall": end_of_speech + 0.1 + EPOCH,
                "t_ack_wall": onset + EPOCH,
                "full_turn_ms": 1000.0,
            })
        t = onset + 3.0
        if resumes is not None and entry is entries[0]:
            t = onset + 1.5 + resumes + 0.1

    largo = t + 1.0 if trims_at_end is None else marks[-1]["measured_end_of_speech"] - EPOCH + trims_at_end
    audio = np.zeros((int(largo * SAMPLE_RATE), 2), dtype=np.int16)
    for canal, start, pcm in blocks:
        i = int(round(start * SAMPLE_RATE))
        pcm = pcm[: max(0, len(audio) - i)]
        audio[i:i + len(pcm), canal] = pcm

    wav = tmp_path / "auditoria_X.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    zeros = tmp_path / "zeros_X.jsonl"
    zeros.write_text(
        "\n".join(json.dumps({"tipo": "mark", "mark": m}) for m in marks), encoding="utf-8"
    )
    live_bridge = tmp_path / "call_X.json"
    live_bridge.write_text(json.dumps({
        "clock": {"interpretable": True},
        "gpt_live_extra": {"wall_axis": {"samples": samples}},
    }), encoding="utf-8")
    return clips, zeros, live_bridge, wav, marks, latencies


def test_finds_the_fast_turns_and_measures_them_with_the_harness_rule(tmp_path):
    clips, zeros, live_bridge, wav, marks, latencies = _call(tmp_path)
    turns, report = tsl.lost_turns(
        call_sid="X", scenario=1, zeros=zeros, live_bridge=live_bridge, wav=wav,
        event_log=None, clips=clips,
    )
    fast = [m["turn"] for m, lat in zip(marks, latencies) if lat < 1.0]
    assert [t["turn"] for t in turns] == fast
    assert report["with_number"] == len(marks) - len(fast)
    for t in turns:
        # 400 ms reales, cuantizados a la ventana de 20 ms.
        assert t["harness_rule_latency_ms"] == pytest.approx(400, abs=25)
        assert t["first_sound_latency_ms"] == pytest.approx(400, abs=25)
        assert t["agent_audible_at_end_of_speech"] == 0
        assert t["motivo"].startswith("el puente no dio muestra")
        mark = next(m for m in marks if m["turn"] == t["turn"])
        voice = mark["measured_end_of_speech"] - mark["start_of_clip"]
        assert t["end_of_speech_second_in_audio"] == pytest.approx(
            max(tsl.ANTES_SEG, voice + tsl.BEFORE_CLIP_S), abs=0.01
        )
        assert t["agent_was_speaking_at_clip_start"] == 0
        assert t["last_turn"] == 0


def test_the_cut_leaves_the_end_of_speech_at_its_second(tmp_path):
    clips, zeros, live_bridge, wav, *_ = _call(tmp_path)
    turns, _ = tsl.lost_turns(
        call_sid="X", scenario=1, zeros=zeros, live_bridge=live_bridge, wav=wav,
        event_log=None, clips=clips,
    )
    audio, sample_rate = ext3.leer_wav(wav)
    t = turns[0]
    since, until = t["cutoff"]
    cutoff = audio[since:until]
    eos = int(t["end_of_speech_second_in_audio"] * sample_rate)
    onset = eos + int(0.4 * sample_rate)
    # Antes del arranque el canal del agente está en silencio; después, no.
    assert np.abs(cutoff[eos:onset - 200, 1]).max() == 0
    assert np.abs(cutoff[onset + 200:onset + 2000, 1]).max() > 0


def test_the_cut_reaches_the_following_line(tmp_path):
    clips, zeros, live_bridge, wav, marks, _ = _call(tmp_path)
    turns, _ = tsl.lost_turns(
        call_sid="X", scenario=1, zeros=zeros, live_bridge=live_bridge, wav=wav,
        event_log=None, clips=clips,
    )
    audio, sample_rate = ext3.leer_wav(wav)
    t = turns[0]
    k = next(i for i, m in enumerate(marks) if m["turn"] == t["turn"])
    since, until = t["cutoff"]
    end_of_speech = marks[k]["measured_end_of_speech"]
    following = marks[k + 1]["start_of_clip"]
    duration = (until - since) / sample_rate
    expected = t["end_of_speech_second_in_audio"] + (following - end_of_speech) + tsl.AFTER_THE_NEXT_S
    assert duration == pytest.approx(expected, abs=0.05)


def test_marks_when_the_agent_resumes_at_the_start_of_the_line(tmp_path):
    clips, zeros, live_bridge, wav, marks, _ = _call(tmp_path, resumes=2.2)
    turns, _ = tsl.lost_turns(
        call_sid="X", scenario=1, zeros=zeros, live_bridge=live_bridge, wav=wav,
        event_log=None, clips=clips,
    )
    t = next(t for t in turns if t["turn"] == marks[1]["turn"])
    assert t["agent_was_speaking_at_clip_start"] == 1
    assert t["agent_pause_before_clip_ms"] == pytest.approx(2200, abs=50)


def test_last_turn_without_speech_is_the_call_closing(tmp_path):
    clips, zeros, live_bridge, wav, marks, _ = _call(tmp_path, trims_at_end=0.6)
    turns, _ = tsl.lost_turns(
        call_sid="X", scenario=1, zeros=zeros, live_bridge=live_bridge, wav=wav,
        event_log=None, clips=clips,
    )
    t = turns[-1]
    assert t["turn"] == marks[-1]["turn"]
    assert t["last_turn"] == 1
    assert t["first_sound_latency_ms"] is None
    assert t["motivo"].startswith("último turno")
    assert t["recording_seconds_after_end_of_speech"] == pytest.approx(0.6, abs=0.01)
