"""The export for the report, on a synthetic campaign with one pair and a broken PoC3."""

from __future__ import annotations

import csv
import datetime as dt
import json

from shared.metrics import export_for_report as ex


def _campana(S):
    lote = S/"lote"; c = lote/"campana200"
    (c/"summaries").mkdir(parents=True); (c/"calls").mkdir()
    base = 1_758_400_000.0
    # The real shape of `por_intervencion` in PoC1 and PoC2 (`marcas.bloque_de_intervenciones`):
    # without `latencia_ms` or `fragmentacion`, which are the keys of the shared join.
    def inter(t, end, voice, lat, inbound): return {"turn":t,"clip_id":f"e1_{t.lower()}","end_of_speech":end,"agent_first_speech":voice,"line_latency_ms":lat,"speech_bias_ms":inbound,"pipeline_events":1}
    progress=[]
    for poc, clave, off in ((1,"0001-poc1-e1-r1",0.0),(2,"0002-poc2-e1-r1",100.0)):
        b = base+off
        ints = [inter("T1", b+1.0, b+3.5, 2500.0, 140.0), inter("T2", b+10.0, b+12.0, 2000.0, 150.0)]
        if poc == 1:
            f = c/f"cero_{clave}.json"
            f.write_text(json.dumps({"audio_outputs":[b+3.497, b+11.998], "lines":{"per_line":ints}}))
            art = {"zero_crossing": str(f), "event_log": "x"}
            # function call in the middle of T1, in local time UTC-6
            loc = dt.datetime.fromtimestamp(b+2.0, dt.timezone.utc) - dt.timedelta(hours=6)
            (c/"calls"/f"{clave}.log").write_text(loc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " | DEBUG | x - OpenAILLMService#0 Calling function [proponer_precio:call_1]\n")
        else:
            f = c/f"call_{clave}.json"
            f.write_text(json.dumps({"livekit_extra":{"per_turn":[{"assistant":{"started_speaking_at":b+3.3}},{"assistant":{"started_speaking_at":b+11.8}}],"harness":{"lines":{"per_line":ints}}}}))
            art = {"worker_json": str(f), "event_log": "x"}
        (c/"summaries"/f"{clave}.json").write_text(json.dumps({"duration_s":90,"lines":{"per_line":ints,"heard_lines":2,"with_latency":2},"fragmentation":1.0,"failed_criteria":0,"artifacts":art}))
        progress.append({"indice":int(clave[:4]),"clave":clave,"poc":poc,"scenario":1,"repetition":1,"estado":"ok","label":clave})
    (c/"progress.jsonl").write_text("\n".join(json.dumps(p) for p in progress))
    # worker.log: local line followed by its JSON in UTC; and a PoC2 tool inside T2
    t = base+100.0+11.0
    loc = dt.datetime.fromtimestamp(t, dt.timezone.utc)-dt.timedelta(hours=6)
    utc = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    (c/"worker.log").write_text(loc.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]+" INFO tool\n"+'{"message": "tool proponer_precio argumentos={} -> ok", "timestamp": "'+utc+'+00:00"}\n')
    p3 = lote/"campana_poc3_pcmu"; (p3/"summaries").mkdir(parents=True)
    (p3/"progress.jsonl").write_text(json.dumps({"indice":1,"clave":"0001-poc3-e1-r1","poc":3,"scenario":1,"repetition":1,"estado":"ok","label":"SID1"}))
    (p3/"summaries"/"0001-poc3-e1-r1.json").write_text(json.dumps({"artifacts":{}}))
    return lote


def test_exports_latencies_transport_and_round_trips(tmp_path):
    lote = _campana(tmp_path)
    ex.main(["--lote", str(lote), "--output", str(tmp_path / "out")])
    rows = list(csv.DictReader((tmp_path / "out" / "lines.csv").open(encoding="utf-8")))
    by = {(f["poc"], f["turn"]): f for f in rows}
    # The PoC1 tool falls in T1 (local time corrected with the worker.log offset).
    assert by[("1", "T1")]["with_round_trip"] == "1"
    assert by[("1", "T2")]["with_round_trip"] == "0"
    # The PoC2 one falls in T2 and is attributed by window, from the single worker.log.
    assert by[("2", "T1")]["with_round_trip"] == "0"
    assert by[("2", "T2")]["with_round_trip"] == "1"
    # Corrected = observed − inbound − outbound, per line.
    assert float(by[("1", "T1")]["outbound_ms"]) == 3.0
    assert float(by[("1", "T1")]["corrected_latency_ms"]) == 2357.0
    assert float(by[("2", "T1")]["outbound_ms"]) == 200.0


def test_a_poc3_run_without_artifacts_is_skipped_with_reason(tmp_path):
    lote = _campana(tmp_path)
    ex.main(["--lote", str(lote), "--output", str(tmp_path / "out")])
    calls = list(csv.DictReader((tmp_path / "out" / "calls.csv").open(encoding="utf-8")))
    poc3 = [l for l in calls if l["poc"] == "3"]
    assert len(poc3) == 1 and poc3[0]["reason_if_omitted"].startswith("missing_items")
    assert "NO CUADRA" in (tmp_path / "out" / "verification.txt").read_text(encoding="utf-8")



def test_the_first_sound_does_not_filter_the_short_filler(tmp_path):
    """A 0.2 s "mm" at 0.8 s counts as first sound; the response at 2.0 s does not.

    Four turns because the anchor requires at least four clips that agree.
    """
    import wave

    import numpy as np

    from shared.metrics import external_latency_poc3 as ext3

    sample_rate = 8000
    rng = np.random.default_rng(7)
    clips = tmp_path / "clips"
    clips.mkdir()
    wav_epoch = 1_758_400_000.0
    total = int(sample_rate * 30)
    canal0 = np.zeros(total, dtype=np.int16)
    canal1 = np.zeros(total, dtype=np.int16)
    tone = (8000 * np.sin(2 * np.pi * 300 * np.arange(sample_rate) / sample_rate)).astype(np.int16)
    marks = []
    for k in range(4):
        clip_id = f"e1_t{k + 1}"
        raw_value = rng.integers(0, 256, size=sample_rate * 2, dtype=np.uint8).tobytes()
        (clips / f"{clip_id}.ulaw").write_bytes(raw_value)
        clip = ext3.ulaw_a_pcm16(raw_value)
        start = 1.0 + 7.0 * k
        end = start + 2.0
        canal0[int(start * sample_rate):int(start * sample_rate) + clip.size] = clip
        canal1[int((end + 0.8) * sample_rate):int((end + 1.0) * sample_rate)] = tone[: int(0.2 * sample_rate)]
        canal1[int((end + 2.0) * sample_rate):int((end + 3.0) * sample_rate)] = tone[:sample_rate]
        marks.append({"tipo": "mark", "mark": {
            "turn": f"T{k + 1}", "clip_id": clip_id,
            "start_of_clip": wav_epoch + start,
            "measured_end_of_speech": wav_epoch + end,
        }})

    wav = tmp_path / "auditoria.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.column_stack([canal0, canal1]).astype(np.int16).tobytes())
    zeros = tmp_path / "zeros.jsonl"
    zeros.write_text("\n".join(json.dumps(m) for m in marks) + "\n", encoding="utf-8")

    primer = ex.first_sound_per_turn(ex._marks_of(zeros), wav, clips)
    assert set(primer) == {"T1", "T2", "T3", "T4"}
    for ms in primer.values():
        assert abs(ms - 800.0) <= 40.0


def test_the_first_sound_never_goes_past_the_response():
    assert ex._no_mayor(2100.0, 2000.0) == 2000.0
    assert ex._no_mayor(800.0, 2000.0) == 800.0
    assert ex._no_mayor(None, 2000.0) is None


def test_poc2_rebuilds_the_marks_and_downsamples_the_wav_to_8khz(tmp_path):
    """PoC2 does not write `ceros_*.jsonl` and its WAV may come at another sample rate."""
    import wave

    import numpy as np
    from scipy.signal import resample_poly

    from shared.metrics import external_latency_poc3 as ext3

    sample_rate, wav_sample_rate = 8000, 48000
    rng = np.random.default_rng(11)
    clips = tmp_path / "clips"
    clips.mkdir()
    epoch = 1_758_500_000.0
    total = sample_rate * 30
    canal0 = np.zeros(total)
    canal1 = np.zeros(total)
    manifest, block = [], []
    for k in range(4):
        clip_id = f"e2_t{k + 1}"
        raw_value = rng.integers(0, 256, size=sample_rate * 2, dtype=np.uint8).tobytes()
        (clips / f"{clip_id}.ulaw").write_bytes(raw_value)
        clip = ext3.ulaw_a_pcm16(raw_value).astype(np.float64)
        start = 1.0 + 7.0 * k
        canal0[int(start * sample_rate):int(start * sample_rate) + clip.size] = clip
        end = start + 1.9  # speech ends 0.1 s before the end of the file
        tone = 8000 * np.sin(2 * np.pi * 300 * np.arange(int(0.2 * sample_rate)) / sample_rate)
        canal1[int((end + 0.5) * sample_rate):int((end + 0.5) * sample_rate) + tone.size] = tone
        manifest.append({"id": clip_id, "end_of_speech_s": 1.9})
        block.append({"turn": f"T{k + 1}", "clip_id": clip_id, "end_of_speech": epoch + end})
    (clips / "manifest.json").write_text(json.dumps({"clips": manifest}), encoding="utf-8")

    arriba = np.column_stack([resample_poly(c, wav_sample_rate // sample_rate, 1) for c in (canal0, canal1)])
    wav = tmp_path / "auditoria.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(wav_sample_rate)
        w.writeframes(np.clip(arriba, -32768, 32767).astype(np.int16).tobytes())

    marks = ex.marks_from_lines(block, clips)
    assert [m["turn"] for m in marks] == ["T1", "T2", "T3", "T4"]
    primer = ex.first_sound_per_turn(marks, wav, clips)
    assert set(primer) == {"T1", "T2", "T3", "T4"}
    for ms in primer.values():
        assert abs(ms - 500.0) <= 40.0


def test_the_stages_only_count_samples_that_ended_in_a_response(tmp_path):
    ruta = tmp_path / "call.json"
    ruta.write_text(json.dumps({"latencies": [
        {"indice": 0, "is_opening": True, "full_turn_ms": 1500.0},
        {"indice": 1, "full_turn_ms": 2400.0, "end_of_turn_ms": 430.0,
         "stt_ms": 380.0, "llm_ttft_ms": 690.0, "tts_ttfb_ms": 300.0},
        # absorbed fragment: the framework closed the turn but did not answer
        {"indice": 2, "full_turn_ms": None, "end_of_turn_ms": 420.0, "stt_ms": 350.0},
    ]}), encoding="utf-8")
    rows = ex.call_stages(ruta, {"poc": 1, "clave": "c", "scenario": 1, "repetition": 1})
    assert [f["indice"] for f in rows] == [1]
    assert rows[0]["llm_ttft_ms"] == 690.0 and rows[0]["end_of_turn_ms"] == 430.0


def test_the_tool_result_after_starting_to_speak():
    from types import SimpleNamespace

    marks = [
        {"turn": "T1", "start_of_clip": 98.0, "measured_end_of_speech": 100.0},
        {"turn": "T2", "start_of_clip": 110.0, "measured_end_of_speech": 112.0},
        {"turn": "T3", "start_of_clip": 120.0, "measured_end_of_speech": 122.0},
    ]
    turns = [
        SimpleNamespace(turn="T1", end_of_speech=100.0, first_speech_heard=102.0),
        SimpleNamespace(turn="T2", end_of_speech=112.0, first_speech_heard=114.0),
        SimpleNamespace(turn="T3", end_of_speech=122.0, first_speech_heard=124.0),
    ]
    # T1: the result arrives 0.5 s after the agent started speaking.
    # T2: it arrives before the agent speaks. T3: none in its window.
    output = ex.results_per_turn(turns, marks, [102.5, 113.0])
    assert output["T1"] == (1, 1, 500.0)
    assert output["T2"] == (1, 0, 0.0)
    assert output["T3"] == (0, None, None)


def test_recovers_the_turns_the_external_instrument_leaves_without_a_number(tmp_path):
    """Filler discarded live and content 2.5 s later; fast response; noise only.

    T1: a 0.8 s "mm, déjame ver" at +2.0 s (the harness discarded it live as a
    backchannel) and the content at +4.5 s. T2: a 1 s response at +0.4 s.
    T3 already has a number. T4: only 0.2 s of noise, no content: not recovered.
    """
    import wave

    import numpy as np

    from shared.metrics import external_latency_poc3 as ext3

    sample_rate = 8000
    rng = np.random.default_rng(11)
    clips = tmp_path / "clips"
    clips.mkdir()
    epoch = 1_758_400_000.0
    canal0 = np.zeros(sample_rate * 45, dtype=np.int16)
    canal1 = np.zeros(sample_rate * 45, dtype=np.int16)
    tone = (8000 * np.sin(2 * np.pi * 300 * np.arange(sample_rate * 2) / sample_rate)).astype(np.int16)

    def voice(since: float, dura: float) -> None:
        canal1[int(since * sample_rate):int((since + dura) * sample_rate)] = tone[: int(dura * sample_rate)]

    marks, ends = [], []
    for k in range(4):
        clip_id = f"e1_t{k + 1}"
        raw_value = rng.integers(0, 256, size=sample_rate * 2, dtype=np.uint8).tobytes()
        (clips / f"{clip_id}.ulaw").write_bytes(raw_value)
        clip = ext3.ulaw_a_pcm16(raw_value)
        start = 1.0 + 10.0 * k
        canal0[int(start * sample_rate):int(start * sample_rate) + clip.size] = clip
        ends.append(start + 2.0)
        marks.append({
            "turn": f"T{k + 1}", "clip_id": clip_id,
            "start_of_clip": epoch + start, "measured_end_of_speech": epoch + start + 2.0,
        })
    voice(ends[0] + 2.0, 0.8)
    voice(ends[0] + 4.5, 1.0)
    voice(ends[1] + 0.4, 1.0)
    voice(ends[2] + 1.5, 1.0)
    voice(ends[3] + 1.0, 0.2)

    wav = tmp_path / "auditoria.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.column_stack([canal0, canal1]).astype(np.int16).tobytes())
    event_log = tmp_path / "event_log.json"
    event_log.write_text(json.dumps({"turns": [
        {"turn": "T1", "discarded_fragments": [
            {"start": ends[0] + 2.0, "duration_ms": 780.0, "motivo": "apoyo verbal", "text": "mm, déjame ver"},
        ]},
    ]}), encoding="utf-8")

    recovered_items = ex.recovered_turns(marks, wav, clips, measured_items={"T3"}, event_log=event_log)
    by = {t.turn: t for t in recovered_items}
    assert set(by) == {"T1", "T2"}
    assert abs(by["T1"].first_sound_latency_ms - 2000.0) <= 40.0
    assert abs(by["T1"].line_latency_ms - 4500.0) <= 40.0
    assert abs(by["T2"].first_sound_latency_ms - 400.0) <= 40.0
    assert abs(by["T2"].line_latency_ms - 400.0) <= 40.0


def test_without_a_response_mid_call_the_closing_does_not_count():
    turns = ["T1", "T2", "T3", "T4", "T5"]
    # T2 mute with the agent answering afterwards; T4 and T5 are the closing.
    assert ex.no_response_midcall(turns, {"T1", "T3"}) == 1
    assert ex.no_response_midcall(turns, set()) == 0
    assert ex.no_response_midcall(turns, set(turns)) == 0


def test_the_livekit_closes_count_each_confirmed_turn_in_its_line(tmp_path):
    # Two lines: in the first, LiveKit confirmed two user turns before
    # answering (one extra end of turn); in the second, one.
    marks = [
        {"turn": "T1", "start_of_clip": 10.0, "measured_end_of_speech": 13.0},
        {"turn": "T2", "start_of_clip": 20.0, "measured_end_of_speech": 22.0},
    ]
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"livekit_extra": {"per_turn": [
        {"is_opening": True, "user_fragments": [{"stopped_speaking_at": 1.0}]},
        {"user_fragments": [{"stopped_speaking_at": 11.5}, {"stopped_speaking_at": 13.3}]},
        {"user_fragments": [{"stopped_speaking_at": 22.4}]},
    ]}}))
    assert ex.livekit_closures(w, marks) == {"T1": 2, "T2": 1}
    assert ex.next_clip(marks) == {"T1": 20.0, "T2": float("inf")}


def test_the_excluded_are_the_first_180_outside_the_pairs():
    rows = [
        {"indice": 1, "clave": "a", "poc": 1, "scenario": 1, "repetition": 1, "estado": "ok"},
        {"indice": 2, "clave": "b", "poc": 2, "scenario": 1, "repetition": 1, "estado": "ok"},
        {"indice": 3, "clave": "c", "poc": 1, "scenario": 2, "repetition": 1, "estado": "ok"},
        {"indice": 4, "clave": "d", "poc": 2, "scenario": 2, "repetition": 1, "estado": "roja"},
        {"indice": 181, "clave": "e", "poc": 1, "scenario": 1, "repetition": 19, "estado": "roja"},
    ]
    # `c` drops out by the pair rule and `d` for failing; 181 is no longer part of the campaign.
    assert [f["clave"] for f in ex.excluded_from_campaign200(rows)] == ["c", "d"]


def test_the_poc3_result_knows_which_function_it_is_from(tmp_path):
    # The result is linked to its call by `call_id`; only those from functions
    # whose data the response needs count toward "up to the result".
    from shared.metrics import round_trips as rt

    census = tmp_path / "census.jsonl"
    events = [
        {"type": "response.output_item.done", "t_local_seg": 1.0,
         "event": {"item": {"type": "function_call", "call_id": "a", "name": "proponer_precio"}}},
        {"type": "delegation.function_call_output.created", "t_local_seg": 1.5,
         "event": {"item": {"type": "function_call_output", "call_id": "a"}}},
        {"type": "response.output_item.done", "t_local_seg": 5.0,
         "event": {"item": {"type": "function_call", "call_id": "b", "name": "registrar_acuerdo"}}},
        {"type": "delegation.function_call_output.created", "t_local_seg": 5.4,
         "event": {"item": {"type": "function_call_output", "call_id": "b"}}},
    ]
    census.write_text("\n".join(json.dumps(e) for e in events))
    results = [e for e in rt.poc3_calls(census, 100.0) if e.is_result]
    assert [(e.when, e.function) for e in results] == [(101.5, "proponer_precio"), (105.4, "registrar_acuerdo")]
    assert [e.function in rt.FUNCTIONS_WITH_DATA for e in results] == [True, False]
