"""El resumen legible a máquina de una llamada del arnés."""

from __future__ import annotations

import json

import numpy as np
import pytest

from shared.carrier_simulator.run import evaluate_run
from shared.carrier_simulator.script import ScriptStep
from shared.carrier_simulator.observed import Observed, ObservedTurn
from shared.carrier_simulator.summary import (
    write_summary,
    quote_summary,
    call_summary,
    run_wer,
)


def _step(turn: str, text: str, *, of_script: bool = True, silence: bool = False):
    return ScriptStep(
        turn=turn,
        text=text,
        samples=None if silence else np.zeros(800, dtype=np.int16),
        clip_id=f"e1_{turn.lower()}",
        of_script=of_script,
        silence_s=2.0 if silence else 0.0,
    )


def _observed(*turns: ObservedTurn) -> Observed:
    obs = Observed(silent=True)
    obs.turns = list(turns)
    return obs


def _turn(step: ScriptStep, *, stt: str, responded: bool = True) -> ObservedTurn:
    return ObservedTurn(
        step=step,
        t_clip_end=1.0,
        t_agent_first_speech=3.0 if responded else None,
        stt=stt,
        response="claro" if responded else "",
    )


# --------------------------------------------------------------------------
# WER
# --------------------------------------------------------------------------


def test_un_stt_perfecto_da_wer_cero():
    step = _step("T1", "quiero hablar sobre un envío")
    block = run_wer(
        _observed(_turn(step, stt="quiero hablar sobre un envío")), 1
    )
    assert block["wer"] == 0.0
    assert block["turns"] == 1


def test_one_misheard_word_out_of_five_gives_twenty_percent():
    step = _step("T1", "quiero hablar sobre un envío")
    block = run_wer(
        _observed(_turn(step, stt="quiero hablar sobre un invento")), 1
    )
    assert block["wer"] == pytest.approx(0.2)


def test_the_turns_that_were_not_reached_do_not_enter():
    """Salieron contra un agente que ya había colgado: su STT vacío hundiría el WER."""
    alive = _turn(_step("T1", "hola que tal"), stt="hola que tal")
    dead = _turn(_step("T2", "y cuanto me paga"), stt="", responded=False)
    block = run_wer(_observed(alive, dead), 1)
    assert block["turns"] == 1
    assert block["wer"] == 0.0


def test_last_transcribed_turn_COUNTS_even_if_the_agent_did_not_answer():
    """El caso que `alcanzados` se comía, y no era simétrico.

    El clip sonó, el STT lo transcribió mal, y el agente ya no contestó. Eso no
    es cola muerta: la transcripción existe y su error es real, y su ausencia
    puede invertir el signo de una brecha de WER chica. Ver
    `docs/METRICS_CONTRACT.md` §2.10.
    """
    alive = _turn(_step("T1", "hola que tal"), stt="hola que tal")
    ultimo = _turn(
        _step("T2", "le bajo a veinticinco mil pero de ahi no me muevo"),
        stt="le bajo a veinticinco mil",
        responded=False,
    )
    block = run_wer(_observed(alive, ultimo), 1)
    assert block["turns"] == 2, "el turno transcrito sin respuesta se perdió"
    # 6 borrados sobre 13 palabras de referencia: el error que se estaba tirando.
    assert block["counts"]["wer"]["deleted"] == 6
    assert block["wer"] == pytest.approx(6 / 13, abs=1e-4)


def test_only_the_tail_is_trimmed_not_a_mute_turn_in_the_middle():
    """Un turno sin STT ni respuesta EN MEDIO es una falla, no cola.

    Control del recorte: sin esto, cortar en el primer turno mudo en vez de
    recortar desde el final pasaría igual.
    """
    alive = _turn(_step("T1", "hola que tal"), stt="hola que tal")
    mute = _turn(_step("T2", "y cuanto me paga"), stt="", responded=False)
    after = _turn(_step("T3", "va pues"), stt="va pues")
    block = run_wer(_observed(alive, mute, after), 1)
    assert block["turns"] == 3
    assert block["counts"]["wer"]["deleted"] == 4  # las de T2, que son reales


def test_a_silence_turn_has_no_line_to_compare():
    spoken = _turn(_step("T1", "hola que tal"), stt="hola que tal")
    silent = _turn(_step("T2", "(silencio)", silence=True), stt="")
    block = run_wer(_observed(spoken, silent), 2)
    assert block["turns"] == 1


def test_a_loose_clip_has_no_reference_and_does_not_enter():
    """Con `--repeat` el texto no es la línea del guion: medir ahí sería ruido."""
    loose = _turn(_step("T1", "(clip suelto)", of_script=False), stt="lo que sea")
    block = run_wer(_observed(loose), 1)
    assert block["turns"] == 0
    assert block["wer"] is None


def test_counts_are_summed_and_rates_are_not_averaged():
    """Un turno de tres palabras no pesa lo mismo que uno de treinta."""
    short = _turn(_step("T1", "sí"), stt="no")  # 1 error sobre 1 palabra
    largo = _turn(
        _step("T2", "uno dos tres cuatro cinco seis siete ocho nueve diez"),
        stt="uno dos tres cuatro cinco seis siete ocho nueve diez",
    )
    block = run_wer(_observed(short, largo), 2)
    # Promediando tasas daría 0.5; sumando conteos, 1 error sobre 11 palabras.
    assert block["wer"] == pytest.approx(1 / 11)


def test_eighteen_thousand_and_18000_are_the_same_word():
    step = _step("T1", "te doy dieciocho mil")
    block = run_wer(_observed(_turn(step, stt="te doy 18,000")), 1)
    assert block["wer"] == 0.0


def test_a_misheard_amount_is_counted_separately():
    step = _step("T1", "te doy dieciocho mil")
    block = run_wer(_observed(_turn(step, stt="te doy dieciséis mil")), 1)
    assert block["amount_error"] == pytest.approx(1.0)
    assert block["counts"]["amounts"]["substitutions"] == 1


def test_without_comparable_turns_the_rates_are_null_and_not_zero():
    """Cero no es lo mismo que "no se pudo medir", y en una tabla se confunden."""
    block = run_wer(_observed(), 0)
    assert block["wer"] is None
    assert block["cer"] is None
    assert block["amount_error"] is None


# --------------------------------------------------------------------------
# El resumen completo
# --------------------------------------------------------------------------


def _call(lines=None):
    obs = _observed(
        _turn(_step("T1", "hola que tal"), stt="hola que tal"),
        _turn(_step("T2", "cuanto me paga"), stt="cuanto me paga"),
    )
    return call_summary(
        obs,
        evaluate_run(obs),
        poc=1,
        scenario=3,
        repetition=2,
        label="arnes-call-x",
        duration_s=118.42,
        cost_usd=0.0831,
        failed_criteria=0,
        lines=lines,
        artifacts={"event_log": "b.json"},
    )


def test_the_summary_carries_the_call_identification():
    summary = _call()
    assert summary["poc"] == 1
    assert summary["scenario"] == 3
    assert summary["repetition"] == 2
    assert summary["label"] == "arnes-call-x"
    assert summary["duration_s"] == 118.4
    assert summary["cost_usd"] == 0.0831


def test_the_p50_and_the_fragmentation_are_promoted_to_the_first_level():
    """El lote los lee sin tener que entrar al bloque de intervenciones."""
    summary = _call(
        {"fragmentation": 1.57, "line_latency_ms": {"n": 7, "p50": 2280.0}}
    )
    assert summary["fragmentation"] == 1.57
    assert summary["line_latency_ms"]["p50"] == 2280.0


def test_without_a_lines_block_the_summary_comes_out_the_same():
    """PoC2 puede quedarse sin el JSON del worker y eso no invalida la llamada."""
    summary = _call(None)
    assert summary["lines"] is None
    assert summary["fragmentation"] is None
    assert summary["line_latency_ms"] is None


def test_without_recording_the_interruptions_are_null_and_not_zero():
    """Cero significa que se midió y no hubo; nulo, que no se grabó nada."""
    assert _call()["agent_interruptions"] is None


def test_the_overlap_promotes_the_interruptions_to_the_first_level():
    obs = _observed(_turn(_step("T1", "hola que tal"), stt="hola que tal"))
    summary = call_summary(
        obs,
        evaluate_run(obs),
        poc=1,
        scenario=1,
        repetition=1,
        label="x",
        duration_s=10.0,
        cost_usd=0.0,
        failed_criteria=0,
        overlap={"the_agent_talked_over": 2, "overlaps": 3},
    )
    assert summary["agent_interruptions"] == 2


def test_without_wav_overlap_is_not_measured(tmp_path):
    from shared.carrier_simulator.summary import measure_overlap

    assert measure_overlap(None) is None
    assert measure_overlap(tmp_path / "does_not_exist.wav") is None


def test_a_wav_that_cannot_be_measured_does_not_bring_down_the_run(tmp_path):
    """Una corrida buena no se pierde porque falle un archivo de diagnóstico."""
    from shared.carrier_simulator.summary import measure_overlap

    roto = tmp_path / "roto.wav"
    roto.write_bytes(b"esto no es un wav")
    assert measure_overlap(roto) is None


def test_the_summary_is_serializable():
    json.dumps(_call({"fragmentation": 1.0}))


def test_the_quote_says_nothing_was_spent():
    quote = quote_summary(
        poc=2,
        scenario=1,
        script=[_step("T1", "hola"), _step("T2", "adiós")],
        seconds=210.0,
        cost_usd=0.1234,
    )
    assert quote["estimate_only"] is True
    assert quote["turns"] == 2
    assert quote["estimated_usd"] == 0.1234
    assert quote["seconds_cap"] == 210.0


# --------------------------------------------------------------------------
# La escritura
# --------------------------------------------------------------------------


def test_the_summary_is_written_and_reread(tmp_path):
    ruta = tmp_path / "sub" / "summary.json"
    write_summary(ruta, {"poc": 1})
    assert json.loads(ruta.read_text(encoding="utf-8")) == {"poc": 1}


def test_writing_it_twice_replaces_it_entirely(tmp_path):
    ruta = tmp_path / "summary.json"
    write_summary(ruta, {"a": "x" * 500})
    write_summary(ruta, {"b": 1})
    assert json.loads(ruta.read_text(encoding="utf-8")) == {"b": 1}


def test_does_not_leave_temp_files_lying_around(tmp_path):
    ruta = tmp_path / "summary.json"
    write_summary(ruta, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]


def test_if_it_cannot_be_serialized_it_leaves_no_half_written_file(tmp_path):
    ruta = tmp_path / "summary.json"
    write_summary(ruta, {"bueno": 1})
    with pytest.raises(TypeError):
        write_summary(ruta, {"malo": object()})
    # El anterior sigue entero y no hay basura al lado.
    assert json.loads(ruta.read_text(encoding="utf-8")) == {"bueno": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]
