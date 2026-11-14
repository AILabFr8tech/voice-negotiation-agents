"""The report's whole-call WER."""

from __future__ import annotations

from shared.metrics import export_wer as ew


def _log(turns):
    return {"scenario": 2, "turns": turns}


def test_counts_the_whole_call_without_pairing_turns():
    event_log = _log([
        {"turn": "T1", "clip_id": "e1_t1", "played_line": "Sí, soy Carlos. Dígame.",
         "stt": "basura de la bitácora que no se usa", "agent": "Hola"},
        {"turn": "T2", "clip_id": "e1_t2", "played_line": "Yo ando pensando en dieciocho mil.",
         "stt": "x", "agent": "Entiendo"},
    ])
    # The recognizer split T2 into two pieces and misheard the amount.
    call = {"transcript": [
        {"role": "agent", "text": "Hola, ¿hablo con Carlos?"},
        {"role": "carrier", "text": "Sí, soy Carlos. Dígame."},
        {"role": "agent", "text": "Le ofrezco dieciséis mil."},
        {"role": "carrier", "text": "Yo ando pensando"},
        {"role": "carrier", "text": "en diecisiete mil."},
    ]}
    c = ew.call_counts(event_log, call)
    assert c["turns_reference"] == 2
    assert c["words_s"] == 1 and c["words_d"] == 0 and c["words_i"] == 0
    assert c["amounts_reference"] == 1 and c["amounts_s"] == 1


def test_the_dead_tail_does_not_count_as_deletion():
    event_log = _log([
        {"turn": "T1", "clip_id": "e1_t1", "played_line": "Sí, soy Carlos.", "stt": "sí soy carlos", "agent": "Hola"},
        {"turn": "T8", "clip_id": "e1_t8", "played_line": "Sí, así quedamos. Confirmado.", "stt": "", "agent": ""},
    ])
    call = {"transcript": [{"role": "carrier", "text": "Sí, soy Carlos."}]}
    c = ew.call_counts(event_log, call)
    assert c["dead_tail_turns"] == 1
    assert c["words_d"] == 0 and c["words_reference"] == 3


def test_uses_the_corrected_reference_of_the_audio():
    """`e2_t3` lost "normalmente" in the trim: the reference does not require it."""
    event_log = _log([
        {"turn": "T3", "clip_id": "e2_t3",
         "played_line": "¿A cómo andan pagando esa ruta ustedes normalmente?",
         "stt": "a", "agent": "b"},
    ])
    call = {"transcript": [{"role": "carrier", "text": "¿A cómo andan pagando esa ruta ustedes?"}]}
    c = ew.call_counts(event_log, call)
    assert c["words_d"] == 0


def _call_cut():
    event_log = _log([
        {"turn": "T1", "clip_id": "e5_t1", "played_line": "Sí, soy Carlos.", "stt": "x", "agent": "Hola"},
        {"turn": "T7", "clip_id": "e5_t7", "played_line": "Órale, pues vaya que sí. Quedamos en 18,500.",
         "stt": "x", "agent": "Perfecto"},
    ])
    call = {"transcript": [
        {"role": "carrier", "text": "Sí, soy Carlos."},
        {"role": "carrier", "text": "Órale, pues vaya que sí. Quedamos en dieciocho mil quin"},
    ]}
    return event_log, call


def test_the_last_line_cut_by_the_closing_does_not_count():
    """With less than 4 s between the end of the line and the end of the call, the line is dropped."""
    event_log, call = _call_cut()
    c = ew.call_counts(event_log, call, slack=1.2)
    assert c["last_line_outside"] == 1
    assert c["amounts_reference"] == 0 and c["amounts_i"] == 0
    assert c["words_s"] + c["words_d"] + c["words_i"] == 0


def test_with_slack_the_last_line_counts():
    event_log, call = _call_cut()
    c = ew.call_counts(event_log, call, slack=6.0)
    assert c["last_line_outside"] == 0
    assert c["amounts_reference"] == 1 and c["amounts_s"] == 1


def test_trim_tail_removes_only_the_piece_of_the_line():
    assert ew.trim_tail("Va. En dieciocho mil quedamos Sale, quedamos en die",
                            "Sale, quedamos en dieciocho mil. Confirmado.") == "va en 18000 quedamos"
    assert ew.trim_tail("es lo máximo que puede pagar Bueno",
                            "Bueno… le bajo a veinticinco mil, pero de ahí…") == "es lo maximo que puede pagar"
    assert ew.trim_tail("ya con la tara no me sale", "Órale, pues vaya que sí.") == "ya con la tara no me sale"


def test_the_census_does_not_split_words_that_arrive_in_two_fragments(tmp_path):
    import json

    rows = [
        {"type": "session.input_transcript.delta", "event": {"delta": " Sí,"}},
        {"type": "session.output_transcript.delta", "event": {"delta": " este"}},
        {"type": "session.input_transcript.delta", "event": {"delta": " dé"}},
        {"type": "session.input_transcript.delta", "event": {"delta": "jame"}},
        {"type": "session.input_transcript.delta", "event": {"delta": " apuntar"}},
    ]
    census = tmp_path / "census.jsonl"
    census.write_text("\n".join(json.dumps(f) for f in rows) + "\n", encoding="utf-8")
    assert ew.census_transcript(census) == "Sí, déjame apuntar"


def test_census_spaces_joins_the_split_and_cuts_where_the_json_ends():
    assert ew.census_spacing("Sí, dé jame apuntar quin", "Sí, déjame apuntar quinientos pesos") == "Sí, déjame apuntar quin"
    assert ew.census_spacing("otra cosa", "Sí, déjame") == "otra cosa"


def test_trim_tail_prefers_the_segment_that_looks_most_alike():
    h = "orale pues vaya que si quedamos en 18500 si asi quedamos confirmado mandeme la"
    linea = "Sí, así quedamos, confirmado. Mándeme la carta porte al correo."
    assert ew.trim_tail(h, linea) == "orale pues vaya que si quedamos en 18500"

