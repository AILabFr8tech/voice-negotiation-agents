"""The return path: from the audio that leaves the pipeline to the audio the harness hears."""

from __future__ import annotations

import json

import pytest

from shared.metrics.return_path import (
    PAIRING_WINDOW_S,
    Outbound,
    compare,
    of_poc1,
    of_poc2,
    pair_outputs,
    leer,
    summarize,
    latency_table,
)


def _line(
    turn: str,
    first_speech: float | None,
    latency_ms: float | None = None,
    ida_ms: float | None = None,
) -> dict:
    return {
        "turn": turn,
        "agent_first_speech": first_speech,
        "line_latency_ms": latency_ms,
        "speech_bias_ms": ida_ms,
    }


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def test_the_outbound_is_how_long_it_took_to_hear_what_had_already_gone_out():
    outbounds, unpaired = pair_outputs([_line("T1", 100.25)], [100.0])
    assert unpaired == []
    assert outbounds[0].ms == pytest.approx(250.0)


def test_a_later_output_belongs_to_the_following_response():
    """Audio cannot be heard before it has gone out."""
    outbounds, _ = pair_outputs([_line("T1", 100.25)], [100.0, 105.0])
    assert outbounds[0].left == 100.0


def test_the_latest_output_that_is_not_later_wins():
    """With several inside the window, the response is the last one, not the first."""
    outbounds, _ = pair_outputs([_line("T2", 200.1)], [198.0, 199.9])
    assert outbounds[0].left == 199.9


def test_an_output_that_is_too_old_is_not_paired():
    """Without a window, an unanswered line would steal another line's output."""
    outbounds, unpaired = pair_outputs(
        [_line("T3", 300.0)], [300.0 - PAIRING_WINDOW_S - 1.0]
    )
    assert outbounds == []
    assert unpaired == ["T3"]


def test_a_line_nobody_answered_neither_contributes_nor_is_missing():
    """Without first speech there is no right endpoint: it is not a pairing failure."""
    outbounds, unpaired = pair_outputs([_line("T8", None)], [100.0])
    assert outbounds == []
    assert unpaired == []


def test_without_outputs_all_are_left_unpaired():
    outbounds, unpaired = pair_outputs([_line("T1", 100.0)], [])
    assert outbounds == []
    assert unpaired == ["T1"]


def test_two_lines_each_take_their_own_output():
    outbounds, _ = pair_outputs(
        [_line("T1", 100.2), _line("T2", 110.3)], [100.0, 110.0]
    )
    assert [round(v.ms) for v in outbounds] == [200, 300]


def test_the_outputs_are_sorted_even_if_they_arrive_reversed():
    """Both fall in the window: without sorting, the last one in the list would win."""
    outbounds, _ = pair_outputs([_line("T1", 110.1)], [110.0, 109.5])
    assert outbounds[0].left == 110.0


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def test_the_summary_without_round_trips_does_not_invent_percentiles():
    assert summarize([]) == {"n": 0}


def test_the_summary_gives_min_p50_and_max():
    outbounds = [Outbound("T1", 0.0, 0.1), Outbound("T2", 0.0, 0.3), Outbound("T3", 0.0, 0.2)]
    assert summarize(outbounds) == {"n": 3, "min": 100.0, "p50": 200.0, "max": 300.0}


# --------------------------------------------------------------------------
# Reading each PoC
# --------------------------------------------------------------------------


def _poc1_crossing(tmp_path, *, outputs, lines, audible=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ruta = tmp_path / "cero_ARNES-call-x.json"
    content = {
        "audio_outputs": outputs,
        "lines": {"per_line": lines},
    }
    if audible is not None:
        content["audible_outputs"] = audible
    ruta.write_text(json.dumps(content), encoding="utf-8")
    return ruta


def _poc2_json(tmp_path, *, outputs, lines):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ruta = tmp_path / "call_sala-x.json"
    ruta.write_text(
        json.dumps(
            {
                "livekit_extra": {
                    "per_turn": [
                        {"assistant": {"started_speaking_at": s}} for s in outputs
                    ],
                    "harness": {"lines": {"per_line": lines}},
                }
            }
        ),
        encoding="utf-8",
    )
    return ruta


def test_poc1_is_read_from_its_crossing(tmp_path):
    ruta = _poc1_crossing(
        tmp_path, outputs=[100.0], lines=[_line("T1", 100.05)]
    )
    data = of_poc1(ruta)
    assert data["poc"] == 1
    assert data["return_path_ms"]["p50"] == pytest.approx(50.0)


def test_poc1_measures_from_the_first_frame_WITH_SPEECH(tmp_path):
    """The right endpoint is "the harness heard it": the left one has to be audible.

    With just the first frame, the TTS onset gets included and reads as
    transport.
    """
    ruta = _poc1_crossing(
        tmp_path,
        outputs=[100.0],  # the TTS started delivering here, in silence
        audible=[100.8],  # and became audible here
        lines=[_line("T1", 100.85)],
    )
    data = of_poc1(ruta)
    assert data["return_path_ms"]["p50"] == pytest.approx(50.0, abs=1.0)
    assert data["same_definition_at_both_ends"] is True
    # And the TTS onset is published separately, because PoC2 cannot separate it.
    assert data["tts_onset_ms"]["p50"] == pytest.approx(800.0, abs=1.0)


def test_an_old_run_is_marked_as_measured_another_way(tmp_path):
    """Runs from before the audible probe are not comparable without saying so."""
    ruta = _poc1_crossing(
        tmp_path, outputs=[100.0], lines=[_line("T1", 100.05)]
    )
    data = of_poc1(ruta)
    assert data["same_definition_at_both_ends"] is False
    assert "tts_onset_ms" not in data


def test_poc2_is_read_from_the_worker_json(tmp_path):
    ruta = _poc2_json(
        tmp_path, outputs=[100.0], lines=[_line("T1", 100.4)]
    )
    data = of_poc2(ruta)
    assert data["poc"] == 2
    assert data["return_path_ms"]["p50"] == pytest.approx(400.0)


def test_read_decides_by_content_and_not_by_name(tmp_path):
    """The two files are named differently in each PoC and that must not matter."""
    uno = _poc1_crossing(
        tmp_path / "a", outputs=[1.0], lines=[_line("T1", 1.1)]
    )
    dos = _poc2_json(
        tmp_path / "b", outputs=[1.0], lines=[_line("T1", 1.2)]
    )
    assert leer(uno)["poc"] == 1
    assert leer(dos)["poc"] == 2


def test_an_old_run_without_the_outputs_cannot_be_measured(tmp_path):
    """Runs from before 20/09/2026 do not carry the left endpoint. It is not made up."""
    ruta = _poc1_crossing(
        tmp_path, outputs=[], lines=[_line("T1", 100.0)]
    )
    assert of_poc1(ruta) is None


def test_a_run_without_per_line_detail_neither(tmp_path):
    ruta = _poc1_crossing(tmp_path, outputs=[100.0], lines=[])
    assert of_poc1(ruta) is None


def test_a_file_that_does_not_exist_does_not_blow_up(tmp_path):
    assert leer(tmp_path / "does_not_exist.json") is None


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def _run(poc: int, values: list[float], *, rows=None) -> dict:
    """`rows` is [(turn, observed_ms, ida_ms)]; `values` are the outbound times."""
    detail = [
        {"turn": f"T{n}", "return_path_ms": v} for n, v in enumerate(values, 1)
    ]
    return {
        "poc": poc,
        "detail": detail,
        "per_line": [
            {"turn": t, "observed_ms": o, "ida_ms": i} for t, o, i in (rows or [])
        ],
    }


def test_the_comparison_gives_the_p50_difference():
    final = compare([_run(1, [10.0, 20.0, 30.0]), _run(2, [200.0, 400.0, 600.0])])
    assert final["per_poc"]["1"]["p50"] == 20.0
    assert final["per_poc"]["2"]["p50"] == 400.0
    assert final["p50_difference_ms"] == 380.0


def test_with_a_single_poc_there_is_no_difference_to_give():
    final = compare([_run(1, [10.0])])
    assert final["p50_difference_ms"] is None


def test_the_comparison_pools_several_runs_of_the_same_poc():
    final = compare([_run(1, [10.0, 20.0]), _run(1, [30.0, 40.0])])
    assert final["per_poc"]["1"]["n"] == 4


# --------------------------------------------------------------------------
# The three figures (contract §2.2.8)
# --------------------------------------------------------------------------


def test_the_table_publishes_the_three_figures_and_not_just_the_corrected_one():
    tabla = latency_table([_run(1, [5.0], rows=[("T1", 2700.0, 141.0)])])
    row = tabla["per_poc"]["1"]
    assert row["observed_p50_ms"] == 2700.0
    assert row["ida_p50_ms"] == 141.0
    assert row["outbound_p50_ms"] == 5.0
    assert row["transport_p50_ms"] == 146.0
    assert row["corrected_p50_ms"] == 2554.0


def test_the_corrected_one_discounts_BOTH_ends_and_not_just_the_inbound():
    """Discounting only the inbound would leave 3,022: the error would be the size of the finding."""
    row = latency_table([_run(2, [202.0], rows=[("T1", 3430.0, 408.0)])])
    assert row["per_poc"]["2"]["transport_p50_ms"] == 610.0
    assert row["per_poc"]["2"]["corrected_p50_ms"] == 2820.0


def test_each_line_is_corrected_with_ITS_transport():
    """With per-turn bias, a slow turn is not corrected with the median.

    T1 pays 100 inbound and T2 pays 900; correcting both with the median (500)
    would give [2400, 2600] and a p50 of 2,500. Per line, both are 2,000.
    """
    tabla = latency_table(
        [_run(1, [0.0, 0.0], rows=[("T1", 2100.0, 100.0), ("T2", 2900.0, 900.0)])]
    )
    row = tabla["per_poc"]["1"]
    assert row["corrected_p50_ms"] == 2000.0
    assert row["n_correctable"] == 2


def test_the_drift_warns_that_subtracting_the_p50_by_hand_does_not_add_up():
    """Without this number, anyone trying to reconcile the table would think there is an error."""
    tabla = latency_table(
        [
            _run(
                1,
                [10.0, 0.0, 0.0],
                rows=[
                    ("T1", 2100.0, 100.0),  # corrected 1,990
                    ("T2", 2900.0, 900.0),  # corrected 2,000
                    ("T3", 3000.0, 100.0),  # corrected 2,900
                ],
            )
        ]
    )
    row = tabla["per_poc"]["1"]
    assert row["observed_p50_ms"] == 2900.0
    assert row["ida_p50_ms"] == 100.0
    assert row["outbound_p50_ms"] == 0.0
    # Subtracting the p50s by hand would give 2,800; the p50 of the differences is 2,000.
    assert row["corrected_p50_ms"] == 2000.0
    assert row["drift_by_subtracting_percentiles_ms"] == 800.0


def test_without_inbound_path_no_corrected_one_is_invented():
    """Half a correction is worse than none: only the observed value is published."""
    row = latency_table([_run(1, [5.0], rows=[("T1", 2700.0, None)])])
    row = row["per_poc"]["1"]
    assert row["observed_p50_ms"] == 2700.0
    assert row["outbound_p50_ms"] == 5.0
    assert row["corrected_p50_ms"] is None
    assert row["n_correctable"] == 0


def test_an_outbound_without_a_line_still_counts_toward_its_own_p50():
    """The outbound is a measurement in its own right, not a derivative of the latency."""
    run = _run(1, [5.0, 400.0], rows=[("T1", 2700.0, 141.0)])
    row = latency_table([run])["per_poc"]["1"]
    assert row["n_round_trips"] == 2
    assert row["outbound_p50_ms"] == 202.5
    assert row["n_correctable"] == 1


def test_the_table_pools_several_runs_of_the_same_poc():
    tabla = latency_table(
        [
            _run(1, [5.0], rows=[("T1", 1000.0, 100.0)]),
            _run(1, [7.0], rows=[("T1", 3000.0, 200.0)]),
        ]
    )
    row = tabla["per_poc"]["1"]
    assert row["n_lines"] == 2
    assert row["observed_p50_ms"] == 2000.0
    assert row["ida_p50_ms"] == 150.0


def test_the_table_declares_that_percentiles_are_not_subtracted():
    tabla = latency_table([_run(1, [5.0], rows=[("T1", 2700.0, 141.0)])])
    assert "no se restan" in tabla["caution"]
    assert set(tabla["what_each_one_answers"]) == {
        "observed_value",
        "transport",
        "corrected",
    }
    assert "§2.2.8" in tabla["how_it_is_published"]


def test_poc1_takes_the_observed_and_the_inbound_from_the_same_file(tmp_path):
    """If they came from different files, the subtraction would mix calls."""
    ruta = _poc1_crossing(
        tmp_path,
        outputs=[100.0],
        lines=[_line("T1", 100.05, latency_ms=2700.0, ida_ms=141.0)],
    )
    assert of_poc1(ruta)["per_line"] == [
        {"turn": "T1", "observed_ms": 2700.0, "ida_ms": 141.0}
    ]


def test_poc2_too(tmp_path):
    ruta = _poc2_json(
        tmp_path,
        outputs=[100.0],
        lines=[_line("T1", 100.4, latency_ms=3430.0, ida_ms=408.0)],
    )
    assert of_poc2(ruta)["per_line"] == [
        {"turn": "T1", "observed_ms": 3430.0, "ida_ms": 408.0}
    ]


def test_the_short_name_of_the_generic_summary_is_also_read(tmp_path):
    """`carrier_simulator.cruce` writes `latency_ms`; each PoC writes the long one."""
    ruta = _poc1_crossing(
        tmp_path,
        outputs=[100.0],
        lines=[
            {"turn": "T1", "agent_first_speech": 100.05, "latency_ms": 2700.0}
        ],
    )
    assert of_poc1(ruta)["per_line"][0]["observed_ms"] == 2700.0


def test_a_line_without_latency_does_not_enter_as_zero(tmp_path):
    ruta = _poc1_crossing(
        tmp_path,
        outputs=[100.0, 110.0],
        lines=[
            _line("T1", 100.05, latency_ms=2700.0, ida_ms=141.0),
            _line("T2", 110.05, latency_ms=None),
        ],
    )
    assert [f["observed_ms"] for f in of_poc1(ruta)["per_line"]] == [2700.0]
