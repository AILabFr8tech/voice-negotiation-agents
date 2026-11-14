"""The decision rules of the report analysis."""

from __future__ import annotations

import numpy as np

from shared.metrics import analysis_report as ar


def test_verdict_against_the_200_ms_margin():
    assert ar.verdict(-577.5, -408.5) == "distinta de cero y mayor que 200 ms"
    assert ar.verdict(26.6, 264.3) == "distinta de cero; no se sabe si pasa de 200 ms"
    # Includes zero but fits within ±200 ms: it may be zero and does not reach the margin.
    assert ar.verdict(-119.0, 33.7) == "equivalentes en la práctica"
    # Includes zero and crosses the margin: unknown.
    assert ar.verdict(-146.3, 267.6) == "no se distingue"
    assert ar.verdict(20.0, 150.0) == "distinta de cero, menor que 200 ms"


def test_the_english_article_translates_each_verdict():
    # docs/paper/en/sn-article.tex translates the verdicts by comparing against the
    # Spanish text; one that changed here would come out as «??» in the paper. The
    # search is done outside the block of embedded figures, which also carries the
    # Spanish verdicts and would let the test pass without the translation.
    article = (ar.RAIZ / "docs" / "paper" / "en" / "sn-article.tex").read_text(encoding="utf-8")
    start = article.index("% --- Cifras generadas")
    end = article.index("% --- fin de las cifras generadas ---")
    translation = article[:start] + article[end:]
    for low, alto in [(-150, 120), (-50, 260), (40, 150), (250, 400), (120, 300)]:
        assert "{" + ar.verdict(low, alto) + "}" in translation


def test_the_bootstrap_of_a_stage_preserves_whole_calls():
    rng = np.random.default_rng(0)
    groups = [np.array([1.0, 1.0]), np.array([5.0, 5.0, 5.0])]
    sample = ar._resample(rng, groups)
    # Only whole calls can come out: sizes 2 or 3 per drawn call.
    assert set(np.unique(sample)) <= {1.0, 5.0}
    assert sample.size in (4, 5, 6)


def test_without_output_subtracts_the_output_median_from_that_of_all_responses():
    # The 1,500 ms response has no measured output, and it still enters the
    # latency median, just as in the latency table.
    calls = [
        (np.array([2000.0, 2100.0]), np.array([100.0, 110.0])),
        (np.array([1500.0, 2200.0]), np.array([120.0])),
    ]
    assert ar.no_output(calls) == 2050.0 - 110.0


def test_the_difference_without_output_resamples_whole_calls():
    rng = np.random.default_rng(0)
    a = [(np.array([2000.0, 2000.0]), np.array([3.0, 3.0]))]
    b = [(np.array([1900.0]), np.array([100.0]))]
    lo, hi = ar.ci_difference_no_exit(a, b, rng)
    assert lo == hi == (2000.0 - 3.0) - (1900.0 - 100.0)


def test_the_output_is_measured_against_the_first_sound():
    # No earlier sound: the output stays as it came.
    assert ar.outbound_to_first_sound(2000.0, 1990.0, 3.0, "en_vivo") == (3.0, "directa")
    # A «claro» 860 ms before the response: the mark is the «claro»'s, and the
    # output measured against the response carries the sound and the pause.
    assert ar.outbound_to_first_sound(2600.0, 1740.0, 863.0, "en_vivo") == (3.0, "corrected")
    # Within the two-frame tolerance, even if it comes out negative.
    assert ar.outbound_to_first_sound(2600.0, 1740.0, 830.0, "en_vivo") == (-30.0, "corrected")
    # The mark falls between the first sound and the response: it belongs to neither.
    assert ar.outbound_to_first_sound(2600.0, 1740.0, 400.0, "en_vivo") == (None, "no_mark")
    # Recovered turns were already measured against the first sound.
    assert ar.outbound_to_first_sound(4300.0, 2300.0, 110.0, "recuperado") == (110.0, "directa")
    assert ar.outbound_to_first_sound(2000.0, None, None, "en_vivo") == (None, "no_value")


def test_the_latency_without_transport_uses_the_output_against_the_first_sound(tmp_path):
    fields = ["poc", "clave", "scenario", "repetition", "observed_latency_ms",
              "first_sound_latency_ms", "ida_ms", "outbound_ms", "corrected_latency_ms",
              "with_round_trip", "origin"]
    ruta = tmp_path / "lines.csv"
    ruta.write_text(
        ",".join(fields) + "\n" + "1,0001-poc1,1,1,2600.0,1740.0,185.0,863.0,1552.0,0,en_vivo\n",
        encoding="utf-8",
    )
    (f,) = ar.read_lines(ruta)
    assert f.outbound == 3.0 and f.raw_outbound == 863.0 and f.outbound_state == "corrected"
    assert f.corrected == 2600.0 - 185.0 - 3.0


def test_box_uses_whiskers_of_1_5_ranges():
    c = ar.box([1, 2, 3, 4, 100])
    assert c["median"] == 3 and c["upper whisker"] == 4
