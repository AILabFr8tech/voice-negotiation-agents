"""Recortes por evento: el evento cae en el segundo 2 y el índice marca discrepancias."""

from __future__ import annotations

import csv

import numpy as np

from shared.audio.wav import read_wav_stereo
from validation import interruption_cuts as rc
from validation.tests.test_pauses_interruptions import SAMPLE_RATE, _campana


def test_trim_fills_outside_the_file():
    canal = np.arange(1, 11, dtype=np.int16)
    out = rc.trim(canal, 1, since=-2, until=3)
    assert out.tolist() == [0, 0, 1, 2, 3]


def test_a_trim_per_event_and_discrepancies(tmp_path):
    c = _campana(tmp_path)
    assert rc.main(["--campana", str(c), "--output", str(tmp_path / "out")]) == 0
    with (tmp_path / "out" / "indice.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["turn"], r["discrepa"]) for r in rows] == [("T4", "0"), ("T5", "0")]
    _, agent, sample_rate = read_wav_stereo(tmp_path / "out" / rows[0]["file"])
    assert sample_rate == SAMPLE_RATE and len(agent) == int(SAMPLE_RATE * (rc.ANTES_SEG + rc.AFTER_S))
    # El apoyo es a los 5 s; el agente calla de 6.0 a 7.2 → en el recorte, de 3.0 a 4.2.
    assert np.abs(agent[int(SAMPLE_RATE * 3.2):int(SAMPLE_RATE * 4.0)]).max() == 0
    assert np.abs(agent[int(SAMPLE_RATE * 2.5):int(SAMPLE_RATE * 2.9)]).max() > 0
