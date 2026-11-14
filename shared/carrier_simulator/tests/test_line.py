"""La intervención como unidad de análisis, y la tasa de fragmentación.

Lo que se prueba es lo que haría que el número mintiera sin que se note: que un
evento se le cuente a la intervención equivocada, que la latencia se mida contra
el fin del archivo en vez del fin de la voz, que un clip soltado contra un
agente que ya colgó diluya la fragmentación hacia 1.0, y que tomar el primer
evento del turno en vez del último convierta una pausa a media frase en un
sesgo enorme con pinta de hallazgo.
"""

from __future__ import annotations

import pytest

from shared.carrier_simulator.cero import ZeroMark
from shared.carrier_simulator.cruce import (
    EstimatedZero,
    Line,
    group,
    lines_summary,
)

SAMPLE_RATE = 8000


def _mark(turn: str, start: float, *, dura: float = 2.0, cola: float = 0.2) -> ZeroMark:
    """Un clip que empieza en `inicio`, dura `dura` y trae `cola` de silencio."""
    return ZeroMark(
        turn=turn,
        clip_id=turn.lower(),
        start_of_clip=start,
        clip_end=start + dura,
        measured_end_of_speech=start + dura - cola,
        drop_db=25.0,
        speech_threshold_dbfs=-27.0,
        sequence=int(turn[1:]),
        publish_sample_rate=SAMPLE_RATE,
    )


def _cero(indice: int, end: float, start: float | None = None) -> EstimatedZero:
    return EstimatedZero(indice=indice, start=start, end=end)


# --------------------------------------------------------------------------
# A quién le toca cada evento
# --------------------------------------------------------------------------


def test_the_events_fall_in_the_line_that_caused_them() -> None:
    marks = [_mark("T1", 100.0), _mark("T2", 110.0)]
    # El VAD partió T1 en dos y cerró T2 de una.
    zeros = [_cero(0, 101.2), _cero(1, 102.1), _cero(2, 112.1)]

    lines, loose_ones = group(marks, zeros)
    assert [i.fragmentation for i in lines] == [2, 1]
    assert loose_ones == []


def test_an_event_before_the_first_clip_belongs_to_nobody() -> None:
    """El saludo del agente puede disparar el VAD antes de que hablemos."""
    marks = [_mark("T1", 100.0)]
    lines, loose_ones = group(marks, [_cero(0, 95.0), _cero(1, 101.9)])
    assert [i.fragmentation for i in lines] == [1]
    assert len(loose_ones) == 1


def test_an_event_long_after_the_last_clip_belongs_to_nobody() -> None:
    marks = [_mark("T1", 100.0)]
    lines, loose_ones = group(
        marks, [_cero(0, 101.9), _cero(1, 500.0)], cola_seg=15.0
    )
    assert [i.fragmentation for i in lines] == [1]
    assert len(loose_ones) == 1


def test_the_generous_tail_does_not_lose_the_last_event() -> None:
    """Cerrar tarde el último turno es normal; perderlo mentiría a la baja."""
    marks = [_mark("T1", 100.0)]
    lines, _ = group(marks, [_cero(0, 110.0)], cola_seg=15.0)
    assert lines[0].fragmentation == 1


def test_a_zero_without_an_end_is_not_attributed() -> None:
    marks = [_mark("T1", 100.0)]
    lines, loose_ones = group(marks, [_cero(0, None)])  # type: ignore[arg-type]
    assert lines[0].fragmentation == 0
    assert len(loose_ones) == 1


def test_unordered_marks_are_sorted_by_time() -> None:
    """Llegan por canal de datos: el orden de llegada no es el del guion."""
    marks = [_mark("T2", 110.0), _mark("T1", 100.0)]
    lines, _ = group(marks, [_cero(0, 101.9)])
    assert [i.turn for i in lines] == ["T1", "T2"]
    assert lines[0].fragmentation == 1


def test_without_marks_everything_is_left_loose() -> None:
    lines, loose_ones = group([], [_cero(0, 1.0)])
    assert lines == []
    assert len(loose_ones) == 1


# --------------------------------------------------------------------------
# La latencia de la intervención
# --------------------------------------------------------------------------


def test_the_latency_goes_from_end_of_speech_to_the_first_audio() -> None:
    """Del fin de MI voz, no del fin del archivo: la cola no es latencia."""
    mark = _mark("T1", 100.0, dura=2.0, cola=0.2)  # voz termina en 101.8
    lines, _ = group([mark], [], first_speech={"T1": 103.0})
    assert lines[0].latency == pytest.approx(1.2)


def test_the_latency_does_not_depend_on_how_many_events_it_was_split_into() -> None:
    """Es la propiedad que hace comparable un framework que corta con uno que no."""
    mark = _mark("T1", 100.0)
    departure, _ = group(
        [mark], [_cero(0, 101.0), _cero(1, 101.5), _cero(2, 102.0)],
        first_speech={"T1": 103.0},
    )
    whole, _ = group([mark], [_cero(0, 102.0)], first_speech={"T1": 103.0})
    assert departure[0].latency == whole[0].latency
    assert departure[0].fragmentation == 3
    assert whole[0].fragmentation == 1


def test_without_first_speech_there_is_no_latency() -> None:
    """Inventar un cero sería peor que declarar que el turno no se midió."""
    lines, _ = group([_mark("T1", 100.0)], [])
    assert lines[0].latency is None


def test_if_the_clip_has_no_end_of_speech_it_falls_back_to_the_end_of_the_file() -> None:
    """Un `--clip` suelto fuera del manifiesto no lo trae; se dice, no se falla."""
    mark = ZeroMark(
        turn="T1",
        clip_id=None,
        start_of_clip=100.0,
        clip_end=102.0,
        measured_end_of_speech=None,
        sequence=1,
        publish_sample_rate=SAMPLE_RATE,
    )
    lines, _ = group([mark], [], first_speech={"T1": 103.0})
    assert lines[0].end_of_speech == pytest.approx(102.0)
    assert lines[0].latency == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Cuál evento cierra la intervención
# --------------------------------------------------------------------------


def test_the_zero_of_the_line_is_the_last_event() -> None:
    """Tomar el primero mediría la primera pausa del clip, no el turno."""
    mark = _mark("T1", 100.0)
    lines, _ = group([mark], [_cero(0, 100.9), _cero(1, 102.0)])
    cero = lines[0].pipeline_zero
    assert cero is not None and cero.end == pytest.approx(102.0)


def test_without_events_there_is_no_pipeline_zero() -> None:
    lines, _ = group([_mark("T1", 100.0)], [])
    assert lines[0].pipeline_zero is None


# --------------------------------------------------------------------------
# La tasa de fragmentación
# --------------------------------------------------------------------------


def test_one_turn_per_line_gives_one_point_zero() -> None:
    # Clips en 100, 110 y 120; cada uno dura 2 s. Un cierre por clip.
    marks = [_mark("T1", 100.0), _mark("T2", 110.0), _mark("T3", 120.0)]
    zeros = [_cero(0, 101.9), _cero(1, 111.9), _cero(2, 121.9)]
    lines, loose_ones = group(marks, zeros)
    data = lines_summary(lines, loose_ones=len(loose_ones))
    assert loose_ones == []
    assert data["heard_lines"] == 3
    assert data["fragmentation"] == pytest.approx(1.0)


def test_splitting_each_line_in_two_gives_two_point_zero() -> None:
    marks = [_mark("T1", 100.0), _mark("T2", 110.0), _mark("T3", 120.0)]
    zeros = []
    for i, base in enumerate((100.0, 110.0, 120.0)):
        zeros.append(_cero(2 * i, base + 1.0))
        zeros.append(_cero(2 * i + 1, base + 1.9))
    lines, loose_ones = group(marks, zeros)
    data = lines_summary(lines, loose_ones=len(loose_ones))
    assert loose_ones == []
    assert data["fragmentation"] == pytest.approx(2.0)


def test_a_clip_against_an_agent_that_hung_up_does_not_dilute_the_fragmentation() -> None:
    """Es el caso real de T8: si contara, haría ver más parejo al que más corta."""
    marks = [_mark("T1", 100.0), _mark("T2", 110.0), _mark("T8", 200.0)]
    # T1 y T2 se partieron en dos; T8 salió contra nadie y no produjo evento.
    zeros = [_cero(0, 101.0), _cero(1, 101.9), _cero(2, 111.0), _cero(3, 111.9)]
    lines, loose_ones = group(marks, zeros)
    data = lines_summary(lines, loose_ones=len(loose_ones))

    assert data["lines"] == 3
    assert data["heard_lines"] == 2
    assert data["fragmentation"] == pytest.approx(2.0)


def test_the_sample_fragmentation_uses_the_same_denominator() -> None:
    marks = [_mark("T1", 100.0), _mark("T8", 200.0)]
    lines, loose_ones = group(marks, [_cero(0, 101.9)])
    data = lines_summary(
        lines, loose_ones=len(loose_ones), pipeline_samples=3
    )
    assert data["heard_lines"] == 1
    assert data["sample_fragmentation"] == pytest.approx(3.0)


def test_without_pipeline_samples_that_row_is_not_published() -> None:
    lines, _ = group([_mark("T1", 100.0)], [_cero(0, 101.9)])
    assert "sample_fragmentation" not in lines_summary(lines)


def test_without_any_heard_line_the_fragmentation_is_none() -> None:
    """Dividir entre cero o publicar 0.0 serían dos mentiras distintas."""
    lines, _ = group([_mark("T1", 100.0)], [])
    assert lines_summary(lines)["fragmentation"] is None


def test_the_summary_carries_the_five_latency_numbers() -> None:
    marks = [_mark("T1", 100.0), _mark("T2", 110.0), _mark("T3", 120.0)]
    zeros = [_cero(0, 101.9), _cero(1, 111.9), _cero(2, 121.9)]
    # La voz de cada clip termina en 101.8, 111.8 y 121.8.
    first_speech = {"T1": 103.0, "T2": 114.0, "T3": 123.0}
    lines, loose_ones = group(marks, zeros, first_speech=first_speech)
    lat = lines_summary(lines, loose_ones=len(loose_ones))[
        "line_latency_ms"
    ]
    assert lat["n"] == 3
    assert lat["min"] == pytest.approx(1200.0, abs=1.0)
    assert lat["max"] == pytest.approx(2200.0, abs=1.0)


def test_loose_events_are_counted_separately() -> None:
    lines, loose_ones = group([_mark("T1", 100.0)], [_cero(0, 50.0)])
    data = lines_summary(lines, loose_ones=len(loose_ones))
    assert data["events_without_line"] == 1
    assert data["pipeline_events"] == 0
