"""The plan order and how a range is read."""

from __future__ import annotations

import pytest

from shared.lote.plan import BatchItem, build_plan, plan_fingerprint, span


# --- range ---


def test_a_range_with_a_dash_is_expanded():
    assert span("1-5") == [1, 2, 3, 4, 5]


def test_a_list_with_commas_keeps_the_written_order():
    assert span("5,1,3") == [5, 1, 3]


def test_a_single_number():
    assert span("2") == [2]


def test_ranges_and_singles_are_mixed():
    assert span("1-2,5") == [1, 2, 5]


def test_a_reversed_range_is_an_error():
    with pytest.raises(ValueError, match="reversed range"):
        span("5-1")


def test_a_text_without_numbers_is_an_error():
    with pytest.raises(ValueError):
        span(" , ")


# --- build_plan ---


def test_the_two_pocs_of_the_same_scenario_end_up_adjacent():
    """The whole point of batch mode: never one campaign and then the other."""
    plan = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=1)
    assert [(i.poc, i.scenario) for i in plan] == [(1, 1), (2, 1), (1, 2), (2, 2)]


def test_no_poc_runs_twice_in_a_row():
    plan = build_plan(pocs=[1, 2], scenarios=[1, 2, 3], repetitions=2)
    pocs = [i.poc for i in plan]
    assert all(a != b for a, b in zip(pocs, pocs[1:]))


def test_the_repetition_is_the_outer_loop():
    """First one full pass through the scenarios, then the next."""
    plan = build_plan(pocs=[1], scenarios=[1, 2], repetitions=2)
    assert [(i.scenario, i.repetition) for i in plan] == [(1, 1), (2, 1), (1, 2), (2, 2)]


def test_the_indices_go_from_one_to_n_without_gaps():
    plan = build_plan(pocs=[1, 2], scenarios=[1, 2, 3], repetitions=3)
    assert [i.indice for i in plan] == list(range(1, 19))


def test_the_key_carries_index_poc_scenario_and_repetition():
    item = BatchItem(indice=7, poc=2, scenario=4, repetition=3)
    assert item.clave == "0007-poc2-e4-r3"


def test_the_key_sorts_alphabetically_like_the_plan():
    plan = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=3)
    keys = [i.clave for i in plan]
    assert keys == sorted(keys)


def test_zero_repetitions_is_an_error():
    with pytest.raises(ValueError):
        build_plan(pocs=[1], scenarios=[1], repetitions=0)


def test_without_scenarios_is_an_error():
    with pytest.raises(ValueError):
        build_plan(pocs=[1], scenarios=[], repetitions=1)


# --- fingerprint ---


def test_the_same_plan_gives_the_same_fingerprint():
    a = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=2)
    b = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=2)
    assert plan_fingerprint(a) == plan_fingerprint(b)


def test_an_extra_scenario_changes_the_fingerprint():
    """Exactly the expensive case: the indices shift and the keys lie."""
    a = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=1)
    b = build_plan(pocs=[1, 2], scenarios=[1, 2, 3], repetitions=1)
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_other_scenarios_with_the_same_shape_give_another_fingerprint():
    """Same number of calls and same indices, but they are not the same calls."""
    a = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=1)
    b = build_plan(pocs=[1, 2], scenarios=[3, 4], repetitions=1)
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_the_order_of_the_pocs_changes_the_fingerprint():
    a = build_plan(pocs=[1, 2], scenarios=[1], repetitions=1)
    b = build_plan(pocs=[2, 1], scenarios=[1], repetitions=1)
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_from_repetition_skips_the_already_run():
    plan = build_plan(pocs=[1, 2], scenarios=[1, 2], repetitions=3, from_repetition=2)
    assert [(i.poc, i.scenario, i.repetition) for i in plan] == [
        (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2),
        (1, 1, 3), (2, 1, 3), (1, 2, 3), (2, 2, 3),
    ]
    assert [i.indice for i in plan] == list(range(1, 9))


def test_from_repetition_out_of_range():
    with pytest.raises(ValueError):
        build_plan(pocs=[1], scenarios=[1], repetitions=2, from_repetition=3)
    with pytest.raises(ValueError):
        build_plan(pocs=[1], scenarios=[1], repetitions=2, from_repetition=0)


def test_by_default_the_plan_does_not_change():
    assert build_plan(pocs=[1, 2], scenarios=[1], repetitions=2) == build_plan(
        pocs=[1, 2], scenarios=[1], repetitions=2, from_repetition=1
    )
