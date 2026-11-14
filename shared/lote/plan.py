"""Which calls will be made and in what order.

The order is this module's measurement decision, and its only one. Everything
else (how many, from which scenario) is arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchItem:
    """One call in the plan. The index is its position, and it does not change on resume."""

    indice: int
    poc: int
    scenario: int
    repetition: int

    @property
    def clave(self) -> str:
        """Stable name of the call. It is the key for resuming.

        It carries the index first so the files sort themselves in the folder,
        and the three values that identify it so it can be read without opening
        the plan.
        """
        return (
            f"{self.indice:04d}-poc{self.poc}-e{self.scenario}-r{self.repetition}"
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "indice": self.indice,
            "poc": self.poc,
            "scenario": self.scenario,
            "repetition": self.repetition,
            "clave": self.clave,
        }


def span(text: str) -> list[int]:
    """Read `"1-5"`, `"1,3,5"` or `"2"` as a list of integers, in that order.

    No `set`: the order written is the order run, and sorting behind the scenes
    would hide that `--scenarios 5,1` does not do what it says.
    """
    number_list: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece[1:]:
            since, _, until = piece.partition("-")
            start, final = int(since), int(until)
            if final < start:
                raise ValueError(f"reversed range: {piece!r}")
            number_list.extend(range(start, final + 1))
        else:
            number_list.append(int(piece))
    if not number_list:
        raise ValueError(f"no number at all in {text!r}")
    return number_list


def build_plan(
    *,
    pocs: Sequence[int],
    scenarios: Sequence[int],
    repetitions: int,
    from_repetition: int = 1,
) -> list[BatchItem]:
    """The campaign order: the PoCs interleaved, not one and then the other.

    The outer loop is the repetition and the inner loop the PoC, so the two
    runs of the same scenario end up **adjacent**: `poc1-e1-r1`, `poc2-e1-r1`,
    `poc1-e2-r1`, … Minutes pass between those two calls, not a day, and
    whatever changes in the network or at the providers affects both almost
    equally.

    Running all 15 of one and then all 15 of the other would produce exactly
    the confound this avoids: any difference between the two batches (time of
    day, provider load, a model version behind the same label) would add to the
    architecture difference with no way to separate them afterwards.

    `from_repetition` starts the plan at a repetition other than 1: with
    `repetitions=3, from_repetition=2` you get 2 and 3. It is used to extend a
    campaign already run, under a new label, without repeating repetition 1,
    which in the interruption scenario carries its own events
    (`attach_events(..., repetition=...)`).
    """
    if repetitions < 1:
        raise ValueError("at least 1 repetition is required")
    if not 1 <= from_repetition <= repetitions:
        raise ValueError(
            f"the first repetition ({from_repetition}) must be between 1 and "
            f"the last ({repetitions})"
        )
    if not pocs or not scenarios:
        raise ValueError("the plan needs at least one PoC and one scenario")

    items: list[BatchItem] = []
    for repetition in range(from_repetition, repetitions + 1):
        for scenario in scenarios:
            for poc in pocs:
                items.append(
                    BatchItem(
                        indice=len(items) + 1,
                        poc=poc,
                        scenario=scenario,
                        repetition=repetition,
                    )
                )
    return items


def plan_fingerprint(items: Sequence[BatchItem]) -> str:
    """A hash of the plan, so that resuming does not mix two different campaigns.

    Resuming skips calls by their `clave` (key), and the key carries the index.
    If someone relaunches the same label with other scenarios, the indices shift
    and `0007` would no longer be the same call that was already done: one that
    never ran would be skipped and another would be repeated. With the
    fingerprint stored in `plan.json`, this is caught before spending.
    """
    raw_value = json.dumps(
        [[i.indice, i.poc, i.scenario, i.repetition] for i in items],
        separators=(",", ":"),
    )
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:16]
