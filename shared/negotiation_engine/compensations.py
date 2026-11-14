"""Catalog of non-monetary compensations. Pure logic: it only reads the YAML.

`compensations.yaml` is the single source of truth and feeds two things at
once, so they cannot get out of sync:

  - the {COMPENSACIONES_DISPONIBLES} placeholder of the system prompt, as
    readable prose for the LLM to say aloud (`render_compensations`);
  - the valid values of `tipo` in ofrecer_compensacion_no_monetaria(tipo)
    (`compensation_ids` / `validate_compensation`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

COMPENSATIONS_PATH: Path = Path(__file__).with_name("compensations.yaml")

_VALID_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Compensation:
    """A non-monetary compensation that can be offered during the negotiation."""

    id: str
    descripcion: str
    # Phrases by which this compensation is recognized when the agent says it
    # **paraphrased**. They are not part of the contract with the LLM: they are
    # the ear of the harness's automatic actor (`shared/test_scenarios/decision.py`),
    # which has to pick the branch of scenario 4 from what it heard.
    #
    # They live here and not in the harness for the same reason as everything
    # else in this file: if the business changes the catalog, the phrases change
    # with it. A second list in another file is exactly the desynchronization
    # this module exists to prevent.
    recognized_by: tuple[str, ...] = ()


@lru_cache(maxsize=None)
def load_compensations(ruta: Path | str | None = None) -> tuple[Compensation, ...]:
    """Load and validate the catalog. Result cached per path.

    Raises FileNotFoundError if the file does not exist and ValueError if the
    structure does not meet the contract (a list of records with a unique
    snake_case `id` and a non-empty `descripcion`).
    """
    destination = Path(ruta) if ruta is not None else COMPENSATIONS_PATH
    content = yaml.safe_load(destination.read_text(encoding="utf-8"))

    if not isinstance(content, list) or not content:
        raise ValueError(
            f"{destination.name} must contain a non-empty list of compensations"
        )

    compensations: list[Compensation] = []
    seen_items: set[str] = set()
    for position, record in enumerate(content, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"{destination.name}: record #{position} is not a mapping with "
                "`id` and `descripcion`"
            )

        identifier = record.get("id")
        descripcion = record.get("descripcion")

        if not isinstance(identifier, str) or not _VALID_ID.match(identifier):
            raise ValueError(
                f"{destination.name}: record #{position} has an invalid `id` "
                f"({identifier!r}). It must be snake_case: ^[a-z][a-z0-9_]*$"
            )
        if identifier in seen_items:
            raise ValueError(
                f"{destination.name}: the `id` {identifier!r} is duplicate. The "
                "ids are the valid values of `tipo` and must be unique"
            )
        if not isinstance(descripcion, str) or not descripcion.strip():
            raise ValueError(
                f"{destination.name}: record {identifier!r} needs a non-empty "
                "`descripcion` (it is what the agent says aloud)"
            )

        phrases = record.get("recognized_by") or ()
        if isinstance(phrases, str):
            phrases = [phrases]
        if not isinstance(phrases, (list, tuple)):
            raise ValueError(
                f"{destination.name}: `recognized_by` of {identifier!r} must be "
                "a list of phrases"
            )
        clean_items = tuple(str(f).strip() for f in phrases if str(f).strip())

        seen_items.add(identifier)
        compensations.append(
            Compensation(
                id=identifier,
                descripcion=descripcion.strip(),
                recognized_by=clean_items,
            )
        )

    return tuple(compensations)


def compensation_ids(ruta: Path | str | None = None) -> tuple[str, ...]:
    """Catalog ids: the only valid values of `tipo`."""
    return tuple(c.id for c in load_compensations(ruta))


def get_compensation(
    tipo: str, ruta: Path | str | None = None
) -> Compensation | None:
    """Return the compensation with that `tipo`, or None if it is not in the catalog."""
    for compensation in load_compensations(ruta):
        if compensation.id == tipo:
            return compensation
    return None


def validate_compensation(tipo: object, ruta: Path | str | None = None) -> bool:
    """True if `tipo` matches a compensation in the catalog.

    Guards against the LLM inventing a compensation Spots does not offer: if
    the `tipo` does not exist, the agent must not promise anything.
    """
    if not isinstance(tipo, str):
        return False
    return get_compensation(tipo, ruta) is not None


def render_compensations(ruta: Path | str | None = None) -> str:
    """Prose for the {COMPENSACIONES_DISPONIBLES} placeholder of the system prompt.

    Joins the descriptions into a natural sentence (no bullets or ids: this is
    synthesized as audio, not read on a screen).
    """
    descriptions = [c.descripcion for c in load_compensations(ruta)]
    if len(descriptions) == 1:
        return descriptions[0]
    return f"{', '.join(descriptions[:-1])} y {descriptions[-1]}"
