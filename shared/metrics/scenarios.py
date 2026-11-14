"""Reading the scenario script as data.

It only loads and validates `shared/test_scenarios/scenarios.yaml`. Its only
reason to exist is so the WER reference file can be pre-filled with the script
lines: without it, what one said would have to be typed by hand after each of
the 15 calls, which is exactly the manual work this module eliminates.

The compensation catalog (`negotiation_engine/compensations.py`) does the same
with its YAML; the same pattern is followed so as not to invent two ways of
reading config in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

SCENARIOS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "test_scenarios" / "scenarios.yaml"
)


# Conditions a branch can declare. Closed on purpose: a key not listed here is
# rejected on load instead of ignored, because a silently ignored condition
# makes the tree always take the default branch and the run looks normal.
BRANCH_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        "reached_amount",
        "turn_amount",
        "compensation_in_call",
        "compensation_in_turn",
        "words",
        "no_words",
    }
)

_BRANCH_KEYS: Final[frozenset[str]] = BRANCH_CONDITIONS | {
    "id",
    "linea",
    "when",
    "by_default",
}


@dataclass(frozen=True)
class ScriptBranch:
    """A branch of a turn's decision tree.

    `linea` has to match the turn's base line or one of its `alternates`: that
    equality is what ties the branch to the recorded clip, and what keeps the
    tree and the prose script from drifting apart.
    """

    id: str
    linea: str
    when: str = ""
    by_default: bool = False
    reached_amount: float | None = None
    turn_amount: float | None = None
    compensation_in_call: bool | None = None
    compensation_in_turn: bool | None = None
    words: tuple[str, ...] = ()
    no_words: tuple[str, ...] = ()

    @property
    def condiciones(self) -> dict[str, Any]:
        """Only what this branch requires. Empty in the default branch."""
        raw_ones = {
            "reached_amount": self.reached_amount,
            "turn_amount": self.turn_amount,
            "compensation_in_call": self.compensation_in_call,
            "compensation_in_turn": self.compensation_in_turn,
            "words": self.words or None,
            "no_words": self.no_words or None,
        }
        return {k: v for k, v in raw_ones.items() if v is not None}


@dataclass(frozen=True)
class ScriptTurn:
    """One line of the actor. Empty `linea` = the turn is a silence."""

    id: str
    linea: str = ""
    when: str = ""
    note: str = ""
    alternates: tuple[str, ...] = ()
    branches: tuple[ScriptBranch, ...] = ()
    silence_s: float | None = None
    hangs_up: bool = False

    @property
    def is_silence(self) -> bool:
        return not self.linea.strip()

    @property
    def branches_out(self) -> bool:
        return bool(self.branches)

    def variant_of(self, linea: str) -> str:
        """Which recorded variant corresponds to that line: `base` or `altN`.

        It is the bridge between the branch and the clip: the base is `eN_tM`
        and the alternate in position k is `eN_tM_altk`, same as in
        `shared/audio/manifest.py::expected_inventory`.
        """
        target = " ".join(linea.split())
        if target == " ".join(self.linea.split()):
            return "base"
        for numero, alternate in enumerate(self.alternates, start=1):
            if target == " ".join(alternate.split()):
                return f"alt{numero}"
        raise ValueError(
            f"el turno {self.id}: la línea {linea!r} no es la base ni ninguna de "
            "sus alternativas, así que no hay clip grabado que le corresponda"
        )


@dataclass(frozen=True)
class Scenario:
    """A script scenario, with its turns in order."""

    id: int
    name: str
    turns: tuple[ScriptTurn, ...] = ()
    mide: str = ""
    expected_result: str = ""
    caution: str = ""

    @property
    def label(self) -> str:
        return f"{self.id} — {self.name}"

    def reference_text(self) -> str:
        """All the scenario's spoken lines, in order.

        It is the seed of the WER reference: what the actor *should* have said
        if they followed the script to the letter. Silence turns contribute no
        text, on purpose.
        """
        return "\n".join(t.linea for t in self.turns if not t.is_silence)


@dataclass(frozen=True)
class Script:
    """The full script: trip config + the 5 scenarios."""

    version: str
    trip: dict[str, Any] = field(default_factory=dict)
    scenarios: tuple[Scenario, ...] = ()

    def scenario(self, scenario_key: int) -> Scenario:
        """The scenario by its number. Raises if it does not exist."""
        for scen in self.scenarios:
            if scen.id == scenario_key:
                return scen
        available = ", ".join(str(e.id) for e in self.scenarios)
        raise KeyError(
            f"no existe el escenario {scenario_key!r}; disponibles: {available}"
        )


def _text(valor: Any) -> str:
    """Any YAML scalar to clean text. None -> empty string."""
    if valor is None:
        return ""
    return str(valor).strip()


def _phrases(valor: Any, where: str, key: str) -> tuple[str, ...]:
    if isinstance(valor, str):
        valor = [valor]
    if not isinstance(valor, (list, tuple)):
        raise ValueError(f"{where}: `{key}` debe ser una lista de frases")
    return tuple(_text(v) for v in valor if _text(v))


def _branch_from_yaml(raw_value: Any, turn_id: str, position: int) -> ScriptBranch:
    where = f"la rama #{position} de {turn_id}"
    if not isinstance(raw_value, dict):
        raise ValueError(f"{where} no es un mapa")

    foreign = set(raw_value) - _BRANCH_KEYS
    if foreign:
        # Rejected instead of ignored: a misspelled condition that gets ignored
        # makes the tree always fall into the default branch, and the run looks
        # perfectly normal.
        raise ValueError(
            f"{where} trae llaves que no existen: {sorted(foreign)}. "
            f"Condiciones válidas: {sorted(BRANCH_CONDITIONS)}"
        )

    linea = _text(raw_value.get("linea"))
    if not linea:
        raise ValueError(f"{where} no trae `linea`")

    def _amount(key: str) -> float | None:
        valor = raw_value.get(key)
        if valor is None:
            return None
        try:
            return float(valor)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: `{key}` debe ser un número") from exc

    def _flag(key: str) -> bool | None:
        valor = raw_value.get(key)
        return None if valor is None else bool(valor)

    return ScriptBranch(
        id=_text(raw_value.get("id")) or f"rama_{position}",
        linea=linea,
        when=" ".join(_text(raw_value.get("when")).split()),
        by_default=bool(raw_value.get("by_default", False)),
        reached_amount=_amount("reached_amount"),
        turn_amount=_amount("turn_amount"),
        compensation_in_call=_flag("compensation_in_call"),
        compensation_in_turn=_flag("compensation_in_turn"),
        words=_phrases(raw_value.get("words") or (), where, "words"),
        no_words=_phrases(raw_value.get("no_words") or (), where, "no_words"),
    )


def _validate_branches(turn: ScriptTurn) -> None:
    """A tree with no way out is worse than no tree: it is rejected on load."""
    branches = turn.branches
    by_default = [r for r in branches if r.by_default]
    if len(by_default) != 1:
        raise ValueError(
            f"el turno {turn.id} tiene {len(by_default)} ramas `by_default` y "
            "necesita exactamente 1: sin ella una respuesta inesperada del "
            "agente dejaría al arnés sin línea que decir a media llamada"
        )
    if not branches[-1].by_default:
        raise ValueError(
            f"el turno {turn.id}: la rama `by_default` tiene que ir al final. "
            "Se evalúan en orden y gana la primera que se cumple, así que una "
            "por defecto en medio tapa a las que vengan después"
        )
    for rama in branches[:-1]:
        if not rama.condiciones:
            raise ValueError(
                f"el turno {turn.id}: la rama {rama.id!r} no declara ninguna "
                "condición y no es la por defecto, así que gana siempre"
            )

    seen_items: set[str] = set()
    for rama in branches:
        if rama.id in seen_items:
            raise ValueError(f"el turno {turn.id}: la rama {rama.id!r} está duplicada")
        seen_items.add(rama.id)
        # Ties the branch to a recorded clip. Raises if the line does not match.
        turn.variant_of(rama.linea)


def _turn_from_yaml(raw_value: Any, position: int) -> ScriptTurn:
    if not isinstance(raw_value, dict):
        raise ValueError(f"el turno en posición {position} no es un mapa")

    raw_alternates = raw_value.get("alternates") or ()
    if isinstance(raw_alternates, str):
        raw_alternates = [raw_alternates]

    identifier = _text(raw_value.get("id")) or f"T{position + 1}"
    raw_branches = raw_value.get("branches") or ()
    if not isinstance(raw_branches, (list, tuple)):
        raise ValueError(f"el turno {identifier}: `ramas` debe ser una lista")

    silence = raw_value.get("silence_s")
    turn = ScriptTurn(
        id=identifier,
        # `linea` can be empty on purpose (silence turn); it is not filled in
        # with anything.
        linea=_text(raw_value.get("linea")),
        when=_text(raw_value.get("when")),
        note=" ".join(_text(raw_value.get("note")).split()),
        alternates=tuple(_text(a) for a in raw_alternates if _text(a)),
        branches=tuple(
            _branch_from_yaml(r, identifier, i)
            for i, r in enumerate(raw_branches, start=1)
        ),
        silence_s=float(silence) if silence is not None else None,
        hangs_up=bool(raw_value.get("cuelga", False)),
    )
    if turn.branches:
        _validate_branches(turn)
    return turn


def load_script(ruta: Path | str | None = None) -> Script:
    """Load and validate the script. No need to cache: it is read once per run."""
    destination = Path(ruta) if ruta else SCENARIOS_PATH
    if not destination.is_file():
        raise FileNotFoundError(f"no encuentro el guion de escenarios en {destination}")

    data = yaml.safe_load(destination.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{destination} debe contener un mapa en la raíz")

    raw_values = data.get("scenarios")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"{destination} no trae la lista `escenarios`")

    scenarios: list[Scenario] = []
    seen_items: set[int] = set()
    for i, raw_value in enumerate(raw_values):
        if not isinstance(raw_value, dict):
            raise ValueError(f"el escenario en posición {i} no es un mapa")
        try:
            scenario_key = int(raw_value["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"el escenario en posición {i} no tiene `id` entero") from exc
        if scenario_key in seen_items:
            raise ValueError(f"el escenario {scenario_key} está duplicado en {destination}")
        seen_items.add(scenario_key)

        raw_turns = raw_value.get("turns") or []
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"el escenario {scenario_key} no trae turnos")

        scenarios.append(
            Scenario(
                id=scenario_key,
                name=_text(raw_value.get("name")) or f"escenario {scenario_key}",
                turns=tuple(
                    _turn_from_yaml(t, j) for j, t in enumerate(raw_turns)
                ),
                mide=" ".join(_text(raw_value.get("mide")).split()),
                expected_result=_text(raw_value.get("expected_result")),
                caution=" ".join(_text(raw_value.get("caution")).split()),
            )
        )

    trip = data.get("trip")
    return Script(
        version=_text(data.get("version")) or "no_version",
        trip=dict(trip) if isinstance(trip, dict) else {},
        scenarios=tuple(sorted(scenarios, key=lambda e: e.id)),
    )
