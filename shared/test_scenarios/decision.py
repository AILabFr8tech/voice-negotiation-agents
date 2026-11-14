"""El actor automático: qué línea toca según lo que dijo el agente.

Los escenarios 2 y 4 no son lineales, y eso no es adorno: **es lo que miden**. El
2 mide si las concesiones decrecientes dejan al agente llegar a 18k, y la
pregunta solo tiene sentido si Carlos reacciona distinto cuando llega y cuando
no. El 4 mide si ofrece una compensación antes de subir precio, y si Carlos dice
"órale, con eso ya cambia la cosa" cuando el agente nunca ofreció nada, el
escenario dejó de medir lo que dice medir.

Eso lo hacía un humano leyendo `guion_escenarios.md`; aquí es dato: las ramas en
`escenarios.yaml` y las frases de cada compensación en
`negotiation_engine/compensations.yaml`, que ya era la fuente única del prompt y
de la función.

Vive en `shared/` porque **las 3 PoC actúan el mismo guion**. Si cada arnés
ramificara con su criterio, la llamada de PoC1 y la de PoC2 dejarían de ser el
mismo escenario y la comparación mediría también la diferencia entre dos actores.

## Decide sobre lo que el agente DIJO, no sobre lo que registró

Es deliberado y es la mitad del valor. Un defecto ya observado es que el agente
**pronuncia** una compensación sin llamar a `ofrecer_compensacion_no_monetaria`:
el motor registra `offered_compensations: []` mientras el transportista la oyó
perfectamente. El árbol rama con lo que se oyó, que es lo que haría un
transportista, así que la discrepancia entre la rama tomada y el registro vacío
queda en el JSON como dos hechos que no cuadran — y ese desacuerdo es justo el
hallazgo que el escenario 4 busca. Ramificar por el registro lo escondería.

## Precisión antes que cobertura, y la decisión queda auditada

Las frases se eligen para no dar falso positivo aunque cueste recall: el agente
menciona "fecha de entrega" también cuando la está negando. Aun así ninguna
heurística sobre texto libre es exacta, y por eso **cada decisión se registra con
su evidencia** (qué frase casó, qué montos se oyeron, qué rama ganó y por qué).
Una rama equivocada se ve después en el reporte; una rama equivocada y muda, no.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

from shared.metrics.scenarios import ScriptBranch, ScriptTurn
from shared.metrics.text import extract_amounts, fold_accents

# Palabras que, justo antes de una frase, la convierten en lo contrario de un
# ofrecimiento: "no tenemos flexibilidad en la fecha", "sin tiempo de espera".
# Es una heurística y se declara como tal; su acierto se audita en la evidencia,
# donde una frase suprimida por negación aparece con su motivo.
NEGATIONS: Final[frozenset[str]] = frozenset(
    {"no", "sin", "tampoco", "ninguna", "ningun", "ninguno", "nunca", "imposible"}
)

# Cuántas palabras antes de la frase se miran buscando una negación. Con 3
# alcanza para "no tenemos flexibilidad" y no para "no hay problema, te damos
# tiempo de espera", que son los dos casos que importan.
NEGATION_WINDOW: Final[int] = 3

_NON_WORD = re.compile(r"[^a-z0-9ñ ]+")


def normalize_for_phrases(text: str) -> str:
    """Minúsculas, sin acentos y sin puntuación, para buscar frases dentro.

    **No** se usa `texto.tokenizar` aquí a propósito: aquel colapsa "dieciocho
    mil" a `18000`, que es lo correcto para el WER y lo incorrecto para buscar
    "tiempo de espera" dentro de una frase. Los montos se leen aparte, con el
    extractor de verdad.
    """
    flat_text = fold_accents(text).lower()
    return " ".join(_NON_WORD.sub(" ", flat_text).split())


def _negated(flat_text: str, start: int) -> bool:
    priors = flat_text[:start].split()[-NEGATION_WINDOW:]
    return any(p in NEGATIONS for p in priors)


def find_phrase(plain_text: str, frase: str) -> tuple[bool, str]:
    """¿Aparece la frase, y no negada? Devuelve `(si, motivo)`.

    El motivo viaja siempre, incluso cuando la respuesta es "sí": es lo que
    después permite releer una rama dudosa sin volver a correr la llamada.
    """
    needle = normalize_for_phrases(frase)
    if not needle:
        return False, "frase vacía"
    position = plain_text.find(needle)
    if position < 0:
        return False, "no aparece"
    if _negated(plain_text, position):
        context = " ".join(plain_text[:position].split()[-NEGATION_WINDOW:])
        return False, f"aparece pero negada ({context!r} antes)"
    return True, "aparece"


# --------------------------------------------------------------------------
# Lo que el agente lleva dicho
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompensationHeard:
    """Una compensación reconocida en el habla del agente, con su evidencia."""

    id: str
    frase: str
    turn: str


@dataclass
class NegotiationState:
    """Lo que el agente lleva dicho, en los dos ámbitos que el árbol usa.

    Hay dos ámbitos y confundirlos rompe ramas distintas en direcciones
    different_ones: "¿ofreció una compensación **en este turno**?" es la pregunta del
    escenario 4 en T4 y T6, donde lo que importa es qué hizo *en respuesta* a lo
    que Carlos acaba de pedir; "¿ofreció alguna **en la llamada**?" es la de T8,
    donde la pregunta es si en algún momento la hubo.
    """

    # Frases de reconocimiento por id de compensación. Sale del catálogo.
    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    amounts: list[float] = field(default_factory=list)
    compensations: list[CompensationHeard] = field(default_factory=list)

    _current_turn: str = ""
    _turn_amounts: list[float] = field(default_factory=list)
    _turn_compensations: list[CompensationHeard] = field(default_factory=list)
    _evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_the_catalog(cls, ruta: Any = None) -> "NegotiationState":
        """Construye el estado con las frases del catálogo de compensaciones."""
        from shared.negotiation_engine.compensations import load_compensations

        return cls(
            phrases={
                c.id: c.recognized_by
                for c in load_compensations(ruta)
                if c.recognized_by
            }
        )

    def compensations_without_phrases(self, ruta: Any = None) -> tuple[str, ...]:
        """Compensaciones del catálogo que el arnés no sabría reconocer.

        Se pregunta **antes** de conectarse. Una compensación sin frases no da
        error: da una rama que nunca se cumple, y el escenario 4 saldría
        diciendo "nunca ofreció nada" sobre un agente que ofreció.
        """
        from shared.negotiation_engine.compensations import load_compensations

        return tuple(c.id for c in load_compensations(ruta) if not c.recognized_by)

    # -- Alimentación --

    def observe(self, turn: str, text: str) -> None:
        """Registra lo que el agente dijo en un turno. Reemplaza el turno actual."""
        self._current_turn = turn
        self._turn_amounts = []
        self._turn_compensations = []
        self._evidence = []

        if not text.strip():
            self._evidence.append("el agente no dijo nada en este turno")
            return

        flat_text = normalize_for_phrases(text)

        for monto in extract_amounts(text):
            self._turn_amounts.append(monto)
            self.amounts.append(monto)

        for compensation_id, phrases in sorted(self.phrases.items()):
            for frase in phrases:
                casa, motivo = find_phrase(flat_text, frase)
                if casa:
                    heard_value = CompensationHeard(
                        id=compensation_id, frase=frase, turn=turn
                    )
                    self._turn_compensations.append(heard_value)
                    self.compensations.append(heard_value)
                    self._evidence.append(f"{compensation_id}: {frase!r} {motivo}")
                    break
                if motivo.startswith("aparece pero negada"):
                    # Se anota aunque no cuente: es lo que explica una rama que
                    # no se tomó.
                    self._evidence.append(f"{compensation_id}: {frase!r} {motivo}")

    # -- Lecturas --

    @property
    def max_amount(self) -> float | None:
        return max(self.amounts) if self.amounts else None

    @property
    def turn_max_amount(self) -> float | None:
        return max(self._turn_amounts) if self._turn_amounts else None

    @property
    def had_compensation(self) -> bool:
        return bool(self.compensations)

    @property
    def had_compensation_in_turn(self) -> bool:
        return bool(self._turn_compensations)

    @property
    def ids_of_compensations(self) -> tuple[str, ...]:
        seen_items: list[str] = []
        for c in self.compensations:
            if c.id not in seen_items:
                seen_items.append(c.id)
        return tuple(seen_items)

    def evidence(self) -> tuple[str, ...]:
        parts = list(self._evidence)
        if self._turn_amounts:
            parts.append(
                "montos del turno: "
                + ", ".join(f"{m:,.0f}" for m in self._turn_amounts)
            )
        if self.amounts:
            parts.append(f"máximo de la llamada: {max(self.amounts):,.0f}")
        return tuple(parts)

    def as_json(self) -> dict[str, Any]:
        return {
            "max_amount_heard": self.max_amount,
            "amounts_heard": list(self.amounts),
            "compensations_heard": [
                {"id": c.id, "frase": c.frase, "turn": c.turn}
                for c in self.compensations
            ],
            "note": (
                "lo que el agente PRONUNCIÓ, reconocido por frases. No es el "
                "registro del motor: si difiere de `offered_compensations`, "
                "el agente dijo una compensación sin llamar a la función, y ese "
                "desacuerdo es el hallazgo del escenario 4"
            ),
        }


# --------------------------------------------------------------------------
# La decisión
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchDecision:
    """Qué rama ganó, con qué línea y por qué."""

    turn: str
    rama: str
    linea: str
    variant: str
    by_default: bool
    motivo: str
    evidence: tuple[str, ...] = ()
    discarded_ones: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "rama": self.rama,
            "variant": self.variant,
            "linea": self.linea,
            "by_default": self.by_default,
            "motivo": self.motivo,
            "evidence": list(self.evidence),
            "discarded_branches": list(self.discarded_ones),
        }


def _meets(rama: ScriptBranch, estado: NegotiationState, flat_text: str) -> tuple[bool, str]:
    """¿Se cumple esta rama? Devuelve `(si, por qué)`, siempre con el porqué."""
    reasons: list[str] = []

    if rama.reached_amount is not None:
        tope = estado.max_amount
        if tope is None or tope < rama.reached_amount:
            visto = "nada" if tope is None else f"{tope:,.0f}"
            return False, f"el máximo oído es {visto}, no llega a {rama.reached_amount:,.0f}"
        reasons.append(f"llegó a {tope:,.0f} >= {rama.reached_amount:,.0f}")

    if rama.turn_amount is not None:
        tope = estado.turn_max_amount
        if tope is None or tope < rama.turn_amount:
            visto = "nada" if tope is None else f"{tope:,.0f}"
            return False, (
                f"en este turno dijo {visto}, no llega a {rama.turn_amount:,.0f}"
            )
        reasons.append(f"en este turno dijo {tope:,.0f}")

    if rama.compensation_in_call is not None:
        occurred = estado.had_compensation
        if occurred != rama.compensation_in_call:
            return False, (
                "hubo compensación en la llamada" if occurred else "no hubo ninguna compensación"
            )
        reasons.append(
            f"compensaciones oídas: {', '.join(estado.ids_of_compensations) or 'ninguna'}"
        )

    if rama.compensation_in_turn is not None:
        occurred = estado.had_compensation_in_turn
        if occurred != rama.compensation_in_turn:
            return False, (
                "ofreció compensación en este turno"
                if occurred
                else "no ofreció compensación en este turno"
            )
        reasons.append("ofreció compensación en este turno" if occurred else "no ofreció ninguna")

    for frase in rama.words:
        casa, motivo = find_phrase(flat_text, frase)
        if not casa:
            return False, f"{frase!r}: {motivo}"
        reasons.append(f"dijo {frase!r}")

    for frase in rama.no_words:
        casa, _ = find_phrase(flat_text, frase)
        if casa:
            return False, f"dijo {frase!r}, que esta rama excluye"

    return True, "; ".join(reasons) or "sin condiciones"


def decide(
    turn: ScriptTurn,
    estado: NegotiationState,
    *,
    agent_text: str = "",
) -> BranchDecision:
    """Qué línea toca decir. Las ramas se evalúan en orden; gana la primera.

    `estado` ya tiene que haber observado el turno del agente: aquí no se
    alimenta, para que quede claro que decidir no tiene efectos y se pueda
    volver a decidir sobre el mismo estado sin contar dos veces los montos.
    """
    if not turn.branches:
        return BranchDecision(
            turn=turn.id,
            rama="no_branches",
            linea=turn.linea,
            variant="base",
            by_default=True,
            motivo="el turno no se ramifica",
        )

    flat_text = normalize_for_phrases(agent_text)
    discarded_ones: list[str] = []
    for rama in turn.branches:
        if rama.by_default:
            break
        meets, motivo = _meets(rama, estado, flat_text)
        if meets:
            return BranchDecision(
                turn=turn.id,
                rama=rama.id,
                linea=rama.linea,
                variant=turn.variant_of(rama.linea),
                by_default=False,
                motivo=motivo,
                evidence=estado.evidence(),
                discarded_ones=tuple(discarded_ones),
            )
        discarded_ones.append(f"{rama.id}: {motivo}")

    byDefault = turn.branches[-1]
    return BranchDecision(
        turn=turn.id,
        rama=byDefault.id,
        linea=byDefault.linea,
        variant=turn.variant_of(byDefault.linea),
        by_default=True,
        motivo="ninguna rama se cumplió",
        evidence=estado.evidence(),
        discarded_ones=tuple(discarded_ones),
    )


def decision_block(
    decisions: Sequence[BranchDecision], estado: NegotiationState
) -> dict[str, Any]:
    """Lo que va al JSON: qué camino se actuó y con qué evidencia."""
    return {
        "what_it_is": (
            "el camino que el arnés actuó dentro del árbol del escenario. Dos "
            "corridas del mismo escenario pueden recorrer ramas distintas, y "
            "comparar su WER o su resultado sin mirar esto compara dos guiones"
        ),
        "branches": [d.as_json() for d in decisions],
        "camino": [f"{d.turn}:{d.rama}" for d in decisions],
        "by_default": sum(1 for d in decisions if d.by_default),
        "what_was_heard": estado.as_json(),
    }


def branching_turns(turns: Iterable[ScriptTurn]) -> tuple[str, ...]:
    return tuple(t.id for t in turns if t.branches_out)
