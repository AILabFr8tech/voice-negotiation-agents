"""Guion de pruebas: las líneas del actor, los clips grabados y el árbol.

`escenarios.yaml` es la fuente única de las líneas y de las ramas; `clips/` son
las grabaciones ya preparadas con su manifiesto; `decision.py` es el actor
automático que elige la rama en los escenarios que no son lineales.
"""

from __future__ import annotations

from shared.test_scenarios.decision import (
    BranchDecision,
    NegotiationState,
    decision_block,
    decide,
)

__all__ = [
    "BranchDecision",
    "NegotiationState",
    "decision_block",
    "decide",
]
