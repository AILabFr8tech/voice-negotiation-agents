"""Mutable state of a negotiation call.

`shared/negotiation_engine/` is pure logic: its functions receive the offer
history and the financial configuration as arguments and remember nothing
between calls. The LLM, on the other hand, only sends `monto` or `tipo`. This
class is the bridge: it keeps what the engine needs to receive and accumulates
what has to be persisted at the end.

It reimplements no negotiation rule: each method delegates to the engine.

It lives in `shared/` because all 3 PoCs need exactly this state and no line
here knows the voice framework: whoever wraps it sets the `close_callback` and
decides how to hang up.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Imported from the sibling modules and NOT from the package: `__init__.py`
# exports `NegotiationSession`, so importing from there would close an import cycle.
from shared.negotiation_engine.compensations import (
    Compensation,
    get_compensation,
)
from shared.negotiation_engine.outcomes import (
    CloseReason,
    NegotiationResult,
    finalizar_sin_acuerdo,
    registrar_acuerdo,
)
from shared.negotiation_engine.pricing import ACCEPTED, proponer_precio
from shared.negotiation_engine.trip import TripData

logger = logging.getLogger(__name__)

# Same tolerance `outcomes` uses when comparing amounts: one centavo.
_AMOUNT_TOLERANCE = 0.01


@dataclass
class NegotiationSession:
    """State of a negotiation in progress. One instance per call."""

    trip: TripData
    call_id: str | None = None
    # Alternative path to the compensation catalog (the tests use it).
    compensations_path: Path | str | None = None

    # Only amounts `proponer_precio` accepted, in chronological order. It is
    # what the engine uses for the decreasing-concessions rule and what makes
    # it possible to verify, on close, that the agreed amount came from a
    # validated offer.
    offer_history: list[float] = field(default_factory=list)
    offered_compensations: list[str] = field(default_factory=list)
    resultado: NegotiationResult | None = None

    # Each PoC sets it to learn about the close and hang up. In the tests and
    # in the smoke check it stays None.
    close_callback: Callable[[NegotiationResult], Awaitable[None]] | None = None
    _closure_notified: bool = field(default=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self.resultado is not None

    def _already_recorded(self, monto: float) -> bool:
        """True if that amount is already in the history (within one centavo).

        Avoids inflating the counteroffer count when the agent revalidates the
        same amount to repeat it at the close (step 6 of the prompt).
        """
        return any(
            math.isclose(float(monto), previous, rel_tol=0.0, abs_tol=_AMOUNT_TOLERANCE)
            for previous in self.offer_history
        )

    def proponer(self, monto: object) -> str:
        """Validate an amount against the engine. Returns "aceptado" or "rechazado".

        If the engine accepts it, the amount enters the history: from then on
        the agent may speak it, and the following validations are measured
        against it.
        """
        verdict = proponer_precio(
            monto,  # type: ignore[arg-type]  # the engine validates the type and rejects
            self.trip.target_price,
            self.trip.max_price,
            self.offer_history,
        )
        if verdict == ACCEPTED and not self._already_recorded(float(monto)):  # type: ignore[arg-type]
            self.offer_history.append(float(monto))  # type: ignore[arg-type]
        return verdict

    def offer_compensation(self, tipo: object) -> Compensation | None:
        """Record a compensation from the catalog. None if `tipo` does not exist.

        Returning the `Compensation` gives the agent the authorized wording to
        say it aloud, instead of letting it invent the scope of what it
        promises.
        """
        if not isinstance(tipo, str):
            return None
        compensation = get_compensation(tipo, self.compensations_path)
        if compensation is None:
            logger.warning(
                "offer_compensation: tipo %r is not in the catalog (call_id=%r)",
                tipo,
                self.call_id,
            )
            return None
        if compensation.id not in self.offered_compensations:
            self.offered_compensations.append(compensation.id)
        return compensation

    def close_with_agreement(
        self, monto_final: object, condiciones: str | Sequence[str]
    ) -> NegotiationResult:
        """Close with agreement and store the record in the session."""
        if self.is_closed:
            return self._closure_repeated()

        self.resultado = registrar_acuerdo(
            monto_final,  # type: ignore[arg-type]  # the engine validates and raises
            condiciones,
            offer_history=self.offer_history,
            offered_compensations=self.offered_compensations,
            call_id=self.call_id,
        )
        return self.resultado

    def close_without_agreement(self, motivo: str | CloseReason) -> NegotiationResult:
        """Close without agreement and store the record in the session."""
        if self.is_closed:
            return self._closure_repeated()

        self.resultado = finalizar_sin_acuerdo(
            motivo,
            offer_history=self.offer_history,
            offered_compensations=self.offered_compensations,
            call_id=self.call_id,
        )
        return self.resultado

    def _closure_repeated(self) -> NegotiationResult:
        """The LLM asked to close twice: the first record is kept.

        Overwriting the result would leave the closing metric at the mercy of a
        spurious second tool call.
        """
        assert self.resultado is not None
        logger.warning(
            "repeated close ignored: the session was already closed as %r "
            "(call_id=%r)",
            self.resultado.resultado,
            self.call_id,
        )
        return self.resultado

    def take_pending_close(self) -> NegotiationResult | None:
        """Return the result only once, to trigger the hang-up.

        The second call returns None: that way the wrapper does not try to end
        the call more than once.
        """
        if self.resultado is None or self._closure_notified:
            return None
        self._closure_notified = True
        return self.resultado

    def summary(self) -> dict[str, object]:
        """Summary for logs and persistence. Without `max_price`, on purpose."""
        return {
            "call_id": self.call_id,
            "company": self.trip.company,
            "origin": self.trip.origin,
            "destination": self.trip.destination,
            "cargo_type": self.trip.cargo_type,
            "target_price": self.trip.target_price,
            "offer_history": list(self.offer_history),
            "offered_compensations": list(self.offered_compensations),
            "resultado": self.resultado.to_dict() if self.resultado else None,
        }
