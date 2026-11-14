"""Price validation for the negotiation. Pure logic: no I/O, no LLM.

NON-NEGOTIABLE RULE (CLAUDE.md): `max_price` does not leave this module.
`proponer_precio` returns *only* the string "aceptado" or "rechazado".
The rejection reason goes to the server log, never to the return value,
because that value does travel into the LLM context and any extra detail
("exceeds the maximum", "you went over by 200") would leak the existence of
the ceiling.

Amounts are in MXN pesos. Integers or decimals are accepted.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Final, Literal, Sequence, Union

logger = logging.getLogger(__name__)

ACCEPTED: Final[str] = "aceptado"
RECHAZADO: Final[str] = "rechazado"

Verdict = Literal["aceptado", "rechazado"]
Amount = Union[int, float]


class RejectionReason(str, Enum):
    """Internal rejection reasons. Only for logging and metrics.

    NEVER returned to the caller nor injected into the LLM context.
    """

    INVALID_AMOUNT = "invalid_amount"
    EXCEEDS_MAX_PRICE = "exceeds_max_price"
    NON_DECREASING_CONCESSION = "non_decreasing_concession"


def _is_valid_amount(valor: object) -> bool:
    """True if `valor` is a numeric, finite, positive amount."""
    # bool is a subclass of int in Python; a True slipped in as an amount is a bug.
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return False
    return math.isfinite(valor) and valor > 0


def _normalize_history(offer_history: Sequence[Amount]) -> list[float]:
    """Validate the history and return it as a list of floats.

    The history is the amounts the agent has already spoken aloud, in
    chronological order. An invalid value here is a caller bug, not a
    negotiation decision: ValueError is raised so it does not go unnoticed.
    """
    normalized: list[float] = []
    for offer in offer_history:
        if not _is_valid_amount(offer):
            raise ValueError(
                "offer_history contains a value that is not a valid amount"
            )
        normalized.append(float(offer))
    return normalized


def _validate_config(target_price: Amount, max_price: Amount) -> None:
    """Validate the financial configuration inherited by the negotiation.

    The error messages do not include the amounts: an exception that bubbles
    up through the tool-calling layer may end up serialized to the LLM.
    """
    if not _is_valid_amount(target_price):
        raise ValueError("target_price must be a finite, positive number")
    if not _is_valid_amount(max_price):
        raise ValueError("max_price must be a finite, positive number")
    if max_price < target_price:
        raise ValueError(
            "invalid configuration: the maximum price cannot be "
            "below the target price"
        )


def last_increment(offer_history: Sequence[Amount]) -> float | None:
    """Last positive increment between two consecutive offers in the history.

    Returns None if the history has fewer than two offers or if there was
    never an increment (for example, if the agent repeated the same amount to
    confirm it, which step 6 of the prompt requires before closing).
    """
    historial = _normalize_history(offer_history)
    for i in range(len(historial) - 1, 0, -1):
        delta = historial[i] - historial[i - 1]
        if delta > 0:
            return delta
    return None


def validate_decreasing_concession(
    monto: Amount, offer_history: Sequence[Amount]
) -> bool:
    """True if `monto` respects the decreasing-concessions rule.

    Each increase must be strictly smaller than the previous increase: less is
    conceded each time, so as not to signal that there is room left. Cases:

    - Empty history: the first offer has no previous increment to compare
      against -> valid.
    - `monto` less than or equal to the highest offer already spoken: it is
      not a new concession, it is a repeat or a decrease. Valid: repeating the
      amount aloud is part of the close (step 6 of the prompt), and going down
      can never break the financial ceiling.
    - `monto` above the highest offer already spoken: the increment is
      measured against that highest offer (not against the last one said, so
      that going down and back up does not reset the concession margin) and
      must be strictly smaller than the last positive increment in the history.
    """
    if not _is_valid_amount(monto):
        return False

    historial = _normalize_history(offer_history)
    if not historial:
        return True

    offered_ceiling = max(historial)
    if monto <= offered_ceiling:
        return True

    previous_increment = last_increment(historial)
    if previous_increment is None:
        # A single previous offer (or no increase): any first increment is
        # admissible; the financial ceiling bounds it anyway.
        return True

    return (monto - offered_ceiling) < previous_increment


def _reject(motivo: RejectionReason, monto: object) -> Verdict:
    """Log the rejection reason and return the bare verdict.

    The log includes the proposed amount but never `max_price`.
    """
    logger.info("proponer_precio: rejected monto=%r reason=%s", monto, motivo.value)
    return RECHAZADO  # type: ignore[return-value]


def proponer_precio(
    monto: Amount,
    target_price: Amount,
    max_price: Amount,
    offer_history: Sequence[Amount] = (),
) -> Verdict:
    """Validate an amount BEFORE the agent speaks it aloud.

    The only point in the system that knows `max_price`. Returns "aceptado" or
    "rechazado" and nothing else: not the ceiling, not the distance to the
    ceiling, not the rejection reason.

    `target_price` is not a hard barrier (the prompt allows exceeding it after
    exhausting the non-monetary compensations), but it is logged for the
    margin-vs-target-price metric.

    Raises ValueError only if the inherited financial configuration is
    inconsistent (an integration bug, not a negotiation decision).
    """
    _validate_config(target_price, max_price)

    if not _is_valid_amount(monto):
        return _reject(RejectionReason.INVALID_AMOUNT, monto)

    # Financial ceiling: the only rule without exceptions. It goes first so
    # that no other path can return "aceptado" above the maximum.
    if monto > max_price:
        return _reject(RejectionReason.EXCEEDS_MAX_PRICE, monto)

    if not validate_decreasing_concession(monto, offer_history):
        return _reject(RejectionReason.NON_DECREASING_CONCESSION, monto)

    if monto > target_price:
        logger.info(
            "proponer_precio: accepted monto=%r above the internal target "
            "price (negative margin vs. target)",
            monto,
        )
    else:
        logger.info("proponer_precio: accepted monto=%r", monto)

    return ACCEPTED  # type: ignore[return-value]
