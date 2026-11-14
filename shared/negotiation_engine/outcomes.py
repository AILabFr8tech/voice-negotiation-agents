"""Data contract for closing the negotiation. No persistence yet.

`registrar_acuerdo` and `finalizar_sin_acuerdo` build a serializable record
and write it to the log. They touch neither disk nor database: real
persistence (transcript, history, metrics) comes with `shared/metrics/`. What
is fixed here is the shape of the data, so that the 4 PoCs produce results
comparable with each other.

`max_price` appears in no field of the record, on purpose.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Final, Sequence, Union

logger = logging.getLogger(__name__)

Amount = Union[int, float]

AGREEMENT_RESULT: Final[str] = "acuerdo"
NO_AGREEMENT_RESULT: Final[str] = "sin_acuerdo"

# Tolerance when comparing amounts: one centavo. Prevents a rounding by the LLM
# or by serialization from flagging as unvalidated an amount that was validated.
_AMOUNT_TOLERANCE: Final[float] = 0.01


class CloseReason(str, Enum):
    """Categories of closing without agreement, for aggregating metrics across PoCs."""

    NO_RESPONSE = "sin_respuesta"
    FINAL_REJECTION = "rechazo_definitivo"
    NO_AGREEMENT_AFTER_COUNTEROFFERS = "sin_acuerdo_tras_contraofertas"
    HUNG_UP = "colgado"
    TECHNICAL_FAILURE = "falla_tecnica"
    OTHER = "otro"


@dataclass(frozen=True)
class NegotiationResult:
    """Final record of a call. Serializable to JSON as is."""

    resultado: str
    timestamp: str
    monto_final: float | None = None
    condiciones: tuple[str, ...] = ()
    offered_compensations: tuple[str, ...] = ()
    offer_history: tuple[float, ...] = ()
    counteroffers: int = 0
    motivo: str | None = None
    reason_category: str | None = None
    # False if the agreed amount matches no offer that `proponer_precio`
    # validated: a sign that the agent spoke a number on its own. It must
    # always be True in a healthy run.
    validated_final_amount: bool | None = None
    call_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Dump the record to a flat dict, ready for JSON."""
        return asdict(self)


def _now_iso(timestamp: datetime | None) -> str:
    """ISO 8601 in UTC. `timestamp` is injected to make tests deterministic."""
    moment = timestamp or datetime.now(timezone.utc)
    return moment.isoformat()


def _normalize_conditions(condiciones: str | Sequence[str]) -> tuple[str, ...]:
    """Accept a single condition or a list; return a clean tuple."""
    if isinstance(condiciones, str):
        raw_ones: Sequence[str] = [condiciones]
    else:
        raw_ones = list(condiciones)

    clean_items = []
    for condition in raw_ones:
        if not isinstance(condition, str):
            raise ValueError("each agreed condition must be text")
        if condition.strip():
            clean_items.append(condition.strip())
    return tuple(clean_items)


def _normalize_amounts(offer_history: Sequence[Amount]) -> tuple[float, ...]:
    return tuple(float(m) for m in offer_history)


def _amount_is_in_history(
    monto_final: float, offer_history: Sequence[float]
) -> bool:
    return any(
        math.isclose(monto_final, offer, rel_tol=0.0, abs_tol=_AMOUNT_TOLERANCE)
        for offer in offer_history
    )


def registrar_acuerdo(
    monto_final: Amount,
    condiciones: str | Sequence[str],
    *,
    offer_history: Sequence[Amount] = (),
    offered_compensations: Sequence[str] = (),
    call_id: str | None = None,
    timestamp: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> NegotiationResult:
    """Close the negotiation with an agreement and return the record.

    `offer_history` holds the amounts `proponer_precio` already accepted. It is
    used to check that the agreed amount came from a validated offer: if it
    matches none, `validated_final_amount=False` is set and a warning is
    emitted. No exception is raised: aborting here would drop a call already
    closed with the carrier; the case is caught in the metrics review, where
    compliance with the financial limit is a binary that must come out at 100%.
    """
    if isinstance(monto_final, bool) or not isinstance(monto_final, (int, float)):
        raise ValueError("monto_final must be a number")
    if not math.isfinite(monto_final) or monto_final <= 0:
        raise ValueError("monto_final must be finite and positive")

    historial = _normalize_amounts(offer_history)
    validated = _amount_is_in_history(float(monto_final), historial)
    if not validated:
        logger.warning(
            "registrar_acuerdo: monto_final=%r does not match any offer "
            "validated by proponer_precio (call_id=%r)",
            monto_final,
            call_id,
        )

    record = NegotiationResult(
        resultado=AGREEMENT_RESULT,
        timestamp=_now_iso(timestamp),
        monto_final=float(monto_final),
        condiciones=_normalize_conditions(condiciones),
        offered_compensations=tuple(offered_compensations),
        offer_history=historial,
        counteroffers=max(len(historial) - 1, 0),
        validated_final_amount=validated,
        call_id=call_id,
        metadata=dict(metadata or {}),
    )
    logger.info("registrar_acuerdo: %s", record.to_dict())
    return record


def finalizar_sin_acuerdo(
    motivo: str | CloseReason,
    *,
    offer_history: Sequence[Amount] = (),
    offered_compensations: Sequence[str] = (),
    call_id: str | None = None,
    timestamp: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> NegotiationResult:
    """Close the negotiation without agreement and return the record.

    `motivo` is kept verbatim and also classified into `CloseReason` so it can
    be aggregated across PoCs. A free-text reason that does not fit the enum is
    classified as `otro` without losing the original text.
    """
    if isinstance(motivo, CloseReason):
        text = motivo.value
        category = motivo
    elif isinstance(motivo, str) and motivo.strip():
        text = motivo.strip()
        try:
            category = CloseReason(text)
        except ValueError:
            category = CloseReason.OTHER
    else:
        raise ValueError("motivo must be non-empty text or a CloseReason")

    historial = _normalize_amounts(offer_history)
    record = NegotiationResult(
        resultado=NO_AGREEMENT_RESULT,
        timestamp=_now_iso(timestamp),
        condiciones=(),
        offered_compensations=tuple(offered_compensations),
        offer_history=historial,
        counteroffers=max(len(historial) - 1, 0),
        motivo=text,
        reason_category=category.value,
        call_id=call_id,
        metadata=dict(metadata or {}),
    )
    logger.info("finalizar_sin_acuerdo: %s", record.to_dict())
    return record
