"""Negotiation engine: layer 4 of the reference architecture.

Pure logic shared by the 3 PoCs. It knows nothing about telephony, STT/TTS or
whichever framework is in use: each PoC wraps these functions in the
tool-calling schemas that fit it.

Besides the stateless engine (`pricing`, `compensations`, `outcomes`), the
package carries the trip data (`trip`) and the per-call state (`session`).
All three are shared by the 3 PoCs: if each carried its own copy, the
financial configuration or the max-price guardrail could diverge between
architectures without the comparison noticing.

Contract with the LLM (see `shared/system_prompt/v0.md`):
    proponer_precio(monto)                     -> "aceptado" | "rechazado"
    ofrecer_compensacion_no_monetaria(tipo)    -> validate_compensation(tipo)
    registrar_acuerdo(monto_final, condiciones)
    finalizar_sin_acuerdo(motivo)
"""

from shared.negotiation_engine.compensations import (
    COMPENSATIONS_PATH,
    Compensation,
    load_compensations,
    compensation_ids,
    get_compensation,
    render_compensations,
    validate_compensation,
)
from shared.negotiation_engine.outcomes import (
    AGREEMENT_RESULT,
    NO_AGREEMENT_RESULT,
    CloseReason,
    NegotiationResult,
    finalizar_sin_acuerdo,
    registrar_acuerdo,
)
from shared.negotiation_engine.pricing import (
    ACCEPTED,
    RECHAZADO,
    RejectionReason,
    proponer_precio,
    last_increment,
    validate_decreasing_concession,
)
from shared.negotiation_engine.trip import (
    TEST_TRIP,
    TripData,
    Amount,
    format_amount_for_voice,
)

# `session` goes last on purpose: it imports from its siblings, and putting it
# before them would leave the import cycle at the mercy of this file's order.
from shared.negotiation_engine.session import NegotiationSession  # noqa: E402

__all__ = [
    # pricing
    "ACCEPTED",
    "RECHAZADO",
    "RejectionReason",
    "proponer_precio",
    "last_increment",
    "validate_decreasing_concession",
    # compensations
    "COMPENSATIONS_PATH",
    "Compensation",
    "load_compensations",
    "compensation_ids",
    "get_compensation",
    "render_compensations",
    "validate_compensation",
    # closing
    "AGREEMENT_RESULT",
    "NO_AGREEMENT_RESULT",
    "CloseReason",
    "NegotiationResult",
    "finalizar_sin_acuerdo",
    "registrar_acuerdo",
    # trip and call state
    "TEST_TRIP",
    "TripData",
    "Monto",
    "NegotiationSession",
    "format_amount_for_voice",
]
