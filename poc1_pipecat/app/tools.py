"""Tool-calling contract between the LLM and the negotiation engine.

Two layers, on purpose:

  - `execute_*(session, parsed_args) -> dict`: the logic, with nothing from
    Pipecat. It can run in a test or in the smoke test without building a
    pipeline or fabricating framework objects.
  - `register_functions(llm, ...)`: the 6-line adapter that hooks those
    functions into Pipecat. It is the only code in this file that knows
    Pipecat exists.

The return payloads are minimal and deliberate: they travel into the LLM's
context, so they cannot contain the maximum price, the distance to the
maximum, or the reason an offer was rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final

from shared.negotiation_engine import (
    CloseReason,
    NegotiationSession,
    compensation_ids,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.services.llm_service import LLMService

logger = logging.getLogger(__name__)

PROPOSE_PRICE: Final[str] = "proponer_precio"
OFFER_COMPENSATION: Final[str] = "ofrecer_compensacion_no_monetaria"
RECORD_AGREEMENT: Final[str] = "registrar_acuerdo"
END_WITHOUT_AGREEMENT: Final[str] = "finalizar_sin_acuerdo"

FUNCTION_NAMES: Final[tuple[str, ...]] = (
    PROPOSE_PRICE,
    OFFER_COMPENSATION,
    RECORD_AGREEMENT,
    END_WITHOUT_AGREEMENT,
)

Handler = Callable[[NegotiationSession, dict[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------
# Function logic (no Pipecat)
# --------------------------------------------------------------------------


def execute_propose_price(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Validate an amount before the agent speaks it.

    Returns exactly `{"resultado": "aceptado"}` or
    `{"resultado": "rechazado"}`. Nothing else: any extra field would be a
    hint about where the ceiling is.
    """
    verdict = session.proponer(parsed_args.get("monto"))
    return {"resultado": verdict}


def execute_offer_compensation(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Record a non-monetary compensation from the catalog.

    By returning the catalog description, the agent says the authorized
    wording out loud instead of making up the scope of what it promises.
    """
    tipo = parsed_args.get("tipo")
    compensation = session.offer_compensation(tipo)
    if compensation is None:
        return {
            "resultado": "no_disponible",
            "compensaciones_validas": list(
                compensation_ids(session.compensations_path)
            ),
        }
    return {"resultado": "ofrecida", "descripcion": compensation.descripcion}


def _normalize_conditions(valor: Any) -> list[str]:
    """Accept whatever the LLM sends and turn it into a list of clean strings."""
    if valor is None:
        return []
    if isinstance(valor, str):
        raw_ones: list[Any] = [valor]
    elif isinstance(valor, (list, tuple)):
        raw_ones = list(valor)
    else:
        raw_ones = [valor]
    return [str(c).strip() for c in raw_ones if str(c).strip()]


def execute_register_agreement(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Close the negotiation with an agreement.

    The prompt requires having repeated the amount and the conditions out loud
    and having explicit confirmation from the carrier before getting here.
    """
    monto_final = parsed_args.get("monto_final")
    condiciones = _normalize_conditions(parsed_args.get("condiciones"))
    try:
        resultado = session.close_with_agreement(monto_final, condiciones)
    except ValueError:
        # An unreadable amount must not bring the call down: the error is
        # returned to the LLM so it retries with the amount it already validated.
        logger.warning(
            "registrar_acuerdo: monto_final ilegible en los argumentos del LLM "
            "(call_id=%r)",
            session.call_id,
        )
        return {
            "resultado": "error",
            "detail": "monto_final no es un monto válido, vuelve a enviarlo",
        }
    return {"resultado": "acuerdo_registrado", "monto_final": resultado.monto_final}


def execute_end_without_agreement(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Close the negotiation without an agreement."""
    motivo = parsed_args.get("motivo")
    try:
        resultado = session.close_without_agreement(motivo)  # type: ignore[arg-type]
    except ValueError:
        resultado = session.close_without_agreement(CloseReason.OTHER)
    return {"resultado": "cierre_registrado", "motivo": resultado.motivo}


HANDLERS: Final[dict[str, Handler]] = {
    PROPOSE_PRICE: execute_propose_price,
    OFFER_COMPENSATION: execute_offer_compensation,
    RECORD_AGREEMENT: execute_register_agreement,
    END_WITHOUT_AGREEMENT: execute_end_without_agreement,
}


# --------------------------------------------------------------------------
# Schemas the LLM sees
# --------------------------------------------------------------------------


def build_schemas(
    compensations_path: Path | str | None = None,
) -> list[FunctionSchema]:
    """The 4 negotiation `FunctionSchema`s.

    The `enum` for `tipo` comes from `compensaciones.yaml` and the one for
    `motivo` from `CloseReason`: neither is written by hand here, so the
    catalog and the contract with the LLM cannot drift apart.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    return [
        FunctionSchema(
            name=PROPOSE_PRICE,
            description=(
                "Valida un monto ANTES de pronunciarlo en voz alta. Responde "
                '"aceptado" o "rechazado". Si responde "rechazado", no digas '
                "ese monto: ajusta a la baja y vuelve a validar. Nunca "
                "menciones un monto que no haya sido aceptado por esta función."
            ),
            properties={
                "monto": {
                    "type": "number",
                    "description": (
                        "Monto en pesos mexicanos que quieres ofrecer al "
                        "transportista por el viaje completo."
                    ),
                }
            },
            required=["monto"],
        ),
        FunctionSchema(
            name=OFFER_COMPENSATION,
            description=(
                "Registra que ofreces una compensación no monetaria del "
                "catálogo autorizado. Responde con la redacción exacta que "
                "debes usar al decirla en voz alta. Ofrece compensaciones "
                "antes de subir el precio."
            ),
            properties={
                "tipo": {
                    "type": "string",
                    "enum": list(compensation_ids(compensations_path)),
                    "description": (
                        "Identificador de la compensación del catálogo. No "
                        "inventes valores fuera de esta lista."
                    ),
                }
            },
            required=["tipo"],
        ),
        FunctionSchema(
            name=RECORD_AGREEMENT,
            description=(
                "Cierra la negociación con acuerdo. Llámala solo después de "
                "haber repetido en voz alta el monto final y las condiciones, "
                "y de que el transportista lo haya confirmado explícitamente."
            ),
            properties={
                "monto_final": {
                    "type": "number",
                    "description": (
                        "Monto acordado en pesos mexicanos. Debe ser uno que "
                        "proponer_precio ya haya aceptado."
                    ),
                },
                "condiciones": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Condiciones acordadas, en las palabras en que se "
                        "pactaron: compensaciones incluidas, fechas, tiempos "
                        "de espera."
                    ),
                },
            },
            required=["monto_final", "condiciones"],
        ),
        FunctionSchema(
            name=END_WITHOUT_AGREEMENT,
            description=(
                "Cierra la negociación sin acuerdo, después de despedirte "
                "cordialmente."
            ),
            properties={
                "motivo": {
                    "type": "string",
                    "enum": [m.value for m in CloseReason],
                    "description": (
                        "Por qué no hubo acuerdo: sin_respuesta si dejó de "
                        "contestar, rechazo_definitivo si rechazó de forma "
                        "tajante, sin_acuerdo_tras_contraofertas si se agotaron "
                        "las contraofertas, colgado si cortó la llamada, "
                        "falla_tecnica si hubo un problema de audio, otro en "
                        "cualquier otro caso."
                    ),
                }
            },
            required=["motivo"],
        ),
    ]


# --------------------------------------------------------------------------
# Pipecat adapter
# --------------------------------------------------------------------------


def register_functions(llm: LLMService) -> None:
    """Hook the handlers into Pipecat's LLM service.

    The session travels through the `PipelineTask`'s `app_resources`, which
    Pipecat delivers in every `FunctionCallParams`: that way the handlers need
    no global state and each call has its own session.
    """
    from pipecat.services.llm_service import FunctionCallParams

    def _adapt(name: str, handler_fn: Handler):
        async def handler(params: FunctionCallParams) -> None:
            session = params.app_resources
            if not isinstance(session, NegotiationSession):
                raise RuntimeError(
                    "el PipelineTask no recibió una SesionNegociacion en "
                    "app_resources: sin ella no hay forma de validar montos"
                )

            response = handler_fn(session, dict(params.arguments or {}))
            logger.info(
                "tool %s argumentos=%r -> %r", name, params.arguments, response
            )
            await params.result_callback(response)

            closure = session.take_pending_close()
            if closure is not None and session.close_callback is not None:
                await session.close_callback(closure)

        return handler

    for name, handler_fn in HANDLERS.items():
        llm.register_function(name, _adapt(name, handler_fn))
