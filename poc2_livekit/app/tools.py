"""The 4 negotiation functions, in LiveKit's tool-calling contract.

The logic does not change between PoCs: the `execute_*` functions here are the
same as in `poc1_pipecat/app/tools.py` and delegate entirely to
`shared/negotiation_engine/`. The only framework-specific part is how they are
registered.

## Why raw schemas and not `@function_tool` with annotations

LiveKit can derive the JSON schema from the function signature (types and
docstring). That would be more idiomatic, but it would produce a schema
**different** from the one the LLM sees in PoC1, and tool calling is one of the
things the comparison measures: turns with a function call cost considerably
more than those without one (CLAUDE.md).

With `raw_schema` the LLM receives in PoC2 **exactly the same JSON** as in
PoC1 —same names, same descriptions, same enums taken from the catalog— so the
latency difference that gets measured comes from the framework and not from
how a function's description happened to be worded.

A tool with `raw_schema` receives the unvalidated arguments in a parameter that
**must be named `raw_arguments`**; LiveKit injects it by name. This also
replicates PoC1's behavior, where the handlers receive the LLM's raw dict and
are the ones that validate it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Final

from shared.negotiation_engine import (
    CloseReason,
    NegotiationSession,
    compensation_ids,
)

logger = logging.getLogger(__name__)

# Names of the 4 functions. They are part of the contract with system prompt v0:
# the prompt names them in its text, so renaming them here breaks the prompt.
PROPOSE_PRICE: Final[str] = "proponer_precio"
OFFER_COMPENSATION: Final[str] = "ofrecer_compensacion_no_monetaria"
RECORD_AGREEMENT: Final[str] = "registrar_acuerdo"
END_WITHOUT_AGREEMENT: Final[str] = "finalizar_sin_acuerdo"

Handler = Callable[[NegotiationSession, dict[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------
# Function logic (without LiveKit)
# --------------------------------------------------------------------------


def execute_propose_price(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Validates an amount before the agent says it out loud.

    Returns exactly `{"resultado": "aceptado"}` or
    `{"resultado": "rechazado"}`. Nothing else: any extra field would be a
    hint about where the ceiling is.
    """
    verdict = session.proponer(parsed_args.get("monto"))
    return {"resultado": verdict}


def execute_offer_compensation(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Records a non-monetary compensation from the catalog.

    By returning the catalog description, the agent says the authorized
    wording out loud instead of inventing the scope of what it promises.
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
    """Accepts whatever the LLM sends and turns it into a list of clean strings."""
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
    """Closes the negotiation with an agreement.

    The prompt requires having repeated the amount and the conditions out loud
    and having explicit confirmation from the carrier before getting here.
    """
    monto_final = parsed_args.get("monto_final")
    condiciones = _normalize_conditions(parsed_args.get("condiciones"))
    try:
        resultado = session.close_with_agreement(monto_final, condiciones)
    except ValueError:
        # An unreadable amount must not take down the call: the error is
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
    """Closes the negotiation without an agreement."""
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

# Closing the negotiation has to hang up the call. It is flagged here and not
# inside the handler so that the handlers remain pure logic.
CLOSING_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {RECORD_AGREEMENT, END_WITHOUT_AGREEMENT}
)


# --------------------------------------------------------------------------
# Schemas the LLM sees — same as PoC1, word for word
# --------------------------------------------------------------------------


def build_schemas(
    compensations_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """The 4 raw schemas, in OpenAI's function-calling format.

    The `enum` of `tipo` comes from `compensaciones.yaml` and that of `motivo`
    from `CloseReason`: neither is written by hand here, so that the catalog
    and the contract with the LLM cannot drift apart.

    The descriptions are **identical** to those in `poc1_pipecat/app/tools.py`.
    If they change in one PoC they must change in the other, or the LLM stops
    seeing the same contract and the tool-calling comparison loses its meaning.
    """
    return [
        {
            "name": PROPOSE_PRICE,
            "description": (
                "Valida un monto ANTES de pronunciarlo en voz alta. Responde "
                '"aceptado" o "rechazado". Si responde "rechazado", no digas '
                "ese monto: ajusta a la baja y vuelve a validar. Nunca "
                "menciones un monto que no haya sido aceptado por esta función."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "monto": {
                        "type": "number",
                        "description": (
                            "Monto en pesos mexicanos que quieres ofrecer al "
                            "transportista por el viaje completo."
                        ),
                    }
                },
                "required": ["monto"],
            },
        },
        {
            "name": OFFER_COMPENSATION,
            "description": (
                "Registra que ofreces una compensación no monetaria del "
                "catálogo autorizado. Responde con la redacción exacta que "
                "debes usar al decirla en voz alta. Ofrece compensaciones "
                "antes de subir el precio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "enum": list(compensation_ids(compensations_path)),
                        "description": (
                            "Identificador de la compensación del catálogo. No "
                            "inventes valores fuera de esta lista."
                        ),
                    }
                },
                "required": ["tipo"],
            },
        },
        {
            "name": RECORD_AGREEMENT,
            "description": (
                "Cierra la negociación con acuerdo. Llámala solo después de "
                "haber repetido en voz alta el monto final y las condiciones, "
                "y de que el transportista lo haya confirmado explícitamente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["monto_final", "condiciones"],
            },
        },
        {
            "name": END_WITHOUT_AGREEMENT,
            "description": (
                "Cierra la negociación sin acuerdo, después de despedirte "
                "cordialmente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["motivo"],
            },
        },
    ]


# --------------------------------------------------------------------------
# LiveKit adapter
# --------------------------------------------------------------------------


def build_tools(
    session: NegotiationSession,
    *,
    compensations_path: Path | str | None = None,
) -> list[Any]:
    """The 4 functions, ready to pass to `Agent(tools=...)`.

    Each one is bound to **this** session through a lexical closure, which is
    the equivalent of what PoC1 achieves with the `PipelineTask`'s
    `app_resources`: no global state and one session per call.

    Closing the negotiation fires `session.close_callback`, which is what hangs
    up. It is done here and not inside the handler so that the `execute_*`
    functions remain pure, testable without LiveKit.
    """
    from livekit.agents import llm as lk_llm

    tools: list[Any] = []

    for schema in build_schemas(compensations_path):
        name = schema["name"]
        handler_fn = HANDLERS[name]

        def _create(name: str, handler_fn: Handler, schema: dict[str, Any]) -> Any:
            async def tool(raw_arguments: dict[str, object]) -> dict[str, Any]:
                parsed_args = dict(raw_arguments or {})
                response = handler_fn(session, parsed_args)
                logger.info("tool %s argumentos=%r -> %r", name, parsed_args, response)

                if name in CLOSING_FUNCTIONS:
                    pending = session.take_pending_close()
                    if pending is not None and session.close_callback is not None:
                        await session.close_callback(pending)

                return response

            # The Python function name is not the one the LLM sees (that one
            # comes from the raw_schema), but it is the one that shows up in the
            # logs and in LiveKit's traces.
            tool.__name__ = name
            return lk_llm.function_tool(tool, raw_schema=schema)

        tools.append(_create(name, handler_fn, schema))

    return tools


def tool_names(tools: list[Any]) -> list[str]:
    """The names the LLM will see. For checking it in the smoke test."""
    names: list[str] = []
    for tool in tools:
        info = getattr(tool, "info", None)
        name = getattr(info, "name", None)
        if name:
            names.append(str(name))
    return names
