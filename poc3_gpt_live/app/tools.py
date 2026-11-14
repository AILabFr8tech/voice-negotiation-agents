"""The 4 negotiation functions, in GPT-Live's tool-calling contract.

The logic does not change between PoCs: the `ejecutar_*` functions here are the same as in
`poc1_pipecat/app/tools.py` and `poc2_livekit/app/tools.py`, and they delegate entirely
to `shared/negotiation_engine/`. The only framework-specific parts are how the
schemas are declared and how the result is returned.

(They are copied rather than imported from another PoC on purpose: one PoC cannot depend
on another, or one's environment would drag in the other's. The real logic lives
in `shared/`, which is what the three have in common.)

## Why `responses` delegation and not `client`

The alpha offers two delegation modes, and only one allows tool calling:

- **`client`** — the model sends a unit of work as **free text**
  (`delegation.created` with `content: [{type: "input_text", ...}]`) and the client
  replies with text via `delegation.context.append`. There is no `call_id`, no
  structured arguments, no schemas to declare. With that **the non-negotiable rule
  in CLAUDE.md cannot be enforced**: the agent never says an
  amount without having called `proponer_precio` and received "aceptado". Free
  text is not a function call and there is no way to require it.
- **`responses`** — the model delegates to a Responses API model, which does
  emit `response.function_call_arguments.done` with `call_id`, `name` and
  `arguments` as JSON, and which is answered with
  `delegation.function_call_output.create`. It is the only documented path that
  preserves the tool-calling contract the comparison measures.

`responses` is chosen. The delegation model is the same `gpt-4.1-mini`
frozen for the 3 PoCs (CLAUDE.md): that way the 4 functions
run against the same model in all three architectures and any measured
difference comes from the voice framework.

Cost consequence, and it must be stated: `docs_alpha/` says that traffic to the
`gpt-live` model is not billed but **delegated calls to the text model
are**, at list price. So in PoC3 the billed line item is exactly the
tool-calling one. See `docs/costos_poc3.md`.

## Differences in shape from PoC1 and PoC2

The schema the model sees is **the same JSON**: same names, same
descriptions word for word, same enums taken from the real catalog. The
only thing that changes is the wrapper, because the Responses API declares tools
flat and with `type: "function"` instead of nesting `function: {...}`.

**`additionalProperties: false` is not added** even though the docs example
includes it: it is not declared as mandatory, and adding it would change the JSON
the model sees relative to PoC1 and PoC2. If the alpha started requiring it, it is
added in all three or in none.
"""

from __future__ import annotations

import json
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

FUNCTION_NAMES: Final[tuple[str, ...]] = (
    PROPOSE_PRICE,
    OFFER_COMPENSATION,
    RECORD_AGREEMENT,
    END_WITHOUT_AGREEMENT,
)

Handler = Callable[[NegotiationSession, dict[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------
# Function logic (nothing alpha-specific)
# --------------------------------------------------------------------------


def execute_propose_price(
    session: NegotiationSession, parsed_args: dict[str, Any]
) -> dict[str, Any]:
    """Validate an amount before the agent says it.

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
    """Accept whatever the model sends and turn it into a list of clean strings."""
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

    The prompt requires having repeated the amount and the conditions out loud and
    having explicit confirmation from the carrier before getting here.
    """
    monto_final = parsed_args.get("monto_final")
    condiciones = _normalize_conditions(parsed_args.get("condiciones"))
    try:
        resultado = session.close_with_agreement(monto_final, condiciones)
    except ValueError:
        # An unreadable amount must not bring down the call: the error is returned to
        # the model so it retries with the amount it already validated.
        logger.warning(
            "registrar_acuerdo: monto_final ilegible en los argumentos del modelo "
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

# Closing the negotiation has to hang up the call. It is flagged here and not inside
# the handler so that the handlers remain pure logic.
CLOSING_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {RECORD_AGREEMENT, END_WITHOUT_AGREEMENT}
)


# --------------------------------------------------------------------------
# Schemas the model sees — same as PoC1 and PoC2, word for word
# --------------------------------------------------------------------------


def build_schemas(
    compensations_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """The 4 tools in Responses API format (flat `type: "function"`).

    The `enum` for `tipo` comes from `compensaciones.yaml` and the one for `motivo` from
    `MotivoCierre`: neither is hand-written here, so that the catalog and the
    contract with the model cannot drift apart.

    The descriptions are **identical** to those in PoC1 and PoC2. If they change in one
    PoC they have to change in all three, or the model stops seeing the same
    contract and the tool-calling comparison loses its meaning.
    """
    return [
        {
            "type": "function",
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
            "type": "function",
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
            "type": "function",
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
            "type": "function",
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


def build_delegation(
    *,
    model_name: str,
    compensations_path: Path | str | None = None,
) -> dict[str, Any]:
    """The complete `delegation` object of the initial `session.start`.

    The delegation is replaced **as a whole object**: nested fields are not
    patched separately (`docs_alpha/Alpha_guide.html`, "Update a live
    session"). That is why this returns the complete thing and not a fragment.

    What is pinned and why:

    - `tool_choice: "auto"` — prompt v0 already says when to call each function;
      forcing a choice would mean rewriting the negotiation strategy from the
      config, and the prompt is frozen for the 3 PoCs.
    - `parallel_tool_calls: false` — the 4 functions **are not commutative**:
      `proponer_precio` feeds the decreasing-concessions rule with the
      history, so two validations in parallel would be evaluated against the
      same state and both could pass. In PoC1 and PoC2 the problem does not exist
      because the frameworks resolve tools serially.
    - **No `web_search` or other hosted tools.** The agent negotiates with what
      the prompt gave it; searching the internet mid-call would add a
      latency source that no other PoC pays.
    - `reasoning` and `text.verbosity` are left **unset**: `gpt-4.1-mini` is not
      a reasoning model (CLAUDE.md), so there is no `effort` to turn off, and
      setting `verbosity` would change response length only in PoC3.
    """
    return {
        "type": "responses",
        "responses": {
            "model": model_name,
            "tools": build_schemas(compensations_path),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
    }


# --------------------------------------------------------------------------
# Adapter to the alpha's event protocol
# --------------------------------------------------------------------------


def parse_arguments(raw_values: Any) -> dict[str, Any]:
    """The `arguments` of `response.function_call_arguments.done`, as a dict.

    They arrive as a **JSON string**, not as an object. Parsing is lenient: a
    malformed JSON returns `{}` and lets the handler reject on its own,
    because blowing up here would kill an already-paid phone call over a model
    error the engine knows how to handle (`proponer_precio` without `monto` replies
    "rechazado", which is the correct answer).
    """
    if isinstance(raw_values, dict):
        return dict(raw_values)
    if not isinstance(raw_values, str) or not raw_values.strip():
        return {}
    try:
        data = json.loads(raw_values)
    except (TypeError, ValueError):
        logger.warning("argumentos de función ilegibles, se tratan como vacíos")
        return {}
    return dict(data) if isinstance(data, dict) else {}


def execute_function(
    session: NegotiationSession, name: str, raw_args: Any
) -> dict[str, Any]:
    """Run the function the model asked for and return its response payload.

    An unknown function does not blow up: the model is told it does not exist,
    along with the list of those that do. It is the same criterion as a compensation
    outside the catalog — the model can correct itself, and an exception here would
    hang up the call.
    """
    handler_fn = HANDLERS.get(name)
    parsed_args = parse_arguments(raw_args)

    if handler_fn is None:
        logger.warning("el modelo llamó a una función que no existe: %r", name)
        return {
            "resultado": "unknown_function",
            "valid_functions": list(FUNCTION_NAMES),
        }

    response = handler_fn(session, parsed_args)
    logger.info("tool %s argumentos=%r -> %r", name, parsed_args, response)
    return response


def build_output_event(
    call_id: str, response: dict[str, Any], *, event_id: str | None = None
) -> dict[str, Any]:
    """The `response.item.create` event carrying the output for one `call_id`.

    `output` travels as a **JSON string**, not as an object. One is sent per
    `call_id`; unknown or already-resolved IDs are rejected on the
    server side, so the caller must not retry the same one. Queuing it does not
    resume the delegated model: that is `response.create`, which the bridge sends
    once all outputs of the response are in
    (`bridge.SeguimientoDeRespuestas`). With the alpha protocol this event
    was `delegation.function_call_output.create`.
    """
    event: dict[str, Any] = {
        "type": "response.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(response, ensure_ascii=False),
        },
    }
    if event_id:
        event["event_id"] = event_id
    return event
