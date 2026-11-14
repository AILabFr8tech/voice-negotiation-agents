"""Chequeo de humo de PoC2. Dos niveles, ninguno gasta una llamada.

    nivel 0  ->  sin red, sin costo. Verifica lo que puede fallar en silencio.
    nivel 1  ->  APIs reales, sin teléfono. LLM y tool calling de verdad.

Equivale al humo de PoC1 y verifica los mismos invariantes, que son los que
protegen las reglas no negociables de CLAUDE.md: que el precio máximo no se
filtre al prompt, que las 4 funciones salgan del catálogo real y que los
manejadores den los veredictos correctos contra el motor.

El nivel 1 aprovecha algo que PoC1 no tiene: LiveKit puede correr una
`AgentSession` **sin sala**, en modo texto. Eso ejercita el LLM y el tool
calling reales —el mismo `Agent`, el mismo prompt, las mismas herramientas que
van a correr en la llamada— sin audio, sin telefonía y por centavos.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable
from typing import Any, Final

from poc2_livekit.app.config import (
    Settings,
    CollisionWithDispatcher,
    IncompleteConfiguration,
    load_settings,
    force_utf8_output,
    verify_for_livekit,
    verify_for_telephony,
    verify_for_voice,
)
from poc2_livekit.app.tools import (
    END_WITHOUT_AGREEMENT,
    OFFER_COMPENSATION,
    PROPOSE_PRICE,
    RECORD_AGREEMENT,
    build_schemas,
    build_tools,
    tool_names,
)
from shared import llm_transport
from shared.negotiation_engine import (
    ACCEPTED,
    RECHAZADO,
    TEST_TRIP,
    TripData,
    CloseReason,
    NegotiationSession,
    load_compensations,
    compensation_ids,
)
from shared.system_prompt import (
    PROMPT_VARIABLES,
    MaxPriceLeak,
    load_template,
    render_system_prompt,
    amount_representations,
    variables_in_template,
    verify_absence_of_max_price,
)

logger = logging.getLogger(__name__)

# Estimación del nivel 1. Es texto contra gpt-4.1-mini: unas 5 respuestas con el
# prompt v0 completo (~1,500 tokens) más las herramientas. Con `--con-voz` se
# suma una síntesis corta de ElevenLabs y su transcripción en Deepgram.
LEVEL_1_COST_USD: Final[float] = 0.01
COST_WITH_VOICE_USD: Final[float] = 0.03


class SmokeFailure(AssertionError):
    """Un invariante del PoC no se cumple."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


# --------------------------------------------------------------------------
# Nivel 0 — sin red
# --------------------------------------------------------------------------


def check_config() -> str:
    settings = load_settings()
    _require(settings.openai_model, "OPENAI_MODEL vacío")
    _require(
        settings.turn_detector_version in ("v1", "v1-mini"),
        f"versión de turn detector inválida: {settings.turn_detector_version!r}",
    )
    return (
        f"config: LLM={settings.openai_model} STT={settings.deepgram_model}/"
        f"{settings.deepgram_language} TTS={settings.eleven_model} "
        f"detector={settings.turn_detector_version}"
    )


def check_models_identical_to_poc1() -> str:
    """La comparativa se cae si PoC2 corre otro modelo que PoC1."""
    settings = load_settings()
    expected_items = {
        "openai_model": "gpt-4.1-mini",
        "deepgram_model": "nova-3",
        "deepgram_language": "multi",
        "eleven_model": "eleven_flash_v2_5",
    }
    for campo, valor in expected_items.items():
        actual = getattr(settings, campo)
        _require(
            actual == valor,
            f"{campo}={actual!r} pero PoC1 corre {valor!r}: con modelos distintos "
            "la comparativa mide el modelo, no el framework",
        )
    return "modelos idénticos a PoC1 (gpt-4.1-mini / nova-3·multi / eleven_flash_v2_5)"


def check_compensation_catalog() -> str:
    compensations = load_compensations()
    _require(compensations, "el catálogo de compensaciones está vacío")
    return f"catálogo: {len(compensations)} compensaciones"


def check_prompt_template() -> str:
    template = load_template()
    found = variables_in_template(template)
    _require(
        found == set(PROMPT_VARIABLES),
        f"v0.md no tiene las 7 variables esperadas: {sorted(found)}",
    )
    return "v0.md: las 7 variables presentes"


def check_prompt_render() -> str:
    render = render_system_prompt(TEST_TRIP)
    _require(not variables_in_template(render), "quedaron placeholders sin sustituir")
    for compensation in load_compensations():
        _require(
            compensation.descripcion in render,
            f"la compensación {compensation.id!r} no llegó al prompt",
        )
    return f"prompt renderizado: {len(render)} caracteres, sin placeholders"


def check_absence_of_max_price() -> str:
    """El guardarraíl, y su control negativo.

    Sin el control negativo, un detector roto pasaría siempre.
    """
    render = render_system_prompt(TEST_TRIP)
    for shape in amount_representations(TEST_TRIP.max_price):
        _require(shape not in render, f"el precio máximo se filtró al prompt: {shape!r}")

    # Control negativo: un prompt con el máximo sembrado tiene que ser detectado.
    seeded = f"{render}\nEl tope real de esta ruta es {TEST_TRIP.max_price:,.0f}."
    try:
        verify_absence_of_max_price(seeded, TEST_TRIP)
    except MaxPriceLeak:
        return "precio máximo ausente del prompt (y el detector de fugas funciona)"
    raise SmokeFailure(
        "el detector de fugas NO detectó un precio máximo sembrado a propósito"
    )


def check_schemas() -> str:
    schemas = build_schemas()
    names = [e["name"] for e in schemas]
    expected_items = [
        PROPOSE_PRICE,
        OFFER_COMPENSATION,
        RECORD_AGREEMENT,
        END_WITHOUT_AGREEMENT,
    ]
    _require(names == expected_items, f"los 4 schemas no son los esperados: {names}")

    by_name = {e["name"]: e for e in schemas}

    enum_kind = by_name[OFFER_COMPENSATION]["parameters"]["properties"]["tipo"]["enum"]
    _require(
        enum_kind == list(compensation_ids()),
        "el enum de compensaciones no sale del catálogo",
    )

    enum_motivo = by_name[END_WITHOUT_AGREEMENT]["parameters"]["properties"]["motivo"]["enum"]
    _require(
        enum_motivo == [m.value for m in CloseReason],
        "el enum de motivos no sale de MotivoCierre",
    )

    # Ningún schema puede insinuar que existe un techo.
    text = str(schemas).lower()
    for forbidden in ("maximum", "máximo", "tope", "limite superior"):
        _require(
            forbidden not in text,
            f"un schema menciona {forbidden!r}: el LLM no debe saber que hay techo",
        )

    return f"4 schemas, enums del catálogo real ({len(enum_kind)} compensaciones)"


def check_handlers() -> str:
    """Una negociación completa contra el motor, sin LLM."""
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo")

    _require(session.proponer(16_000) == ACCEPTED, "16,000 debería aceptarse")
    _require(
        session.proponer(TEST_TRIP.max_price + 1) == RECHAZADO,
        "un monto por encima del máximo debería rechazarse",
    )
    _require(
        session.offer_history == [16_000.0],
        f"el historial solo debe traer lo aceptado: {session.offer_history}",
    )

    compensation = session.offer_compensation(next(iter(compensation_ids())))
    _require(compensation is not None, "la compensación del catálogo no se registró")
    _require(
        session.offer_compensation("does_not_exist") is None,
        "una compensación fuera del catálogo debería devolver None",
    )

    _require(session.proponer(17_000) == ACCEPTED, "17,000 debería aceptarse")
    resultado = session.close_with_agreement(17_000, ["dos horas de espera pagadas"])
    _require(resultado.validated_final_amount, "el monto final debería estar validado")
    _require(session.is_closed, "la sesión debería quedar cerrada")

    return (
        f"manejadores: historial={session.offer_history} "
        f"compensaciones={session.offered_compensations} "
        f"resultado={resultado.resultado}"
    )


def check_no_max_price_in_persistence() -> str:
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo")
    session.proponer(16_000)
    text = str(session.summary())
    for shape in amount_representations(TEST_TRIP.max_price):
        _require(
            shape not in text,
            f"el resumen que se persiste contiene el precio máximo: {shape!r}",
        )
    return "el resumen persistido no contiene el precio máximo"


def check_assembly() -> str:
    """Que la sesión de voz completa se arme: servicios, VAD y turn detector.

    Es el chequeo que atrapa un cambio de API del SDK sin gastar una llamada.
    Construir los servicios no abre ninguna conexión.
    """
    from poc2_livekit.app.agent import (
        build_agent,
        build_voice_session,
    )

    settings = load_settings()
    verify_for_voice(settings)

    session = NegotiationSession(trip=TEST_TRIP, call_id="humo")
    tools = build_tools(session)
    names = tool_names(tools)
    _require(
        names
        == [PROPOSE_PRICE, OFFER_COMPENSATION, RECORD_AGREEMENT, END_WITHOUT_AGREEMENT],
        f"las herramientas registradas no son las 4 esperadas: {names}",
    )

    agent = build_agent(session)
    _require(agent is not None, "el Agent no se construyó")

    voice_session, detector = build_voice_session(settings)
    _require(voice_session is not None, "la AgentSession no se construyó")

    model_name = getattr(detector, "model", None)
    expected = (
        "turn-detector-v1"
        if settings.turn_detector_version == "v1"
        else "turn-detector-v1-mini"
    )
    _require(
        model_name == expected,
        f"el turn detector resolvió como {model_name!r} y se pidió {expected!r}",
    )

    return (
        f"ensamblado ok: 4 herramientas, detector={model_name} "
        f"({'remoto' if model_name == 'turn-detector-v1' else 'local'})"
    )


def check_llm_policy() -> str:
    """Los timeouts del LLM son los de `shared/`, no los del plugin.

    Vale un chequeo propio porque falla en silencio: si LiveKit mueve
    `SessionConnectOptions` de módulo o le cambia los nombres a los campos, la
    sesión se arma igual y la corrida sale con los defaults del framework —
    reintentos distintos a los de PoC1 y la comparativa midiendo eso.
    """
    from poc2_livekit.app.agent import build_conn_options

    connections = build_conn_options()
    llm = connections.llm_conn_options
    _require(
        llm.max_retry == llm_transport.retries_after_first(),
        f"max_retry={llm.max_retry} y la política pide "
        f"{llm_transport.retries_after_first()}",
    )
    _require(
        llm.retry_interval == llm_transport.WAIT_BETWEEN_ATTEMPTS_S,
        f"retry_interval={llm.retry_interval} y la política pide "
        f"{llm_transport.WAIT_BETWEEN_ATTEMPTS_S}",
    )
    # STT y TTS se quedan en los defaults del framework, a propósito.
    return llm_transport.summary()


def check_failure_capture() -> str:
    """El evento `error` de la sesión llega al colector, sin gastar una llamada.

    Es instrumentación que **solo corre durante una llamada real**, así que sin
    este chequeo se descubriría rota justo cuando ya se pagó el minuto. Se le
    emite a la sesión un error sintético con la forma del `ErrorEvent` de
    LiveKit y se comprueba que aterrizó en `technical_failures`.
    """
    from poc2_livekit.app.agent import (
        CallRecord,
        connect_observers,
        build_voice_session,
    )
    from poc2_livekit.app.latency import LatencyCollector

    class _SyntheticError:
        type = "llm_error"
        recoverable = True
        error = "Request timed out."

    class _SyntheticSource:
        model = "gpt-4.1-mini"
        provider = "openai"

    class _SyntheticEvent:
        error = _SyntheticError()
        source = _SyntheticSource()

    voice_session, _ = build_voice_session(load_settings(), with_voice=False)
    collector = LatencyCollector()
    collector.begin()
    connect_observers(voice_session, CallRecord(), collector)
    voice_session.emit("error", _SyntheticEvent())

    _require(collector.failures, "el evento `error` de la sesión no llegó al colector")
    failure = collector.failures[0]
    _require(
        failure["tipo"] == "llm_error" and failure["model_name"] == "gpt-4.1-mini",
        f"el fallo llegó incompleto: {failure}",
    )
    return "los fallos de proveedor se capturan y quedan en el JSON"


CHECKS_LEVEL_0: Final[tuple[tuple[str, Callable[[], str]], ...]] = (
    ("configuración", check_config),
    ("modelos idénticos a PoC1", check_models_identical_to_poc1),
    ("catálogo de compensaciones", check_compensation_catalog),
    ("plantilla del prompt", check_prompt_template),
    ("render del prompt", check_prompt_render),
    ("precio máximo ausente", check_absence_of_max_price),
    ("schemas de tool calling", check_schemas),
    ("manejadores contra el motor", check_handlers),
    ("persistencia sin fugas", check_no_max_price_in_persistence),
    ("política de timeouts del LLM", check_llm_policy),
    ("captura de fallos de proveedor", check_failure_capture),
    ("ensamblado de la sesión", check_assembly),
)


def run_level_0() -> int:
    print("Humo de PoC2 — nivel 0 (sin red, sin costo)\n")
    faults = 0
    for label, check in CHECKS_LEVEL_0:
        try:
            detail = check()
        except SmokeFailure as error:
            faults += 1
            print(f"  [FALLA] {label}: {error}")
        except Exception as error:  # noqa: BLE001 - el humo reporta, no propaga
            faults += 1
            print(f"  [ERROR] {label}: {type(error).__name__}: {error}")
        else:
            print(f"  [ok]    {label} — {detail}")

    print()
    if faults:
        print(f"{faults} chequeo(s) fallaron.")
        return 1

    print("Nivel 0 completo. Lo que falta para la llamada:")
    settings = load_settings()
    for label, check in (
        ("LiveKit", verify_for_livekit),
        ("telefonía", verify_for_telephony),
    ):
        try:
            check(settings)
        except (IncompleteConfiguration, CollisionWithDispatcher) as error:
            print(f"  [falta] {label}: {error}")
        else:
            print(f"  [ok]    {label}")
    return 0


# --------------------------------------------------------------------------
# Nivel 1 — APIs reales, sin teléfono
# --------------------------------------------------------------------------

# Guion de 4 turnos. Empuja al agente por el camino que interesa: pedir más,
# recibir compensación, subir precio y cerrar. Es el mismo recorrido que hace el
# humo de PoC1.
SMOKE_SCRIPT: Final[tuple[str, ...]] = (
    "Bueno, sí, habla Carlos. ¿De qué se trata?",
    "Mire, por esa ruta yo no bajo de veinte mil pesos. Está muy cargado el tráfico.",
    "Pues el tiempo de espera me sirve, pero necesito que suba algo el flete.",
    "Órale, si me lo deja así con esas condiciones, sale. Quedamos.",
)


async def run_level_1(*, with_voice: bool) -> int:
    print("Humo de PoC2 — nivel 1 (APIs reales, sin teléfono)\n")

    from poc2_livekit.app.agent import build_agent, build_voice_session

    settings = load_settings()
    verify_for_voice(settings)

    negotiation = NegotiationSession(trip=TEST_TRIP, call_id="humo-nivel1")
    agent = build_agent(negotiation)

    # Sin sala y sin STT/TTS: LiveKit corre la sesión en modo texto. Es el mismo
    # Agent, el mismo prompt y las mismas herramientas de la llamada real.
    voice_session, detector = build_voice_session(settings, with_voice=False)
    print(f"  turn detector construido: {getattr(detector, 'model', '?')}")

    function_calls: list[str] = []
    await voice_session.start(agent)
    try:
        for i, linea in enumerate(SMOKE_SCRIPT, start=1):
            print(f"\n  [{i}] transportista: {linea}")
            resultado = await voice_session.run(user_input=linea)
            # `events` es propiedad, no método, y trae ChatMessageEvent /
            # FunctionCallEvent / FunctionCallOutputEvent.
            for event in resultado.events:
                tipo = getattr(event, "type", "")
                item = getattr(event, "item", None)
                if tipo == "function_call":
                    name = getattr(item, "name", "")
                    if name:
                        function_calls.append(str(name))
                        print(f"      → {name}({getattr(item, 'arguments', '')})")
                elif tipo == "function_call_output":
                    print(f"      ← {getattr(item, 'output', '')}")
                elif tipo == "message":
                    text = getattr(item, "text_content", None)
                    if text and getattr(item, "role", "") == "assistant":
                        print(f"      agent: {text}")
    finally:
        await voice_session.aclose()

    print("\n  ---")
    print(f"  funciones llamadas   {function_calls or '(ninguna)'}")
    print(f"  historial validado   {negotiation.offer_history or '(ninguno)'}")
    print(f"  compensaciones       {negotiation.offered_compensations or '(ninguna)'}")
    resultado = negotiation.resultado
    print(f"  resultado            {resultado.resultado if resultado else '(sin cierre)'}")

    faults = 0
    if not function_calls:
        faults += 1
        print(
            "\n  [FALLA] el agente no llamó ninguna función: el tool calling no "
            "está llegando al LLM"
        )
    if PROPOSE_PRICE not in function_calls:
        faults += 1
        print(
            f"  [FALLA] nunca llamó a {PROPOSE_PRICE}: no puede pronunciar "
            "montos sin validarlos"
        )

    if with_voice:
        faults += await _check_voice(settings)

    print()
    if faults:
        print(f"{faults} chequeo(s) fallaron.")
        return 1
    print("Nivel 1 completo. Falta la llamada de diagnóstico para cerrar el")
    print("contrato de métricas (ver CAPTURA_METRICAS.md).")
    return 0


async def _check_voice(settings: Settings) -> int:
    """Ida y vuelta por el camino de voz: ElevenLabs sintetiza, Deepgram oye.

    No prueba turn-taking (eso necesita una llamada), pero sí que las dos
    credenciales, los dos modelos y la voz existan y respondan.
    """
    from poc2_livekit.app.agent import build_stt, build_tts

    print("\n  --- camino de voz ---")
    frase = "Le ofrezco dieciséis mil quinientos pesos por el viaje completo."

    tts = build_tts(settings)
    marcos: list[Any] = []
    async with tts.synthesize(frase) as stream:
        async for event in stream:
            marcos.append(event.frame)
    _require(marcos, "el TTS no devolvió audio")
    samples = sum(getattr(m, "samples_per_channel", 0) for m in marcos)
    sample_rate = getattr(marcos[0], "sample_rate", 0) or 1
    print(f"  TTS: {len(marcos)} marcos, {samples / sample_rate:.1f} s de audio")

    stt = build_stt(settings)
    try:
        recognized = await stt.recognize(buffer=marcos)
    except Exception as error:  # noqa: BLE001
        print(f"  [FALLA] el STT no pudo transcribir: {type(error).__name__}: {error}")
        return 1

    text = ""
    alternates = getattr(recognized, "alternatives", None) or []
    if alternates:
        text = getattr(alternates[0], "text", "") or ""
    print(f"  STT: {text!r}")
    if not text.strip():
        print("  [FALLA] el STT devolvió texto vacío")
        return 1
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_cost_warning(*, with_voice: bool) -> None:
    total = LEVEL_1_COST_USD + (COST_WITH_VOICE_USD if with_voice else 0.0)
    print("El nivel 1 consume APIs de pago (no telefonía).")
    print(f"  estimado: ~${total:.2f} USD por corrida")
    print("    LLM   gpt-4.1-mini, ~5 respuestas con el prompt v0 completo")
    if with_voice:
        print("    TTS   ElevenLabs Flash, una frase corta")
        print("    STT   Deepgram nova-3, esa misma frase")
    print("\nVuelve a correrlo con --accept-cost para ejecutarlo.")


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.smoke",
        description="Chequeo de humo de PoC2. Nivel 0 sin red; nivel 1 con APIs reales.",
    )
    arg_parser.add_argument(
        "--online", action="store_true", help="corre el nivel 1 (APIs reales)"
    )
    arg_parser.add_argument(
        "--with-voice",
        action="store_true",
        help="en el nivel 1, agrega una ida y vuelta de TTS + STT",
    )
    arg_parser.add_argument(
        "--accept-cost",
        action="store_true",
        help="obligatorio para el nivel 1: confirma que aceptas el costo",
    )
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING)

    code_value = run_level_0()
    if code_value or not args.online:
        return code_value

    print()
    if not args.accept_cost:
        _print_cost_warning(with_voice=args.with_voice)
        return 0

    return asyncio.run(run_level_1(with_voice=args.with_voice))


if __name__ == "__main__":
    raise SystemExit(main())
