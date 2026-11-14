"""Chequeo de humo de PoC1, en dos niveles.

**Nivel 0** (default): sin red, sin audio, sin Twilio, costo cero. Verifica que
las piezas que van a colgar del pipeline estén bien conectadas antes de gastar
un solo crédito:

  - el `.env` carga y las tres llaves de IA están presentes;
  - el system prompt v0 se lee, se interpola y no deja placeholders vivos;
  - el precio máximo **no** aparece en el prompt renderizado ni en el registro
    que se va a persistir (y el detector de fugas sí detecta cuando hay una);
  - los 4 schemas de tool-calling salen del catálogo y del enum reales;
  - los 4 manejadores producen los veredictos correctos contra el motor;
  - el pipeline entero ensambla, con los modelos ONNX de VAD y fin de turno
    cargados y las 4 funciones registradas en el LLM.

**Nivel 1** (`--online`): negocia un guion fijo contra el LLM de `OPENAI_MODEL`
(gpt-4.1-mini) y ElevenLabs reales, y opcionalmente pasa el audio por Deepgram. Sigue sin haber teléfono:
no se marca a nadie. Cuesta dinero, así que exige `--accept-cost`; el
desglose está en `docs/costos_poc1.md`.

    uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke
    uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke --online \\
        --accept-cost --loopback-stt
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import re
import sys
from collections.abc import Callable, Generator, Sequence

from poc1_pipecat.app.config import (
    Settings,
    IncompleteConfiguration,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc1_pipecat.app.tools import (
    END_WITHOUT_AGREEMENT,
    FUNCTION_NAMES,
    OFFER_COMPENSATION,
    PROPOSE_PRICE,
    RECORD_AGREEMENT,
    build_schemas,
    execute_offer_compensation,
    execute_propose_price,
    execute_register_agreement,
)
from shared import llm_transport
from shared.negotiation_engine import (
    ACCEPTED,
    RECHAZADO,
    TEST_TRIP,
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

# Marcadores en ASCII puro: la consola de Windows no siempre trae UTF-8.
OK_MARK = "[ok]  "
FAILURE_MARK = "[X]   "
WARNING_MARK = "[!]   "


class SmokeFailure(AssertionError):
    """Un chequeo no pasó."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


@contextlib.contextmanager
def _silence_expected_warnings() -> Generator[None]:
    """Calla los warnings de los casos negativos que este chequeo provoca a propósito.

    Una compensación inventada y un cierre repetido *deben* generar warning:
    eso es lo que se está probando. Verlos en el reporte haría pensar que algo
    salió mal.
    """
    noisy = [
        logging.getLogger("shared.negotiation_engine.session"),
        logging.getLogger("poc1_pipecat.app.tools"),
    ]
    previous_ones = [(log, log.level) for log in noisy]
    for log, _ in previous_ones:
        log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for log, level in previous_ones:
            log.setLevel(level)


# --------------------------------------------------------------------------
# Chequeos
# --------------------------------------------------------------------------


def check_config() -> str:
    settings = load_settings()
    _require(settings.openai_model, "OPENAI_MODEL quedó vacío")
    return (
        f"llaves de IA presentes | LLM={settings.openai_model} "
        f"STT={settings.deepgram_model}/{settings.deepgram_language} "
        f"TTS={settings.eleven_model}"
    )


def check_compensation_catalog() -> str:
    compensations = load_compensations()
    _require(compensations, "el catálogo de compensaciones quedó vacío")
    ids = ", ".join(c.id for c in compensations)
    return f"{len(compensations)} compensaciones en el catálogo: {ids}"


def check_prompt_template() -> str:
    template = load_template()
    found = variables_in_template(template)
    expected_variables = set(PROMPT_VARIABLES)
    _require(
        found == expected_variables,
        f"v0.md no tiene las 7 variables esperadas: faltan "
        f"{sorted(expected_variables - found)}, desconocidas "
        f"{sorted(found - expected_variables)}",
    )
    return f"v0.md: {len(template)} caracteres, las 7 variables presentes"


def check_prompt_render() -> str:
    render = render_system_prompt(TEST_TRIP)
    _require(
        not variables_in_template(render),
        "quedaron placeholders sin sustituir en el prompt renderizado",
    )
    for expected in (TEST_TRIP.company, TEST_TRIP.origin):
        _require(expected in render, f"el prompt renderizado no contiene {expected!r}")
    return f"prompt renderizado: {len(render)} caracteres, 0 placeholders vivos"


def check_absence_of_max_price() -> str:
    """El máximo no está en el prompt, y el detector de fugas de verdad detecta.

    El segundo caso es el que importa: sin el control negativo, un detector
    roto pasaría este chequeo siempre.
    """
    render = render_system_prompt(TEST_TRIP)
    verify_absence_of_max_price(render, TEST_TRIP)

    for shape in amount_representations(TEST_TRIP.max_price):
        contaminated = f"{render}\nEl tope de esta ruta es {shape} y no lo digas."
        try:
            verify_absence_of_max_price(contaminated, TEST_TRIP)
        except MaxPriceLeak:
            continue
        raise SmokeFailure(
            f"el detector de fugas no reconoce el máximo escrito como {shape!r}"
        )

    return (
        "el máximo no aparece en el prompt, y el detector reconoce sus "
        f"{len(amount_representations(TEST_TRIP.max_price))} "
        "representaciones"
    )


def check_schemas() -> str:
    schemas = build_schemas()
    _require(len(schemas) == 4, f"se esperaban 4 schemas, hay {len(schemas)}")

    by_name = {e.name: e for e in schemas}
    _require(
        set(by_name) == set(FUNCTION_NAMES),
        f"los nombres de los schemas no coinciden con el contrato: "
        f"{sorted(by_name)}",
    )

    proponer = by_name[PROPOSE_PRICE]
    _require(
        set(proponer.properties) == {"monto"},
        "proponer_precio debe exponer exactamente un parámetro `monto` al LLM, "
        f"expone {sorted(proponer.properties)}",
    )

    enum_kinds = by_name[OFFER_COMPENSATION].properties["tipo"]["enum"]
    _require(
        list(enum_kinds) == list(compensation_ids()),
        "el enum de `tipo` no coincide con los ids de compensaciones.yaml",
    )

    enum_reasons = by_name[END_WITHOUT_AGREEMENT].properties["motivo"]["enum"]
    _require(
        list(enum_reasons) == [m.value for m in CloseReason],
        "el enum de `motivo` no coincide con MotivoCierre",
    )

    _require(
        set(by_name[RECORD_AGREEMENT].required) == {"monto_final", "condiciones"},
        "registrar_acuerdo debe requerir monto_final y condiciones",
    )

    # Ningún schema puede mencionar un límite superior: el texto de las
    # descripciones viaja al contexto del LLM igual que el prompt.
    for schema in schemas:
        text = json.dumps(schema.to_default_dict(), ensure_ascii=False).lower()
        for word in ("máximo", "maximum", "tope", "límite", "limit"):
            _require(
                word not in text,
                f"el schema {schema.name} menciona {word!r}: insinúa que "
                "existe un techo",
            )

    return "4 schemas válidos, enums sincronizados con el catálogo y MotivoCierre"


def check_handlers() -> str:
    """Corre una negociación completa contra el motor, sin LLM y sin red."""
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-offline")
    target = TEST_TRIP.target_price
    maximum = TEST_TRIP.max_price

    # 1. Oferta de anclaje, por debajo del objetivo -> aceptada.
    response = execute_propose_price(session, {"monto": target - 2_000})
    _require(
        response == {"resultado": ACCEPTED},
        f"la oferta de anclaje debió ser aceptada, respondió {response!r}",
    )

    # 2. Por encima del máximo -> rechazada, y el payload no dice nada más.
    response = execute_propose_price(session, {"monto": maximum + 4_000})
    _require(
        response == {"resultado": RECHAZADO},
        f"una oferta sobre el máximo debió ser rechazada sin detalles, "
        f"respondió {response!r}",
    )

    # 3. Primera concesión -> aceptada. Incremento de 1000.
    response = execute_propose_price(session, {"monto": target - 1_000})
    _require(response == {"resultado": ACCEPTED}, "la primera concesión debió pasar")

    # 4. Concesión que no decrece (2000 > 1000) -> rechazada, aunque esté bajo
    #    el máximo. Aísla la regla de incrementos decrecientes.
    response = execute_propose_price(session, {"monto": target + 1_000})
    _require(
        response == {"resultado": RECHAZADO},
        "una concesión no decreciente debió ser rechazada",
    )

    # 5. Concesión decreciente (500 < 1000) -> aceptada.
    monto_final = target - 500
    response = execute_propose_price(session, {"monto": monto_final})
    _require(
        response == {"resultado": ACCEPTED}, "la concesión decreciente debió pasar"
    )

    # 6. Monto ilegible del LLM -> rechazado, sin excepción.
    response = execute_propose_price(session, {"monto": "diecisiete mil"})
    _require(
        response == {"resultado": RECHAZADO},
        "un monto no numérico debió ser rechazado, no reventar",
    )

    _require(
        session.offer_history
        == [target - 2_000, target - 1_000, monto_final],
        f"el historial solo debe tener las ofertas aceptadas, tiene "
        f"{session.offer_history!r}",
    )

    # 7. Compensación del catálogo -> devuelve la redacción autorizada.
    primer_id = compensation_ids()[0]
    response = execute_offer_compensation(session, {"tipo": primer_id})
    _require(
        response.get("resultado") == "ofrecida" and response.get("descripcion"),
        f"la compensación {primer_id!r} debió ofrecerse con su descripción, "
        f"respondió {response!r}",
    )

    # 8. Compensación inventada -> no disponible.
    response = execute_offer_compensation(session, {"tipo": "secret_discount"})
    _require(
        response.get("resultado") == "no_disponible",
        "una compensación fuera del catálogo no debe registrarse",
    )

    # 9. Cierre con el último monto validado.
    response = execute_register_agreement(
        session,
        {"monto_final": monto_final, "condiciones": ["dos horas de espera pagadas"]},
    )
    _require(
        response.get("resultado") == "acuerdo_registrado",
        f"el cierre debió registrarse, respondió {response!r}",
    )
    _require(
        session.resultado is not None and session.resultado.validated_final_amount is True,
        "el monto del acuerdo debió quedar marcado como validado",
    )

    # 10. Segundo cierre espurio -> se ignora, el registro no se reescribe.
    execute_register_agreement(
        session, {"monto_final": maximum + 1_000, "condiciones": "otra cosa"}
    )
    _require(
        session.resultado is not None and session.resultado.monto_final == monto_final,
        "un segundo cierre no debe reescribir el resultado ya registrado",
    )

    return (
        f"10 pasos de negociación correctos | historial="
        f"{[int(m) for m in session.offer_history]} "
        f"compensaciones={session.offered_compensations}"
    )


def check_persistence_without_leaks() -> str:
    """El registro que se va a persistir tampoco puede contener el máximo."""
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-offline")
    execute_propose_price(session, {"monto": TEST_TRIP.target_price - 500})
    execute_register_agreement(
        session,
        {
            "monto_final": TEST_TRIP.target_price - 500,
            "condiciones": "sin condiciones extra",
        },
    )

    dumped = json.dumps(session.summary(), ensure_ascii=False, default=str)
    for shape in amount_representations(TEST_TRIP.max_price):
        _require(
            shape not in dumped,
            f"el resumen persistible contiene el máximo escrito como {shape!r}",
        )
    _require(
        "max_price" not in dumped,
        "el resumen persistible expone la clave max_price",
    )
    return f"resumen persistible limpio: {len(dumped)} caracteres, sin el máximo"


def check_llm_policy() -> str:
    """Los timeouts del LLM son los de `shared/`, no los defaults de pipecat.

    Chequeo propio porque falla en silencio y caro: con el default
    (`retry_on_timeout=False`) no hay reintento **ni tope propio** —la única
    cota es la del SDK de OpenAI, 600 s— así que una petición atorada deja la
    llamada muda hasta diez minutos, y encima con otra política que PoC2.
    """
    from poc1_pipecat.app.pipeline import build_llm

    llm = build_llm(load_settings())
    _require(
        getattr(llm, "_retry_on_timeout", False) is True,
        "el LLM quedó sin reintento por timeout: un atorón perdería el turno",
    )
    _require(
        getattr(llm, "_retry_timeout_secs", None)
        == llm_transport.FIRST_BYTE_TIMEOUT_S,
        f"el tope del primer intento es "
        f"{getattr(llm, '_retry_timeout_secs', None)!r} y la política pide "
        f"{llm_transport.FIRST_BYTE_TIMEOUT_S}",
    )
    return llm_transport.summary()


def check_pipeline_assembly() -> str:
    """Arma el pipeline completo sin conectarse a nada.

    Los servicios de Pipecat no abren conexiones al instanciarse, así que esto
    verifica gratis lo que si no se descubriría hasta la primera llamada: que
    los modelos ONNX de VAD y fin de turno cargan en esta máquina, que el
    contexto se construye con los 4 schemas, y que el LLM quedó con las 4
    funciones registradas.
    """
    from poc1_pipecat.app.pipeline import (
        build_turn_analyzer,
        build_context,
        build_llm,
        build_text_pipeline,
        build_stt,
        build_vad,
        verify_registered_functions,
    )

    settings = load_settings()

    build_stt(settings)
    build_llm(settings)
    build_vad()
    build_turn_analyzer()

    context = build_context(TEST_TRIP)
    messages = context.get_messages() if hasattr(context, "get_messages") else []
    _require(messages, "el contexto quedó sin el mensaje de sistema")

    try:
        verify_for_voice(settings)
    except IncompleteConfiguration:
        return (
            "STT, LLM, VAD, fin de turno y contexto ensamblan | pipeline "
            "completo no verificado: falta ELEVENLABS_VOICE_ID"
        )

    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-ensamblado")
    pipeline = build_text_pipeline(settings, session)

    unrecorded = verify_registered_functions(pipeline.llm)
    _require(
        not unrecorded,
        f"el LLM no tiene registradas estas funciones: {unrecorded}",
    )

    return (
        "pipeline completo ensambla | STT, LLM, TTS, VAD, fin de turno local, "
        "contexto con system prompt y 4 funciones registradas"
    )


def _no_warnings(check: Callable[[], str]) -> Callable[[], str]:
    """Envuelve un chequeo cuyos casos negativos son intencionales."""

    def wrapped() -> str:
        with _silence_expected_warnings():
            return check()

    return wrapped


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("configuración y llaves", check_config),
    ("catálogo de compensaciones", check_compensation_catalog),
    ("plantilla del system prompt", check_prompt_template),
    ("interpolación del prompt", check_prompt_render),
    ("ausencia del precio máximo", check_absence_of_max_price),
    ("schemas de tool-calling", check_schemas),
    ("manejadores contra el motor", _no_warnings(check_handlers)),
    ("registro persistible", _no_warnings(check_persistence_without_leaks)),
    ("política de timeouts del LLM", check_llm_policy),
    ("ensamblado del pipeline", check_pipeline_assembly),
]


# --------------------------------------------------------------------------
# Avisos: lo que falta para los niveles siguientes, sin marcar falla
# --------------------------------------------------------------------------


def pending_warnings(settings: Settings) -> list[str]:
    warning_list: list[str] = []
    try:
        verify_for_voice(settings)
    except IncompleteConfiguration as error:
        warning_list.append(f"TTS real todavía no se puede correr: {error}")
    try:
        verify_for_telephony(settings)
    except IncompleteConfiguration as error:
        warning_list.append(f"llamada real todavía no se puede correr: {error}")
    return warning_list


def run_level_0() -> int:
    """Corre todos los chequeos offline. Devuelve el número de fallas."""
    print("Chequeo de humo PoC1 - nivel 0 (sin red, sin audio, costo cero)\n")

    faults = 0
    for name, check in CHECKS:
        try:
            detail = check()
        except Exception as error:  # noqa: BLE001 - el reporte los muestra todos
            faults += 1
            print(f"{FAILURE_MARK}{name}")
            print(f"        {type(error).__name__}: {error}")
        else:
            print(f"{OK_MARK}{name}")
            print(f"        {detail}")

    print()
    try:
        for aviso in pending_warnings(load_settings()):
            print(f"{WARNING_MARK}{aviso}")
    except IncompleteConfiguration:
        pass  # ya se reportó como falla en el chequeo de configuración

    if faults:
        print(f"\n{faults} de {len(CHECKS)} chequeos fallaron.")
    else:
        print(
            f"\n{len(CHECKS)} de {len(CHECKS)} chequeos pasaron. "
            "Nada de esto consumió créditos."
        )
    return faults


# --------------------------------------------------------------------------
# Nivel 1: LLM, tool calling y TTS reales. Sin teléfono.
# --------------------------------------------------------------------------

# Guion fijo del transportista. No es el `carrier_simulator` ni ninguno de los
# 5 escenarios formales: son 4 turnos escritos a mano que fuerzan los caminos
# que interesa ver conectados — un ancla muy por encima del techo, una presión
# que debería sacar una compensación antes que dinero, y un cierre.
CARRIER_SCRIPT: tuple[str, ...] = (
    "Sí, soy Carlos, dígame.",
    "Está muy bajo ese precio para Monterrey Laredo. "
    "Yo necesito treinta mil pesos por ese viaje.",
    "Pues el diésel está carísimo. Necesito más dinero, "
    "no nada más el tiempo de espera.",
    "Va, quedamos así entonces.",
)

# Nombre fijo: la carpeta está gitignoreada y cada corrida reemplaza la anterior.
WAV_NAME: str = "humo_nivel1.wav"

# Timing del guion. **Solo del humo**: la llamada real no usa esta espera, ahí
# los turnos los deciden el VAD y el analizador de fin de turno.
#
# 6 s de silencio es deliberadamente generoso. La primera síntesis de
# ElevenLabs incluye abrir el WebSocket y puede tardar varios segundos, y entre
# dos tramos de voz de un mismo turno cabe una llamada a función. Cortar antes
# manda el siguiente turno del transportista encima del agente.
END_OF_TURN_SILENCE: float = 6.0
TURN_TIMEOUT: float = 45.0

# Separación entre los dos fines de habla de cada turno. Simula la pausa a media
# frase que dispara un fin de habla espurio con `stop_secs=0.2`. Tiene que ser
# medible en el JSON: si el arreglo funciona, el cero del turno es el segundo
# fin de habla y esta pausa **no** aparece sumada en `turno_completo_ms`.
PAUSE_MID_SENTENCE: float = 0.4

# Probabilidad que se le pone al veredicto sintético del analizador. El valor
# exacto da igual; lo que se verifica es que llega al JSON del turno correcto.
SMOKE_VERDICT: float = 0.87

_NUMBERS_PATTERN = re.compile(r"\d[\d.,]*")


def amounts_in_digits(text: str) -> list[float]:
    """Montos de 4 cifras o más escritos con dígitos en el texto del agente.

    Heurística deliberadamente simple: sirve para detectar que el agente
    pronunció un número que `proponer_precio` nunca validó. Si el modelo dice
    los montos con letras ("diecisiete mil"), esto no los ve — por eso el
    reporte dice cuántos encontró.
    """
    amounts: list[float] = []
    for raw in _NUMBERS_PATTERN.findall(text):
        cleaned = raw.replace(",", "").replace(".", "")
        if cleaned.isdigit() and len(cleaned) >= 4:
            amounts.append(float(cleaned))
    return amounts


def check_spoken_amounts(
    agent_text: str, historial: Sequence[float]
) -> tuple[bool, str]:
    """Verifica la regla no negociable contra lo que el agente realmente dijo."""
    spoken_items = amounts_in_digits(agent_text)
    if not spoken_items:
        return True, (
            "no se detectaron montos en dígitos en la voz del agente "
            "(pudo decirlos con letras: revisar la transcripción de arriba)"
        )

    unvalidated = [
        monto
        for monto in spoken_items
        if not any(abs(monto - previous) < 0.01 for previous in historial)
    ]
    if unvalidated:
        return False, (
            f"el agente pronunció montos que proponer_precio nunca validó: "
            f"{sorted(set(unvalidated))}"
        )
    return True, (
        f"los {len(spoken_items)} montos pronunciados salieron todos del historial "
        "validated"
    )


def _ms(valor: object) -> str:
    """Milisegundos para la tabla. `None` sale como raya, no como cero."""
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return f"{valor:,.0f} ms"
    return "-"


def verify_blockers(
    session: NegotiationSession,
    record: object,
    collector: object,
    *,
    base_path: object = None,
) -> int:
    """Comprueba los dos bloqueantes del pipeline telefónico. Devuelve fallas.

    Está fuera de `run_level_1` a propósito: así se puede probar con datos
    sintéticos, sin gastar los ~0.10 USD del nivel 1. Un bug aquí costaba una
    corrida entera para descubrirse.

    1. La latencia se registra y sale al JSON con la llave `latencias`.
    2. La transcripción trae lo que dijeron **las dos** partes, y lo del agente
       reensamblado después del TTS (lo que se pronunció, no lo que el LLM quiso
       decir).
    """
    from poc1_pipecat.app.pipeline import RECORDINGS_PATH
    from poc1_pipecat.app.telephony import save_run
    from shared.metrics.calls import compute

    faults = 0
    print("\n" + "-" * 70)
    print("Bloqueantes del pipeline telefonico\n")

    collector.close_pending()
    samples = collector.samples

    print("1) Latencia instrumentada")
    if not samples:
        faults += 1
        print(f"{FAILURE_MARK}el colector no produjo ninguna muestra")
    else:
        with_measurement = [m for m in samples if m.full_turn_ms is not None]
        print(
            f"{OK_MARK}{len(samples)} muestra(s), {len(with_measurement)} con "
            "full_turn_ms"
        )
        print(
            "     {:<7}{:<12}{:>10}{:>11}{:>10}{:>10}{:>9}".format(
                "turn", "bucket", "full", "fin turno", "TTFT", "TTS", "veredic"
            )
        )
        for sample in samples:
            print(
                "     {:<7}{:<12}{:>10}{:>11}{:>10}{:>10}{:>9}".format(
                    str(sample.indice),
                    sample.bucket,
                    _ms(sample.full_turn_ms),
                    _ms(sample.end_of_turn_ms),
                    _ms(sample.llm_ttft_ms),
                    _ms(sample.tts_ttfb_ms),
                    ",".join(f"{v:.2f}" for v in sample.end_of_turn_verdicts) or "-",
                )
            )
        if not with_measurement:
            faults += 1
            print(
                f"{FAILURE_MARK}ninguna muestra trae full_turn_ms: el cero del "
                "turno o el primer audio no se estan registrando"
            )

        # -- Los cuatro arreglos de la instrumentación de latencia --

        # a) Un turno por intervencion, no uno por pausa del VAD. El guion manda
        #    DOS fines de habla por linea; si cada uno abriera turno saldrian el
        #    doble de muestras.
        expected_variables = 1 + len(CARRIER_SCRIPT)
        if len(samples) == expected_variables:
            print(
                f"{OK_MARK}sin turnos fantasma: {len(samples)} muestras para "
                f"{len(CARRIER_SCRIPT)} intervenciones + apertura, con dos "
                "fines de habla cada una"
            )
        else:
            faults += 1
            print(
                f"{FAILURE_MARK}se esperaban {expected_variables} muestras (apertura + "
                f"{len(CARRIER_SCRIPT)} intervenciones) y hay {len(samples)}: "
                "las pausas a media frase estan abriendo turnos de mas"
            )

        # b) El cero del turno es el ULTIMO fin de habla. Si ganara el primero,
        #    `turno_completo_ms` traeria sumada la pausa a media frase.
        of_regime = [
            m for m in samples if not m.is_opening and m.full_turn_ms is not None
        ]
        pause_ms = PAUSE_MID_SENTENCE * 1000
        contaminated_ones = [m for m in of_regime if m.end_of_turn_ms and m.end_of_turn_ms > pause_ms]
        if of_regime and not contaminated_ones:
            print(
                f"{OK_MARK}el cero del turno es el ultimo fin de habla "
                f"(ningun end_of_turn_ms pasa de los {pause_ms:.0f} ms de la pausa)"
            )
        elif contaminated_ones:
            faults += 1
            print(
                f"{FAILURE_MARK}{len(contaminated_ones)} turno(s) traen la pausa a media "
                "frase sumada: gano el primer fin de habla, no el ultimo"
            )

        # c) La bandera de interrupcion ya no se enciende sola. En el humo nadie
        #    habla encima del agente, asi que TODAS deben salir en False.
        interrupted_turns = [m for m in samples if m.interrupted]
        if not interrupted_turns:
            print(f"{OK_MARK}ningun turno marcado como interrumpido (correcto: no hubo barge-in)")
        else:
            faults += 1
            print(
                f"{FAILURE_MARK}{len(interrupted_turns)} turno(s) marcados como "
                "interrumpidos sin que nadie interrumpiera: la bandera se "
                "vuelve a encender sola"
            )

        # d) Los veredictos del analizador llegan al turno correcto.
        with_verdict = [m for m in samples if m.end_of_turn_verdicts]
        if len(with_verdict) == len(CARRIER_SCRIPT):
            print(
                f"{OK_MARK}veredictos del analizador capturados en "
                f"{len(with_verdict)} turno(s)"
            )
        else:
            faults += 1
            print(
                f"{FAILURE_MARK}se esperaban {len(CARRIER_SCRIPT)} turnos con "
                f"veredicto y hay {len(with_verdict)}: TurnMetricsData no esta "
                "llegando al colector"
            )

        negatives = [m for m in samples if m.notes]
        if negatives:
            print(
                f"{WARNING_MARK}{len(negatives)} muestra(s) con etapas descartadas "
                "por salir negativas:"
            )
            for sample in negatives[:3]:
                print(f"       turno {sample.indice}: {'; '.join(sample.notes)}")

        print(
            "     stt_ms sale vacio aqui a proposito: el agregador consume los\n"
            "     TranscriptionFrame sin pasarlos rio abajo y el humo no tiene STT\n"
            "     de entrada, asi que no hay sonda que los vea. Se cubre con las\n"
            "     pruebas del colector y se confirma en la llamada real."
        )

    print("\n2) Dialogo completo en la transcripcion")
    roles = record.roles()
    if roles == {"carrier", "agent"}:
        of_agent = [t for t in record.turns if t["role"] == "agent"]
        print(
            f"{OK_MARK}las dos partes quedaron registradas "
            f"({len(of_agent)} intervencion(es) del agente, reensambladas "
            "despues del TTS)"
        )
        for turn in of_agent[:2]:
            print(f"     AGENTE  {turn['text'][:88]}")
    else:
        faults += 1
        print(
            f"{FAILURE_MARK}la transcripcion solo tiene {sorted(roles) or 'nada'}: "
            "falta una de las dos partes"
        )

    # -- El JSON, con la misma forma que la llamada real --

    # En `grabaciones/humo/`, no en `grabaciones/`: el consolidado por PoC
    # recoge todo `llamada_*.json` de la carpeta, y una corrida de humo colada
    # entre las 15 reales ensuciaría los percentiles y la tasa de cierre.
    ruta_json = save_run(
        session,
        record,
        base_path=base_path or (RECORDINGS_PATH / "humo"),
        collector=collector,
        identification={"poc": "1", "scenario": "0", "repetition": "0"},
    )
    print(f"\n3) JSON de la corrida: {ruta_json}")
    content = json.loads(ruta_json.read_text(encoding="utf-8"))
    for key in ("latencies", "transcript", "summary", "scenario"):
        if key in content and content[key] not in ([], {}, None):
            print(f"{OK_MARK}llave `{key}` presente y con contenido")
        else:
            faults += 1
            print(f"{FAILURE_MARK}llave `{key}` ausente o vacia en el JSON")

    # Y que `shared/metrics/` lo lea sin tocar nada mas.
    metrics = compute(content, max_price=session.trip.max_price)
    buckets = metrics.latency.get("buckets", {})
    regimen = (buckets.get("regimen") or {}).get("full_turn") or {}
    opening = (buckets.get("apertura") or {}).get("full_turn") or {}
    print(
        f"{OK_MARK}shared/metrics lo lee: apertura n={opening.get('n', 0)} "
        f"p50={opening.get('p50')} ms | regimen n={regimen.get('n', 0)} "
        f"p50={regimen.get('p50')} ms"
    )
    for aviso in metrics.warning_list:
        print(f"       - {aviso}")

    return faults


async def run_level_1(*, with_loopback: bool, diagnostics: bool = False) -> int:
    """Negocia el guion contra las APIs reales, sin teléfono. Devuelve fallas.

    Además de comprobar que las tres APIs responden, este nivel **ejercita la
    instrumentación de la llamada real**: el mismo `ColectorDeLatencia`, las
    mismas `SondaDeLatencia` y el mismo `AnotadorDeVozDelAgente` que cuelga el
    pipeline telefónico, y escribe un JSON con la misma forma que
    `save_run`. Así se verifica sin teléfono que el JSON sale con la llave
    `latencias` llena y con el diálogo de las dos partes.

    Cada turno del transportista se inyecta con los mismos frames que emitiría el
    transporte real, en el mismo orden:

    1. **dos** `VADUserStoppedSpeakingFrame`, no uno. El primero simula una pausa
       a media frase, que no debe abrir un turno fantasma. Si el arreglo
       funciona, los dos producen **un** turno y el cero es el del segundo.
    2. un `MetricsFrame` con `TurnMetricsData`, que es como el analizador publica
       su probabilidad de fin de turno.
    3. un `UserStoppedSpeakingFrame`, que es la frontera del turno y lo que llena
       `end_of_turn_ms`.

    Lo único que aquí **no** se puede ejercitar es `stt_ms`: el agregador de
    usuario consume los `TranscriptionFrame` sin pasarlos río abajo, y sin STT no
    hay sonda arriba del agregador que los vea. Inyectar uno además duplicaría el
    texto en el contexto. Esa etapa se cubre con las pruebas unitarias del
    colector y se confirma en la llamada real.
    """
    from pipecat.frames.frames import (
        LLMMessagesAppendFrame,
        LLMRunFrame,
        MetricsFrame,
        UserStoppedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.metrics.metrics import TurnMetricsData
    from pipecat.pipeline.runner import PipelineRunner

    from poc1_pipecat.app.latency import (
        AgentSpeechAnnotator,
        LatencyCollector,
        LatencyProbe,
    )
    from poc1_pipecat.app.pipeline import (
        RECORDINGS_PATH,
        build_text_pipeline,
        transcribe_with_deepgram,
    )
    from poc1_pipecat.app.telephony import CallRecord, save_run

    settings = load_settings()
    verify_for_voice(settings)

    # Mismo `stop_secs` que el VAD de la llamada real: el cero del turno se
    # calcula restándolo, así que usar otro valor mediría otra cosa.
    stop_secs = 0.2

    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-nivel1")
    # Se llama `dialogo` y no `registro` porque más abajo el reporte ya usa
    # `registro` para el `ResultadoNegociacion`, y lo tapaba.
    dialogue = CallRecord()
    collector = LatencyCollector()
    annotator = AgentSpeechAnnotator(lambda text: dialogue.annotate("agent", text))

    pipeline = build_text_pipeline(
        settings,
        session,
        diagnostics=diagnostics,
        extras_before_tts=[LatencyProbe(collector, label="llm")],
        extras_after_tts=[annotator, LatencyProbe(collector, label="output")],
    )
    capturer = pipeline.capturer
    runner = PipelineRunner(handle_sigint=False)

    stated_items: list[str] = []
    chronology: list[dict[str, object]] = []

    async def _agent_turn(label: str) -> None:
        motivo = await capturer.wait_for_end_of_turn(
            timeout=TURN_TIMEOUT, silence=END_OF_TURN_SILENCE
        )
        text = capturer.turn_text
        stated_items.append(text)
        chronology.append(
            {
                "turn": label,
                "text_to_tts": pipeline.probe.t_first_text,
                "primer_audio": capturer.t_primer_audio,
                "bytes": capturer.turn_bytes,
                "motivo": motivo,
            }
        )
        print(f"  AGENTE      {text or '(sin voz en este turno)'}")
        if motivo == "no_speech":
            print(
                f"  {WARNING_MARK.strip()} turno {label}: se agotaron "
                f"{TURN_TIMEOUT:.0f} s sin cerrar el turno"
            )

    async def script() -> None:
        print("\nNegociación (guion fijo, 4 turnos del transportista):\n")

        # En una llamada saliente el agente abre. El cero de la llamada es aquí:
        # el turno de apertura mide el arranque en frío.
        collector.begin()
        pipeline.restart_turn()
        await pipeline.task.queue_frame(LLMRunFrame())
        await _agent_turn("apertura")

        for numero, linea in enumerate(CARRIER_SCRIPT, start=1):
            if session.is_closed:
                print("\n  (el agente ya cerró la negociación; se corta el guion)")
                break
            print(f"\n  TRANSPORTISTA  {linea}")
            dialogue.annotate("carrier", linea)
            pipeline.restart_turn()

            # Pausa a media frase: el VAD emite un fin de habla que NO es el
            # final de la intervención, y no debe abrir un turno fantasma.
            await pipeline.task.queue_frame(
                VADUserStoppedSpeakingFrame(stop_secs=stop_secs)
            )
            await asyncio.sleep(PAUSE_MID_SENTENCE)
            # El fin de habla de verdad. Este es el cero que debe ganar.
            await pipeline.task.queue_frame(
                VADUserStoppedSpeakingFrame(stop_secs=stop_secs)
            )
            # El veredicto del analizador, como lo publica pipecat.
            await pipeline.task.queue_frame(
                MetricsFrame(
                    data=[
                        TurnMetricsData(
                            processor="BaseSmartTurn",
                            is_complete=True,
                            probability=SMOKE_VERDICT,
                            e2e_processing_time_ms=12.0,
                        )
                    ]
                )
            )
            # La frontera del turno. Va antes del texto para que el turno esté en
            # medición antes de que el LLM empiece a responder.
            await pipeline.task.queue_frame(UserStoppedSpeakingFrame())
            await pipeline.task.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": linea}], run_llm=True
                )
            )
            await _agent_turn(str(numero))

        await pipeline.task.stop_when_done()

    await asyncio.gather(runner.run(pipeline.task), script())

    # -- Reporte --

    print("\n" + "-" * 70)
    print("Resultado de la negociación\n")
    print(f"  historial validado      {[int(m) for m in session.offer_history]}")
    print(f"  compensaciones          {session.offered_compensations or '(ninguna)'}")

    faults = 0
    if session.resultado is None:
        print("  resultado               (el agente no cerró la negociación)")
        print(f"\n{WARNING_MARK}el guion terminó sin que el agente llamara a ninguna "
              "función de cierre")
    else:
        record = session.resultado
        print(f"  resultado               {record.resultado}")
        print(f"  monto final             {record.monto_final}")
        print(f"  condiciones             {list(record.condiciones) or '(ninguna)'}")
        print(f"  motivo                  {record.motivo or '(no aplica)'}")
        if record.monto_final is not None and not record.validated_final_amount:
            faults += 1
            print(
                f"\n{FAILURE_MARK}el monto del acuerdo no salió de ninguna oferta "
                "validada por proponer_precio"
            )

    ok_amounts, amounts_detail = check_spoken_amounts(
        " ".join(stated_items), session.offer_history
    )
    print("\nCumplimiento del límite financiero")
    print(f"{OK_MARK if ok_amounts else FAILURE_MARK}{amounts_detail}")
    if not ok_amounts:
        faults += 1

    # -- Diagnóstico de tiempos: dónde se va el tiempo en cada turno --

    print("\nTiempos por turno (segundos desde que arranca el turno)")
    header = ("turn", "texto→TTS", "1er audio", "síntesis", "audio", "cierre")
    print(
        "  {:<10}{:>12}{:>12}{:>12}{:>12}  {}".format(*header)
    )
    for row in chronology:
        t_text = row["text_to_tts"]
        t_audio = row["primer_audio"]
        col_text = f"{t_text:.2f}s" if isinstance(t_text, float) else "—"
        col_audio = f"{t_audio:.2f}s" if isinstance(t_audio, float) else "—"
        if isinstance(t_text, float) and isinstance(t_audio, float):
            col_synthesis = f"{t_audio - t_text:.2f}s"
        else:
            col_synthesis = "—"
        col_bytes = f"{row['bytes']} B"
        print(
            "  {:<10}{:>12}{:>12}{:>12}{:>12}  {}".format(
                row["turn"],
                col_text,
                col_audio,
                col_synthesis,
                col_bytes,
                row["motivo"],
            )
        )
    print(
        f"\n  La columna «síntesis» es lo que tardó ElevenLabs en devolver el "
        f"primer chunk\n  desde que recibió el texto. El turno se da por cerrado "
        f"tras {END_OF_TURN_SILENCE:.0f} s de\n  silencio; el timeout duro es "
        f"{TURN_TIMEOUT:.0f} s."
    )

    # -- Qué frames llegaron: responde "¿nunca llegó audio o llegó y se perdió?" --

    print("\nFrames que llegaron al capturador (después del TTS)")
    if capturer.frame_count:
        for name, quantity in sorted(capturer.frame_count.items()):
            print(f"  {name:<34}{quantity:>6}")
    else:
        print("  (ninguno: el pipeline no llegó a correr)")

    if capturer.ttfb:
        print("\nTTFB por servicio (primera respuesta, en segundos)")
        for processor, valor in sorted(capturer.ttfb.items()):
            print(f"  {processor:<28} {valor:.3f}")

    faults += verify_blockers(session, dialogue, collector)

    ruta_wav = capturer.write_wav(RECORDINGS_PATH / WAV_NAME)
    if ruta_wav:
        print(
            f"\nAudio del agente: {ruta_wav} "
            f"({capturer.audio_seconds:.1f} s, {capturer.sample_rate} Hz)"
        )
    else:
        faults += 1
        print(f"\n{FAILURE_MARK}el TTS no produjo audio")

    if with_loopback and ruta_wav and capturer.sample_rate:
        print("\nLoopback de STT: pasando el audio del agente por Deepgram...")
        texts = await transcribe_with_deepgram(
            settings, capturer.raw_audio, capturer.sample_rate
        )
        if texts:
            for text in texts:
                print(f"  DEEPGRAM    {text}")
        else:
            faults += 1
            print(f"{FAILURE_MARK}Deepgram no devolvió ninguna transcripción")

    return faults


def _print_cost_warning() -> None:
    print(
        "\n".join(
            (
                "",
                "El nivel 1 usa las tres APIs reales (OpenAI, ElevenLabs y, con",
                "--loopback-stt, Deepgram). No hay telefonía: no se marca a nadie.",
                "",
                "Costo estimado por corrida: ~0.07 USD (~0.08 con loopback).",
                "El desglose y los supuestos están en docs/costos_poc1.md.",
                "",
                "Para correrlo:",
                "  uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke "
                "--online --accept-cost",
                "",
                "Si algo falla, repite con --diagnostics: traza cada frame con su",
                "tiempo y sube los logs de Pipecat a DEBUG.",
                "",
            )
        )
    )


# Módulos de Pipecat que sí vale la pena escuchar en DEBUG cuando algo falla.
# Todo lo demás se queda en WARNING: subir el nivel a lo bruto vuelca los
# encabezados HTTP, cada trama del WebSocket y el system prompt completo en
# cada petición, y el diagnóstico se vuelve ilegible.
_MODULES_TO_DIAGNOSE: tuple[str, ...] = (
    "pipecat.services.tts_service",
    "pipecat.services.elevenlabs.tts",
    "pipecat.services.llm_service",
    "pipecat.processors.metrics.frame_processor_metrics",
)


def _configure_logs(*, diagnostics: bool = False) -> None:
    """Baja el ruido para que el reporte se lea.

    Son dos sistemas de log distintos: el motor de negociación usa `logging`
    y Pipecat usa loguru. Además, en `logging` viven `httpx`, `httpcore`,
    `openai` y `websockets`, que en DEBUG imprimen el prompt completo y cada
    trama de audio en base64 — por eso el root se queda en WARNING **incluso
    en modo diagnóstico**, y solo se sube lo del propio proyecto.

    De Pipecat sí se suben a DEBUG los módulos de la lista de arriba: ahí están
    los mensajes que importan cuando el audio no aparece ("Generating TTS
    [...]", "Received final message for context", "Calling function [...]").
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("poc1_pipecat").setLevel(
        logging.DEBUG if diagnostics else logging.WARNING
    )
    logging.getLogger("shared").setLevel(
        logging.INFO if diagnostics else logging.WARNING
    )

    try:
        from loguru import logger as logger_pipecat
    except ImportError:  # pragma: no cover - loguru viene con pipecat
        return

    logger_pipecat.remove()
    if not diagnostics:
        logger_pipecat.add(sys.stderr, level="WARNING")
        return

    # El dict de `filter` de loguru es por módulo: todo en WARNING salvo los
    # que interesan.
    filter_rule: dict[str, str] = {"": "WARNING"}
    for modulo in _MODULES_TO_DIAGNOSE:
        filter_rule[modulo] = "DEBUG"
    logger_pipecat.add(sys.stderr, level="DEBUG", filter=filter_rule)


def main(argv: Sequence[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.smoke",
        description=(
            "Chequeo de humo de PoC1. Sin argumentos corre el nivel 0: sin red, "
            "sin audio y sin costo."
        ),
    )
    arg_parser.add_argument(
        "--online",
        action="store_true",
        help="corre además el nivel 1: LLM, tool calling y TTS reales, sin teléfono",
    )
    arg_parser.add_argument(
        "--accept-cost",
        action="store_true",
        help="confirma el gasto del nivel 1 (ver docs/costos_poc1.md)",
    )
    arg_parser.add_argument(
        "--loopback-stt",
        action="store_true",
        help="pasa el audio sintetizado por Deepgram para verificar también el STT",
    )
    arg_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "traza cada frame que cruza el pipeline con su tiempo, y sube los "
            "logs de Pipecat a DEBUG. Úsalo cuando el nivel 1 falle."
        ),
    )
    args = arg_parser.parse_args(argv)
    force_utf8_output()
    _configure_logs(diagnostics=args.diagnostics)

    faults = run_level_0()
    if not args.online:
        return 1 if faults else 0

    if faults:
        print("\nEl nivel 0 falló: no se corre el nivel 1 para no gastar de más.")
        return 1

    if not args.accept_cost:
        _print_cost_warning()
        return 2

    print("\n" + "=" * 70)
    print("Chequeo de humo PoC1 - nivel 1 (APIs reales, sin teléfono)")
    print("=" * 70)

    faults = asyncio.run(
        run_level_1(
            with_loopback=args.loopback_stt, diagnostics=args.diagnostics
        )
    )

    print("\n" + "-" * 70)
    if faults:
        print(f"Nivel 1: {faults} problema(s). Revisa el detalle de arriba.")
        return 1
    print("Nivel 1: todo conectado. STT/LLM/tool calling/TTS responden.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
