"""Chequeo de humo de PoC3, en dos niveles.

**Nivel 0** (default): sin red, sin audio, sin Twilio, costo cero. Verifica que
las piezas estén bien conectadas antes de gastar un solo crédito:

  - el `.env` carga y la llave de OpenAI está presente;
  - el system prompt v0 se lee, se interpola y no deja placeholders vivos;
  - el precio máximo **no** aparece en el prompt renderizado ni en el registro
    que se va a persistir (y el detector de fugas sí detecta cuando hay una);
  - los 4 schemas salen del catálogo y del enum reales, en el formato de la
    Responses API;
  - la delegación queda en `responses` con el modelo congelado de las 3 PoC;
  - los 4 manejadores producen los veredictos correctos contra el motor;
  - **el contrato de tool-calling completo, con un stub**: un evento
    `response.function_call_arguments.done` sintético entra, corre contra el
    motor y sale como `delegation.function_call_output.create` bien formado;
  - el códec μ-law y el puente de audio, ida y vuelta;
  - la captura de latencia produce un JSON que `shared/metrics/` lee, con las 4
    etapas ausentes en vez de nulas;
  - la sesión inicial lleva `model`, PCM16 24 kHz y nada más (el esquema es estricto).

**Nivel 1** (`--online`): sesión **real** contra el alpha, sin teléfono. Abre el
WebSocket, configura, alimenta audio desde un WAV pasándolo por el mismo puente
que usaría Twilio, y verifica el flujo de eventos completo hasta que
`turno_completo_ms` salga distinto de `None`.

    uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke
    uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke --online \\
        --accept-cost
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
import wave
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Final

import numpy as np

from poc3_gpt_live.app import bridge
from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    RECORDINGS_PATH,
    Settings,
    IncompleteConfiguration,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc3_gpt_live.app.latency import LatencyCollector
from poc3_gpt_live.app.tools import (
    END_WITHOUT_AGREEMENT,
    FUNCTION_NAMES,
    OFFER_COMPENSATION,
    PROPOSE_PRICE,
    RECORD_AGREEMENT,
    build_delegation,
    build_schemas,
    build_output_event,
    execute_function,
    execute_offer_compensation,
    execute_propose_price,
    execute_register_agreement,
)
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

# Estimación del nivel 1. `gpt-live-1` se cobra por tiempo de sesión, y lo que
# OpenAI cuenta como sesión no es exactamente el tiempo con el WebSocket abierto
# (ver `docs/costos_poc3.md`). Se deja en el extremo alto de lo medido,
# redondeado hacia arriba: un aviso de costo que se queda corto es peor que uno
# que sobra.
LEVEL_1_COST_USD: Final[float] = 0.06

# WAV que se alimenta por default en el nivel 1: la voz del agente de PoC1,
# hablando español del dominio. No es un transportista actuando el guion, pero
# es habla real en el idioma correcto y ya está en el repo — sirve para verificar
# que el camino de audio entero funciona sin tener que grabar nada.
WAV_DEFAULT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "poc1_pipecat"
    / "recordings"
    / "humo_nivel1.wav"
)

# Twilio manda tramas de 20 ms; el nivel 1 imita ese ritmo para que el puente
# vea exactamente el patrón de llegada de una llamada real.
MS_PER_FRAME: Final[int] = 20
SAMPLES_PER_FRAME: Final[int] = bridge.TELEPHONY_SAMPLE_RATE * MS_PER_FRAME // 1000

# RMS por debajo del cual una trama de PCM16 es silencio y no voz. El habla
# normal anda en miles; el silencio digital es 0 y el ruido de fondo, decenas.
SPEECH_THRESHOLD_RMS: Final[float] = 200.0

# Cuánto se espera a que el modelo diga algo antes de darlo por mudo.
FIRST_AUDIO_TIMEOUT_S: Final[float] = 30.0
# Cuánto audio del WAV se alimenta como máximo, para no alargar la corrida.
MAX_AUDIO_SECONDS: Final[float] = 45.0

# El WAV del humo es un monólogo corrido, sin una sola pausa. Contra un modelo
# full-duplex eso es entrada adversarial: nunca le cede el turno, así que el
# agente casi no habla y la latencia por energía sale sin sentido (turnos de
# 0-60 ms, todos marcados como interrumpidos). Se intercalan silencios para que
# haya turnos de verdad que medir.
#
# Es andamiaje del humo, no configuración de la medición: en una llamada real
# las pausas las hace quien actúa el guion.
SPEECH_SECONDS_PER_TURN: Final[float] = 6.0
PAUSE_SECONDS_BETWEEN_TURNS: Final[float] = 3.0

# Criterios del renglón 3 de §2.6. Un turno por debajo del piso no es una
# respuesta rápida: es el hueco entre dos palabras del transportista colándose
# como latencia. Ningún agente de voz contesta en 60 ms — hay red, modelo y
# síntesis de por medio.
MIN_MEASURED_SAMPLES: Final[int] = 3
PLAUSIBLE_LATENCY_FLOOR_MS: Final[float] = 300.0


class SmokeFailure(AssertionError):
    """Un chequeo no pasó."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


@contextlib.contextmanager
def _silence_expected_warnings() -> Generator[None]:
    """Calla los warnings de los casos negativos que este chequeo provoca a propósito.

    Una compensación inventada, una función desconocida, un cierre repetido y un
    payload de audio impar *deben* generar warning: eso es lo que se está
    probando. Verlos en el reporte haría pensar que algo salió mal.
    """
    noisy = [
        logging.getLogger("shared.negotiation_engine.session"),
        logging.getLogger("poc3_gpt_live.app.tools"),
        logging.getLogger("poc3_gpt_live.app.bridge"),
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
# Nivel 0 — chequeos
# --------------------------------------------------------------------------


def check_config() -> str:
    settings = load_settings()
    _require(settings.live_model, "OPENAI_LIVE_MODEL quedó vacío")
    _require(
        settings.url_live() == "wss://api.openai.com/v1/live/sessions",
        f"la URL de GPT-Live no tiene la forma esperada: {settings.url_live()}",
    )
    headers = settings.headers_live()
    _require(
        set(headers) == {"Authorization"},
        "la API pública solo lleva `Authorization`: el header del alpha ya no va",
    )
    return (
        f"live={settings.live_model} delegación={settings.delegation_model} "
        f"voz={settings.voice}"
    )


def check_frozen_delegation_model() -> str:
    """El tool calling de PoC3 tiene que correr el mismo modelo que PoC1 y PoC2.

    Protege la decisión de correr el mismo LLM en las tres PoC. Si PoC3
    delegara a otro modelo, la comparación del tool calling —que sí es un
    diferenciador entre frameworks— mediría el modelo.
    """
    settings = load_settings()
    _require(
        settings.delegation_model == DEFAULT_DELEGATION_MODEL,
        f"la delegación corre {settings.delegation_model!r} y PoC1/PoC2 corren "
        f"{DEFAULT_DELEGATION_MODEL!r}: con modelos distintos la comparativa del "
        "tool calling mide el modelo, no el framework",
    )
    return f"delegación congelada en {DEFAULT_DELEGATION_MODEL} (igual que PoC1 y PoC2)"


def check_compensation_catalog() -> str:
    compensations = load_compensations()
    _require(compensations, "el catálogo de compensaciones quedó vacío")
    return f"{len(compensations)} compensations: {', '.join(c.id for c in compensations)}"


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

    El segundo caso es el que importa: sin el control negativo, un detector roto
    pasaría este chequeo siempre.
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
        f"{len(amount_representations(TEST_TRIP.max_price))} representaciones"
    )


def check_schemas() -> str:
    schemas = build_schemas()
    _require(len(schemas) == 4, f"se esperaban 4 schemas, hay {len(schemas)}")

    names = [e["name"] for e in schemas]
    _require(
        names == list(FUNCTION_NAMES),
        f"los nombres de los schemas no coinciden con el contrato: {names}",
    )
    for schema in schemas:
        _require(
            schema.get("type") == "function",
            f"el schema {schema['name']} no declara type=function: la Responses "
            "API declara los tools planos",
        )

    by_name = {e["name"]: e for e in schemas}

    proponer = by_name[PROPOSE_PRICE]["parameters"]
    _require(
        set(proponer["properties"]) == {"monto"},
        "proponer_precio debe exponer exactamente un parámetro `monto` al modelo",
    )

    enum_kinds = by_name[OFFER_COMPENSATION]["parameters"]["properties"]["tipo"]["enum"]
    _require(
        list(enum_kinds) == list(compensation_ids()),
        "el enum de `tipo` no coincide con los ids de compensaciones.yaml",
    )

    enum_reasons = by_name[END_WITHOUT_AGREEMENT]["parameters"]["properties"]["motivo"]["enum"]
    _require(
        list(enum_reasons) == [m.value for m in CloseReason],
        "el enum de `motivo` no coincide con MotivoCierre",
    )

    _require(
        set(by_name[RECORD_AGREEMENT]["parameters"]["required"])
        == {"monto_final", "condiciones"},
        "registrar_acuerdo debe requerir monto_final y condiciones",
    )

    # Ningún schema puede mencionar un límite superior: el texto de las
    # descripciones viaja al contexto del modelo igual que el prompt.
    text = json.dumps(schemas, ensure_ascii=False).lower()
    for word in ("máximo", "maximum", "tope", "límite", "limit"):
        _require(
            word not in text,
            f"un schema menciona {word!r}: insinúa que existe un techo",
        )

    return "4 schemas válidos, enums sincronizados con el catálogo y MotivoCierre"


def check_delegation() -> str:
    """La delegación es la decisión de arquitectura de PoC3: se verifica entera."""
    settings = load_settings()
    delegation = build_delegation(model_name=settings.delegation_model)

    _require(
        delegation.get("type") == "responses",
        f"la delegación quedó en {delegation.get('type')!r}. Con `client` no hay "
        "call_id ni argumentos estructurados, así que no se puede garantizar que "
        "ningún monto se pronuncie sin que proponer_precio lo valide",
    )
    responses = delegation["responses"]
    _require(
        responses.get("model") == DEFAULT_DELEGATION_MODEL,
        f"la delegación corre {responses.get('model')!r} y no el modelo congelado",
    )
    _require(
        responses.get("parallel_tool_calls") is False,
        "parallel_tool_calls debe ir en False: proponer_precio evalúa contra el "
        "historial, y dos validaciones en paralelo verían el mismo estado",
    )
    kinds = {t.get("type") for t in responses.get("tools", [])}
    _require(
        kinds == {"function"},
        f"la delegación declara tools que no son funciones: {sorted(kinds)}. Un "
        "hosted tool metería latencia que ninguna otra PoC paga",
    )
    return (
        f"delegación responses -> {responses['model']}, "
        f"{len(responses['tools'])} funciones, sin hosted tools, en serie"
    )


def check_initial_session() -> str:
    """La sesión inicial es estricta: un campo de más y el arranque falla."""
    from poc3_gpt_live.app.telephony import build_initial_session

    settings = load_settings()
    negotiation = NegotiationSession(trip=TEST_TRIP, call_id="humo")
    session = build_initial_session(settings, negotiation)

    _require(
        session.get("model") == settings.live_model,
        "`session.start` debe llevar el modelo: en la API pública no va en la URL",
    )
    _require(
        set(session) == {"model", "instructions", "audio", "delegation"},
        f"la sesión inicial trae campos inesperados: {sorted(session)}",
    )
    _require(session["instructions"].strip(), "la sesión inicial va sin instructions")
    _require(
        session["audio"]["output"]["voice"] == settings.voice,
        "la voz de la sesión no es la de la config",
    )
    _require(
        session["audio"]["format"] == bridge.session_format(settings.audio_format),
        f"el formato de audio no es el de la config ({settings.audio_format}): "
        f"{session['audio'].get('format')!r}",
    )
    # El guardarraíl estructural, otra vez, sobre lo que de verdad se manda.
    verify_absence_of_max_price(session["instructions"], TEST_TRIP)
    return (
        f"sesión inicial válida: {len(session['instructions'])} caracteres de "
        f"instructions, voz={settings.voice}, modelo={settings.live_model}, PCM16 24 kHz"
    )


def check_handlers() -> str:
    """Corre una negociación completa contra el motor, sin modelo y sin red."""
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-offline")
    target = TEST_TRIP.target_price
    maximum = TEST_TRIP.max_price

    response = execute_propose_price(session, {"monto": target - 2_000})
    _require(
        response == {"resultado": ACCEPTED},
        f"la oferta de anclaje debió ser aceptada, respondió {response!r}",
    )

    response = execute_propose_price(session, {"monto": maximum + 4_000})
    _require(
        response == {"resultado": RECHAZADO},
        f"una oferta sobre el máximo debió ser rechazada sin detalles, respondió "
        f"{response!r}",
    )

    response = execute_propose_price(session, {"monto": target - 1_000})
    _require(response == {"resultado": ACCEPTED}, "la primera concesión debió pasar")

    response = execute_propose_price(session, {"monto": target + 1_000})
    _require(
        response == {"resultado": RECHAZADO},
        "una concesión no decreciente debió ser rechazada",
    )

    monto_final = target - 500
    response = execute_propose_price(session, {"monto": monto_final})
    _require(response == {"resultado": ACCEPTED}, "la concesión decreciente debió pasar")

    _require(
        session.offer_history == [target - 2_000, target - 1_000, monto_final],
        f"el historial solo debe tener las ofertas aceptadas, tiene "
        f"{session.offer_history!r}",
    )

    primer_id = compensation_ids()[0]
    response = execute_offer_compensation(session, {"tipo": primer_id})
    _require(
        response.get("resultado") == "ofrecida" and response.get("descripcion"),
        f"la compensación {primer_id!r} debió ofrecerse con su descripción",
    )
    response = execute_offer_compensation(session, {"tipo": "secret_discount"})
    _require(
        response.get("resultado") == "no_disponible",
        "una compensación fuera del catálogo no debe registrarse",
    )

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

    return (
        f"negociación completa correcta | historial="
        f"{[int(m) for m in session.offer_history]} "
        f"compensaciones={session.offered_compensations}"
    )


def check_tool_calling_contract() -> str:
    """El camino completo del tool calling, con eventos sintéticos de GPT-Live.

    Es el chequeo que más vale de este nivel: verifica **sin gastar nada** lo que
    de otro modo se descubriría roto a media llamada telefónica. Entra un
    `response.event` con un `response.output_item.done` de tipo
    `function_call` —con `arguments` como cadena JSON, que es el detalle que más
    fácil se implementa mal—, se desenvuelve como lo hace `escuchar()`, y tiene
    que salir un `response.item.create` bien formado.
    """
    session = NegotiationSession(trip=TEST_TRIP, call_id="humo-stub")

    on_top = {
        "type": bridge.EV_RESPONSE_EVENT,
        "delegation_id": "of_smoke",
        "event": {
            "type": bridge.EV_OUTPUT_ITEM_DONE,
            "output_index": 0,
            "item": {
                "id": "fc_smoke",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_smoke_1",
                "name": PROPOSE_PRICE,
                # Cadena JSON, no objeto.
                "arguments": '{"monto": 16000}',
            },
        },
    }
    event = bridge.unwrap_response(on_top)
    _require(
        event.get("delegation_id") == "of_smoke",
        "el sobre `response.event` perdió su `delegation_id` al desenvolverse",
    )

    data = bridge.function_call(event)
    _require(data is not None, "el evento de llamada a función no se reconoció")
    call_id, name, parsed_args = data  # type: ignore[misc]
    _require(call_id == "call_smoke_1", f"call_id mal leído: {call_id!r}")
    _require(name == PROPOSE_PRICE, f"nombre mal leído: {name!r}")

    response = execute_function(session, name, parsed_args)
    _require(
        response == {"resultado": ACCEPTED},
        f"16,000 debió aceptarse, respondió {response!r}",
    )
    _require(
        session.offer_history == [16_000.0],
        f"el monto validado no entró al historial: {session.offer_history!r}",
    )

    output = build_output_event(call_id, response)
    _require(
        output["type"] == bridge.EV_RESPONSE_ITEM_CREATE,
        f"el evento de salida tiene el tipo equivocado: {output['type']!r}",
    )
    item = output["item"]
    _require(
        item["type"] == "function_call_output" and item["call_id"] == call_id,
        f"el item de salida está mal formado: {item!r}",
    )
    _require(
        isinstance(item["output"], str),
        "`output` debe viajar como cadena JSON, no como objeto",
    )
    _require(
        json.loads(item["output"]) == response,
        "el `output` serializado no coincide con la respuesta del motor",
    )

    # Argumentos ilegibles: no deben reventar. Una excepción aquí colgaría una
    # llamada telefónica ya pagada por un error que el motor sabe manejar.
    bad_response = execute_function(session, PROPOSE_PRICE, "{esto no es json")
    _require(
        bad_response == {"resultado": RECHAZADO},
        f"un `arguments` ilegible debió rechazarse, no reventar: {bad_response!r}",
    )

    # Función inventada: se le contesta al modelo, no se levanta excepción.
    unknown_one = execute_function(session, "delete_all", "{}")
    _require(
        unknown_one.get("resultado") == "unknown_function",
        f"una función fuera del contrato debió contestarse, no reventar: {unknown_one!r}",
    )

    return (
        "contrato completo: response.event -> motor -> "
        "response.item.create, con argumentos ilegibles y "
        "función inventada manejados"
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


def check_codec_and_bridge() -> str:
    """El códec μ-law y el remuestreo, ida y vuelta.

    Vale un chequeo propio porque falla en silencio y caro: un códec mal
    implementado no revienta, solo hace que el transportista oiga ruido y que el
    modelo no entienda nada — y eso solo se descubre con el teléfono en la mano.
    """
    # Ida y vuelta exacta sobre los 256 códigos, salvo el ±0 de μ-law: el código
    # 127 y el 255 decodifican al mismo cero y solo uno sobrevive.
    codes = bytes(range(256))
    outbound = bridge.pcm16_a_ulaw(bridge.ulaw_a_pcm16(codes))
    different_items = [i for i in range(256) if outbound[i] != codes[i]]
    _require(
        different_items == [127],
        f"el códec no es reversible fuera del ±0 de μ-law: difieren {different_items}",
    )

    # Un tono por el puente completo: 8 k -> 24 k -> 8 k.
    live_bridge = bridge.AudioBridge()
    t = np.arange(bridge.TELEPHONY_SAMPLE_RATE) / bridge.TELEPHONY_SAMPLE_RATE
    tone = (np.sin(2 * np.pi * 440 * t) * 12_000).astype(np.int16)
    entry = bridge.pcm16_a_ulaw(tone)

    upsampled = bytearray()
    for i in range(0, len(entry), SAMPLES_PER_FRAME):
        upsampled += live_bridge.from_twilio(bytes(entry[i : i + SAMPLES_PER_FRAME]))
    _require(
        len(upsampled) % bridge.BYTES_PER_SAMPLE == 0,
        "el PCM16 que sale del puente tiene un número impar de bytes: el alpha "
        "lo rechazaría con invalid_audio",
    )
    samples_24k = len(upsampled) // bridge.BYTES_PER_SAMPLE
    expected_variables = len(entry) * (bridge.LIVE_SAMPLE_RATE // bridge.TELEPHONY_SAMPLE_RATE)
    _require(
        0.9 <= samples_24k / expected_variables <= 1.0,
        f"el remuestreo perdió demasiado: {samples_24k} muestras de {expected_variables}",
    )

    lowered = live_bridge.a_twilio(bytes(upsampled))
    _require(lowered, "el camino de vuelta no produjo audio")

    # Un payload impar se sanea en vez de desalinear el resto de la llamada: se
    # descarta el byte suelto, no el payload entero.
    pair = np.zeros(4_800, dtype=np.int16).tobytes()
    _require(
        live_bridge.a_twilio(pair + b"\x7f") != b"",
        "un payload de tamaño impar debe sanearse, no descartarse entero",
    )

    return (
        f"códec exacto contra G.711 | puente 8k->24k->8k: "
        f"{samples_24k} muestras a 24 kHz, {len(lowered)} bytes μ-law de vuelta | "
        f"retraso de régimen {bridge.RESAMPLING_LAG_MS} ms por sentido"
    )


def check_latency_contract() -> str:
    """La captura produce un JSON que `shared/metrics/` lee, sin etapas nulas.

    Se corre con un reloj falso para que el resultado sea determinista: lo que se
    verifica es la **forma** del contrato, no los milisegundos.
    """
    from poc3_gpt_live.app import latency as latency_mod
    from shared.metrics.latency import STAGE_FIELDS
    from shared.metrics.calls import compute

    now = [0.0]

    def clock() -> float:
        return now[0]

    collector = LatencyCollector(clock=clock)
    collector.begin()
    collector.anchor_server({"model": "gpt-live-1-boulder-alpha"})

    def incoming_streak(since: float, until: float) -> None:
        """El transportista hablando, trama por trama.

        Se alimenta con la cadencia real de Twilio (20 ms) porque el detector
        agrupa la voz en **rachas**: una marca suelta es ruido de línea y no
        puede ser el cero de un turno.
        """
        t = since
        while t <= until + 1e-9:
            now[0] = round(t, 6)
            collector.inbound_speech()
            t += latency_mod.INBOUND_FRAME_S

    # Apertura: el agente saluda 1.2 s después de conectar.
    now[0] = 1.2
    collector.outbound_speech()

    # Turno de régimen: el transportista habla de 3.5 a 4.0 s y el agente
    # contesta 2.5 s después de que se calla. Las marcas salen de la energía del
    # audio, no de los eventos del alpha (ver el encabezado de app/latencia.py).
    incoming_streak(3.5, 4.0)
    now[0] = 6.5
    collector.outbound_speech()

    samples = collector.samples
    _require(len(samples) == 2, f"se esperaban 2 muestras, hay {len(samples)}")
    _require(samples[0].is_opening, "la primera muestra debe ser la apertura")
    _require(
        samples[0].full_turn_ms == 1_200.0,
        f"la apertura midió {samples[0].full_turn_ms} y debía medir 1200",
    )
    _require(
        samples[1].full_turn_ms == 2_500.0,
        f"el turno de régimen midió {samples[1].full_turn_ms} y debía medir 2500",
    )
    _require(
        not samples[1].interrupted,
        "un turno sin solape no debe salir marcado como interrumpido",
    )

    # Las 4 etapas se omiten del JSON, no salen como null.
    rows = collector.to_json()
    for row in rows:
        present_items = [c for c in STAGE_FIELDS if c in row]
        _require(
            not present_items,
            f"el JSON de PoC3 trae etapas que no mide: {present_items}. Deben omitirse, "
            "no escribirse como null",
        )
        _require("full_turn_ms" in row, "falta full_turn_ms en el JSON")

    # Y que `shared/metrics/` lo lea sin tocar nada más.
    content = {
        "call_sid": "humo-latencia",
        "poc": 3,
        "scenario": 0,
        "repetition": 0,
        "latencies": rows,
        "transcript": [
            {"t": 0.0, "role": "agent", "text": "Hola, ¿hablo con Carlos?"},
            {"t": 4.0, "role": "carrier", "text": "Sí, soy Carlos."},
        ],
    }
    metrics = compute(content)
    buckets = metrics.latency.get("buckets", {})
    regimen = buckets.get("regimen") or {}
    _require(
        (regimen.get("full_turn") or {}).get("n") == 1,
        f"shared/metrics no vio el turno de régimen: {regimen}",
    )
    _require(
        "stages" not in regimen,
        "el resumen de régimen trae desglose por etapa y PoC3 no tiene etapas",
    )

    verdict = collector.clock_verdict()
    return (
        f"contrato de captura correcto | apertura 1,200 ms, régimen 2,500 ms | "
        f"4 etapas omitidas del JSON | veredicto del reloj: {verdict['verdict']}"
    )


def check_barge_in() -> str:
    """El criterio de `interrumpido`: corte sí, solape full-duplex no.

    Es el chequeo que evita repetir el error que PoC1 ya cometió una vez (marcar
    todos los turnos como interrumpidos) y que aquí sería peor, porque en una
    arquitectura full-duplex el solape es el modo normal de operación.
    """
    from poc3_gpt_live.app.latency import INBOUND_FRAME_S, CUTOFF_WINDOW_S

    def launch(*, go_silent_after: float, overlap_at: float = 2.05) -> bool:
        """Un turno donde el transportista habla encima y el agente calla después.

        Las dos rachas se alimentan trama por trama: el detector agrupa la voz, y
        una marca suelta es ruido de línea que no puede ser el cero de un turno.
        """
        now = [0.0]
        collector = LatencyCollector(clock=lambda: now[0])
        collector.begin()
        collector.anchor_server({})

        def streak(since: float, until: float, step: float, mark) -> None:
            t = since
            while t <= until + 1e-9:
                now[0] = round(t, 6)
                mark()
                t += step

        # El agente habla corrido de 1.0 a 2.0 (los deltas traen 100 ms).
        streak(1.0, 2.0, 0.1, collector.outbound_speech)

        # El transportista se le encima y habla hasta los 2.5.
        streak(overlap_at, 2.5, INBOUND_FRAME_S, collector.inbound_speech)

        # El agente sigue sonando `callar_tras` segundos después del solape.
        streak(2.1, overlap_at + go_silent_after, 0.1, collector.outbound_speech)

        # Calla, y contesta bastante después.
        now[0] = overlap_at + go_silent_after + 2.0
        collector.outbound_speech()

        return collector.samples[-1].interrupted

    cutoff = launch(go_silent_after=CUTOFF_WINDOW_S / 2)
    _require(cutoff, "un barge-in que calló al agente debió marcar el turno")

    overlap = launch(go_silent_after=CUTOFF_WINDOW_S * 4)
    _require(
        not overlap,
        "un solape donde el agente siguió hablando NO debe marcar el turno: en "
        "full-duplex escuchar mientras se habla es el modo normal",
    )

    # El signo de la brecha: el transportista arranca cuando el agente YA se había
    # callado, así que no cortó a nadie. Sin cota inferior en la brecha esto
    # saldría `interrumpido`.
    late = launch(go_silent_after=0.0, overlap_at=2.4)
    _require(
        not late,
        "hablar DESPUÉS de que el agente terminó no es un corte: la brecha entre "
        "el fin de su voz y el solape es negativa y debe descartarse",
    )
    return (
        f"barge-in que corta -> interrumpido | solape que no corta -> no "
        f"(ventana {CUTOFF_WINDOW_S} s)"
    )


def _no_warnings(check: Callable[[], str]) -> Callable[[], str]:
    """Envuelve un chequeo cuyos casos negativos son intencionales."""

    def wrapped() -> str:
        with _silence_expected_warnings():
            return check()

    return wrapped


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("configuración y endpoint de GPT-Live", check_config),
    ("modelo de delegación congelado", check_frozen_delegation_model),
    ("catálogo de compensaciones", check_compensation_catalog),
    ("plantilla del system prompt", check_prompt_template),
    ("interpolación del prompt", check_prompt_render),
    ("ausencia del precio máximo", check_absence_of_max_price),
    ("schemas de tool-calling", check_schemas),
    ("objeto de delegación", check_delegation),
    ("sesión inicial (session.start)", check_initial_session),
    ("manejadores contra el motor", _no_warnings(check_handlers)),
    ("contrato de tool-calling (stub)", _no_warnings(check_tool_calling_contract)),
    ("registro persistible", _no_warnings(check_persistence_without_leaks)),
    ("códec μ-law y puente de audio", _no_warnings(check_codec_and_bridge)),
    ("contrato de captura de latencia", check_latency_contract),
    ("criterio de barge-in", check_barge_in),
]


def pending_warnings(settings: Settings) -> list[str]:
    warning_list: list[str] = []
    try:
        verify_for_voice(settings)
    except IncompleteConfiguration as error:
        warning_list.append(f"sesión real del alpha todavía no se puede correr: {error}")
    try:
        verify_for_telephony(settings)
    except IncompleteConfiguration as error:
        warning_list.append(f"llamada real todavía no se puede correr: {error}")
    return warning_list


def run_level_0() -> int:
    """Corre todos los chequeos offline. Devuelve el número de fallas."""
    print("Chequeo de humo PoC3 - nivel 0 (sin red, sin audio, costo cero)\n")

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
# Sonda de `include` — cuesta $0
# --------------------------------------------------------------------------

# Valor deliberadamente inválido. La sesión inicial del alpha es estricta y
# rechaza lo que no conoce; el sobre de error trae `param` y `message`, y con
# suerte enumera lo aceptado.
PROBE_VALUE_INCLUDE: Final[str] = "__invalid_probe__"


async def do_probe_include(settings: Settings) -> dict[str, Any]:
    """Le pregunta al servidor qué acepta en `include`. No cuesta nada.

    `include` es campo real de la sesión —`session.started` lo devuelve, y la
    guía dice explícito que "`model` and `include` are not update fields", o sea
    que solo se fija al inicializar— pero **sus valores válidos no están
    documentados en ningún lado**. Es el sospechoso de que no lleguen
    `output_transcript.added`.

    Adivinar un valor sería inventar schema del alpha, que es justo lo que el
    guardarraíl prohíbe. Preguntarle al servidor no lo es: se manda un valor
    imposible y se lee el error. Un error de arranque **impide** `session.started`
    (`docs_alpha/Event_API_reference.html`), así que la sonda no abre sesión, no
    delega y no factura.

    Si en vez de rechazar arranca, también es hallazgo: significa que `include`
    no se valida, y entonces no es la causa del bug #2.
    """
    from poc3_gpt_live.app.telephony import build_initial_session

    negotiation = NegotiationSession(trip=TEST_TRIP, call_id="sonda-include")
    session = build_initial_session(settings, negotiation)
    session["include"] = [PROBE_VALUE_INCLUDE]

    client = bridge.GptLiveClient(
        settings.url_live(), settings.headers_live(), session
    )
    try:
        confirmed = await client.connect()
    except bridge.AlphaError as error:
        return {"rechazada": True, "detail": str(error)}
    except Exception as error:  # noqa: BLE001 - la sonda nunca debe tumbar el humo
        return {"rechazada": None, "detail": f"{type(error).__name__}: {error}"}

    # Arrancó: `include` no se valida contra un enum. Se cierra de inmediato.
    with contextlib.suppress(Exception):
        await client.shut()
    return {
        "rechazada": False,
        "include_confirmed": confirmed.get("include"),
        "detail": "el alpha aceptó un valor imposible: `include` no se valida",
    }


def _report_probe_include(resultado: dict[str, Any]) -> None:
    print("\n" + "-" * 70)
    print("Sonda de `include` (costo $0 — un error de arranque no abre sesión)\n")
    if resultado.get("rechazada") is True:
        print(f"{OK_MARK}el alpha rechazó el valor inválido, como se esperaba")
        print(f"        {resultado['detail']}")
        print(
            "\n        Si el mensaje de arriba enumera los valores aceptados, ese\n"
            "        es el arreglo del bug #2. Si no los enumera, `include` no se\n"
            "        puede resolver por documentación ni por sonda."
        )
    elif resultado.get("rechazada") is False:
        print(f"{WARNING_MARK}el alpha ACEPTÓ un `include` imposible")
        print(f"        confirmado: {resultado.get('include_confirmed')!r}")
        print(
            "        No hay validación de enum, así que `include` no explica la\n"
            "        ausencia de output_transcript.added."
        )
    else:
        print(f"{WARNING_MARK}la sonda no concluyó: {resultado['detail']}")


# --------------------------------------------------------------------------
# Nivel 1: sesión real del alpha, sin teléfono
# --------------------------------------------------------------------------


def read_wav_as_ulaw(ruta: Path, *, max_seconds: float) -> bytes:
    """Un WAV cualquiera -> μ-law 8 kHz, como lo entregaría Twilio.

    Se convierte a lo que manda el teléfono **antes** de meterlo al puente, para
    que el nivel 1 ejercite el camino completo —incluido el remuestreo— y no una
    versión de laboratorio con audio de banda ancha.
    """
    with wave.open(str(ruta), "rb") as file:
        channel_list = file.getnchannels()
        width = file.getsampwidth()
        sample_rate = file.getframerate()
        frames = file.readframes(file.getnframes())

    if width != bridge.BYTES_PER_SAMPLE:
        raise SmokeFailure(f"{ruta.name} no es PCM de 16 bits (ancho={width})")

    samples = np.frombuffer(frames, dtype="<i2")
    if channel_list > 1:
        samples = samples.reshape(-1, channel_list).mean(axis=1).astype(np.int16)

    import soxr

    if sample_rate != bridge.TELEPHONY_SAMPLE_RATE:
        samples = soxr.resample(samples, sample_rate, bridge.TELEPHONY_SAMPLE_RATE, quality="HQ")
        samples = np.asarray(samples, dtype=np.int16)

    tope = int(max_seconds * bridge.TELEPHONY_SAMPLE_RATE)
    return bridge.pcm16_a_ulaw(samples[:tope])


async def run_level_1(*, ruta_wav: Path) -> int:
    """Abre una sesión real del alpha y le alimenta audio. Devuelve fallas.

    No hay teléfono: el WAV entra por el mismo `PuenteDeAudio` que usaría el
    Media Stream, y el audio del agente sale por el otro extremo del puente en
    vez de irse a Twilio. Todo lo demás —la sesión, la delegación, el tool
    calling, la captura de latencia— es exactamente lo que corre en la llamada.
    """
    settings = load_settings()
    verify_for_voice(settings)

    if not ruta_wav.is_file():
        print(f"{FAILURE_MARK}no existe el WAV de entrada: {ruta_wav}")
        print(
            "        Pásale uno con --wav. Sirve cualquier grabación de voz en "
            "español;\n        la del humo de PoC1 "
            "(poc1_pipecat/grabaciones/humo_nivel1.wav) es la\n        más a mano."
        )
        return 1

    negotiation = NegotiationSession(trip=TEST_TRIP, call_id="humo-nivel1")
    collector = LatencyCollector()
    collector.begin()

    from poc3_gpt_live.app.telephony import (
        OPENING_CONTEXT,
        CallRecord,
        build_initial_session,
        save_run,
    )

    record = CallRecord()
    live_bridge = bridge.AudioBridge()
    # El censo registra todo lo que llegue, no solo lo que el despacho reconoce:
    # es la prueba de qué eventos manda de verdad el servidor, en vez de otra
    # ronda de suposiciones.
    census = bridge.EventCensus(RECORDINGS_PATH / "humo" / "census_humo-nivel1.jsonl")
    client = bridge.GptLiveClient(
        settings.url_live(),
        settings.headers_live(),
        build_initial_session(settings, negotiation),
        census=census,
    )

    print(f"  conectando a {settings.url_live()} …")
    session_confirmed = await client.connect()
    collector.anchor_server(session_confirmed)
    print(f"{OK_MARK}sesión iniciada: {session_confirmed.get('id', '?')}")

    audio_bytes = 0
    functions: list[str] = []
    agent_with_turn = [False]
    replies = bridge.ResponseTracker()
    # Energía del audio de salida. Sin esto no se puede distinguir "el alpha no
    # manda output_transcript.added" de "el agente casi no habló": las dos cosas
    # producen una transcripción vacía del agente y solo una es un bug.
    energy: list[float] = []

    async def listen() -> None:
        nonlocal audio_bytes
        async for event in client.listen():
            tipo = str(event.get("type", ""))
            if tipo == bridge.EV_OUTPUT_AUDIO_DELTA:
                start_ms = event.get("start_ms")
                end_ms = event.get("end_ms")
                if agent_with_turn[0]:
                    collector.agent_audio(start_ms=start_ms, end_ms=end_ms)
                else:
                    collector.agent_first_audio(start_ms=start_ms, end_ms=end_ms)
                    agent_with_turn[0] = True
                raw_value = bridge.event_audio(event)
                if raw_value:
                    rms = bridge.rms_pcm16(raw_value)
                    energy.append(rms)
                    # Marca de latencia. Igual que en la llamada real.
                    if rms >= bridge.SPEECH_THRESHOLD_RMS:
                        collector.outbound_speech()
                audio_bytes += len(live_bridge.a_twilio(raw_value))
            elif tipo == bridge.EV_INPUT_TRANSCRIPT:
                collector.user_started_speaking()
                record.annotate("carrier", bridge.transcript_text(event))
            elif tipo == bridge.EV_OUTPUT_TRANSCRIPT:
                record.annotate("agent", bridge.transcript_text(event))
            elif tipo == bridge.EV_RESPONSE_CREATED:
                replies.created(event)
            elif tipo in bridge.RESPONSE_END_EVENTS:
                if replies.finished(event):
                    await client.proceed_delegation()
            elif tipo == bridge.EV_OUTPUT_ITEM_DONE:
                data = replies.new_call(event)
                if data is None:
                    continue
                call_id, name, parsed_args = data
                functions.append(name)
                print(f"      -> {name}({parsed_args})")
                response = execute_function(negotiation, name, parsed_args)
                print(f"      <- {response}")
                await client.answer_function(
                    build_output_event(call_id, response)
                )
                if replies.answered(call_id):
                    await client.proceed_delegation()
            elif tipo == bridge.EV_ERROR:
                detail = event.get("error") or {}
                collector.record_failure(
                    str(detail.get("type") or "error"),
                    code_value=str(detail.get("code") or ""),
                    message=str(detail.get("message") or ""),
                )
                print(f"{WARNING_MARK}error de GPT-Live: {detail}")

    async def feed() -> None:
        """Manda el WAV al ritmo de Twilio: una trama de 20 ms cada 20 ms."""
        # Primero el hecho de apertura: es lo único que hace hablar al agente.
        await client.add_context(OPENING_CONTEXT)

        ulaw = read_wav_as_ulaw(ruta_wav, max_seconds=MAX_AUDIO_SECONDS)
        print(
            f"  alimentando {len(ulaw) / bridge.TELEPHONY_SAMPLE_RATE:.1f} s de audio "
            f"desde {ruta_wav.name} …"
        )
        speech_frames = int(SPEECH_SECONDS_PER_TURN * 1000 / MS_PER_FRAME)
        pause_frames = int(PAUSE_SECONDS_BETWEEN_TURNS * 1000 / MS_PER_FRAME)
        silence = bridge.pcm16_a_ulaw(
            np.zeros(SAMPLES_PER_FRAME, dtype=np.int16)
        )

        async def send(trama: bytes) -> None:
            pcm24 = live_bridge.from_twilio(trama)
            if not pcm24:
                return
            # Marca de latencia entrante, igual que en la llamada real.
            if bridge.has_speech(pcm24):
                collector.inbound_speech()
            await client.send_audio(pcm24)

        n = 0
        for i in range(0, len(ulaw), SAMPLES_PER_FRAME):
            await send(ulaw[i : i + SAMPLES_PER_FRAME])
            await asyncio.sleep(MS_PER_FRAME / 1000.0)
            n += 1
            if n % speech_frames:
                continue
            # Le cede el turno al agente.
            for _ in range(pause_frames):
                await send(silence)
                await asyncio.sleep(MS_PER_FRAME / 1000.0)
        # Un respiro para que el modelo cierre el último turno y conteste.
        await asyncio.sleep(5.0)

    listen_task = asyncio.create_task(listen())
    try:
        await asyncio.wait_for(
            feed(), timeout=MAX_AUDIO_SECONDS + FIRST_AUDIO_TIMEOUT_S
        )
    finally:
        listen_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listen_task
        # Cerrar PRIMERO y copiar el uso después. `cerrar` manda `session.close`
        # y drena hasta `session.closed`, que es el evento que trae el total de
        # la sesión: copiarlo antes deja el penúltimo `session.usage.updated`, y
        # con la tarifa por sesión ese contador es con lo que se cotiza.
        #
        # Y el cierre va envuelto: si truena, lo que se pierde es el drenado, no
        # el uso que ya se tenía ni el censo, que es la traza de la corrida.
        with contextlib.suppress(Exception):
            await client.shut()
        collector.usage = dict(client.usage)
        census.shut()

    # -- Reporte --

    print("\n" + "-" * 70)
    print("Flujo de eventos\n")

    faults = 0
    collector.close_pending_ones()
    samples = collector.samples

    print(f"  turnos capturados    {len(samples)}")
    print(f"  audio del agente     {audio_bytes} bytes μ-law")
    print(f"  funciones llamadas   {functions or '(ninguna)'}")
    print(f"  transcripción        {sorted(record.roles()) or '(vacía)'}")

    # ¿El agente habló de verdad, o mandó silencio? Es lo que decide si una
    # transcripción vacía del agente es un bug del alpha o comportamiento.
    if energy:
        with_voice = [e for e in energy if e >= SPEECH_THRESHOLD_RMS]
        print(
            f"  energía del audio    RMS mediana {median(energy):,.0f} | "
            f"{len(with_voice)}/{len(energy)} tramas por encima de "
            f"{SPEECH_THRESHOLD_RMS} ({100 * len(with_voice) / len(energy):.0f}%)"
        )
        if not with_voice:
            print(
                f"{WARNING_MARK}el agente emitió audio pero TODO es silencio: la "
                "transcripción vacía\n        del agente no prueba que falte "
                "session.output_transcript.delta"
            )

    if not audio_bytes:
        faults += 1
        print(f"{FAILURE_MARK}el modelo no produjo audio")
    if not samples:
        faults += 1
        print(f"{FAILURE_MARK}el colector no produjo ninguna muestra de latencia")

    measured_items = [m for m in samples if m.full_turn_ms is not None]
    if measured_items:
        print(f"\n  {'turn':<8}{'bucket':<14}{'full_turn':>16}")
        for sample in samples:
            valor = (
                f"{sample.full_turn_ms:,.0f} ms"
                if sample.full_turn_ms is not None
                else "-"
            )
            print(f"  {sample.indice:<8}{sample.bucket:<14}{valor:>16}")
    else:
        faults += 1
        print(
            f"{FAILURE_MARK}ninguna muestra trae full_turn_ms: el anclaje del "
            "reloj o el primer audio no se están registrando"
        )

    verdict = collector.clock_verdict()
    print(f"\n  cero del turno       {verdict.get('cero', '?')}")
    print(f"  veredicto del cero   {verdict['verdict']}")
    print(f"                       {verdict['detail']}")
    if verdict.get("residual_median_ms") is not None:
        print(
            f"  residual             mediana {verdict['residual_median_ms']:,.0f} ms"
            f" | deriva {verdict['drift_ms']:,.0f} ms"
        )

    usage = collector.usage or {}
    if usage:
        print(
            # La API pública reporta segundos de sesión (`usage.seconds`), que
            # es lo que se factura; el alpha reportaba tokens.
            f"  uso                  {usage['seconds']} s de sesión"
            if "seconds" in usage
            else f"  uso                  {usage.get('total_tokens', '?')} tokens "
            f"({usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out)"
        )

    # El JSON, con la misma forma que la llamada real. En `grabaciones/humo/`,
    # no en `grabaciones/`: el consolidado recoge todo `llamada_*.json` de la
    # carpeta, y una corrida de humo colada entre las 15 reales ensuciaría los
    # percentiles.
    destination = save_run(
        negotiation,
        record,
        base_path=RECORDINGS_PATH / "humo",
        collector=collector,
        identification={"poc": "3", "scenario": "0", "repetition": "0"},
    )
    print(f"\n  corrida guardada en  {destination}")

    content = json.loads(destination.read_text(encoding="utf-8"))
    for key in ("latencies", "transcript", "summary", "gpt_live_extra"):
        if content.get(key) not in ([], {}, None):
            print(f"{OK_MARK}llave `{key}` presente y con contenido")
        else:
            faults += 1
            print(f"{FAILURE_MARK}llave `{key}` ausente o vacía en el JSON")

    if not functions:
        print(
            f"{WARNING_MARK}el agente no llamó ninguna función. Con un WAV que no "
            "actúa el guion\n        es esperable: el tool calling se verifica en "
            "el nivel 0 con el stub."
        )

    faults += _verify_campaign_contract(
        record=record,
        negotiation=negotiation,
        samples=samples,
        energy=energy,
        collector=collector,
    )
    _report_census(census)
    return faults


def _verify_campaign_contract(
    *,
    record: Any,
    negotiation: NegotiationSession,
    samples: list[Any],
    energy: list[float],
    collector: LatencyCollector,
) -> int:
    """Los 4 renglones de `docs/METRICS_CONTRACT.md` §2.6. Devuelve fallas.

    Verifica que el JSON traiga lo que hace falta para medir, no solo que las
    llaves existan: un humo puede salir con exit 0 sin transcripción del agente
    y sin una sola función ejecutada.
    """
    print("\n" + "-" * 70)
    print("Contrato de campaña (CONTRATO_METRICAS.md §2.6)\n")

    faults = 0

    def table_row(ok: bool, title: str, detail: str) -> None:
        nonlocal faults
        print(f"{OK_MARK if ok else FAILURE_MARK}{title}")
        print(f"        {detail}")
        if not ok:
            faults += 1

    roles = record.roles()
    table_row(
        "agent" in roles,
        "1. la transcripción del agente llega",
        f"roles: {sorted(roles) or '(vacío)'}",
    )

    offers = list(negotiation.offer_history)
    table_row(
        bool(offers),
        "2. el tool calling ejecuta",
        f"offer_history: {[int(o) for o in offers] or '(vacío)'}",
    )

    # Se piden números **creíbles**, no una proporción. El WAV del humo le habla
    # encima al agente, así que muchos turnos son legítimamente no medibles y una
    # regla de mayoría solo mediría el andamiaje. Lo que sí tiene que cumplirse:
    # que haya suficientes para un percentil y que ninguno sea un artefacto.
    values = [m.full_turn_ms for m in samples if m.full_turn_ms is not None]
    lows = [v for v in values if v < PLAUSIBLE_LATENCY_FLOOR_MS]
    table_row(
        len(values) >= MIN_MEASURED_SAMPLES and not lows,
        "3. la latencia produce números creíbles",
        f"{len(values)}/{len(samples)} muestras medidas "
        f"(mínimo {MIN_MEASURED_SAMPLES}) | cero: {collector.zero_used()}"
        + (
            f" | {len(lows)} por debajo de {PLAUSIBLE_LATENCY_FLOOR_MS:.0f} ms: "
            f"{[round(v) for v in lows]}"
            if lows
            else f" | rango {min(values):,.0f}-{max(values):,.0f} ms"
            if values
            else ""
        ),
    )

    with_voice = [e for e in energy if e >= bridge.SPEECH_THRESHOLD_RMS]
    table_row(
        bool(with_voice),
        "4. el agente habló de verdad",
        f"{len(with_voice)}/{len(energy)} tramas sobre {bridge.SPEECH_THRESHOLD_RMS} RMS",
    )

    # Y el que salió de la primera llamada real: un handoff sin resolver deja al
    # modelo esperando una respuesta que nunca llega.
    handoffs = [
        f for f in collector.failures if "handoff" in str(f.get("detail", "")).lower()
    ]
    table_row(
        not handoffs,
        "5. sin errores de handoff sin resolver",
        f"{len(handoffs)} error(es) de handoff"
        + (f": {handoffs[0]['detail']}" if handoffs else ""),
    )

    return faults


def _report_census(census: bridge.EventCensus) -> None:
    """El inventario de lo que el alpha mandó de verdad.

    Es el entregable principal de una corrida instrumentada: sin él, cada
    discrepancia entre `docs_alpha/` y el servicio cuesta otra corrida.
    """
    summary = census.summary()

    print("\n" + "-" * 70)
    print("Censo de eventos del alpha\n")
    print(f"  archivo              {summary['file']}")
    print(f"  eventos              {summary['events']}")

    print(f"\n  {'tipo':<48}{'n':>6}  manejado")
    for tipo, n in summary["count_by_kind"].items():
        mark = "sí" if tipo in bridge.HANDLED_EVENTS else "no"
        print(f"  {tipo:<48}{n:>6}  {mark}")

    print("\n  orden de primera aparición:")
    for i, tipo in enumerate(summary["first_appearance"], 1):
        print(f"    {i:>2}. {tipo}")

    unknown_ones = summary["unknown_ones"]
    if unknown_ones:
        print(
            f"\n{WARNING_MARK}{len(unknown_ones)} tipo(s) que no están en "
            "docs_alpha/ ni se manejan:"
        )
        for tipo, n in sorted(unknown_ones.items()):
            print(f"        {tipo} (x{n})")
    else:
        print("\n  no llegó ningún tipo fuera de lo documentado")


def _print_cost_warning(ruta_wav: Path) -> None:
    print(
        "\n".join(
            (
                "",
                "El nivel 1 abre una sesión REAL contra el alpha de GPT-Live.",
                "No hay telefonía: no se marca a nadie.",
                "",
                f"  audio de entrada  {ruta_wav}",
                f"  costo estimado    ~${LEVEL_1_COST_USD:.2f} USD por corrida",
                "",
                "Con el modelo en GA se facturan DOS renglones y el grande es la",
                "sesión de voz: $0.05 por minuto de `gpt-live-1`, cobrado por",
                "segundo. La delegación al modelo de texto, que es donde corre el",
                "tool calling, se cobra aparte y pesa mucho menos. Desglose,",
                "supuestos y tope de gasto en docs/costos_poc3.md.",
                "",
                "La corrida abre sesión ~70 s, casi todos de espera: el aviso",
                "está calculado sobre eso y no sobre el audio que se alimenta.",
                "",
                "Para correrlo:",
                "  uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke "
                "--online --accept-cost",
                "",
            )
        )
    )


def _configure_logs(*, diagnostics: bool = False) -> None:
    """Baja el ruido para que el reporte se lea.

    El root se queda en WARNING **incluso en diagnóstico**: ahí viven `websockets`
    y `httpx`, que en DEBUG vuelcan cada trama del WebSocket —incluidos los
    payloads de audio en base64— y convierten el diagnóstico en ilegible. Solo se
    sube lo del propio proyecto, que ya sabe reemplazar el audio por su tamaño
    (`bridge.summarize_for_log`).
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("poc3_gpt_live").setLevel(
        logging.DEBUG if diagnostics else logging.INFO
    )
    logging.getLogger("shared").setLevel(
        logging.INFO if diagnostics else logging.WARNING
    )


def main(argv: Sequence[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.app.smoke",
        description=(
            "Chequeo de humo de PoC3. Sin argumentos corre el nivel 0: sin red, "
            "sin audio y sin costo."
        ),
    )
    arg_parser.add_argument(
        "--online",
        action="store_true",
        help="corre además el nivel 1: sesión real de GPT-Live, sin teléfono",
    )
    arg_parser.add_argument(
        "--accept-cost",
        action="store_true",
        help="confirma el gasto del nivel 1 (ver docs/costos_poc3.md)",
    )
    arg_parser.add_argument(
        "--wav",
        type=Path,
        default=WAV_DEFAULT,
        help="WAV de entrada para el nivel 1 (default: el del humo de PoC1)",
    )
    arg_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="sube los logs del proyecto a DEBUG. Úsalo cuando el nivel 1 falle.",
    )
    arg_parser.add_argument(
        "--probe-include",
        action="store_true",
        help=(
            "antes del nivel 1, pregunta al alpha qué acepta en `include` "
            "mandando un valor imposible. No abre sesión: cuesta $0."
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
        _print_cost_warning(args.wav)
        return 2

    print("\n" + "=" * 70)
    print("Chequeo de humo PoC3 - nivel 1 (sesión real de GPT-Live, sin teléfono)")
    print("=" * 70)

    if args.probe_include:
        _report_probe_include(asyncio.run(do_probe_include(load_settings())))

    start = time.monotonic()
    faults = asyncio.run(run_level_1(ruta_wav=args.wav))

    print("\n" + "-" * 70)
    print(f"Nivel 1 en {time.monotonic() - start:.1f} s.")
    if faults:
        print(f"Nivel 1: {faults} problema(s). Revisa el detalle de arriba.")
        return 1
    print("Nivel 1: el flujo de eventos de GPT-Live funciona de extremo a extremo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
