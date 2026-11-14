"""Telefonía de PoC3: Twilio Media Streams sobre WebSocket, servido por FastAPI.

**La capa de telefonía es idéntica a la de PoC1, a propósito**: mismo TwiML,
mismo `<Connect><Stream>`, mismo número, mismo destino. Así la diferencia que
mida la comparativa es del motor conversacional y no del bus de medios.

Lo que cambia por completo es lo que hay del otro lado. En PoC1 el WebSocket de
Twilio entra a un `Pipeline` de pipecat con STT, LLM, TTS, VAD y analizador de
fin de turno colgando de él. Aquí no hay pipeline: hay **dos WebSockets y un
puente entre ellos**.

    Twilio  --μ-law 8k-->  puente  --PCM16 24k-->  GPT-Live
    Twilio  <--μ-law 8k--  puente  <--PCM16 24k--  GPT-Live

Tres endpoints, los mismos de PoC1:

    POST/GET /twiml  -> el TwiML que Twilio pide al contestar la llamada
    WS       /ws     -> el Media Stream bidireccional
    GET      /salud  -> para que el marcador confirme el túnel antes de gastar

Sobre la cuenta de Twilio (CLAUDE.md): este módulo **nunca** modifica la
configuración de un número ni de un trunk. Solo sirve TwiML por HTTP y habla por
el WebSocket. La `url` del TwiML viaja en la llamada saliente, así que el webhook
configurado en el número del agente despachador no se toca.

## Tres cosas que GPT-Live obliga a resolver distinto

**1. No hay un "habla ahora".** Ningún evento de cliente dispara una respuesta
del modelo de voz (`response.create` es solo para el modelo delegado). Pero PoC3
hace llamadas **salientes**, donde el agente abre — y los 5 escenarios del guion
lo asumen. Con el alpha el único canal era `session.context.append`; en la API
pública es `session.commentary.append`, que es el que la propia API destina a
"esto dilo, con tus palabras" y el que usa la integración de referencia de
pipecat para que el modelo hable primero. Se le manda **un hecho, no una
instrucción**: "El transportista acaba de contestar." El comportamiento sigue
saliendo entero de `v0.md`, que es lo que tiene que quedar idéntico en las 3 PoC.
Es una desviación real y está declarada en el README y en el contrato.

**2. No existe `output_audio.done`.** El agente termina de hablar cuando dejan de
llegar deltas, y no hay evento que lo diga. Para colgar tras cerrar la
negociación —el `stop_when_done()` de PoC1, el `drain()` de PoC2— hay que
esperar un silencio: `SILENCE_TO_HANG_UP_S`. Es un temporizador donde las
otras dos PoC tienen una señal, y por eso el cuelgue de PoC3 es más lento por
construcción.

**3. El audio que ya salió no se recupera.** GPT-Live genera más rápido que el
tiempo real, así que Twilio acumula cola. Cuando el modelo se calla a media
frase por un barge-in, lo que ya se le mandó a Twilio se sigue oyendo.
`{"event":"clear"}` lo tiraría — y **no se usa**: sería imponerle a una
arquitectura full-duplex la política de interrupción de PoC1, y entonces la
comparativa mediría nuestra política y no el framework. Se deja que el modelo
gobierne su propia voz, y el efecto se anota al puntuar turn-taking.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from poc3_gpt_live.app import bridge, marks
from poc3_gpt_live.app.config import (
    RECORDINGS_PATH,
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc3_gpt_live.app.latency import LatencyCollector
from poc3_gpt_live.app.clock import SAMPLING_PERIOD_S, AnchoredClock
from poc3_gpt_live.app.tools import (
    CLOSING_FUNCTIONS,
    build_delegation,
    build_output_event,
    execute_function,
)
from poc3_gpt_live.diagnostics import audit_recorder, energy_probe
from shared.negotiation_engine import (
    TEST_TRIP,
    TripData,
    CloseReason,
    NegotiationSession,
)
from shared.system_prompt import render_system_prompt

logger = logging.getLogger(__name__)

RUTA_TWIML: Final[str] = "twiml"
RUTA_WS: Final[str] = "ws"
HEALTH_PATH: Final[str] = "health"

DEFAULT_PORT: Final[int] = 8000

# Etiqueta de la corrida. Viaja del marcador -> query string del TwiML ->
# `<Parameter>` del `<Stream>` -> `customParameters` del WebSocket -> JSON. Son
# las tres llaves que `shared.metrics.llamadas.identificar()` busca primero.
RUN_PARAMETERS: Final[tuple[str, ...]] = ("poc", "scenario", "repetition")

CARRIER_ROLE_JSON: Final[str] = "carrier"
AGENT_ROLE_JSON: Final[str] = "agent"

# --------------------------------------------------------------------------
# El canal lateral de texto hacia el arnés
# --------------------------------------------------------------------------
#
# Mismas etiquetas que `poc1_pipecat/app/arnes.py:117-125`: si cada PoC usara
# nombres distintos para el mismo evento, el arnés de cada una tendría que
# traducirlos y la comparación empezaría por el transporte.
#
# **Va ENCENDIDO por default, incluida la campaña.** La campaña de PoC3 corre
# con el arnés por WebSocket local —no hay Twilio en ninguna llamada de
# campaña—, así que el otro lado del cable es siempre quien espera este evento.
# Se apaga con `POC3_SIN_ECO_DE_TEXTO=1`, y el único caso que lo justifica es
# una llamada telefónica real: ahí el otro extremo es Twilio, que no lo espera.
#
# Es la diferencia de forma con PoC1, donde el eco es un `FrameProcessor` que
# el arnés inserta y que en campaña simplemente no existe. Aquí el puente es un
# bucle, no un pipeline, así que el gancho vive dentro y lo que cambia es una
# bandera.
TEXT_EVENT: Final[str] = "harness.text"
FROM_CARRIER: Final[str] = "carrier"
FROM_AGENT: Final[str] = "agent"

# El campo que dice si el fragmento ya trae el separador que lo une al anterior.
# En PoC1 vale `False` —ElevenLabs manda un frame por palabra SIN los espacios— y
# en PoC3 vale `True`. Es el mismo campo con valores distintos, y esa es justo la
# gracia: el arnés reensambla leyendo la bandera y no sabiendo contra qué PoC
# habla. Poner aquí `False` pegaría "teestán" y daría un WER falso; poner `True`
# en PoC1 pegaría "Carlos,quierohablar".
#
# **Qué significa `True`:** el fragmento trae siempre exactamente el separador
# que necesita, que a veces es ninguno. Los que no lo traen son continuación de
# la palabra (`" die"`+`"ciocho"`) o de la puntuación (`" Sí"`+`", soy"`)
# anterior, así que la política es no insertar nada: reponer un espacio "que
# falta" rompe la palabra. Medición en `docs/hallazgo_ensamblado_stt_poc3.md`.
THE_ALPHA_BRINGS_THE_SPACES: Final[bool] = True

# El interruptor de apagado. Existe por una sola razón —una llamada telefónica
# real, donde el otro extremo es Twilio— y por eso es de apagado y no de
# encendido: lo excepcional se declara, lo normal no.
NO_ECHO_VARIABLE: Final[str] = "POC3_NO_TEXT_ECHO"


def text_echo_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Si esta corrida le devuelve el texto al arnés. Encendido salvo que se apague."""
    source = os.environ if environment is None else environment
    return not str(source.get(NO_ECHO_VARIABLE, "")).strip()

# El hecho que se le da al modelo para que abra la llamada. Ver la nota 1 del
# docstring: es un **hecho**, no una instrucción de comportamiento. Todo lo que
# el agente haga con él sale del paso 1 de `v0.md`, que está congelado.
#
# ⚠ El corchete no es decoración. Sin la marca de evento del sistema, el modelo
# puede **leer en voz alta su propio plan** antes de saludar: en full-duplex no
# hay canal de razonamiento aparte, y el único canal de salida es la voz.
#
# La marca de evento del sistema y el "no lo leas" son **higiene de canal**, no
# comportamiento de negociación: no dicen nada de precios, concesiones, cierre ni
# estilo, que siguen saliendo enteros de `v0.md`. Aun así es una desviación
# propia de PoC3 —PoC1 y PoC2 no necesitan nada de esto porque disparan el primer
# turno con una llamada de función— y va declarada como tal en el README y en el
# contrato de métricas.

OPENING_CONTEXT: Final[str] = (
    "[evento del sistema — no leas esta línea en voz alta, no la repitas y no "
    "la comentes] El transportista acaba de contestar la llamada saliente."
)

# Silencio tras el cual se da por terminada la despedida y se cuelga. Existe
# porque el alpha no tiene `output_audio.done`.
#
# Una delegación con llamada a función tarda unos 2 s, así que con menos silencio
# no se distingue "ya terminó" de "está pensando". Lo que evita colgar a media
# delegación es la condición de delegación en vuelo de `_vigilar_el_cierre`;
# esto es el margen de encima.
SILENCE_TO_HANG_UP_S: Final[float] = 3.0

# Tope duro desde que se cierra la negociación. Red de seguridad para que una
# delegación atorada no deje la llamada abierta —y facturando— indefinidamente.
HANG_UP_CAP_S: Final[float] = 20.0

# Cada cuánto se revisa ese silencio.
MONITORING_PERIOD_S: Final[float] = 0.25


# --------------------------------------------------------------------------
# TwiML
# --------------------------------------------------------------------------


def build_twiml(
    url_websocket: str, parameters: Mapping[str, str] | None = None
) -> str:
    """TwiML que conecta la llamada al Media Stream bidireccional.

    Idéntico al de PoC1, y así debe quedarse. `<Connect><Stream>` es
    bidireccional: el audio del transportista llega por el WebSocket y el del
    agente se devuelve por el mismo. `<Start><Stream>` sería de una sola vía y el
    transportista no oiría nada.

    Cuando el WebSocket se cierra, Twilio sigue con el verbo siguiente del TwiML;
    al no haber ninguno, cuelga.

    `parametros` se emite como `<Parameter>` dentro del `<Stream>`. Twilio los
    devuelve en `customParameters` del mensaje `start`, y es así como el escenario
    y la repetición llegan hasta el JSON de la corrida.
    """
    labels = ""
    for name, valor in (parameters or {}).items():
        text = str(valor).strip()
        if not text:
            continue
        labels += (
            f"<Parameter name={quoteattr(str(name))} value={quoteattr(text)} />"
        )

    stream = f"<Stream url={quoteattr(url_websocket)}>{labels}</Stream>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Connect>{stream}</Connect>"
        "</Response>"
    )


# --------------------------------------------------------------------------
# Registro del diálogo
# --------------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """Colapsa las rachas de espacios a uno solo y recorta las puntas.

    Es lo único que se le hace al texto del alpha, y se hace **al final**, nunca
    por fragmento: los fragmentos traen su propio espacio inicial y dos seguidos
    partirían un token de más al calcular el WER. La puntuación y las mayúsculas
    se dejan como están — son la hipótesis del STT y normalizarlas aquí falsearía
    la métrica.

    Vive suelta y no dentro de `RegistroDeLlamada` porque la usan dos: el JSON de
    la llamada y `media_streams.ObservadoDePoC3.assemble_agent`. Es el mismo
    motivo por el que PoC1 reusa `pipeline.join_tts_fragments` desde su arnés —
    si cada lado normalizara por su cuenta, el texto del arnés y el del JSON
    podrían discrepar y el WER dependería de cuál se mirara.
    """
    return " ".join(str(text).split())


@dataclass
class CallRecord:
    """Transcripción de la llamada, con el segundo relativo de cada intervención.

    Misma forma que PoC1 y PoC2, para que `shared/metrics/` la lea sin ramas.

    En PoC3 las dos partes salen del **mismo** motor: `input_transcript.added` es
    lo que el modelo entendió del transportista y `output_transcript.added` es lo
    que él mismo dijo. En PoC1 y PoC2 la primera la produce Deepgram y la segunda
    se reconstruye del TTS. Es una asimetría del WER que hay que tener presente
    al leerlo: aquí la hipótesis la escribe el propio sistema que se evalúa.

    ## Los fragmentos se reensamblan en turnos

    `input_transcript.added` y `output_transcript.added` no entregan turnos:
    entregan **fragmentos por cadencia**, y el propio alpha lo advierte
    ("fragment boundaries reflect cadence, not semantic turn boundaries"): unas
    pocas letras cada uno, muchos por turno.

    Anotarlos uno por uno haría inservible el archivo de referencia del WER y
    desalinearía la comparación contra el guion. Aquí se pegan mientras no
    cambie el rol, que es la misma solución que `AnotadorDeVozDelAgente` de PoC1.

    El texto se concatena **crudo**, sin tocar los espacios: el alpha manda el
    espacio dentro del fragmento (`" Hola,"`), y cualquier regla propia de
    separación partiría las palabras que llegan en dos trozos (`"diecis"` +
    `"iete mil"`). Se limpia al final, en `normalizar()`.

    ⚠ Eso obliga a que `bridge.transcript_text` devuelva el fragmento **sin
    `strip()`**: con él salen `"teestán"` y `"soyCarlos"`, y un WER absurdo sobre
    una transcripción casi perfecta. Si el WER de una corrida sale absurdamente
    alto, mira primero los espacios de este campo.
    """

    start: float = field(default_factory=time.monotonic)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def annotate(self, role: str, text: str) -> None:
        if not text:
            return

        same_turn = bool(self.turns) and self.turns[-1]["role"] == role

        # Un fragmento de PURO ESPACIO se conserva si continúa un turno: el alpha
        # manda el separador como fragmento aparte (`" "` entre `" a"` y `"28"`),
        # y tirarlo pega la palabra con el número (`"a28,000"`): el extractor de
        # montos lo pierde aunque el STT no haya fallado nada.
        #
        # Lo que sí se descarta es el espacio que abriría turno nuevo, porque
        # crearía un turno sin contenido.
        if not text.strip() and not same_turn:
            return

        if same_turn:
            self.turns[-1]["text"] += text
            return

        self.turns.append(
            {
                # El segundo del PRIMER fragmento: es cuando arrancó el turno.
                "t": round(time.monotonic() - self.start, 3),
                "role": role,
                "text": text,
            }
        )
        logger.info("[%s] (turno nuevo)", role)

    def normalize(self) -> None:
        """Limpia los turnos ya cerrados. Se llama antes de persistir.

        Colapsa las rachas de espacios a uno solo: los fragmentos traen su propio
        espacio inicial y dos seguidos partirían un token de más al calcular el
        WER. No se toca nada más — la puntuación y las mayúsculas son la
        hipótesis del STT y normalizarlas aquí falsearía la métrica.
        """
        for turn in self.turns:
            turn["text"] = normalize_text(turn["text"])
        self.turns[:] = [t for t in self.turns if t["text"]]

    def as_text(self) -> str:
        return "\n".join(
            f"[{t['t']:>7.3f}s] {t['role']:>13}  {str(t['text']).strip()}"
            for t in self.turns
        )

    def roles(self) -> set[str]:
        return {str(t.get("role", "")) for t in self.turns}


def read_identification(start: Mapping[str, Any]) -> dict[str, str]:
    """Saca `poc`/`escenario`/`repeticion` del mensaje `start` de Twilio.

    Se lee de forma tolerante —claves en cualquier caja, valores vacíos fuera—
    porque un error aquí no debe tumbar una llamada que ya está sonando.
    """
    raw_values = start.get("customParameters") or {}
    if not isinstance(raw_values, Mapping):
        return {}

    by_lowercase = {str(k).strip().lower(): v for k, v in raw_values.items()}
    label: dict[str, str] = {}
    for clave in RUN_PARAMETERS:
        valor = by_lowercase.get(clave)
        if valor is None:
            continue
        text = str(valor).strip()
        if text:
            label[clave] = text
    return label


# --------------------------------------------------------------------------
# La sesión inicial del alpha
# --------------------------------------------------------------------------


def build_initial_session(
    settings: Settings, negotiation: NegotiationSession
) -> dict[str, Any]:
    """El objeto `session` del `session.start` inicial.

    **Con `model`**: en la API pública el modelo va aquí y no en la URL (con el
    alpha era al revés). El esquema es estricto con los campos desconocidos, así
    que lleva solo `model`, `instructions`, `audio` y `delegation`.

    `audio.format` se pide **explícito**, según `ajustes.audio_format`:
    `pcmu` (μ-law 8 kHz, el de Twilio, sin remuestreo) o `pcm24` (PCM16 24 kHz,
    el de la campaña de septiembre y la de interrupciones de octubre). Fijarlo
    evita que un cambio de default del servidor cambie el camino del audio sin
    que nadie se entere.

    `instructions` es el prompt v0 con las 7 variables resueltas, y es
    **inmutable tras el arranque**: no hay forma de reescribirlo a media llamada,
    solo de agregar con `session.instructions.append`. Da igual para la
    comparativa (el prompt está congelado) pero conviene saberlo.

    `render_system_prompt` verifica en cada render que el precio máximo no se
    haya filtrado, así que ese guardarraíl corre también aquí y no solo en los
    tests.
    """
    return {
        "model": settings.live_model,
        "instructions": render_system_prompt(
            negotiation.trip, compensations_path=negotiation.compensations_path
        ),
        "audio": {
            "output": {"voice": settings.voice},
            "format": bridge.session_format(settings.audio_format),
        },
        "delegation": build_delegation(
            model_name=settings.delegation_model,
            compensations_path=negotiation.compensations_path,
        ),
    }


# --------------------------------------------------------------------------
# El puente de una llamada
# --------------------------------------------------------------------------


class CallBridge:
    """Bombea audio en los dos sentidos y traduce eventos a estado de la llamada.

    Una instancia por llamada. Corre dos tareas concurrentes —una por sentido— y
    termina cuando cualquiera de las dos se acaba: si Twilio cuelga no hay a
    quién hablarle, y si el alpha cierra no hay quién hable.
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        client: bridge.GptLiveClient,
        negotiation: NegotiationSession,
        record: CallRecord,
        collector: LatencyCollector,
        stream_sid: str,
        clock: AnchoredClock | None = None,
        text_echo: bool = False,
        audio_format: str = bridge.PCM24_FORMAT,
    ) -> None:
        self._ws = websocket
        self._client = client
        self._negotiation = negotiation
        self._record = record
        self._collector = collector
        self._stream_sid = stream_sid
        # El eje de pared de esta llamada. `None` solo en pruebas que no lo
        # ejercitan; en producción lo construye `crear_app` ANTES de conectar
        # contra el alpha, para que el ancla quede antes del primer evento.
        self._clock = clock
        # El canal lateral de texto hacia el arnés. Encendido salvo que
        # `POC3_SIN_ECO_DE_TEXTO` lo apague: la campaña de PoC3 corre con el
        # arnés por WebSocket local y el otro lado del cable siempre lo espera.
        self._text_echo = text_echo
        # Lo que se PIDIÓ, que no es lo mismo que `_eco_de_texto` al final de la
        # llamada: una falla lo apaga, y entonces el veredicto tiene que poder
        # distinguir "nunca se encendió" de "se encendió y murió".
        self.text_echo_requested = text_echo
        # Cuántos fragmentos salieron de verdad hacia el arnés. Cero con el eco
        # encendido es una corrida **sorda**, y eso descalifica la llamada: ver
        # `bloque_del_eco_de_texto`.
        self.emitted_text_fragments: int = 0
        self._audio = bridge.AudioBridge(audio_format)

        # Sonda de calibración del umbral de voz. `None` salvo que
        # `POC3_SONDA_DE_ENERGIA` esté puesta, y entonces no se ejecuta ni una
        # línea de ese módulo dentro del ciclo de audio. No decide nada: solo
        # vuelca energía para poder calibrar `bridge.UMBRAL_DE_VOZ_RMS`, que hoy
        # es un valor heredado que nunca vio la portadora del alpha.
        self._probe = energy_probe.begin(negotiation.call_id or stream_sid)

        # Grabación de auditoría de las dos direcciones, para poder oír lo que
        # la sonda dice con números. `None` salvo que `POC3_GRABA_AUDITORIA`
        # esté puesta. Mismo trato de LFPDPPP que el volcado de audio crudo: se
        # enciende a mano, no entra al repo y se borra al congelar.
        self._auditoria = audit_recorder.begin(negotiation.call_id or stream_sid)

        # Fallas del diagnóstico dentro del ciclo de audio. Vacío es lo normal;
        # una entrada aquí significa que esta llamada NO entra al análisis del
        # colgajo, y por eso viaja al JSON en vez de quedarse en el log.
        self.diagnostics_failures: list[dict[str, Any]] = []
        self._t_bridge_start = time.monotonic()

        # Se levanta cuando la negociación cierra: a partir de ahí se espera el
        # silencio de la despedida y se cuelga.
        self._closing = asyncio.Event()
        self._t_last_outbound_audio: float = time.monotonic()
        # Delegaciones en vuelo y cuándo terminó la última. Solo las lee el
        # cuelgue: mientras el modelo delegado esté trabajando, el agente está por
        # hablar y el temporizador de silencio no puede correr.
        self._in_flight_responses: int = 0
        # `response.create` mandados cuyo `response.created` todavía no llega.
        # Ver `_seguir_la_delegacion`.
        self._pending_continuations: int = 0
        self._t_delegation_end: float = time.monotonic()
        # Si ya llegó el primer delta del agente. Con el alpha `turn.done` lo
        # reabría por turno; la API pública no tiene `turn.*`, así que solo
        # distingue el primer delta de la llamada (la apertura) del resto.
        self._agent_with_open_turn: bool = False
        # Llamadas a función por respuesta del delegado, y cuándo reanudarlo.
        self._responses = bridge.ResponseTracker()

    # -- Twilio -> GPT-Live --

    def _turn_off_diagnostics(self, sitio: str, error: BaseException) -> None:
        """Primera falla del diagnóstico: se apaga para el resto de la llamada.

        Tres decisiones, y las tres tienen el mismo motivo — el audio es lo que
        se paga y el diagnóstico es lo que sobra:

        1. **No se reintenta.** Reintentar por trama sería repetir la misma
           excepción 50 veces por segundo y llenar el log mientras la llamada
           corre. Se apaga y ya.
        2. **Se apagan los dos**, sonda y grabadora, aunque falle uno. Comparten
           el decodificado del saliente, y dejar medio diagnóstico vivo produce
           un artefacto a medias sin nada que diga cuál mitad falta.
        3. **No se escribe el artefacto al colgar.** Un WAV que se cortó a los
           30 s de una llamada de 157 se ve igual de entero que uno completo, y
           el modo de falla que se teme es justamente que alguien lo lea como
           bueno. La falla queda en el JSON, que es donde se mira antes de
           contar una llamada.
        """
        self.diagnostics_failures.append(
            {
                "sitio": sitio,
                "excepcion": type(error).__name__,
                "message": str(error)[:300],
                "call_second": round(
                    time.monotonic() - self._t_bridge_start, 3
                ),
                "probe_on": self._probe is not None,
                "audit_enabled": self._auditoria is not None,
            }
        )
        self._probe = None
        self._auditoria = None
        logger.error(
            "el diagnóstico falló en %s y queda apagado el resto de la llamada: "
            "%r. El audio sigue saliendo; esta llamada se excluye del análisis "
            "del colgajo",
            sitio,
            error,
        )

    def _turn_off_text_echo(self, error: BaseException) -> None:
        """Primera falla del eco: se apaga para el resto de la llamada.

        Hermano de `_turn_off_diagnostics` y por el mismo motivo, pero con
        interruptor aparte: el eco no comparte nada con la sonda ni con la
        grabadora, y apagar los tres juntos perdería dos artefactos sanos.

        La falla se anota en la MISMA lista, así que `captura_completa` sale en
        `false` y la llamada queda excluida por escrito. Eso es lo conservador:
        sin eco, el arnés se queda sin la transcripción del agente y su WER de
        esa llamada está incompleto aunque el audio haya salido entero.
        """
        self.diagnostics_failures.append(
            {
                "sitio": "text_echo",
                "excepcion": type(error).__name__,
                "message": str(error)[:300],
                "call_second": round(
                    time.monotonic() - self._t_bridge_start, 3
                ),
                "echo_enabled": True,
                "consecuencia": (
                    "el arnés dejó de recibir texto: su transcripción de esta "
                    "llamada está truncada y el WER derivado no se usa. El audio "
                    "NO se vio afectado"
                ),
            }
        )
        self._text_echo = False
        logger.error(
            "el eco de texto hacia el arnés falló y queda apagado el resto de la "
            "call: %r. El audio sigue saliendo",
            error,
        )

    async def _send_text_to_harness(
        self, of: str, text: str, event: dict[str, Any]
    ) -> None:
        """Un fragmento de transcripción por el mismo cable que el audio.

        Va **después** de que el fragmento ya se anotó en el registro, igual que
        el `EcoDeTexto` de PoC1 empuja su mensaje después de reenviar el frame:
        primero lo que la llamada necesita, después lo que el arnés observa.

        El `texto` viaja **crudo**, sin `strip()`. El alpha manda el espacio
        separador al principio del fragmento y recortarlo aquí sería reponer a
        mano el defecto que dio el WER falso de 84.5%.
        """
        if not self._text_echo or not text:
            return
        try:
            await self._ws.send_text(
                json.dumps(
                    {
                        "event": TEXT_EVENT,
                        "de": of,
                        "text": text,
                        "has_spaces": THE_ALPHA_BRINGS_THE_SPACES,
                        # Siempre `False`: los fragmentos siguen la cadencia del
                        # habla y **no** las fronteras de turno.
                        "final": False,
                        # Propio de PoC3 y sin equivalente en PoC1: permite
                        # cruzar el fragmento contra el eje del servidor sin
                        # abrir el censo. La API pública no manda `item`, así
                        # que `item_id` va vacío; se conserva la llave.
                        "item_id": "",
                        "start_ms": event.get("start_ms"),
                        "end_ms": event.get("end_ms"),
                    }
                )
            )
        except Exception as error:  # noqa: BLE001 - diagnóstico
            self._turn_off_text_echo(error)
            return
        self.emitted_text_fragments += 1

    async def _from_twilio_to_live(self) -> None:
        """Lee el Media Stream y le manda el audio remuestreado al alpha."""
        while True:
            try:
                raw_value = await self._ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                logger.info("Twilio cerró el Media Stream")
                return

            try:
                message = json.loads(raw_value)
            except ValueError:
                logger.warning("mensaje de Twilio ilegible, se ignora")
                continue

            event = str(message.get("event", ""))
            if event == "mean":
                payload = (message.get("mean") or {}).get("payload") or ""
                if not payload:
                    continue
                incoming = base64.b64decode(payload)
                # Canal 0 de la auditoría, fechado aquí: es lo más cerca de la
                # llegada que se puede estar. Twilio manda una trama de 20 ms
                # por mensaje, así que el cabezal no debería correr nada — y si
                # corre, el JSON lo dice.
                if self._auditoria is not None:
                    # El `try` no cuesta nada cuando no hay excepción (CPython
                    # 3.11+), y lo que protege es el reenvío de abajo: una falla
                    # del diagnóstico no puede dejar al alpha sin audio.
                    try:
                        self._auditoria.annotate_carrier(
                            bridge.ulaw_a_pcm16(incoming).tobytes()
                        )
                    except Exception as error:  # noqa: BLE001 - diagnóstico
                        self._turn_off_diagnostics("incoming", error)
                shipment = self._audio.entry(incoming)
                if shipment:
                    # Marca de latencia por energía, antes de mandarlo. SOLO
                    # marca tiempo: el fin de turno lo sigue decidiendo el
                    # servidor sobre este mismo audio.
                    if bridge.has_speech(
                        self._audio.to_measure(shipment),
                        samples_per_window=self._audio.samples_per_window,
                    ):
                        self._collector.inbound_speech()
                    await self._client.send_audio(shipment)
            elif event == "stop":
                logger.info("Twilio mandó `stop`: la llamada terminó")
                return
            elif event == "mark":
                # No se usan marks: el alpha no las necesita y Twilio no las
                # exige. Se registran en debug por si hace falta diagnosticar.
                logger.debug("mark de Twilio: %s", message.get("mark"))

    # -- GPT-Live -> Twilio --

    async def _from_live_to_twilio(self) -> None:
        """Consume los eventos del alpha y despacha cada uno a donde toca."""
        async for event in self._client.listen():
            tipo = str(event.get("type", ""))

            if tipo == bridge.EV_OUTPUT_AUDIO_DELTA:
                await self._handle_audio(event)
            elif tipo == bridge.EV_INPUT_TRANSCRIPT:
                self._handle_incoming_transcript(event)
                await self._send_text_to_harness(
                    FROM_CARRIER, bridge.transcript_text(event), event
                )
            elif tipo == bridge.EV_OUTPUT_TRANSCRIPT:
                text = bridge.transcript_text(event)
                self._record.annotate(AGENT_ROLE_JSON, text)
                await self._send_text_to_harness(FROM_AGENT, text, event)
            elif tipo == bridge.EV_OUTPUT_ITEM_DONE:
                # El item `function_call` completo: trae `call_id`, `name` y
                # `arguments` juntos. Es la señal con la que se ejecuta.
                await self._handle_function_call(event)
                logger.debug("responses: %s", bridge.summarize_for_log(event))
            elif tipo == bridge.EV_ERROR:
                self._handle_error(event)
            elif tipo == bridge.EV_SESSION_CLOSED:
                logger.info("el alpha cerró la sesión")
                return
            elif tipo == bridge.EV_DELEGATION_CREATED:
                # Con delegación `responses` esto solo anuncia que arranca un
                # turno del modelo de texto; el trabajo llega por `response.*`.
                self._collector.delegation_started()
                logger.debug("delegación creada: %s", bridge.summarize_for_log(event))
            elif tipo.startswith(bridge.RESPONSE_PREFIX):
                # La API pide tolerar tipos nuevos de `response.*`. Se loguean
                # completos: son la traza de lo que hizo el modelo delegado.
                if self._track_delegation(event):
                    await self._client.proceed_delegation()
                logger.debug("responses: %s", bridge.summarize_for_log(event))
            else:
                logger.debug("evento no manejado: %s", bridge.summarize_for_log(event))

    async def _handle_audio(self, event: dict[str, Any]) -> None:
        """Un `output_audio.delta`: se marca, se remuestrea y se manda a Twilio.

        La marca de latencia se toma **cuando el delta llega al puente**, no
        cuando Twilio lo reproduce: es la misma convención que PoC1, que ancla en
        `TTSAudioRawFrame` justamente para no depender del buffer del transporte.
        """
        # Instante de llegada del delta: el mismo que fecha la marca del
        # colector, dos líneas abajo. Se toma aquí y no al grabar porque entre
        # este punto y la conversión hay ~24 ms de remuestreo, y el canal del
        # agente quedaría tarde respecto de la marca que este archivo sirve
        # para auditar.
        t_arrival = time.time() if self._auditoria is not None else 0.0

        start_ms = event.get("start_ms")
        end_ms = event.get("end_ms")
        if self._agent_with_open_turn:
            self._collector.agent_audio(start_ms=start_ms, end_ms=end_ms)
        else:
            self._collector.agent_first_audio(start_ms=start_ms, end_ms=end_ms)
            self._agent_with_open_turn = True

        pcm24 = bridge.event_audio(event)
        if not pcm24:
            return

        # Marca de latencia por energía. SOLO marca tiempo: no decide turnos ni
        # interrumpe nada — eso lo sigue haciendo el servidor.
        if bridge.has_speech(
            self._audio.to_measure(pcm24),
            samples_per_window=self._audio.samples_per_window,
        ):
            self._collector.outbound_speech()

        ulaw = self._audio.output(pcm24)
        if not ulaw:
            return

        # Diagnóstico, después de convertir y antes de mandar: así se mide el
        # mismo μ-law que viaja al transportista. Es un `append` a una lista.
        # El decodificado lo comparten la sonda y la grabadora, así que se hace
        # una sola vez y solo si alguna de las dos está encendida.
        if self._probe is not None or self._auditoria is not None:
            # Un solo `try` para los dos: comparten el decodificado, y si el
            # diagnóstico se cae da igual cuál de los dos lo hizo — se apagan
            # juntos. Lo que no puede fallar es el `send_text` de abajo.
            try:
                pcm8 = bridge.ulaw_a_pcm16(ulaw).tobytes()
                # La sonda mide ventanas de 24 kHz: solo tiene sentido en PCM16.
                if self._probe is not None and self._audio.es_pcm:
                    self._probe.annotate(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        pcm24=pcm24,
                        ulaw=ulaw,
                        pcm8=pcm8,
                    )
                if self._auditoria is not None:
                    self._auditoria.annotate_agent(
                        pcm8, t=t_arrival, start_ms=start_ms, end_ms=end_ms
                    )
            except Exception as error:  # noqa: BLE001 - diagnóstico
                self._turn_off_diagnostics("saliente", error)

        self._t_last_outbound_audio = time.monotonic()
        await self._ws.send_text(
            json.dumps(
                {
                    "event": "mean",
                    "streamSid": self._stream_sid,
                    "mean": {"payload": base64.b64encode(ulaw).decode("ascii")},
                }
            )
        )

    def _handle_incoming_transcript(self, event: dict[str, Any]) -> None:
        """Un fragmento de lo que el modelo entendió del transportista.

        Los fragmentos siguen la cadencia del habla, **no las fronteras de
        turno**, así que se anotan tal cual y `RegistroDeLlamada` los junta por
        rol. Es también la señal de que el transportista está hablando, que es
        lo que alimenta la detección de solape.
        """
        self._collector.user_started_speaking()
        self._record.annotate(
            CARRIER_ROLE_JSON, bridge.transcript_text(event)
        )

    async def _handle_function_call(self, event: dict[str, Any]) -> None:
        """Corre una de las 4 funciones y devuelve el resultado por el mismo stream.

        Se manda **un evento por `call_id`** (`response.item.create`). Los IDs
        desconocidos o ya resueltos los rechaza el servidor, así que no se
        reintenta. Si con esta salida la respuesta ya tiene todas las suyas y ya
        terminó, se reanuda al delegado con `response.create`.
        """
        data = self._responses.new_call(event)
        if data is None:
            return
        call_id, name, parsed_args = data
        # Marca el round trip en curso: los turnos con función cuestan más, y esa
        # diferencia es lo que distingue a las 3 arquitecturas.
        self._collector.delegation_with_function()

        response = execute_function(self._negotiation, name, parsed_args)
        await self._client.answer_function(
            build_output_event(call_id, response)
        )
        if self._responses.answered(call_id):
            # La respuesta ya había terminado y esta era la última salida que
            # faltaba: vuelve a quedar una delegación en vuelo (la continuación).
            self._in_flight_responses += 1
            self._pending_continuations += 1
            await self._client.proceed_delegation()

        if name in CLOSING_FUNCTIONS:
            closure = self._negotiation.take_pending_close()
            if closure is not None:
                logger.info(
                    "negociación cerrada como %r; se cuelga al terminar de hablar",
                    closure.resultado,
                )
                # No se cuelga aquí: el paso 6 del prompt pide despedirse, y el
                # alpha no avisa cuándo terminó de hablar. La vigilancia del
                # silencio lo resuelve sin bloquear este manejador.
                self._closing.set()

    def _track_delegation(self, event: dict[str, Any]) -> bool:
        """Cuenta las delegaciones en vuelo. ¿Toca reanudar al delegado?

        Una delegación en vuelo significa que el agente **está por hablar**: el
        modelo delegado ya recibió el turno y su texto todavía viene en camino. El
        temporizador de silencio del cuelgue no puede correr durante eso.

        Devuelve `True` al terminar una respuesta que pidió funciones y ya tiene
        todas contestadas: entonces quien despacha manda `response.create` para
        que el delegado siga con los resultados (`bridge.SeguimientoDeRespuestas`).

        ## La continuación sigue en vuelo

        Con el alpha, contestar la función reanudaba al delegado solo y su
        segundo `response.created` llegaba antes de que el contador bajara a
        cero. En la API pública la continuación la pide el puente **después**
        del `response.completed`, así que contar ingenuamente dejaría un hueco
        en cero entre las dos respuestas: el cuelgue podría disparar ahí y el
        round trip se cerraría a la mitad. Por eso, cuando toca continuar, el
        contador no baja, y el `response.created` de la continuación no sube.
        """
        tipo = str(event.get("type", ""))
        if tipo == bridge.EV_RESPONSE_CREATED:
            self._responses.created(event)
            if self._pending_continuations:
                self._pending_continuations -= 1
            else:
                self._in_flight_responses += 1
            return False
        if tipo not in bridge.RESPONSE_END_EVENTS:
            return False
        if self._responses.finished(event):
            self._pending_continuations += 1
            return True
        self._in_flight_responses = max(0, self._in_flight_responses - 1)
        self._t_delegation_end = time.monotonic()
        if self._in_flight_responses == 0:
            # Cerró el intercambio completo con el delegado —incluida la
            # continuación tras una llamada a función— así que el round trip ya
            # se puede medir.
            self._collector.delegation_finished()
        return False

    def _handle_error(self, event: dict[str, Any]) -> None:
        detail = event.get("error") or {}
        self._collector.record_failure(
            str(detail.get("type") or "unknown_error"),
            code_value=str(detail.get("code") or ""),
            message=str(detail.get("message") or ""),
            param=detail.get("param"),
            event_id=detail.get("event_id"),
        )

    # -- Cuelgue --

    async def _watch_the_close(self) -> None:
        """Tras cerrar la negociación, espera el silencio y termina la llamada.

        Es el sustituto del `stop_when_done()` de pipecat y del `drain()` de
        LiveKit. Los dos tienen una señal de "ya terminó de hablar"; el alpha no
        —**no existe `output_audio.done`**— así que aquí se infiere de que dejen
        de llegar deltas durante `SILENCE_TO_HANG_UP_S`.

        La consecuencia es que PoC3 cuelga sistemáticamente más tarde que las
        otras dos. Es un costo de la arquitectura, no un parámetro a afinar.

        ## Por qué no basta con el silencio

        Tras `registrar_acuerdo` el agente suele decir un relleno ("Un
        segundo...") y callarse mientras el modelo delegado compone la
        despedida. Un temporizador de silencio solo se cumple **a media
        delegación** y cuelga antes de que el transportista oiga la despedida.

        Así que el silencio no alcanza por sí solo: hay que esperar también a que
        no haya delegación en vuelo, y un poco después de que la última termine. `HANG_UP_CAP_S` es la red de seguridad para que una
        delegación atorada no deje la llamada abierta —y facturando— para siempre.
        """
        await self._closing.wait()
        limit = time.monotonic() + HANG_UP_CAP_S
        while True:
            await asyncio.sleep(MONITORING_PERIOD_S)
            now = time.monotonic()
            silent = now - self._t_last_outbound_audio

            if now >= limit:
                logger.warning(
                    "se alcanzó el tope de %.0f s esperando la despedida "
                    "(callado %.1f s, %d delegación(es) en vuelo): se cuelga",
                    HANG_UP_CAP_S,
                    silent,
                    self._in_flight_responses,
                )
                return

            if self._in_flight_responses > 0:
                # El modelo delegado está trabajando: el agente está por hablar.
                continue
            if now - self._t_delegation_end < SILENCE_TO_HANG_UP_S:
                # La delegación acabó hace poco; su audio todavía puede venir.
                continue
            if silent >= SILENCE_TO_HANG_UP_S:
                logger.info(
                    "el agente lleva %.1f s callado tras el cierre: se cuelga",
                    silent,
                )
                return

    async def _sample_drift(self) -> None:
        """Vigila el ancla de reloj mientras la llamada corre.

        Con solo dos lecturas —al abrir y al colgar— el criterio acotaría la
        deriva **final** y no el peor momento, y un ajuste de hora que después
        se revierte deja igual de inservibles las marcas que se fecharon
        mientras duraba. El estadístico es el máximo absoluto, así que hay que
        estar mirando.

        Es una resta de dos lecturas de reloj una vez por segundo. No toca el
        audio, no toca disco y corre en su propia tarea: si muriera, las otras
        tres siguen — `asyncio.wait` termina con la PRIMERA que acabe, así que
        esta no puede ser una de las que terminan, y por eso nunca sale del
        bucle por su cuenta.
        """
        if self._clock is None:
            # Sin eje de pared no hay nada que vigilar. Se duerme para siempre
            # en vez de devolver: devolver despertaría a `asyncio.wait` y
            # colgaría la llamada al instante.
            await asyncio.Event().wait()
        while True:
            await asyncio.sleep(SAMPLING_PERIOD_S)
            try:
                self._clock.take_samples()
            except Exception as error:  # noqa: BLE001 - diagnóstico
                self._turn_off_diagnostics("drift_sampling", error)
                await asyncio.Event().wait()

    # -- Orquestación --

    async def launch(self) -> None:
        """Corre los dos sentidos y la vigilancia del cierre hasta que uno acabe."""
        tasks = [
            asyncio.create_task(self._from_twilio_to_live(), name="twilio->live"),
            asyncio.create_task(self._from_live_to_twilio(), name="live->twilio"),
            asyncio.create_task(self._watch_the_close(), name="cierre"),
            asyncio.create_task(self._sample_drift(), name="drift"),
        ]
        try:
            finished_ones, pending_items = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for tarea in finished_ones:
                # Re-lanza la excepción si alguna murió por error, en vez de
                # dejarla enterrada en la tarea.
                tarea.result()
        finally:
            for tarea in tasks:
                tarea.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # El diagnóstico nunca tumba la corrida: perder el volcado cuesta
            # una calibración, perder la llamada cuesta la llamada.
            if self._probe is not None:
                try:
                    self._probe.shut()
                except Exception as error:  # noqa: BLE001 - diagnóstico
                    logger.warning("no se pudo escribir la sonda: %r", error)
            if self._auditoria is not None:
                try:
                    self._auditoria.shut(call_id=self._negotiation.call_id or "")
                except Exception as error:  # noqa: BLE001 - diagnóstico
                    logger.warning("no se pudo escribir la auditoría: %r", error)
            if self._clock is not None:
                # Segunda medida de R y última muestra de deriva. Si truena, la
                # llamada ya terminó y lo que se pierde es la mitad de la
                # auditoría del reloj: queda registrada como falla y el bloque
                # sale con `al_colgar_ms: null`.
                try:
                    self._clock.shut()
                except Exception as error:  # noqa: BLE001 - diagnóstico
                    self._turn_off_diagnostics("clock_closure", error)


# --------------------------------------------------------------------------
# Persistencia de la corrida
# --------------------------------------------------------------------------


def diagnostics_block(
    faults: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Si la captura de diagnóstico sobrevivió la llamada, y por qué no.

    Va **siempre**, también cuando todo salió bien: una llave que solo aparece
    al fallar se lee como ausente cuando falta el código que la escribe, y ahí
    la llamada rota vuelve a contar como sana. Con la llave siempre puesta, la
    pregunta "¿esta llamada entra al análisis del colgajo?" se contesta abriendo
    el archivo.
    """
    faults = list(faults or [])
    return {
        "full_capture": not faults,
        "eligible_for_hang_up_analysis": not faults,
        "faults": faults,
        "what_it_means": (
            "`captura_completa: false` quiere decir que la sonda o la grabadora "
            "reventaron dentro del ciclo de audio y se apagaron ahí mismo. El "
            "audio de la llamada NO se vio afectado —sale antes que nada— pero "
            "el WAV y el volcado de energía de esta llamada no existen o están "
            "truncados, así que NO entra al análisis"
        ),
    }


def text_echo_block(
    *,
    enabled: bool,
    fragments: int,
    faults: Sequence[Mapping[str, Any]] | None,
    session_initiated: bool = True,
) -> dict[str, Any]:
    """Si el arnés oyó el texto de esta llamada, y si no, por qué no es apta.

    Va **siempre**, igual que `diagnostics_block` y por el mismo motivo:
    una llave que solo aparece cuando algo sale mal se lee como ausente cuando
    falta el código que la escribe, y ahí la llamada rota vuelve a contar como
    sana.

    `deaf_to_text` es la condición que importa: el eco estaba encendido y no
    salió ni un fragmento. Sin texto, el arnés no puede derivar la referencia
    del WER del agente, `turno.respuesta` sale en blanco y las ramas del guion
    caen todas a su línea por defecto — o sea que la llamada **no es apta**
    aunque su audio y sus latencias estén intactos. Es la misma distinción que
    `ObservadoDePoC3.deaf_to_text` hace del otro lado del cable: un agente que
    habló y no se registró no es lo mismo que un agente que no habló.
    """
    died = any(
        str(f.get("sitio")) == "text_echo" for f in (faults or ())
    )
    deaf = enabled and fragments == 0
    if not session_initiated:
        # Sin sesión no hubo puente, y sin puente no hay eco que medir. Antes
        # esto caía en la rama de abajo y decía que el eco "se apagó a mano",
        # que es falso y mandaba a buscar una variable que nadie puso.
        motivo = (
            "la sesión de GPT-Live no llegó a iniciar (ver `technical_failures`): "
            "no hubo llamada y por lo tanto tampoco eco de texto"
        )
    elif not enabled:
        motivo = (
            f"el eco se apagó a mano con `{NO_ECHO_VARIABLE}`. Solo se justifica "
            "en una llamada telefónica real, donde el otro extremo es Twilio; la "
            "campaña corre con el arnés por WebSocket local"
        )
    elif deaf:
        motivo = (
            "el eco estaba encendido y no salió un solo fragmento: o GPT-Live no "
            "mandó `session.output_transcript.delta`, o murió antes del primero (ver "
            "`diagnostico.fallas`). El arnés corrió sordo, así que esta llamada "
            "no sirve para el WER del agente ni para las ramas del guion"
        )
    elif died:
        motivo = (
            "el eco salió y después murió: el texto de esta llamada está "
            "truncado y no se usa para el WER"
        )
    else:
        motivo = ""
    return {
        "enabled": enabled,
        "emitted_fragments": fragments,
        "deaf_to_text": deaf,
        "died_mid_call": died,
        "eligible": enabled and not deaf and not died,
        "motivo": motivo,
    }


def save_run(
    negotiation: NegotiationSession,
    record: CallRecord,
    *,
    base_path: Path | None = None,
    collector: LatencyCollector | None = None,
    identification: Mapping[str, str] | None = None,
    diagnostics_failures: Sequence[Mapping[str, Any]] | None = None,
    clock: AnchoredClock | None = None,
    live_bridge: "PuenteDeLlamada | None" = None,
) -> Path:
    """Vuelca resultado, transcripción y latencia a JSON.

    Misma forma que el JSON de PoC1 y PoC2, para que `shared/metrics/` lo lea sin
    una sola rama por PoC. Cae en `grabaciones/`, que está gitignoreada: la
    transcripción puede contener voz y datos de un transportista real (LFPDPPP).

    El resumen de la sesión no incluye `max_price`, por diseño de
    `SesionNegociacion.resumen()`. Tampoco se agrega aquí.
    """
    # Cierra los turnos que se venían acumulando por fragmentos.
    record.normalize()

    # Con el puente a la mano, las fallas salen de él: pasarlas por separado
    # deja dos fuentes para el mismo dato y la que se olvide gana en silencio.
    if diagnostics_failures is None and live_bridge is not None:
        diagnostics_failures = live_bridge.diagnostics_failures

    folder = base_path or RECORDINGS_PATH
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"call_{negotiation.call_id or 'no_sid'}.json"

    content: dict[str, Any] = {"call_sid": negotiation.call_id}

    # La identificación va antes del resumen para que se vea al abrir el archivo.
    for clave in RUN_PARAMETERS:
        valor = (identification or {}).get(clave)
        if valor is None or str(valor).strip() == "":
            continue
        try:
            content[clave] = int(str(valor).strip())
        except ValueError:
            # Un valor no numérico se conserva tal cual en vez de tirarse: es
            # dato de trazabilidad y perderlo es peor que guardarlo raro.
            content[clave] = str(valor).strip()

    content["summary"] = negotiation.summary()
    content["transcript"] = record.turns
    content["diagnostics"] = diagnostics_block(diagnostics_failures)
    content["text_echo"] = text_echo_block(
        # Sin puente la llamada ni siquiera corrió: no se inventa un eco sano.
        enabled=bool(live_bridge is not None and live_bridge.text_echo_requested),
        fragments=(live_bridge.emitted_text_fragments if live_bridge is not None else 0),
        faults=diagnostics_failures,
        session_initiated=live_bridge is not None,
    )
    # Al mismo nivel que `diagnostico` y no dentro de `gpt_live_extra`: es un
    # veredicto por llamada —"¿las marcas de esta llamada se pueden restar
    # contra las del arnés?"— y se tiene que ver al abrir el archivo, no
    # escarbando. Sale también cuando no hubo reloj, diciendo que no lo hubo.
    content["clock"] = (
        clock.block()
        if clock is not None
        else {
            "requisito": marks.STEP_3_REQUIREMENT,
            "presente": False,
            "interpretable": False,
            "what_it_means": (
                "esta llamada corrió sin ancla de reloj en el puente, así que "
                "su acuse está solo en el eje interno del colector "
                f"(`{marks.ACKNOWLEDGEMENT_CLOCK}`) y no se puede restar contra los "
                f"ceros del arnés, que van en `{marks.HARNESS_CLOCK}`"
            ),
        }
    )

    if collector is not None:
        collector.close_pending_ones()
        content["latencies"] = collector.to_json()
        content["gpt_live_extra"] = collector.extra_block()
        content["technical_failures"] = collector.failures
        # Llave neutral, no `gpt_live_extra`: `shared/metrics/` la agrega al CSV
        # sin saber de PoC3, y el día que PoC1 o PoC2 quieran publicar el round
        # trip de sus tool calls la llenan igual y las columnas se comparan.
        round_trips = collector.llm_round_trips()
        if round_trips:
            content["llm_round_trips"] = round_trips
        degradation = collector.measurement_degradation()
        if degradation is not None:
            content["exclude_from_pooled"] = degradation
            logger.warning(
                "llamada excluida de la poza de percentiles: %s", degradation["detail"]
            )

    destination.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def print_summary(
    negotiation: NegotiationSession,
    record: CallRecord,
    collector: LatencyCollector,
    destination: Path,
) -> None:
    """Lo mismo que imprimen PoC1 y PoC2 al colgar, más el veredicto del reloj."""
    print("\n" + "=" * 70)
    print(f"Llamada terminada: {negotiation.call_id}")
    print("=" * 70)
    print(record.as_text() or "(sin transcripción)")
    print()

    samples = collector.samples
    measured_items = [
        m.full_turn_ms
        for m in samples
        if m.full_turn_ms is not None and not m.is_opening
    ]
    print(f"  turnos con latencia  {len(samples)}")
    if measured_items:
        print(
            f"  turno completo       régimen: "
            f"{min(measured_items):,.0f}-{max(measured_items):,.0f} ms ({len(measured_items)} turnos)"
        )
    opening = next((m for m in samples if m.is_opening), None)
    if opening and opening.full_turn_ms is not None:
        print(f"  apertura (frío)      {opening.full_turn_ms:,.0f} ms")
    print(f"  roles registrados    {sorted(record.roles()) or '(ninguno)'}")

    extra = collector.extra_block()
    verdict = extra["zero_verdict"]
    print(f"  cero del turno       {verdict.get('cero', '?')}")
    print(f"  veredicto del cero   {verdict['verdict']}")
    print(f"                       {verdict['detail']}")
    turn_taking = extra["turn_taking"]
    print(
        f"  turn-taking          {turn_taking['overlaps']} solape(s), "
        f"{turn_taking['cortes']} corte(s)"
    )
    usage = extra.get("usage") or {}
    if usage:
        print(
            # La API pública reporta segundos de sesión (`usage.seconds`), que
            # es lo que se factura; el alpha reportaba tokens.
            f"  uso                  {usage['seconds']} s de sesión"
            if "seconds" in usage
            else f"  uso                  {usage.get('total_tokens', '?')} tokens "
            f"({usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out)"
        )

    failures = collector.failures
    if not failures:
        print("  errores de GPT-Live  ninguno")
    else:
        print(f"  errores de GPT-Live  {len(failures)}")
        for failure in failures:
            mark = f"{failure['t']:.1f}s" if failure.get("t") is not None else "?"
            print(f"                       [{mark}] {failure['tipo']}: {failure['detail']}")

    resultado = negotiation.resultado
    print(f"  historial validado   {negotiation.offer_history or '(ninguno)'}")
    print(f"  compensaciones       {negotiation.offered_compensations or '(ninguna)'}")
    print(f"  resultado            {resultado.resultado if resultado else '(sin cierre)'}")
    print(f"\n  corrida guardada en  {destination}")


# --------------------------------------------------------------------------
# Servidor
# --------------------------------------------------------------------------


def create_app(
    settings: Settings,
    *,
    trip: TripData = TEST_TRIP,
    text_echo: bool | None = None,
    on_finish: Callable[[str, Path], None] | None = None,
) -> FastAPI:
    """La app de FastAPI que atiende el TwiML y el Media Stream.

    `text_echo` en `None` —lo normal— lo decide el entorno:
    `text_echo_enabled()`, o sea encendido salvo `POC3_SIN_ECO_DE_TEXTO`.
    Se puede forzar desde el código para una corrida concreta.

    `on_finish` lo usa el arnés y nadie más. Se llama con el `call_sid` y la
    ruta del JSON, **después** de escribirlo, y es lo que le permite al
    servidor de una sola llamada apagarse solo. Dejarlo vivo obligaría al arnés
    a matarlo, que es como se pierde el volcado. Es el gemelo de
    `GanchosDelArnes.on_save` de PoC1; aquí va suelto porque PoC3 no le
    inserta nada al camino caliente: su gancho de texto ya vive en el puente.
    """
    if text_echo is None:
        text_echo = text_echo_enabled()
    app = FastAPI(title="PoC3 — agente de voz negociador (GPT-Live alpha)")

    @app.get(f"/{HEALTH_PATH}")
    async def health() -> JSONResponse:
        """Para que el marcador confirme el túnel antes de gastar una llamada."""
        return JSONResponse(
            {
                "ok": True,
                "websocket": settings.url_websocket(RUTA_WS),
                "modelos": {
                    "live": settings.live_model,
                    "delegation": settings.delegation_model,
                    "voice": settings.voice,
                },
                "alpha": True,
            }
        )

    @app.api_route(f"/{RUTA_TWIML}", methods=["GET", "POST"])
    async def twiml(request: Request) -> Response:
        """El TwiML que Twilio pide al contestar. Igual que el de PoC1."""
        try:
            form = await request.form()
            call_sid = str(form.get("CallSid", "")) or "?"
        except Exception:  # noqa: BLE001 - un GET de prueba no trae formulario
            call_sid = "?"

        label = {
            clave: request.query_params[clave]
            for clave in RUN_PARAMETERS
            if request.query_params.get(clave)
        }
        url_ws = settings.url_websocket(RUTA_WS)
        logger.info(
            "TwiML pedido (call_sid=%s, corrida=%s) -> %s", call_sid, label, url_ws
        )
        return Response(
            content=build_twiml(url_ws, label), media_type="application/xml"
        )

    @app.websocket(f"/{RUTA_WS}")
    async def media_stream(websocket: WebSocket) -> None:
        """El Media Stream: una llamada, una sesión de negociación."""
        await websocket.accept()

        # Los dos primeros mensajes de Twilio se leen a mano: hacen falta el
        # streamSid (para poder devolver audio) y el callSid (para nombrar la
        # corrida) antes de abrir nada contra el alpha.
        await websocket.receive_text()  # {"event": "connected"}
        raw_start = await websocket.receive_text()
        start = json.loads(raw_start).get("start", {})
        stream_sid = start.get("streamSid", "")
        call_sid = start.get("callSid", "")
        identification = read_identification(start)

        if not stream_sid or not call_sid:
            logger.error("el mensaje `start` de Twilio no trae streamSid/callSid")
            await websocket.close()
            return

        logger.info(
            "llamada entrando al puente: call_sid=%s corrida=%s",
            call_sid,
            identification or "(sin etiquetar)",
        )
        if not identification:
            logger.warning(
                "esta llamada no trae scenario/repetition: pásalos al marcar "
                "(--scenario/--repetition/--poc) o habrá que etiquetarla a mano "
                "en el archivo de referencia"
            )

        negotiation = NegotiationSession(trip=trip, call_id=call_sid)
        record = CallRecord()
        # El ancla se toma ANTES de conectar contra el alpha: así todo lo que
        # esta llamada feche —incluida la apertura— cae después del ancla y no
        # hay un tramo inicial sin eje. Medir R cuesta hasta 50 ms y se paga
        # aquí, donde todavía no hay nada sonando.
        clock = AnchoredClock()
        collector = LatencyCollector(wall_clock=clock.now)
        # Cero de la llamada: el Media Stream conectó. El turno de apertura se
        # mide desde aquí, porque el saludo del agente no tiene fin de habla
        # antes y paga el arranque en frío completo.
        collector.begin()

        # El censo de eventos también en llamadas reales: es la única traza de
        # qué manda el alpha de verdad, y sin él un fallo como el
        # `Unknown handoff function call id` de la llamada del 12 de agosto de
        # 2026 no se puede diagnosticar sin volver a gastar telefonía.
        census = bridge.EventCensus(
            RECORDINGS_PATH / f"census_{negotiation.call_id or 'no_sid'}.jsonl"
        )
        client = bridge.GptLiveClient(
            settings.url_live(),
            settings.headers_live(),
            build_initial_session(settings, negotiation),
            census=census,
        )

        live_bridge: CallBridge | None = None
        try:
            session_confirmed = await client.connect()
            collector.anchor_server(session_confirmed)

            # El agente abre la llamada. Ver la nota 1 del docstring del módulo:
            # va como `commentary`, y como hecho, no como instrucción.
            await client.add_context(OPENING_CONTEXT)

            live_bridge = CallBridge(
                websocket=websocket,
                client=client,
                negotiation=negotiation,
                record=record,
                collector=collector,
                stream_sid=stream_sid,
                clock=clock,
                text_echo=text_echo,
                audio_format=settings.audio_format,
            )
            await live_bridge.launch()
        except bridge.AlphaError as error:
            logger.error("la sesión de GPT-Live falló: %s", error)
            collector.record_failure("session_not_initiated", message=str(error))
        finally:
            # Cerrar PRIMERO y copiar el uso después. `cerrar` manda
            # `session.close` y drena hasta `session.closed`, y ese evento es el
            # que trae el `usage` final de la sesión: copiarlo antes deja en el
            # JSON el penúltimo `session.usage.updated`, o sea una duración
            # sistemáticamente corta. Y la tarifa de `gpt-live-1` se cobra por
            # tiempo de sesión, así que ese sesgo es dinero. Cuánto, en
            # `docs/costos_poc3.md`.
            with contextlib.suppress(Exception):
                await client.shut()
            collector.usage = dict(client.usage)
            with contextlib.suppress(Exception):
                await websocket.close()
            census.shut()
            logger.info("censo de eventos en %s", census.destination)

            if not negotiation.is_closed:
                # Se cortó sin que el agente cerrara.
                negotiation.close_without_agreement(CloseReason.HUNG_UP)

            destination = save_run(
                negotiation,
                record,
                collector=collector,
                identification=identification,
                # `puente` puede no existir si el alpha ni siquiera conectó: no
                # se inventa un diagnóstico sano para una llamada que no corrió.
                diagnostics_failures=(
                    live_bridge.diagnostics_failures if live_bridge is not None else None
                ),
                clock=clock,
                live_bridge=live_bridge,
            )
            print_summary(negotiation, record, collector, destination)

            if on_finish is not None:
                # Lo último de todo, y sin `try`: si el arnés se equivoca aquí,
                # la llamada ya quedó escrita entera y lo que se ve es su error,
                # no una llamada a medias.
                on_finish(call_sid, destination)

    return app


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.app.telephony",
        description=(
            "Servidor de telefonía de PoC3. Sirve el TwiML y el Media Stream; "
            "no marca. Para marcar, usa `poc3_gpt_live.app.llamada`."
        ),
    )
    arg_parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="puerto local"
    )
    args = arg_parser.parse_args(argv)

    # Antes de cualquier print: el volcado de la transcripción ocurre después de
    # colgar una llamada ya pagada, y morir ahí por el encoding de la consola
    # perdería la evidencia.
    force_utf8_output()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = load_settings()
    verify_for_telephony(settings)
    verify_for_voice(settings)

    print("Servidor de telefonía de PoC3 — GPT-Live")
    print(f"  local          http://localhost:{args.port}")
    print(f"  TwiML          {settings.url_webhook(RUTA_TWIML)}")
    print(f"  Media Stream   {settings.url_websocket(RUTA_WS)}")
    print(f"  salud          {settings.url_webhook(HEALTH_PATH)}")
    print(f"  modelo         {settings.live_model}")
    print(f"  delegación     {settings.delegation_model}")
    print(
        f"\nEl túnel debe apuntar a este puerto:\n"
        f"  ngrok http {args.port} --domain={settings.public_host()}\n"
    )

    import uvicorn

    # 127.0.0.1 y no 0.0.0.0: ngrok corre en esta misma máquina, así que no hay
    # razón para exponer el puerto a la red local.
    uvicorn.run(
        create_app(settings), host="127.0.0.1", port=args.port, log_level="info"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
