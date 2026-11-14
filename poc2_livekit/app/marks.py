"""El cero real del turno, del arnés al worker.

El arnés es el único que sabe **con certeza** cuándo salió el audio del
carrier: él mismo empujó los samples. El worker no lo sabe, lo
*reconstruye* — Silero declara el fin de habla tras acumular
`min_silence_duration` y LiveKit le resta esa espera para estimar el instante
real (`audio_recognition.py`):

    speech_end_time = time.time() - ev.silence_duration - ev.inference_duration

Eso es lo que se publica como `stopped_speaking_at`, y es el cero del que cuelgan
`transcription_delay`, `end_of_turn_delay` y `e2e_latency`. O sea: **el cero de
toda la latencia de PoC2 es una estimación**. Este módulo la mide contra el
instante real y publica la diferencia en `livekit_extra.arnes`.

## Los DOS ceros, y por qué no basta con uno

- **`clip_end`** — el último sample salió por la pista. Es un **hecho
  físico**, no depende de ningún criterio.
- **`measured_end_of_speech`** — dónde dejó de haber voz *dentro* del clip. Es una
  **medición** y depende del umbral, así que viaja con su `drop_db` y no se
  puede leer sin él. Se mide al preparar el clip
  (`shared/audio/level.py::end_of_speech`) y vive en el manifiesto: es propiedad
  del clip, y las 3 PoC reproducen el mismo clip.

    sesgo_de_clip_ms = stopped_speaking_at - clip_end
    sesgo_de_voz_ms  = stopped_speaking_at - measured_end_of_speech

El segundo es el que quiere decir algo. Publicar solo el del archivo fue el
primer diseño y produjo sesgos negativos imposibles: un clip arrastra hasta 1.3 s
de respiración y cola de sala después de la última sílaba, y el cero estaba
fechado en el fin del *archivo*. Tampoco sirve corregir con el margen de recorte
—30 ms fijos contra colas reales mucho mayores es una cota inferior, no una
corrección—, y por eso **no se publica**: un campo así se acaba usando como si
corrigiera. Mediciones y ajuste en `docs/comparativo_results.md`.

Para qué sirve: si el sesgo saliera del orden de `min_silence_duration` (550 ms),
significaría que LiveKit **no** descuenta la espera del VAD y que toda la latencia
de PoC2 está inflada medio segundo. `zero_verdict()` hoy decide eso por
magnitud, que es un argumento indirecto; esto lo mide de frente.

## Dos relojes

Arnés y worker son procesos distintos: restar sus marcas solo tiene sentido si
comparten reloj. Cada marca viaja con el `host` del arnés; si no coincide con el
del worker, el bloque marca el sesgo como **no interpretable** y no se corrige —
una corrección de reloj sin sincronización real sería inventar.

## Por qué el canal confiable y por qué antes de desconectar

`reliable=True` retransmite; `lossy` descarta bajo congestión, y una marca
perdida no se nota — el turno simplemente aparecería sin cero. Todo se manda
**antes** de desconectarse, porque desconectarse es lo que le dispara al worker
su `shutdown` y la escritura del JSON.

El cierre manda además un **resumen** con todas las marcas repetidas. Cuesta un
paquete y compra dos cosas: una marca perdida llega en la segunda oportunidad, y
el worker puede reportar cobertura honesta (`3 de 5`) en vez de publicar tres y
callar que faltan dos.

`latencias` no cambia y `end_of_turn_verdicts` sigue vacío por contrato: todo
lo de aquí es diagnóstico.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from shared.carrier_simulator.cero import (
    ZERO_PROTOCOL,
    MARK_KIND,
    SUMMARY_KIND,
    PROTOCOL_VERSION,
    ZeroEmitter,
    ZeroMark,
    Envelope,
    encode_mark,
    encode_summary,
    decode,
)
from shared.carrier_simulator.cruce import (
    PAIRING_TOLERANCE_S,
    EstimatedZero,
    PairedTurn,
    group,
    pair,
    ms,
    lines_summary,
)
from shared.carrier_simulator.cruce import ms as _ms

logger = logging.getLogger(__name__)

# El nombre viejo del tópico. Se sigue escuchando —abajo— solo para poder gritar
# si un arnés sin actualizar publica en él; si no, un desajuste de versiones se
# vería como "no llegó ninguna marca", que es la conclusión equivocada.
OLD_TOPIC: Final[str] = "harness.end_of_speech"

# Espera tras publicar el resumen, antes de desconectarse. `publish_data`
# devuelve cuando el lado Rust aceptó el paquete, no cuando el SFU lo entregó;
# desconectarse en ese instante puede tirarlo. No hay acuse que esperar, así que
# es un margen y se declara como tal.
FLUSH_WAIT_S: Final[float] = 0.25


# --------------------------------------------------------------------------
# El paquete
# --------------------------------------------------------------------------
#
# Vive en `shared/carrier_simulator/cero.py`: el formato tiene que ser el mismo
# en las 3 PoC o los sesgos no se pueden leer juntos. Aquí queda solo lo de
# LiveKit — el tópico del canal de datos y cómo se publica.
#
# `TOPICO_MARCAS` es el mismo string que el identificador del protocolo, a
# propósito: el sobre dice en qué canal viaja.

MARKS_TOPIC: Final[str] = ZERO_PROTOCOL


# --------------------------------------------------------------------------
# Lado del arnés
# --------------------------------------------------------------------------


class MarkEmitter(ZeroEmitter):
    """Publica el cero de cada turno y, al cerrar, el resumen.

    **Nunca propaga una falla de publicación.** Un paquete de diagnóstico que no
    salió no es motivo para tumbar una corrida que ya está gastando STT, TTS y
    LLM; se anota y el reporte lo dice.
    """

    # `publish_data` vuelve cuando el lado Rust aceptó el paquete, no cuando el
    # SFU lo entregó, y desconectarse en ese instante puede tirarlo. No hay
    # acuse que esperar, así que es un margen y se declara como tal.
    flush_wait_s = FLUSH_WAIT_S

    def __init__(
        self,
        participant: Any,
        *,
        publish_sample_rate: int,
        clock: Any = None,
        host: str | None = None,
    ) -> None:
        super().__init__(
            publish_sample_rate=publish_sample_rate, clock=clock, host=host
        )
        self._participant = participant

    async def _deliver(self, carga: bytes) -> None:
        await self._participant.publish_data(
            carga,
            # El canal `_reliable`: ordenado y con retransmisión. Con `lossy`
            # una marca perdida no se notaría.
            reliable=True,
            topic=MARKS_TOPIC,
        )


# --------------------------------------------------------------------------
# Lado del worker
# --------------------------------------------------------------------------


# El emparejamiento, los dos sesgos y la tolerancia viven en
# `shared/carrier_simulator/cruce.py`: si PoC1 casara por índice y PoC2 por
# cercanía, las dos publicarían un `sesgo_de_voz_ms` con el mismo nombre medido
# de formas que no se comparan. Aquí queda el nombre local y la fila del JSON,
# que sí es de PoC2 porque sus llaves son su contrato de captura.
LiveKitZero = EstimatedZero


def add_lines(
    json_path: Path | str,
    *,
    first_speech: Mapping[str, float],
) -> dict[str, Any] | None:
    """Le agrega al JSON del worker la cuenta por intervención. Corre tras colgar.

    Va en el proceso del **arnés** y no en el del worker porque hace falta lo
    que solo el arnés sabe: cuándo se oyó al agente. El worker ya dejó los ceros
    crudos y las marcas en `livekit_extra.arnes`; aquí se agrupan.

    Es el gemelo de `poc1_pipecat.app.arnes.add_lines`, y los
    dos arman el bloque con la misma función compartida — que es lo que hace
    comparable la tasa de fragmentación entre arquitecturas (§2.2.5).
    """
    ruta = Path(json_path)
    if not ruta.is_file():
        return None
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # noqa: BLE001 - diagnóstico
        logger.warning("no se pudo leer %s: %r", ruta, error)
        return None

    harness = data.get("livekit_extra", {}).get("harness")
    if not isinstance(harness, dict):
        return None

    marks = [
        ZeroMark.from_dict(dict(m))
        for m in harness.get("marks_of_harness", [])
        if isinstance(m, dict)
    ]
    zeros = [
        LiveKitZero(
            indice=int(c["indice"]), start=c.get("start"), end=c.get("end")
        )
        for c in harness.get("zeros_of_pipeline", [])
    ]
    if not marks:
        return None

    samples = data.get("latencies")
    of_pipeline = (
        len([m for m in samples if not (isinstance(m, dict) and m.get("is_opening"))])
        if isinstance(samples, list)
        else None
    )

    block = lines_block(
        marks, zeros, first_speech=first_speech, pipeline_samples=of_pipeline
    )
    harness["lines"] = block
    ruta.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return block


def lines_block(
    marks: Sequence[ZeroMark],
    zeros: Sequence[LiveKitZero],
    *,
    first_speech: Mapping[str, float] | None = None,
    pipeline_samples: int | None = None,
) -> dict[str, Any]:
    """La corrida contada por intervención. Mismo armado que en PoC1.

    Vive duplicado en las dos PoC solo como envoltura: el agrupamiento, la
    latencia y el resumen son de `shared/carrier_simulator/cruce.py`, que es lo
    que no puede divergir.
    """
    lines, loose_ones = group(marks, zeros, first_speech=first_speech)
    rows: list[dict[str, Any]] = []
    for i in lines:
        cero = i.pipeline_zero
        rows.append(
            {
                "indice": i.indice,
                "turn": i.turn,
                "clip_id": i.clip_id,
                "end_of_speech": round(i.end_of_speech, 6),
                "agent_first_speech": (
                    None
                    if i.agent_first_speech is None
                    else round(i.agent_first_speech, 6)
                ),
                "line_latency_ms": ms(i.latency),
                "pipeline_events": i.fragmentation,
                "pipeline_zero": None if cero is None else cero.end,
                "speech_bias_ms": (
                    None
                    if cero is None or cero.end is None
                    else ms(cero.end - i.end_of_speech)
                ),
            }
        )
    data = lines_summary(
        lines, loose_ones=len(loose_ones), pipeline_samples=pipeline_samples
    )
    data["per_line"] = rows
    return data


def _turn_row(
    turn: PairedTurn, *, no_audio: Sequence[str] = ()
) -> dict[str, Any]:
    """Un turno emparejado como fila de `livekit_extra.arnes.per_turn`.

    **Las llaves son contrato**: `shared/metrics/` y el comparativo las leen por
    nombre. Renombrar una aquí no rompe ningún import.
    """
    data: dict[str, Any] = {"indice": turn.indice}
    if turn.mark is None or turn.cero is None:
        data["paired"] = False
        return data

    mark, cero = turn.mark, turn.cero
    mute = mark.turn in set(no_audio)
    data.update(
        {
            "paired": True,
            "harness_turn": mark.turn,
            "clip_id": mark.clip_id,
            # El agente escribió y no sonó: la muestra de latencia de este turno
            # mide un TTS que nadie oyó. No se borra —el dato de que ocurrió es
            # el hallazgo— pero queda marcada para que no entre al percentil.
            "agent_audio": not mute,
            "exclude_from_percentile": mute,
            "stopped_speaking_at": cero.end,
            # --- el cero medido: la última voz dentro del clip ---
            "measured_end_of_speech": mark.measured_end_of_speech,
            "speech_bias_ms": _ms(turn.speech_bias),
            # El umbral viaja pegado al número que produjo. Si alguien lo cambia,
            # el JSON lo grita en vez de callarlo.
            "drop_db": mark.drop_db,
            "speech_threshold_dbfs": mark.speech_threshold_dbfs,
            # --- el cero de archivo: hecho físico, para control ---
            "clip_end": mark.clip_end,
            "clip_bias_ms": _ms(turn.clip_bias),
            "cola_ms": mark.cola_ms,
            # --- inicio, con su advertencia ---
            "start_of_clip": mark.start_of_clip,
            "started_speaking_at": cero.start,
            "onset_bias_ms": _ms(turn.onset_bias),
            "clip_duration_ms": mark.duration_ms,
        }
    )
    return data


def _rounded(summary: dict[str, Any]) -> dict[str, Any]:
    """Los cuartiles interpolados salen con cola de punto flotante.

    `118.0000000003929` en un JSON que alguien va a leer a ojo sugiere una
    precisión que la medición no tiene: el sesgo se mide en milisegundos y su
    piso de ruido es el jitter de la red.
    """
    return {
        k: round(v, 1) if isinstance(v, float) else v for k, v in summary.items()
    }


@dataclass
class MarksRecord:
    """Lo que el worker recibió por el canal de datos del arnés.

    Se instala siempre: en una llamada telefónica real nadie publica en este
    tópico y el registro se queda vacío, así que no cambia nada de lo que se
    mide ni de lo que se guarda.
    """

    clock: Any = None
    host: str = ""
    _by_sequence: dict[int, ZeroMark] = field(default_factory=dict)
    _arrival: dict[int, float] = field(default_factory=dict)
    declared_total: int | None = None
    packets: int = 0
    rejected_items: list[str] = field(default_factory=list)
    # Turnos que el arnés declaró mudos. Llegan en el resumen, o sea al final:
    # si la corrida se cae antes de publicarlo, esta lista queda vacía y el
    # bloque lo dice en vez de afirmar que todos los turnos se oyeron.
    turns_without_audio: list[str] = field(default_factory=list)
    summary_received: bool = False

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = time.time
        if not self.host:
            self.host = socket.gethostname()

    @property
    def marks(self) -> list[ZeroMark]:
        return [self._by_sequence[k] for k in sorted(self._by_sequence)]

    @property
    def received(self) -> int:
        return len(self._by_sequence)

    def receive(self, carga: bytes | str, *, topic: str, identity: str) -> int:
        """Procesa un paquete. Devuelve cuántas marcas **nuevas** trajo.

        El resumen final repite las marcas que ya llegaron: se deduplican por
        `secuencia` y se conserva la hora de llegada de la **primera** copia,
        que es la que mide el transporte de verdad.
        """
        from poc2_livekit.app.config import CARRIER_IDENTITY

        if topic == OLD_TOPIC:
            # Un arnés sin actualizar. Se anota fuerte en vez de ignorarlo: si
            # no, el desajuste se vería como "no llegó ninguna marca", que es la
            # conclusión equivocada y manda a buscar el problema en la red.
            aviso = (
                f"llegó un paquete en el tópico viejo {OLD_TOPIC!r}: el arnés "
                f"que lo publica es anterior a la v{PROTOCOL_VERSION} del "
                "protocolo y su cero fecha el fin del archivo, no el de la voz. "
                "Actualízalo; no se lee"
            )
            self.rejected_items.append(aviso)
            logger.warning("%s", aviso)
            return 0
        if topic != MARKS_TOPIC:
            return 0
        if identity != CARRIER_IDENTITY:
            # Solo el arnés publica ceros. Otro participante en este tópico es
            # ruido, y tomarlo por bueno daría un sesgo contra un clip ajeno.
            self.rejected_items.append(f"marca de un participante ajeno: {identity!r}")
            return 0

        self.packets += 1
        try:
            on_top = decode(carga)
        except Exception as error:  # noqa: BLE001 - un paquete malo no tumba la llamada
            self.rejected_items.append(repr(error))
            logger.warning("paquete de marcas descartado: %r", error)
            return 0

        if on_top.tipo == SUMMARY_KIND:
            self.summary_received = True
            self.turns_without_audio = list(on_top.turns_without_audio)
        if on_top.declared_total is not None:
            self.declared_total = on_top.declared_total

        new_items = 0
        now = float(self.clock())
        for mark in on_top.marks:
            if mark.sequence in self._by_sequence:
                continue
            self._by_sequence[mark.sequence] = mark
            self._arrival[mark.sequence] = now
            new_items += 1
        return new_items

    # -- Lectura --

    def transport_ms(self) -> list[float]:
        """Cuánto tardó cada marca en llegar, según los dos relojes.

        Mide el canal de datos, **no** el de audio: van por caminos distintos y
        con buffers distintos. Sirve de orden de magnitud para leer el sesgo, no
        para restarlo.
        """
        return [
            round((self._arrival[s] - m.sent_at) * 1000.0, 1)
            for s, m in sorted(self._by_sequence.items())
            if m.sent_at
        ]

    def same_host(self) -> bool | None:
        hosts = {m.host for m in self._by_sequence.values() if m.host}
        if not hosts:
            return None
        return hosts == {self.host}

    def block(self, zeros: Iterable[LiveKitZero]) -> dict[str, Any]:
        """Lo que va a `livekit_extra.arnes`. Nunca a `latencias`."""
        from poc2_livekit.app.statistical_tests import describe

        # Una sola vez: `ceros` puede ser un generador y abajo se vuelve a usar.
        all_zeros = list(zeros)
        paired_ones = pair(self.marks, all_zeros)
        with_mark = [e for e in paired_ones if e.mark is not None and e.cero is not None]
        of_voice = [e.speech_bias * 1000.0 for e in with_mark if e.speech_bias is not None]
        of_clip = [e.clip_bias * 1000.0 for e in with_mark if e.clip_bias is not None]
        tails = [e.mark.cola_ms for e in with_mark if e.mark and e.mark.cola_ms is not None]
        # El umbral con el que se midió el fin de voz. Si hubiera más de uno, la
        # comparación entre turnos no valdría y hay que verlo.
        criteria = sorted(
            {e.mark.drop_db for e in with_mark if e.mark and e.mark.drop_db is not None}
        )

        same = self.same_host()
        interpretable = same is True
        used = {
            e.mark.sequence for e in with_mark if e.mark is not None
        }

        def _summary(values: list[float]) -> dict[str, Any]:
            if not interpretable:
                return {"n": 0, "note": "no se resume: los relojes no son el mismo"}
            return _rounded(describe(values))

        return {
            "protocolo": MARKS_TOPIC,
            "version": PROTOCOL_VERSION,
            "what_it_is": (
                "los dos ceros reales del turno, medidos por quien reprodujo el "
                "clip, contra el que LiveKit estimó. Diagnóstico del arnés: no "
                "entra en `latencias` ni cambia ninguna muestra"
            ),
            "cobertura": {
                "packets": self.packets,
                "marks_received": self.received,
                "marks_declaradas_por_el_arnes": self.declared_total,
                "full": (
                    None
                    if self.declared_total is None
                    else self.received >= self.declared_total
                ),
                "paired_turns": len(with_mark),
                "turns_without_mark": [
                    e.indice for e in paired_ones if e.mark is None
                ],
                "marks_no_turn": [
                    m.sequence for m in self.marks if m.sequence not in used
                ],
                "discarded_packets": list(self.rejected_items),
            },
            "clock": {
                "host_of_harness": next(
                    (m.host for m in self.marks if m.host), None
                ),
                "worker_host": self.host,
                "same_host": same,
                "interpretable_bias": interpretable,
                "note": (
                    "los dos procesos comparten `time.time()`, así que la resta "
                    "vale"
                    if interpretable
                    else "el arnés y el worker NO corrieron en la misma máquina: "
                    "la diferencia lleva dentro el desfase entre relojes y no se "
                    "puede leer como sesgo"
                ),
                "data_transport_ms": _rounded(describe(self.transport_ms())),
            },
            # La medida que quiere decir algo: contra la última voz del clip.
            "speech_bias_ms": _summary(of_voice),
            # Control: contra el fin del archivo. Se mantiene publicado porque
            # es el hecho físico y porque su diferencia con el de arriba ES la
            # cola, que es lo que explicó el -707 ms del 15 de septiembre.
            "clip_bias_ms": _summary(of_clip),
            "clips_tail_ms": _rounded(describe(tails)),
            "end_of_speech_criterion": {
                "drop_db": criteria[0] if len(criteria) == 1 else criteria,
                "single": len(criteria) <= 1,
                "note": (
                    "caída en dB bajo el pico por debajo de la cual una ventana "
                    "deja de contar como voz. El mismo audio mide hasta 9 dB "
                    "distinto según este número, así que va pegado a la medida y "
                    "no se puede leer sin él. Si aquí aparece más de un valor, "
                    "los turnos se midieron con criterios distintos y NO se "
                    "comparan entre sí"
                ),
            },
            "how_to_read": (
                "speech_bias = stopped_speaking_at - última voz del clip: es la "
                "medida. clip_bias = contra el fin del archivo, y su "
                "diferencia con la anterior es la cola del clip, no latencia. El "
                "sesgo de voz lleva dentro el encoder Opus, el jitter buffer y la "
                "ventana de Silero — NO la red: el canal de datos de este mismo "
                "módulo viaja por el mismo camino y se mide aparte en "
                "reloj.transporte_de_datos_ms. Un sesgo del orden de "
                "min_silence_duration (550 ms) significaría que LiveKit no "
                "descuenta la espera del VAD y que toda la latencia de PoC2 está "
                "inflada medio segundo."
            ),
            "reliability": {
                "turns_without_agent_audio": list(self.turns_without_audio),
                "n": len(self.turns_without_audio),
                "declared": self.summary_received,
                "note": (
                    "turnos donde el agente produjo texto y el arnés no oyó una "
                    "sola trama con voz: el TTS no sonó. Su muestra de latencia "
                    "mide audio que nadie escuchó, así que NO entra al percentil "
                    "— va a confiabilidad, como fallo técnico. Se marcan en "
                    "per_turn con excluir_del_percentil"
                    if self.summary_received
                    else "el resumen del arnés no llegó, así que NO se sabe si "
                    "algún turno quedó mudo. Una lista vacía aquí no es "
                    "evidencia de que todos se oyeron"
                ),
            },
            "per_turn": [
                _turn_row(e, no_audio=self.turns_without_audio)
                for e in paired_ones
            ],
            # Los ceros **crudos**, todos, incluidos los que no emparejaron.
            # Viven en el worker y solo él los tiene; el arnés los agrupa por
            # intervención después de colgar, cuando ya sabe cuándo se oyó al
            # agente. Los turnos que el VAD cerró de más son justamente los que
            # no tienen marca, y son el dato de la fragmentación (§2.2.5).
            "zeros_of_pipeline": [
                {"indice": c.indice, "start": c.start, "end": c.end}
                for c in all_zeros
            ],
            # Las marcas tal como llegaron. `por_turno` ya las lleva mezcladas
            # con el cruce y con otros nombres; estas se reconstruyen sin
            # pérdida, que es lo que necesita el agrupamiento por intervención.
            "marks_of_harness": [m.a_dict() for m in self.marks],
        }
