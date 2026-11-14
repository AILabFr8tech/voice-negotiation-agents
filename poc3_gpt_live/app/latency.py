"""PoC3 latency capture, over the bridge audio and the model events.

`shared/metrics/` **computes**; this module **measures**. It is the equivalent of
`poc1_pipecat/app/latency.py` and `poc2_livekit/app/latency.py`, and the shortest
of the three — because it is the one that can measure the least.

## PoC3's capture contract is reduced, and that is the result

| Field | Captured? | Why |
|---|---|---|
| `full_turn_ms` | yes | end of the carrier's speech -> first speech of the agent |
| `end_of_turn_ms` | **no** | VAD and turn-taking run inside the server |
| `stt_ms` | **no** | there is no discrete STT: the model listens directly |
| `llm_ttft_ms` | **no** | there is no LLM that starts at an observable instant |
| `tts_ttfb_ms` | **no** | there is no discrete TTS |
| `interrumpido` | yes | see the criterion below |
| `is_opening` | yes | cold-start turn |
| `end_of_turn_verdicts` | empty | the model exposes no end-of-turn probability |

Absent stages are **omitted from the JSON** instead of being written as `null`
(`to_json`), so the PoC3 report does not carry empty columns.

## The turn zero: local energy, not server events

`full_turn_ms` is measured by external observation: from the end of the last
**streak** of speech coming in through the transport to the first frame with speech going
out to it, all on the local clock. The model events are useless as a zero: the carrier's
end of turn arrives late, `output_audio.delta` is a continuous
stream that includes silence, and the server timeline advances at a different
pace from the wall clock. The measurements that showed this are in
`docs/METRICS_CONTRACT.md` §2.7.

**This is not a VAD.** `bridge.has_speech` decides nothing about the conversation: end
of turn and interruptions are resolved by the server, which is what is being compared.
Energy only sets timestamps, so the number includes network, bridge and
resampling, same as in PoC1 and PoC2. What it lacks is a per-stage breakdown.

Zero rules:

- Incoming speech is grouped into **streaks**. A streak closes when
  `SILENCE_FOR_TURN_DONE_S` passes without speech, and the zero is its **last frame
  with speech**. The hangover only decides where the streak ends; it is not added to the
  number.
- A streak shorter than `MIN_TURN_SPEECH_S` is not a line: it is
  a click or a breath, and cannot be the zero of anything.
- **One sample per carrier streak.** If the agent starts again without
  the carrier having said anything in between, it is the continuation of its own
  line: it is counted in `gpt_live_extra.turn_taking.reanudaciones_del_agente`
  and does **not** produce a sample.
- If the agent starts while the carrier's streak is still open, there is no
  response latency to measure: the sample comes out with `full_turn_ms = None`
  and a note, and is counted in `onsets_over_carrier`. It is the same criterion
  PoC1 applies to its negative deltas: never a number, always `None` with a
  reason.
- **`is_opening` is the first sample of the call.** Over the phone the
  carrier answers by speaking, so "the carrier had not been heard yet"
  cannot be used to recognize it.

Numbers derived from events are kept in `gpt_live_extra.per_turn` for
comparison, but they do **not** feed `latencias[]`. `delivery_lag` publishes
separately how far the server falls behind in delivering audio: it is reliability, not
latency. And `clock_series` keeps the raw relationship between the two clocks,
because a gap between `output_audio` ranges is omitted silence and not
elapsed time: no linear translation holds.

## `interrumpido` is barge-in that cuts

In PoC1 and PoC2 talking over the agent is rare. In GPT-Live it is **the normal
mode of operation**: the model is full-duplex and keeps listening while it speaks. With
the literal criterion almost every turn would go into the `interrumpido` bucket and PoC3
would be left with no steady-state p50 to compare.

So here `interrumpido` requires two things: that the carrier spoke
while the agent was sounding (`OVERLAP_WINDOW_S`) **and** that the agent kept
sounding afterwards and went quiet within `CUTOFF_WINDOW_S` — the gap has
to fall in `[0, CUTOFF_WINDOW_S]`. Overlap that cuts nothing does not flag the
turn, but it is counted separately in `gpt_live_extra`: it is a data point specific to the
full-duplex architecture.

## The bridge delay is included

`bridge.RESAMPLING_LAG_MS` is what resampling adds to
`full_turn_ms` in each direction. It is not subtracted: it is part of what this
architecture costs over telephony. It is noted in the JSON so that whoever reads the p50
knows it is included.
"""

from __future__ import annotations

import logging
import time
from statistics import median
from typing import Any, Final

from poc3_gpt_live.app import bridge
from shared.metrics.latency import STAGE_FIELDS, TurnSample

logger = logging.getLogger(__name__)

# How long the agent may take to go quiet after an overlap for it to count as a
# cut and not as full-duplex listening. Half a second is generous relative to human
# reaction time and short relative to the length of an agent sentence.
CUTOFF_WINDOW_S: Final[float] = 0.5

# How much time may separate the agent's speech and the carrier's for it to
# count as overlap. It means "the agent was sounding", so it is short: with the
# cutoff window (half a second) speaking **after** the agent finished would
# count as overlap.
OVERLAP_WINDOW_S: Final[float] = 0.15

# How much agent silence separates two of its lines. Below this
# the outbound speech continues the current turn; above it, it opens a new one. It is a
# measurement knob, not a conversational one: the agent does not stop or start because of
# this number.
SILENCE_BETWEEN_TURNS_S: Final[float] = 0.6

# How much silence closes the carrier's speech streak. It is the energy detector's
# hangover: it only decides **where the streak ends**, and the turn zero
# is the last frame with speech of that streak, so this number is **not added** to
# `turno_completo_ms`.
#
# If the agent starts before the streak closes, the carrier was still
# speaking: the sample comes out with `turno_completo_ms = None` and a note, never with the
# gap between two words as if it were a latency.
#
# With less, a normal pause mid-sentence closes the streak and builds a zero in
# the middle of the line. The value comes from the distribution of internal gaps
# in the carrier's speech and is frozen for the campaign; the JSON publishes
# that distribution in `gpt_live_extra.turn_taking.huecos_entrantes_ms` so it
# can be re-derived for every call. A longer hangover moves the zero
# later and therefore **lowers** the measured latency: that bias is declared.
#
# **It is not compared against PoC1's `stop_secs` or PoC2's `min_silence_duration`:**
# those are *conversational* knobs —they decide when the
# agent answers— and this one is for *measurement* only. Calibration and asymmetry in
# docs/CONTRATO_METRICAS.md §2.7.
SILENCE_FOR_TURN_DONE_S: Final[float] = 0.80

# Duration of a Twilio frame. It keeps a single-frame streak from
# measuring zero when compared against the minimum below.
INBOUND_FRAME_S: Final[float] = 0.02

# How much speech a streak must contain to count as a carrier line
# and be eligible as a turn zero. Below this it is a click on the
# line or a breath.
#
# 150 ms is below the shortest word in the script ("sí", "ajá", which
# are around 250-300 ms) and above a click. It is a knob: the JSON publishes
# `rachas_del_transportista` and `rachas_descartadas` so it can be calibrated with
# data instead of blindly.
MIN_TURN_SPEECH_S: Final[float] = 0.15

# Gap window published to calibrate the hangover. Below the
# minimum it is normal speech cadence and above the maximum it is a turn
# boundary no reasonable hangover would dispute: the decision lives in between.
MIN_GAP_OF_INTEREST_S: Final[float] = 0.1
MAX_GAP_OF_INTEREST_S: Final[float] = 3.0

# Maximum acceptable residual between the local arrival of an audio delta and its
# translated `start_ms`, for the clock anchoring to be accepted. It is on the order of
# "network trip plus queuing", not a fine measurement.
UMBRAL_RESIDUAL_SEG: Final[float] = 2.0

# How far the median residual may drift between the first and second
# half of the call before declaring that the origin moved.
DRIFT_THRESHOLD_S: Final[float] = 1.0

_LOCAL_ENERGY: Final[str] = "measured_by_local_energy"
_CLEAN_AXIS: Final[str] = "measured_on_the_server_axis"
_ANCHORED_CLOCK: Final[str] = "anchored_on_first_output_audio"
_CLOCK_DRIFT: Final[str] = "the_server_origin_drifted"
_CLOCK_NO_DATA: Final[str] = "no_data"

# Where the zero of `turno_completo_ms` came from. Today it is always local
# energy; the other names remain for reading old JSONs.
ZERO_LOCAL_ENERGY: Final[str] = "energy_local"
ZERO_SERVER_AXIS: Final[str] = "server_axis"
ZERO_MIXED: Final[str] = "mixed_server_and_local_clock"
CERO_PRIMER_AUDIO: Final[str] = "primer_output_audio_delta"
ZERO_SESSION_STARTED: Final[str] = "session_started"
ZERO_UNANCHORED: Final[str] = "unanchored"

# How many points of the series (local arrival vs `start_ms`) go into the JSON. This
# is enough to fit a line and to see whether the relationship is stepped rather than
# linear; storing a call's ~400 deltas would only bloat the file.
SERIES_POINTS: Final[int] = 120

USER_ROLE: Final[str] = "user"
AGENT_ROLE: Final[str] = "assistant"


def _ms(seconds: Any) -> float | None:
    """Seconds -> milliseconds, to one decimal. None if there is no data."""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return None
    return round(float(seconds) * 1000.0, 1)


class LatencyCollector:
    """Accumulates each turn's marks and produces the `MuestraDeTurno` samples.

    It is fed from the event loop in `app/telephony.py`. It does not touch the
    audio: it only reads what the alpha already published, plus the local instant at which
    each event reached the bridge.
    """

    def __init__(self, clock: Any = None, wall_clock: Any = None) -> None:
        self._clock = clock or time.monotonic
        # SECOND axis, which is ADDED to the one above and does not replace it. `_reloj`
        # is still the one that produces `turno_completo_ms` and the census `t_local_seg`,
        # which is what `diagnostico/paso0_latencias.py` reads: changing it
        # would move every PoC3 number already published.
        #
        # This other one exists for one thing only: subtracting the bridge acknowledgement
        # from the zeros the harness timestamps with `time.time()`. It is
        # `reloj.RelojAnclado.ahora` in production —epoch with `perf_counter`
        # granularity— and `None` in any run that does not pass it, and
        # then the clock block comes out declaring that there was no wall axis
        # instead of inventing marks.
        self._wall_clock = wall_clock
        self._t_start: float | None = None
        # Local instant of `session.started`. It was the original anchor and remains
        # as a fallback: it is **not** the origin of the server timeline
        # (docs/CONTRATO_METRICAS.md §2.7).
        self._t_session_started: float | None = None
        # Effective origin: the first `output_audio.delta` with a numeric
        # `start_ms`. Both examples in `docs_alpha/` show that first delta
        # at `start_ms: 0`, and the real run confirmed it. The origin of the server
        # timeline **is not documented**, so this is the
        # cheapest inference that holds up with the call's own
        # data — and `veredicto_del_cero` puts it to the test again in every
        # run instead of taking it for granted.
        self._t_audio_origin: float | None = None
        # (relative local t, start_ms, end_ms) of each delta, so the
        # relationship between the two clocks can be reconstructed offline.
        self._serie: list[tuple[float, Any, Any]] = []
        # Which axis each steady-state sample was measured on. This decides whether the
        # call enters the percentile pool.
        self._on_server_axis: int = 0
        self._on_local_clock: int = 0
        self._on_local_energy: int = 0

        # Energy marks. They are the source of `latencias[]`. Timing only: they
        # decide nothing about the conversation.
        #
        # The incoming speech streak in progress: first and last frame with speech. It
        # is closed lazily —in `voz_entrante` and in `voz_saliente`— because
        # `voz_entrante` is only called with frames WITH speech and nobody signals when the
        # carrier goes quiet.
        self._streak_start: float | None = None
        self._streak_end: float | None = None
        # End of the last closed streak the agent has not yet answered. It is the
        # zero of the next turn. It is **overwritten**: if the carrier made
        # two streaks with no response in between, the turn the agent answers is
        # the last one — same rule as `_turnos_pendientes` with the `turn.done` events.
        self._pending_zero: float | None = None
        self._t_last_outbound_speech: float | None = None

        # The same three instants, on the wall axis. They are kept separate and not as
        # a pair inside each field so that the path producing
        # `turno_completo_ms` stays byte for byte as it was: if the wall
        # axis is missing, these stay `None` and there is not a single new branch in
        # the latency computation.
        self._wall_streak_start: float | None = None
        self._wall_streak_end: float | None = None
        self._wall_zero_pending: float | None = None
        self._wall_last_outgoing_speech: float | None = None
        # One row per sample with a number: the zero and the acknowledgement in epoch time. It is what
        # the cross-check against `marcas.jsonl` needs and the only thing this axis
        # produces.
        self._wall_per_sample: list[dict[str, Any]] = []
        # The failure that turned off the wall axis, if any. `None` is normal.
        self._wall_clock_axis_failure: dict[str, Any] | None = None
        self._wall_start: float | None = None

        # Counts from the energy measurement, so the knobs can be calibrated
        # with data. They go to `gpt_live_extra`, not to the samples.
        self._incoming_streaks: int = 0
        # Gaps between the carrier's frames with speech, so that
        # `SILENCIO_PARA_TURNO_TERMINADO_SEG` can be re-derived from each call's data
        # instead of defended from memory. Those falling in the window
        # where the decision lives are kept: below it is speech cadence and
        # above it are obvious turn boundaries.
        self._incoming_gaps: list[float] = []
        self._discarded_streaks: int = 0
        self._streaks_without_response: int = 0
        self._agent_resumptions: int = 0
        self._onsets_over_carrier: int = 0
        self._expired_zeros: int = 0
        self._zeros_absorbed_by_opening: int = 0
        self._zeros_hung_at_end: int = 0

        # Round trips to the delegated model (`gpt-4.1-mini`). It is the only thing in the
        # "LLM latency" family that PoC3 **can** isolate, and it is not the
        # same as PoC1's and PoC2's `llm_ttft_ms`: there the LLM is what produces
        # the response and its TTFT falls inside the measured turn; here the delegate is
        # a secondary brain the voice model consults on some
        # turns, and meanwhile it **keeps speaking**. That is why it gets its own
        # column and never `llm_ttft_ms`: conflating them would compare two different
        # things across PoCs.
        self._t_delegation: float | None = None
        self._delegation_with_function: bool = False
        self._round_trips_llm: list[dict[str, Any]] = []

        self._samples: list[TurnSample] = []
        self._extras: list[dict[str, Any]] = []
        self._indice: int = 0

        # Carrier turns closed by the server and still without a
        # response from the agent. It is a list, not a single turn: a line
        # can arrive split across several `turn.done` events when the model does not
        # respond to the first one.
        self._pending_turns: list[dict[str, Any]] = []
        self._pending_opening: bool = True

        # Barge-in. `_t_solape` is when the carrier started talking
        # over; `_t_ultimo_audio_agente` is the last delta that went out.
        self._t_overlap: float | None = None
        self._t_last_agent_audio: float | None = None
        self._total_overlaps: int = 0
        self._total_cuts: int = 0

        # Clock-anchoring residuals, in arrival order.
        self._residuals: list[float] = []

        # The contract's "reliability" family: the alpha's `error` events. They go at
        # call level, not inside `latencias`, because a failure can
        # fall between turns and have no sample to attach to.
        self._failures: list[dict[str, Any]] = []
        self._turn_failures: list[dict[str, Any]] = []

        # Lo que el servidor confirmó y el uso acumulado, para el bloque extra.
        self.session_confirmed: dict[str, Any] = {}
        self.usage: dict[str, Any] = {}

    # -- Ciclo de vida --

    def begin(self) -> None:
        """Cero de la llamada: el Media Stream de Twilio conectó.

        Es el equivalente del `iniciar()` de PoC1 y del "contestaron" de PoC2:
        desde aquí se mide la apertura, que por definición del contrato paga el
        arranque en frío (abrir el WebSocket del alpha y la primera respuesta con
        el prompt completo).
        """
        self._t_start = self._clock()
        # El mismo cero, en época: es contra esto que se mide la apertura, cuyo
        # cero no es una racha del transportista sino el `connected` del Media
        # Stream.
        self._wall_start = self._now_on_wall()

    def anchor_server(self, session: dict[str, Any] | None = None) -> None:
        """Llegó `session.started`. Se guarda como **respaldo** del anclaje.

        Ya no es el origen por defecto: la primera corrida real mostró que la
        línea de tiempo del servidor no arranca aquí. Solo se usa si en toda la
        llamada no llegó un `output_audio.delta` con `start_ms` numérico, y en
        ese caso `zero_verdict` lo dice.
        """
        self._t_session_started = self._clock()
        if session:
            self.session_confirmed = dict(session)

    @property
    def samples(self) -> list[TurnSample]:
        return list(self._samples)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return list(self._failures)

    def to_json(self) -> list[dict[str, Any]]:
        """Las muestras para el JSON, **sin las etapas que PoC3 no mide**.

        `MuestraDeTurno.to_dict()` usa `asdict()`, que emite todos los campos
        incluidos los `None`. Aquí se filtran las cuatro etapas que en PoC3 no
        existen, para que el JSON diga "no aplica" por ausencia en vez de decir
        "no se midió" por nulo — son cosas distintas y el reporte las lee
        distinto. `MuestraDeTurno.from_dict` ignora las llaves que faltan, así
        que `shared/metrics/` lo lee sin una sola rama por PoC.
        """
        output: list[dict[str, Any]] = []
        for sample in self._samples:
            row = sample.to_dict()
            for campo in STAGE_FIELDS:
                row.pop(campo, None)
            output.append(row)
        return output

    # -- Traducción entre relojes --

    @property
    def _origin(self) -> float | None:
        """El origen efectivo de la línea de tiempo del servidor, en reloj local."""
        if self._t_audio_origin is not None:
            return self._t_audio_origin
        return self._t_session_started

    def _a_local(self, ms_servidor: Any) -> float | None:
        """Un `*_ms` de la línea de tiempo del servidor, en este reloj."""
        origin = self._origin
        if origin is None:
            return None
        if not isinstance(ms_servidor, (int, float)) or isinstance(ms_servidor, bool):
            return None
        return origin + float(ms_servidor) / 1000.0

    def _anchor_with_audio(self, now: float, start_ms: Any) -> None:
        """Fija el origen con el primer delta de audio que traiga `start_ms`.

        Se hace una sola vez y **antes** de anotar el residual, para que el
        primer punto de la serie salga en cero por construcción y la deriva se
        mida contra un origen y no contra dos.
        """
        if self._t_audio_origin is not None:
            return
        if not isinstance(start_ms, (int, float)) or isinstance(start_ms, bool):
            return
        self._t_audio_origin = now - float(start_ms) / 1000.0

    # -- Eventos del transportista --

    def closed_user_turn(self, turn: dict[str, Any]) -> None:
        """Un `turn.done` con `role: "user"`. Es la frontera del turno.

        Se guardan **todos** los que lleguen sin respuesta en medio: si el modelo
        no contestó al primero, el turno que el agente acaba respondiendo es el
        último, y los de en medio son parte de la misma intervención.
        """
        self._pending_turns.append(dict(turn))

    def user_started_speaking(self) -> None:
        """El transportista arrancó. Si el agente sonaba, es un solape.

        Solape no es lo mismo que interrupción: en una arquitectura full-duplex
        el modelo escucha mientras habla, así que solaparse es lo normal. Que
        haya sido corte se decide después, en `agent_first_audio`.
        """
        now = self._clock()
        if self._t_last_agent_audio is None:
            return
        if now - self._t_last_agent_audio <= CUTOFF_WINDOW_S:
            self._t_overlap = now
            self._total_overlaps += 1

    # -- Eventos del agente --

    # -- Energía local: la fuente de las muestras --

    def _now_on_wall(self) -> float | None:
        """El instante en el eje de época, o `None` si no hay eje de pared.

        Protegido igual que el resto del diagnóstico del camino caliente (ver
        `telefonia.PuenteDeLlamada._turn_off_diagnostics`): esto lo llama
        `inbound_speech`, o sea una vez por trama de 20 ms, y una excepción aquí
        no puede dejar de reenviar audio. A la primera falla el eje se apaga
        para el resto de la llamada y queda registrada.
        """
        if self._wall_clock is None:
            return None
        try:
            return float(self._wall_clock())
        except Exception as error:  # noqa: BLE001 - diagnóstico
            self._wall_clock = None
            self._wall_clock_axis_failure = {
                "sitio": "wall_clock",
                "excepcion": type(error).__name__,
                "message": str(error)[:300],
                "samples_before_failing": len(self._wall_per_sample),
            }
            logger.error(
                "el eje de pared falló y queda apagado el resto de la llamada: "
                "%r. La latencia por energía NO se ve afectada; lo que se pierde "
                "es el cruce contra los ceros del arnés",
                error,
            )
            return None

    def inbound_speech(self) -> None:
        """Una trama de Twilio con energía de voz. **Solo marca tiempo.**

        No decide nada: no abre turnos, no los cierra y no interrumpe al agente.
        Eso lo sigue haciendo el servidor. Aquí sólo se agrupa la voz en rachas,
        y el fin de la última racha cerrada es el cero del turno.
        """
        now = self._clock()
        wall = self._now_on_wall()
        if self._streak_end is not None:
            hole = now - self._streak_end
            if MIN_GAP_OF_INTEREST_S < hole < MAX_GAP_OF_INTEREST_S:
                self._incoming_gaps.append(hole)
        self._close_incoming_streak(now)

        if self._streak_start is None:
            self._streak_start = now
            self._wall_streak_start = wall
        self._streak_end = now
        self._wall_streak_end = wall

        # Solape: el transportista habla mientras el agente SUENA. Se cuenta
        # aparte; en full-duplex es el modo normal y no marca la muestra.
        if (
            self._t_last_outbound_speech is not None
            and now - self._t_last_outbound_speech <= OVERLAP_WINDOW_S
            and self._t_overlap is None
        ):
            self._t_overlap = now
            self._total_overlaps += 1

    def _close_incoming_streak(self, now: float) -> None:
        """Cierra la racha de voz del transportista si ya cumplió el hangover.

        Perezoso a propósito: nadie avisa cuando el transportista se calla, así
        que se revisa cada vez que llega una marca de energía en cualquiera de los
        dos sentidos.
        """
        if self._streak_end is None:
            return
        if now - self._streak_end <= SILENCE_FOR_TURN_DONE_S:
            return

        start = float(self._streak_start if self._streak_start is not None else self._streak_end)
        end = self._streak_end
        # El mismo fin, en época. Se toma ANTES de limpiar: es el instante de la
        # última trama con voz del transportista, o sea el cero del turno.
        wall_end = self._wall_streak_end
        self._streak_start = self._streak_end = None
        self._wall_streak_start = self._wall_streak_end = None
        self._incoming_streaks += 1

        # Una racha demasiado corta no es una intervención: es un clic de la
        # línea o una respiración, y no puede ser el cero de un turno.
        if (end - start) + INBOUND_FRAME_S < MIN_TURN_SPEECH_S:
            self._discarded_streaks += 1
            return

        if self._pending_zero is not None:
            # La racha anterior se quedó sin respuesta: o el transportista pausó
            # a media intervención, o el agente no contestó. En los dos casos el
            # turno que el agente acaba respondiendo es el último.
            self._streaks_without_response += 1
        self._pending_zero = end
        self._wall_zero_pending = wall_end

    def outbound_speech(self) -> None:
        """Una trama hacia Twilio con energía de voz. Cierra la muestra si abre turno."""
        now = self._clock()
        wall = self._now_on_wall()
        prior = self._t_last_outbound_speech
        self._t_last_outbound_speech = now
        self._wall_last_outgoing_speech = wall
        self._close_incoming_streak(now)

        # ¿El agente venía hablando? Entonces esto no abre turno, lo continúa.
        if prior is not None and (now - prior) < SILENCE_BETWEEN_TURNS_S:
            return

        self._close_sample_by_energy(
            now, previous_voice_end=prior, acknowledgement_wall=wall
        )

    def _close_sample_by_energy(
        self,
        now: float,
        *,
        previous_voice_end: float | None,
        acknowledgement_wall: float | None = None,
    ) -> None:
        """El agente arrancó a hablar. Produce muestra si hay algo que medir.

        Tres casos y sólo tres:

        - es la primera voz del agente en la llamada -> **apertura**;
        - hay una racha del transportista cerrada y sin contestar -> turno de
          régimen con número;
        - el agente arrancó encima de una racha abierta -> muestra sin número.

        Cualquier otro arranque de voz saliente es el agente reanudando su propia
        intervención tras una pausa larga (el relleno de la delegación), y **no
        produce muestra**: si la produjera, una intervención contaría varias veces.
        """
        # El barge-in se evalúa siempre, incluso si esto no va a producir muestra:
        # el corte existió y `gpt_live_extra` lo cuenta.
        interrupted = False
        if self._t_overlap is not None and previous_voice_end is not None:
            gap = previous_voice_end - self._t_overlap
            # Con cota inferior: una brecha negativa es el agente que ya se había
            # callado antes del solape, o sea que nadie lo cortó.
            interrupted = 0.0 <= gap <= CUTOFF_WINDOW_S
            if interrupted:
                self._total_cuts += 1
        self._t_overlap = None

        is_opening = self._indice == 0
        notes: list[str] = []
        full_turn: float | None = None

        # ¿El cero pendiente sigue siendo válido? Sólo si el agente estuvo callado
        # desde que terminó la racha del transportista. Si el agente ya había
        # hablado **después** de esa racha, la racha fue un apoyo verbal a media
        # intervención suya —un "ajá", un "sí"— y esto de ahora es la reanudación
        # de esa misma intervención, no una respuesta. Sin esta guarda, la frase
        # siguiente del agente consumiría ese cero como si fuera una latencia.
        cero = self._pending_zero
        zero_wall = self._wall_zero_pending
        if cero is not None and previous_voice_end is not None and previous_voice_end > cero:
            self._expired_zeros += 1
            self._pending_zero = cero = None
            self._wall_zero_pending = zero_wall = None

        if is_opening:
            if self._t_start is not None:
                full_turn = _ms(now - self._t_start)
            notes.append(
                "apertura: del `connected` del Media Stream a la primera voz del "
                "agente. Paga el arranque en frío (WebSocket del alpha, sesión y "
                "primera respuesta con el prompt completo)"
            )
            if self._pending_zero is not None or self._streak_end is not None:
                notes.append(
                    "el transportista ya había hablado antes del saludo: contestó "
                    "el teléfono. Sigue siendo arranque en frío, no una respuesta"
                )
            # Su cero es el `connected`, no la racha que el transportista
            # pudo haber hecho al contestar el teléfono: el número que publica
            # `turno_completo_ms` para la apertura se mide contra `_t_inicio`.
            zero_wall = self._wall_start
            if self._pending_zero is not None:
                # La apertura se comió una racha contestada. Se cuenta para que el
                # cuadre de `turn_taking` cierre exacto en vez de salir sobrado por
                # uno.
                self._zeros_absorbed_by_opening += 1
            self._pending_zero = None
            self._wall_zero_pending = None
        elif cero is not None:
            full_turn = _ms(now - cero)
            self._pending_zero = None
            self._wall_zero_pending = None
            notes.append(
                "medido por energía local: fin de la última racha de voz del "
                "transportista -> primera voz del agente. Incluye red, puente y "
                "remuestreo, igual que PoC1 y PoC2 (docs/METRICS_CONTRACT.md §2.7)"
            )
        elif self._streak_end is not None:
            self._onsets_over_carrier += 1
            notes.append(
                "el agente arrancó encima del transportista, que seguía hablando "
                f"(su racha llevaba {(now - self._streak_end) * 1000:.0f} ms de "
                f"silencio, mínimo {SILENCE_FOR_TURN_DONE_S * 1000:.0f} "
                "ms): no hay turno cerrado contra el cual medir y no se reporta "
                "número. Es el modo full-duplex, no un fallo"
            )
        else:
            # El agente reanuda su propia intervención. No hay turno que medir.
            self._agent_resumptions += 1
            return

        notes.extend(self._notes_per_failure())

        self._samples.append(
            TurnSample(
                indice=self._indice,
                full_turn_ms=full_turn,
                interrupted=interrupted,
                is_opening=is_opening,
                end_of_turn_verdicts=(),
                notes=tuple(notes),
            )
        )
        # El mismo turno en el eje de época, aparte. NO va dentro de
        # `MuestraDeTurno`: ese contrato vive en `shared/metrics/latencia.py` y
        # `desde_dict` tira en silencio las claves que no conoce, así que meterlo
        # ahí sería perderlo sin que nadie se entere.
        if self._wall_clock is not None or acknowledgement_wall is not None:
            self._wall_per_sample.append(
                {
                    "indice": self._indice,
                    "is_opening": is_opening,
                    # Fin de la última racha de voz del transportista, que es lo
                    # que el arnés fecha como `fin_de_voz_medido` en su marca.
                    "t_zero_wall": (
                        None if zero_wall is None else round(zero_wall, 6)
                    ),
                    # Primera trama con voz del agente hacia Twilio: el acuse.
                    "t_ack_wall": (
                        None if acknowledgement_wall is None else round(acknowledgement_wall, 6)
                    ),
                    # Se repite a propósito: permite verificar que el eje de
                    # pared y el interno dan la misma diferencia sin tener que
                    # cruzar dos archivos.
                    "full_turn_ms": full_turn,
                }
            )
        self._indice += 1
        self._on_local_energy += 1

    def agent_first_audio(self, *, start_ms: Any = None, end_ms: Any = None) -> None:
        """Llegó audio del agente. Cierra la muestra del turno que lo esperaba.

        Sin turno pendiente esto es la continuación del mismo turno del agente
        —un segundo tramo de voz después de un tool call— y se ignora, porque
        `full_turn` es hasta el **primer** audio. Es la misma regla de PoC1
        y PoC2.
        """
        now = self._clock()
        self._annotate_residual(now, start_ms, end_ms)
        self._t_last_agent_audio = now

        if self._pending_turns:
            self._close_turn(now, start_ms)
            return

        if self._pending_opening:
            self._pending_opening = False
            self._close_opening(now, start_ms)

    def agent_audio(self, *, start_ms: Any = None, end_ms: Any = None) -> None:
        """Un delta de audio que no abre turno. Solo mueve el fin de habla.

        Hace falta para el criterio de barge-in: sin esto, "el agente dejó de
        hablar" se mediría contra el primer delta de la frase y no contra el
        último.
        """
        now = self._clock()
        self._annotate_residual(now, start_ms, end_ms)
        self._t_last_agent_audio = now

    # -- Delegación al modelo de la Responses API --

    def delegation_started(self) -> None:
        """Llegó `delegation.created`: el modelo de voz entregó el turno."""
        self._t_delegation = self._clock()
        self._delegation_with_function = False

    def delegation_with_function(self) -> None:
        """El delegado emitió una llamada a función en este round trip."""
        self._delegation_with_function = True

    def delegation_finished(self) -> None:
        """El delegado cerró todas sus respuestas. Se guarda el round trip.

        El cero es `delegation.created` y el fin es el último `response.completed`
        del mismo intercambio, así que el número incluye los dos `response.*` que
        salen cuando hay llamada a función (los argumentos y luego el texto).
        """
        if self._t_delegation is None:
            return
        ms = _ms(self._clock() - self._t_delegation)
        self._t_delegation = None
        if ms is None:
            return
        self._round_trips_llm.append(
            {"ms": ms, "with_function": self._delegation_with_function}
        )

    def llm_round_trips(self) -> list[dict[str, Any]]:
        """Los round trips al delegado, crudos. `shared/metrics/` los agrega."""
        return [dict(r) for r in self._round_trips_llm]

    def _annotate_residual(self, now: float, start_ms: Any, end_ms: Any = None) -> None:
        """Cuánto tardó este delta respecto de su `start_ms` traducido.

        Ancla primero, si hacía falta: el origen sale del primer delta útil.
        Guarda además el punto crudo, que es lo único con lo que se puede
        decidir después si la relación entre los dos relojes es lineal o
        escalonada — los docs avisan que un hueco entre rangos es **silencio
        omitido**, y si el servidor no cuenta ese silencio las dos líneas de
        tiempo no son la misma magnitud y no hay traducción que valga.
        """
        self._anchor_with_audio(now, start_ms)
        if self._t_start is not None:
            self._serie.append((round(now - self._t_start, 6), start_ms, end_ms))
        expected = self._a_local(start_ms)
        if expected is not None:
            self._residuals.append(now - expected)

    # -- Fallos --

    def record_failure(
        self,
        tipo: str,
        *,
        code_value: str = "",
        message: str = "",
        param: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Un evento `error` del alpha.

        Se guardan **todos**. Un error de comando no necesariamente cierra la
        sesión —lo dice el propio alpha— pero sí puede haberle costado segundos
        de silencio al transportista y ensuciado la latencia del turno. Es el
        dato con el que después se sabe si una corrida está limpia. Mismo
        criterio que `technical_failures` de PoC2.

        Recordatorio del contexto: el alpha declara que su confiabilidad y su
        latencia **no cumplen estándares de producción**, así que aquí se espera
        más ruido que en PoC1 y PoC2 y por eso hace falta contarlo.
        """
        record: dict[str, Any] = {
            "t": (
                round(self._clock() - self._t_start, 3)
                if self._t_start is not None
                else None
            ),
            "tipo": tipo,
            # El alpha no clasifica los errores como recuperables; se deja la
            # llave por compatibilidad con el JSON de PoC2 y se marca None para
            # no inventar un dato que la API no da.
            "recuperable": None,
            "provider": "openai",
            "model_name": str(self.session_confirmed.get("model") or ""),
            "code_value": code_value,
            "param": param,
            "event_id": event_id,
            "detail": message,
        }
        self._failures.append(record)
        self._turn_failures.append(record)
        logger.warning("error del alpha: %s / %s — %s", tipo, code_value, message)

    def _notes_per_failure(self) -> list[str]:
        """Nota para la muestra que se cierra, si hubo fallos en su turno.

        La muestra **no se descarta ni se anula**: el número medido sigue siendo
        cierto. Lo que se agrega es la advertencia de que ese turno no describe
        al framework, para que nadie lo cite y para poder sacarlo a mano.
        """
        if not self._turn_failures:
            return []
        kinds = sorted({str(f["tipo"]) for f in self._turn_failures})
        n = len(self._turn_failures)
        self._turn_failures = []
        return [
            f"turno con {n} error(es) del alpha ({', '.join(kinds)}): la latencia "
            "de este turno no describe al motor conversacional"
        ]

    # -- Construcción de la muestra --

    def _close_opening(self, now: float, start_ms: Any) -> None:
        """Diagnóstico de la apertura según los eventos. **No produce muestra.**

        `latencias[]` sale de la energía local; esto queda en
        `gpt_live_extra.per_turn` para poder contrastar los dos métodos sin
        repetir llamadas.
        """
        full_turn = None
        if self._t_start is not None:
            full_turn = _ms(now - self._t_start)
            if full_turn is not None and full_turn < 0:
                full_turn = None

        self._extras.append(
            {
                "indice": None,
                "source": "alpha_events",
                "is_opening": True,
                "audio_start_ms": start_ms,
                "user_turns": [],
                "full_turn_event_ms": full_turn,
            }
        )

    def _close_turn(self, now: float, start_ms: Any) -> None:
        """Diagnóstico del turno según los eventos. **No produce muestra.**

        Los dos números que salen de aquí —eje del servidor y reloj local— se
        conservan en `gpt_live_extra.per_turn` porque permiten contrastar los
        tres métodos de medición con los datos de una sola llamada. La muestra
        que va a `latencias[]` la produce la energía local.
        """
        turns = self._pending_turns
        self._pending_turns = []

        ultimo = turns[-1]
        local_speech_end = self._a_local(ultimo.get("end_ms"))

        on_server = self._server_only(ultimo, start_ms)
        if on_server is not None:
            self._on_server_axis += 1
        elif local_speech_end is not None:
            self._on_local_clock += 1

        self._extras.append(
            {
                "indice": None,
                "source": "alpha_events",
                "is_opening": False,
                "audio_start_ms": start_ms,
                "user_turns": turns,
                "full_turn_server_ms": self._server_only(ultimo, start_ms),
                # La versión con reloj local, que es la que arrastra el retraso
                # de entrega. Se conserva para poder restar una de otra sin
                # repetir llamadas.
                "full_turn_local_ms": (
                    _ms(now - local_speech_end) if local_speech_end is not None else None
                ),
            }
        )

    def _server_only(self, turn: dict[str, Any], start_ms: Any) -> float | None:
        """`start_ms` del audio menos `end_ms` del turno, los dos del servidor.

        Es el número sin ruido de red ni de puente. **No es la métrica de la
        comparativa** —deja fuera justo lo que el transportista sí percibe— pero
        permite separar "tardó el modelo" de "tardó la red" sin repetir llamadas.
        """
        end = turn.get("end_ms")
        for valor in (end, start_ms):
            if not isinstance(valor, (int, float)) or isinstance(valor, bool):
                return None
        delta = round(float(start_ms) - float(end), 1)  # type: ignore[arg-type]
        return delta if delta >= 0 else None

    def close_pending_ones(self) -> None:
        """Al colgar: vuelca los turnos que se quedaron sin respuesta.

        También cierra el cero de energía que quedó armado sin contestar. Pasa de
        truth: si el transportista dice la última palabra de la llamada y se
        cuelga, su racha nunca recibe respuesta y no genera muestra. Sin contarlo,
        el cuadre de `turn_taking` sale sobrado por uno y parece un error de
        medición cuando es la llamada terminándose.

        No hay muestra que cerrar —el agente nunca contestó, así que no hay nada
        que medir— pero el turno existió y perderlo escondería que el modelo se
        quedó callado, que es un dato de confiabilidad.
        """
        # La racha abierta al colgar sí terminó —la llamada se acabó— así que se
        # cierra sin esperar el hangover. Sólo cuenta como colgada si dejó un cero
        # sin contestar: si el agente ya había arrancado encima de ella, esa vía ya
        # está contabilizada en `arranques_sobre_el_transportista`.
        if self._streak_end is not None:
            self._close_incoming_streak(
                float(self._streak_end) + SILENCE_FOR_TURN_DONE_S + 1e-6
            )
        if self._pending_zero is not None:
            self._zeros_hung_at_end += 1
            self._pending_zero = None

        if not self._pending_turns:
            return
        self._extras.append(
            {
                "indice": None,
                "no_agent_response": True,
                "user_turns": list(self._pending_turns),
            }
        )
        self._pending_turns = []

    # -- Diagnóstico --

    def zero_used(self) -> str:
        """Qué eje se usó para `full_turn_ms` de los turnos de régimen."""
        if self._on_local_energy:
            return ZERO_LOCAL_ENERGY
        return ZERO_UNANCHORED

    def clock_series(self) -> list[dict[str, Any]]:
        """La serie (llegada local, `start_ms`, `end_ms`), submuestreada.

        Es la evidencia cruda del anclaje. Con ella se puede ajustar fuera de
        línea y decidir si la relación entre los dos relojes es una recta con
        deriva o una escalera —lo segundo si el servidor descuenta el silencio—
        sin tener que pagar otra corrida.
        """
        if not self._serie:
            return []
        step = max(1, len(self._serie) // SERIES_POINTS)
        return [
            {"t_local_seg": t, "start_ms": start, "end_ms": end}
            for t, start, end in self._serie[::step]
        ]

    def delivery_lag(self) -> dict[str, Any]:
        """Cuánto se atrasa el servidor en **entregar** el audio que ya generó.

        El desfase entre su línea de tiempo y el reloj de pared no es un error de
        anclaje que otro origen arregle: es el servicio entregando cada vez más
        tarde. Por eso `full_turn_ms` se mide por energía local y esto se
        publica aparte, como confiabilidad: sobre telefonía, un atraso que crece
        así es lo que el transportista percibe como que el agente se quedó
        colgado. **No es una latencia de respuesta** (docs/METRICS_CONTRACT.md
        §2.7).
        """
        if len(self._serie) < 2:
            return {"n": len(self._serie), "detail": "sin suficientes deltas de audio"}

        points = [
            (t, float(start))
            for t, start, _ in self._serie
            if isinstance(start, (int, float)) and not isinstance(start, bool)
        ]
        if len(points) < 2:
            return {"n": len(points), "detail": "ningún delta trajo start_ms numérico"}

        t0, s0 = points[0]
        delays = [((t - t0) * 1000.0) - (s - s0) for t, s in points]
        span_servidor = points[-1][1] - points[0][1]
        span_local = (points[-1][0] - points[0][0]) * 1000.0

        return {
            "n": len(points),
            "initial_delay_ms": round(delays[0], 1),
            "final_delay_ms": round(delays[-1], 1),
            "median_delay_ms": round(median(delays), 1),
            "span_servidor_ms": round(span_servidor, 1),
            "span_local_ms": round(span_local, 1),
            "delivery_pacing": (
                round(span_servidor / span_local, 3) if span_local else None
            ),
            "note": (
                "`ritmo_de_entrega` es cuántos ms de audio entrega el alpha por "
                "cada ms de reloj de pared. NO está dentro de full_turn_ms: eso "
                "se mide por energía local sobre el audio del puente y no traduce "
                "nada del eje del servidor."
            ),
            "NOT_LATENCY": (
                "⚠ `atraso_*` NO es un retraso de respuesta y no se cita como tal. "
                "Es la distancia entre el eje `start_ms` del alpha y el reloj de "
                "pared, y crece a lo largo de la llamada porque el eje del servidor "
                "avanza más despacio que el reloj: en la llamada 1.2 del 12 de "
                "agosto de 2026 llegó a 31,312 ms mientras los turnos medidos por "
                "energía local salían en 1,300-1,700 ms. Los dos números son "
                "correctos y miden cosas distintas. La lectura defendible es la de "
                "`ritmo_de_entrega`: el eje del alpha NO es un reloj de pared, así "
                "que ninguna marca suya se puede traducir a uno — que es exactamente "
                "la razón por la que la medida principal se mide por fuera."
            ),
        }

    def clock_verdict(self) -> dict[str, Any]:
        """¿El anclaje elegido describe esta llamada?

        Se decide con los datos de la propia corrida, sin creerle a nadie — el
        mismo patrón que `zero_verdict` de PoC2.

        El discriminante es el residual: cuánto tardó cada delta de audio en
        llegar respecto de su `start_ms` traducido a reloj local. Si el origen
        está bien puesto, ese residual es el viaje de red más el encolado: chico
        y **estable**. Si el origen del servidor no es `session.started`, o si las
        dos líneas de tiempo corren a ritmos distintos, el residual **deriva** a
        lo largo de la llamada — y entonces `full_turn_ms` arrastra esa
        deriva y no describe nada.
        """
        if not self._residuals:
            return {
                "cero": self.zero_used(),
                "verdict": _CLOCK_NO_DATA,
                "n_deltas": 0,
                "detail": (
                    "no llegó ningún output_audio.delta con start_ms: sin ellos no "
                    "se puede verificar el anclaje del reloj"
                ),
            }

        half = len(self._residuals) // 2 or 1
        first = median(self._residuals[:half])
        second_one = median(self._residuals[half:] or self._residuals[:half])
        drift = abs(second_one - first)
        median_value = median(self._residuals)

        # Con la medida en el eje del servidor, la deriva del reloj local ya no
        # invalida `turno_completo_ms`: es el retraso de entrega y va aparte.
        # Solo hay problema si algún turno tuvo que caer al reloj local.
        on_axis = self._on_local_energy > 0
        stable = drift <= DRIFT_THRESHOLD_S and abs(median_value) <= UMBRAL_RESIDUAL_SEG
        return {
            "cero": self.zero_used(),
            "verdict": (
                _LOCAL_ENERGY
                if on_axis
                else (_ANCHORED_CLOCK if stable else _CLOCK_DRIFT)
            ),
            "turns_measured_by_local_energy": self._on_local_energy,
            # Los dos de abajo cuentan el diagnóstico de `por_turno`, NO las
            # muestras de `latencias[]`. Están para poder contrastar los métodos.
            "diagnostics_on_server_axis": self._on_server_axis,
            "diagnostics_on_local_clock": self._on_local_clock,
            "n_deltas": len(self._residuals),
            "residual_median_ms": _ms(median_value),
            "residual_first_half_ms": _ms(first),
            "residual_second_half_ms": _ms(second_one),
            "drift_ms": _ms(drift),
            # Los umbrales sólo aplican a la traducción de relojes que se abandonó.
            # Con la medida por energía local se publican como referencia histórica
            # y **no** son un criterio de aprobación.
            "umbral_residual_ms": _ms(UMBRAL_RESIDUAL_SEG),
            "drift_threshold_ms": _ms(DRIFT_THRESHOLD_S),
            "thresholds_apply": not on_axis,
            "detail": (
                "full_turn_ms se midió por energía local sobre el audio que "
                "entra y sale del puente, así que no depende de la línea de "
                "tiempo del servidor ni de su deriva. Incluye red, puente y "
                "remuestreo, igual que PoC1 y PoC2: ver "
                "docs/METRICS_CONTRACT.md §2.7. El residual y la deriva de aquí "
                "abajo son la EVIDENCIA de por qué se mide así —el eje del alpha no "
                "es un reloj de pared— y no un veredicto sobre la latencia: no los "
                "cites como retraso"
                if on_axis
                else "el residual entre la llegada local y el start_ms traducido es "
                "estable a lo largo de la llamada: el origen del servidor está "
                "donde se supuso y full_turn_ms es comparable con PoC1 y PoC2"
                if stable
                else "el residual DERIVA durante la llamada: el origen elegido no "
                "es el de la línea de tiempo del servidor, o los dos relojes corren "
                "distinto. Mira clock_series: si la relación es escalonada y no "
                "una recta, el servidor descuenta el silencio y las dos líneas de "
                "tiempo no son la misma magnitud. Usa turno_completo_servidor_ms de "
                "per_turn y no cites ningún percentil"
            ),
        }

    def measurement_degradation(self) -> dict[str, Any] | None:
        """La llamada no entra a la poza de percentiles, o `None` si está bien.

        El caso a evitar es mezclar ejes: un turno medido con reloj local trae
        adentro el retraso de entrega del servidor, y juntarlo con turnos medidos
        en el eje del servidor contaminaría el p50 sin dejar rastro.

        **Ya no se excluye por deriva del reloj.** Esa deriva existe y está
        medida, pero desde que `full_turn_ms` se calcula con los dos
        extremos en el eje del servidor no entra en la métrica: vive en
        `delivery_lag`, como el hallazgo propio que es.

        Igual que en PoC2, la llamada **no se descarta**: sigue en el detalle, su
        WER sigue contando, y `shared/metrics/` respeta esta llave al consolidar.
        """
        unmeasured = [m for m in self._samples if m.full_turn_ms is None]
        if not self._samples:
            return {
                "motivo": "no se capturó ninguna muestra de latencia",
                "detail": (
                    "no se detectó voz saliente en toda la llamada: revisa el "
                    "umbral de energía o si el agente llegó a hablar"
                ),
            }
        # ⚠ Un turno sin número NO es lo mismo que una captura fallida, y
        # confundirlos costaba caro: con el umbral viejo ("más de la mitad sin
        # número") 4 de las primeras 8 llamadas salían de la poza con el mensaje
        # "arquitectura distinta de la que se quería medir" — que en PoC3 es
        # **falso**. La arquitectura era la correcta; lo que pasa es que el modelo
        # es full-duplex y arranca encima del transportista, y ahí no hay turno
        # cerrado contra el cual medir. En esas 8 llamadas **el 100%** de los
        # turnos sin número eran `arranques_sobre_el_transportista`.
        #
        # Los turnos que sí traen número siguen siendo mediciones válidas de la
        # misma arquitectura, así que no contaminan ningún percentil y la llamada
        # se queda en la poza. Lo que falta no se esconde: la cobertura sube a la
        # tabla de confiabilidad como métrica propia (`shared/metrics/`).
        by_overlap = sum(
            1
            for m in unmeasured
            if any("encima del transportista" in note for note in m.notes)
        )
        unexplained = len(unmeasured) - by_overlap
        if unexplained * 2 > len(self._samples):
            return {
                "motivo": "la captura de latencia falló en más de la mitad de los turnos",
                "detail": (
                    f"{unexplained} de {len(self._samples)} muestras sin "
                    "full_turn_ms por una razón que NO es solape full-duplex: "
                    "eso apunta a un problema de captura, no a la arquitectura. "
                    f"(Otros {by_overlap} sí son solape y no cuentan aquí.) Revisa "
                    "las notas de esas muestras antes de citar cualquier percentil"
                ),
            }
        return None

    def measurement_coverage(self) -> dict[str, Any]:
        """Cuántos turnos admitieron medición, y por qué los otros no.

        Es **resultado de la comparativa**, no un hueco de instrumentación: que un
        modelo full-duplex arranque encima del interlocutor en una fracción alta de
        los turnos es un hallazgo sobre la arquitectura, y probablemente más
        interesante que el p50. Antes vivía enterrado en una advertencia de
        exclusión que además decía otra cosa.

        Ojo con el sesgo que introduce, y hay que declararlo al citar el p50: si el
        agente se encima justo cuando responde rápido, los turnos que quedan
        medidos son los lentos y el p50 sale **hacia arriba**. No se corrige
        excluyendo llamadas —el sesgo es del mecanismo, no de la muestra— se
        declara y se lee junto con la cobertura.
        """
        total = len(self._samples)
        no_number = [m for m in self._samples if m.full_turn_ms is None]
        by_overlap = sum(
            1
            for m in no_number
            if any("encima del transportista" in note for note in m.notes)
        )
        return {
            "turns": total,
            "with_number": total - len(no_number),
            "no_number": len(no_number),
            "no_number_due_to_full_duplex_overlap": by_overlap,
            "no_number_for_other_reason": len(no_number) - by_overlap,
            "cobertura": round((total - len(no_number)) / total, 4) if total else None,
        }

    def extra_block(self) -> dict[str, Any]:
        """Todo lo que el alpha da de más, para el diagnóstico.

        Va en la llave `gpt_live_extra` del JSON, **nunca** dentro de
        `latencias`: `MuestraDeTurno.from_dict` ignora las claves que no
        conoce, así que meterlas ahí las tiraría en silencio.
        """
        return {
            "session_confirmed": self.session_confirmed,
            "usage": self.usage,
            # El nombre sigue el patrón de `veredicto_del_cero` de PoC2: la
            # decisión de dónde está el cero se documenta en el JSON de cada
            # llamada y se vuelve a poner a prueba con los datos de esa llamada.
            "zero_verdict": self.clock_verdict(),
            "delivery_lag": self.delivery_lag(),
            "clock_series": self.clock_series(),
            "turn_taking": {
                "overlaps": self._total_overlaps,
                "cortes": self._total_cuts,
                "cutoff_window_s": CUTOFF_WINDOW_S,
                "overlap_window_s": OVERLAP_WINDOW_S,
                "note": (
                    "`solapes` es cuántas veces el transportista habló mientras el "
                    "agente sonaba; `cortes` es cuántas de esas callaron al agente "
                    "dentro de la ventana. Solo los cortes marcan `interrumpido` en "
                    "la muestra: en una arquitectura full-duplex el solape es el "
                    "modo normal de operación y no dice nada del framework."
                ),
                # Contabilidad de la medición por energía. Sirve para calibrar las
                # perillas con datos y para ver de un golpe si el número de
                # muestras cuadra con la conversación: muestras = apertura +
                # rachas contestadas, y nada más.
                "carrier_streaks": self._incoming_streaks,
                "discarded_streaks": self._discarded_streaks,
                "streaks_without_response": self._streaks_without_response,
                "agent_resumptions": self._agent_resumptions,
                "onsets_over_carrier": (
                    self._onsets_over_carrier
                ),
                "zeros_vencidos_por_apoyo_verbal": self._expired_zeros,
                "zeros_absorbidos_por_la_apertura": (
                    self._zeros_absorbed_by_opening
                ),
                "zeros_colgados_al_final": self._zeros_hung_at_end,
                "min_turn_speech_s": MIN_TURN_SPEECH_S,
                "incoming_hangover_s": SILENCE_FOR_TURN_DONE_S,
                "incoming_gaps_ms": sorted(
                    round(h * 1000.0) for h in self._incoming_gaps
                ),
                "hangover_note": (
                    "`incoming_gaps_ms` son los silencios del transportista "
                    f"entre {MIN_GAP_OF_INTEREST_S * 1000:.0f} y "
                    f"{MAX_GAP_OF_INTEREST_S * 1000:.0f} ms: la ventana donde "
                    "vive la decisión del hangover. Cuenta cuántos pasan de "
                    "`incoming_hangover_s`: esos son los que PARTEN una frase en "
                    "dos y arman un cero falso. Con 250 ms, la llamada del 12 de "
                    "agosto de 2026 partía 26 de 42 y su p50 de régimen bajaba de "
                    "2,156 a 1,117 ms. Si en una corrida salen muchos por encima, "
                    "el hangover quedó corto — pero NO se cambia a media campaña: "
                    "se anota y se repiten las llamadas ya hechas."
                ),
                "streaks_note": (
                    "`rachas_descartadas` son rachas de voz entrante más cortas que "
                    "`voz_minima_del_turno_seg`: ruido de línea, no habla. "
                    "`reanudaciones_del_agente` son arranques de voz del agente que "
                    "NO produjeron muestra porque continuaban su propia "
                    "intervención tras una pausa larga — sin este filtro la llamada "
                    "1.1 reportó 19 muestras para 8 turnos. "
                    "`ceros_vencidos_por_apoyo_verbal` son rachas válidas que el "
                    "agente no contestó porque ya estaba hablando encima de ellas: "
                    "eran apoyos verbales ('ajá', 'sí'), y medir contra ellas dio un "
                    "falso 8,266 ms en la llamada 1.2. "
                    "`zeros_colgados_al_final` es la última racha de la llamada "
                    "cuando el agente ya había colgado y no la contestó: existió "
                    "como turno pero no produce muestra. "
                    "CUADRE ESPERADO: muestras = 1 apertura + rachas contestadas, "
                    "donde rachas contestadas = rachas_del_transportista - "
                    "rachas_descartadas - rachas_sin_respuesta - "
                    "ceros_vencidos_por_apoyo_verbal "
                    "- ceros_absorbidos_por_la_apertura "
                    "- zeros_colgados_al_final "
                    "(+ onsets_over_carrier, que salen sin número). "
                    "`ceros_absorbidos_por_la_apertura` existe porque el "
                    "transportista contesta el teléfono hablando: esa racha la "
                    "absorbe la apertura y no genera muestra propia."
                ),
            },
            "audio_bridge": {
                "resampling_lag_ms_per_direction": bridge.RESAMPLING_LAG_MS,
                "peak_lag_ms_per_direction": bridge.PEAK_RESAMPLING_LAG_MS,
                "note": (
                    "El remuestreo μ-law 8 kHz <-> PCM16 24 kHz está DENTRO de "
                    "full_turn_ms y no se resta: es parte de lo que cuesta "
                    "esta arquitectura sobre telefonía."
                ),
            },
            "unmeasurable_stages": {
                "fields": list(STAGE_FIELDS),
                "note": (
                    "Se omiten del JSON en vez de escribirse como null. No es "
                    "captura incompleta: el modelo es full-duplex y no tiene STT, "
                    "LLM ni TTS aislables, así que esas etapas no existen. Ver "
                    "docs/METRICS_CONTRACT.md §3."
                ),
            },
            "measurement_coverage": self.measurement_coverage(),
            "wall_axis": self.wall_timeline_block(),
            "per_turn": list(self._extras),
        }

    def wall_timeline_block(self) -> dict[str, Any]:
        """El cero y el acuse de cada muestra en época, para cruzar con el arnés.

        Es la mitad del requisito del paso 3 que vive en el colector; la otra
        —el ancla, su resolución y su deriva— la publica `reloj.RelojAnclado` en
        la llave `reloj` del mismo JSON.

        Va siempre, también cuando no hay eje: una llave que solo aparece
        cuando funciona se lee como ausente cuando falta el código que la
        escribe, y ahí una llamada sin cruce vuelve a contar como cruzable.
        """
        return {
            "presente": self._wall_clock is not None
            or bool(self._wall_per_sample),
            "fault": self._wall_clock_axis_failure,
            "samples": list(self._wall_per_sample),
            "what_it_is": (
                "`t_cero_pared` es el fin de la última racha de voz del "
                "transportista y `t_acuse_pared` la primera trama con voz del "
                "agente hacia Twilio, las dos en segundos desde la época, en el "
                "MISMO eje en el que el arnés fecha sus ceros "
                "(`time.time()` anclado una vez + `perf_counter`). Es lo que "
                "hace restable el acuse del puente contra `marcas.jsonl`. El "
                "eje interno del colector, `time.monotonic`, no cambió: sigue "
                "produciendo `full_turn_ms` y el `t_local_seg` del censo"
            ),
        }
