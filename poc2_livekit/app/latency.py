"""PoC2 latency capture, on top of LiveKit's per-turn metrics.

`shared/metrics/` **computes**; this module **measures**. Where PoC1 spies on
frames from three probes in the pipeline, LiveKit already publishes per-stage
latency in `ChatMessage.metrics`: here it is only translated into the
`TurnSample` contract.

Full mapping and its justification: `CAPTURE_METRICS.md`.

## How a turn is built

A contract turn is one exchange, and in LiveKit that is **two**
`ChatMessage`s with the metrics split between them:

    user message            -> stopped_speaking_at, transcription_delay,
                               end_of_turn_delay
    assistant message       -> llm_node_ttft, tts_node_ttfb, e2e_latency,
                               playback_latency, started_speaking_at

The collector accumulates the user ones and closes the sample when the
assistant one arrives. **It accumulates, plural, on purpose:** a line often
arrives split into several `ChatMessage`s and all of them stay in `per_turn`,
so the silence *within* a turn can be reconstructed by subtracting consecutive
ones. The sample does not change — its stages hang off the last anchored
fragment, which is the zero from which LiveKit measures `e2e_latency`.

## The turn's zero: why the VAD silence is NOT subtracted

This is the decision that could most bias the comparison. PoC1 has to subtract
its `stop_secs` (0.2 s) from the VAD frame, because pipecat delivers the event
when it **confirms** the silence; the question was whether LiveKit does the
same, in which case its 0.55 s would have to be subtracted.

**Nothing needs to be subtracted.** LiveKit already does it internally
(`livekit/agents/voice/audio_recognition.py`, `VADEventType.END_OF_SPEECH` branch):

    speech_end_time = time.time() - ev.silence_duration - ev.inference_duration
    self._last_speaking_time = speech_end_time

That `_last_speaking_time` is what gets published as `stopped_speaking_at`. In
other words, LiveKit already anchors on the real end of speech, just like PoC1
after its subtraction: both frameworks measure from the same physical event.
Subtracting 0.55 here would have inflated **all** of PoC2's latency by half a
second.

Since it is a claim about the internals of an SDK, `zero_verdict()` verifies
it with the call's own data: a diagnostic run confirms or refutes it.

## The opening turn

In an outbound call the agent speaks first, so its greeting has no user
message before it and LiveKit does not compute an `e2e_latency` for it. It is
measured by hand, as in PoC1: from `begin()` to the greeting's
`started_speaking_at`.

That is why the clock is `time.time()` and not `time.monotonic()`: we have to
subtract against marks that LiveKit stamps with `time.time()`. We lose the
immunity to clock adjustments that PoC1 does have; in exchange, the
subtractions are valid.
"""

from __future__ import annotations

import logging
import time
from statistics import median
from typing import Any, Final

from shared.metrics.latency import TurnSample

logger = logging.getLogger(__name__)

# LiveKit's Silero default. It is not changed (each framework runs detection
# the way its maker recommends), but it is needed as a reference for the zero
# verdict.
MIN_SILENCE_DURATION: Final[float] = 0.55

# `min_delay` of the endpointing defaults with a streaming turn detector.
MIN_ENDPOINTING_DELAY: Final[float] = 0.3

# Margin for the verdict: the comparison is between orders of magnitude
# (≈850 ms against ≈300 ms), not a fine measurement.
_VERDICT_MARGIN: Final[float] = 0.9

_ANCHOR_CORRECTED: Final[str] = "anchored_on_real_end_of_speech"
_ANCHOR_CONFIRMATION: Final[str] = "anchored_on_vad_confirmation"
_ANCHOR_NO_DATA: Final[str] = "no_data"


def _ms(valor: Any) -> float | None:
    """Seconds -> milliseconds, to one decimal. None if there is no data."""
    if not isinstance(valor, (int, float)) or isinstance(valor, bool):
        return None
    return round(float(valor) * 1000.0, 1)


class LatencyCollector:
    """Pairs LiveKit messages and produces the `TurnSample`s.

    It is fed from the `AgentSession`'s `conversation_item_added` event. It
    touches neither the audio nor the pipeline: it only reads what the session
    has already published.
    """

    def __init__(
        self,
        clock: Any = None,
        end_of_turn_capture: Any = None,
        marks: Any = None,
    ) -> None:
        self._clock = clock or time.time
        # Real zeros the harness published over the data channel, if any. In a
        # phone call nobody publishes on that topic and this stays empty. See
        # `app/marks.py`: it goes to `livekit_extra.harness`, never to
        # `latencies`.
        self._marks = marks
        # Why each turn closed: acoustic detector confidence or timeout of the
        # long wait. See `app/end_of_turn.py`. It goes to
        # `livekit_extra.per_turn`, NEVER to `latencies`, and
        # `end_of_turn_verdicts` stays empty: the contract is not touched.
        self._end_of_turn_capture = end_of_turn_capture
        self._t_start: float | None = None
        self._samples: list[TurnSample] = []
        self._indice: int = 0
        self._extras: list[dict[str, Any]] = []
        # Fragments of the carrier's current turn, IN ORDER. It is a list and
        # not a single dict because a line often arrives split into several
        # messages (a pause, "déjeme apuntar", self-correcting an amount
        # mid-sentence), and LiveKit publishes one `ChatMessage` for each, with
        # its own speech start and end marks.
        #
        # Keeping only the last one —what used to be done— throws away the
        # silence gaps in between, and those CANNOT be reconstructed later: it
        # is exactly the data PoC1 lacks and the reason its scenario 5 cannot
        # be re-read.
        self._pending_fragments: list[dict[str, Any]] = []
        # The agent has not spoken yet: its first message is the opening.
        self._pending_opening: bool = True
        # Set by the previous assistant message if it was interrupted: the
        # upcoming user turn is the one that talked over it.
        self._next_interrupted: bool = False
        # Turn detector version, at start and at end (it can change on its
        # own: the detector falls back from the remote model to the local one
        # if the cloud fails).
        self.initial_detector: str | None = None
        self.final_detector: str | None = None
        self.requested_detector: str | None = None
        # Provider errors (LLM/STT/TTS) that LiveKit reported during the call.
        # "Technical failures" is one of the contract's four metric families,
        # and until now it only existed in the console — which is not
        # evidence. See `record_failure`.
        self._failures: list[dict[str, Any]] = []
        # Those of the current turn, so the sample they tainted can be flagged.
        self._turn_failures: list[dict[str, Any]] = []

    # -- Lifecycle --

    def begin(self) -> None:
        """The call's zero. Called when the SIP participant answered."""
        self._t_start = self._clock()

    @property
    def samples(self) -> list[TurnSample]:
        return list(self._samples)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return list(self._failures)

    def to_json(self) -> list[dict[str, Any]]:
        """The samples in the shape `shared/metrics/` expects."""
        return [m.to_dict() for m in self._samples]

    # -- Data input --

    def record_message(self, role: str, metrics: Any, *, interrupted: bool) -> None:
        """Processes a `ChatMessage` from the conversation.

        `metrics` is the message's `MetricsReport` (a dict). It can be empty:
        for messages that do not come from a voice turn —for example the text
        the agent generates after a tool call— LiveKit fills nothing in, and
        those produce no sample.
        """
        data = dict(metrics or {})

        if role == "user":
            self._record_user(data)
        elif role == "assistant":
            self._record_assistant(data, interrupted=interrupted)

    def record_failure(
        self,
        tipo: str,
        *,
        recuperable: bool,
        provider: str | None = None,
        model_name: str | None = None,
        detail: str = "",
    ) -> None:
        """A provider error that LiveKit reported through the `error` event.

        **All** of them are kept, recoverable ones included. A retry that
        succeeded did not break the call, but it did cost the carrier seconds
        of silence and tainted the turn's latency: it is exactly the data that
        later tells whether a run is clean or whether its p95 was set by the
        LLM provider and not by the voice framework. Without this, several
        recoverable `llm_error`s in a row can leave tens of seconds of silence
        without a single trace in the JSON.
        """
        record: dict[str, Any] = {
            "t": (
                round(self._clock() - self._t_start, 3)
                if self._t_start is not None
                else None
            ),
            "tipo": tipo,
            "recuperable": bool(recuperable),
            "provider": provider,
            "model_name": model_name,
            "detail": detail,
        }
        self._failures.append(record)
        self._turn_failures.append(record)
        logger.warning(
            "provider failure: %s (recoverable=%s) %s", tipo, recuperable, detail
        )

    def _notes_per_failure(self) -> list[str]:
        """Note for the sample being closed, if there were failures in its turn.

        The sample is **neither discarded nor voided**: the number LiveKit
        measured is still true. What gets added is the warning that this turn
        does not describe the framework, so nobody cites it as PoC2 latency and
        so it can be taken out of the pool by hand.
        """
        if not self._turn_failures:
            return []
        kinds = sorted({str(f["tipo"]) for f in self._turn_failures})
        n = len(self._turn_failures)
        self._turn_failures = []
        return [
            f"turno con {n} fallo(s) del proveedor ({', '.join(kinds)}): la "
            "latencia de este turno no describe al framework de voz"
        ]

    def _record_user(self, metrics: dict[str, Any]) -> None:
        """Accumulates a fragment of the carrier's turn. ALL of them are kept.

        Including those that arrive **without an anchor**: LiveKit leaves
        `stopped_speaking_at` out when its anchor turned out inconsistent
        (`_compute_end_of_turn_metrics`). That fragment does not contribute to
        the turn's computation, but it must be recorded — if it were silently
        dropped, whoever reconstructs the intra-turn silence later would
        measure the gap between the fragments that survived and read it as a
        single pause, when in fact there was speech inside it.

        The dicts are stored **raw, as the SDK delivered them**: no flags
        computed here. The reader checks whether the key is present, and the
        JSON remains a faithful copy of what LiveKit gave.
        """
        if "stopped_speaking_at" not in metrics:
            logger.debug(
                "user fragment without stopped_speaking_at: recorded for "
                "diagnostics, but it does not contribute to the turn's computation"
            )
        self._pending_fragments.append(dict(metrics))

    def _record_assistant(
        self, metrics: dict[str, Any], *, interrupted: bool
    ) -> None:
        if self._pending_opening:
            self._pending_opening = False
            # The greeting only counts as the opening if it really came first.
            # If the carrier managed to say "¿bueno?" before it, the turn
            # already has a paired user and is measured as a normal turn — but
            # it inherits the opening flag, because it pays for the cold start.
            if not self._pending_fragments:
                self._close_opening(metrics)
                return
            self._close_turn(metrics, is_opening=True, interrupted=interrupted)
            return

        if not self._pending_fragments:
            # Second speech segment of the same agent turn (typically after a
            # tool call). `full_turn` runs up to the FIRST audio, so this one
            # does not open a sample.
            logger.debug("assistant message without a user turn: produces no sample")
            return

        self._close_turn(metrics, is_opening=False, interrupted=interrupted)

    # -- Building the sample --

    def _close_opening(self, assistant: dict[str, Any]) -> None:
        """The greeting: measured from the call's zero to the first audio."""
        full_turn = None
        speech_start = assistant.get("started_speaking_at")
        if self._t_start is not None and isinstance(speech_start, (int, float)):
            full_turn = _ms(float(speech_start) - self._t_start)
            if full_turn is not None and full_turn < 0:
                full_turn = None

        self._samples.append(
            TurnSample(
                indice=self._indice,
                full_turn_ms=full_turn,
                llm_ttft_ms=_ms(assistant.get("llm_node_ttft")),
                tts_ttfb_ms=_ms(assistant.get("tts_node_ttfb")),
                interrupted=False,
                is_opening=True,
                # LiveKit does not expose the audio turn detector's probability.
                end_of_turn_verdicts=(),
                notes=(
                    "apertura medida desde que la llamada fue contestada: el "
                    "saludo no tiene turno de usuario antes",
                    *self._notes_per_failure(),
                ),
            )
        )
        self._extras.append(
            {"indice": self._indice, "is_opening": True, "assistant": assistant}
        )
        self._indice += 1

    def _close_turn(
        self, assistant: dict[str, Any], *, is_opening: bool, interrupted: bool
    ) -> None:
        fragments = self._pending_fragments
        self._pending_fragments = []

        # The sample's stages hang off the LAST anchored fragment: its end of
        # speech is the turn's zero, the same instant from which LiveKit
        # measures `e2e_latency`. Taking the first one would misalign the
        # per-stage breakdown relative to `full_turn_ms`, and with it the
        # comparison with PoC1. The accumulation is for `per_turn`, it does NOT
        # change what is measured.
        anchored = [f for f in fragments if "stopped_speaking_at" in f]
        user = anchored[-1] if anchored else {}

        # The contract's `interrupted` = the carrier talked over the agent.
        # LiveKit marks `interrupted` on the AGENT message that got cut off, so
        # the flag belongs to the user turn that comes after — which is
        # exactly PoC1's criterion.
        mark = self._next_interrupted
        self._next_interrupted = bool(interrupted)

        notes: list[str] = []
        full_turn = _ms(assistant.get("e2e_latency"))
        if full_turn is None:
            notes.append(
                "sin e2e_latency: LiveKit no pudo anclar el turno (ver "
                "livekit_extra)"
            )
        notes.extend(self._notes_per_failure())

        self._samples.append(
            TurnSample(
                indice=self._indice,
                full_turn_ms=full_turn,
                end_of_turn_ms=_ms(user.get("end_of_turn_delay")),
                stt_ms=_ms(user.get("transcription_delay")),
                llm_ttft_ms=_ms(assistant.get("llm_node_ttft")),
                tts_ttfb_ms=_ms(assistant.get("tts_node_ttfb")),
                interrupted=mark,
                is_opening=is_opening,
                end_of_turn_verdicts=(),
                notes=tuple(notes),
            )
        )
        self._extras.append(
            {
                "indice": self._indice,
                "is_opening": is_opening,
                # `user` stays as it was —the last anchored fragment— so as not
                # to break anyone who already reads this block.
                "user": user,
                "user_fragments": fragments,
                "assistant": assistant,
                "end_of_turn": self._take_end_of_turn(),
            }
        )
        self._indice += 1

    def _take_end_of_turn(self) -> dict[str, Any] | None:
        """Why this turn closed, if the capture is installed.

        It is what separates the two modes of `end_of_turn_ms` (~577 ms against
        ~2,500 ms): without this, a p50 mixes "the detector was sure" with "the
        detector was not convinced and the wait ran out", which are different
        phenomena.
        """
        if self._end_of_turn_capture is None:
            return None
        from poc2_livekit.app.end_of_turn import summarize

        return summarize(self._end_of_turn_capture.take())

    def close_pending_ones(self) -> None:
        """Dumps the fragments that were left without an agent response.

        It happens when the call ends while the carrier is still talking.
        There is no sample to close —the agent never answered, so there is
        nothing to measure— but the silences WITHIN that last turn did exist,
        and if they are not dumped here they go away with the process.
        """
        if not self._pending_fragments:
            return
        self._extras.append(
            {
                # No `indice`: no `TurnSample` corresponds to it.
                "indice": None,
                "no_agent_response": True,
                "user_fragments": self._pending_fragments,
            }
        )
        self._pending_fragments = []

    # -- Diagnostics --

    def zero_verdict(self) -> dict[str, Any]:
        """Does LiveKit anchor on the real end of speech or on the VAD confirmation?

        It is decided with the call's own data, without taking anyone's word
        for it.

        The discriminant is `end_of_turn_ms` (= `end_of_turn_delay`), which
        measures from the zero to the moment the detector closed the turn:

        - If the zero is at the **real end of speech**, that delta must include
          the VAD's silence wait (0.55 s) plus the endpointing `min_delay`
          (0.3 s): on the order of 850 ms or more.
        - If the zero were at the **VAD confirmation**, the silence wait would
          already have elapsed and the delta would hover around just 300 ms.

        They are different orders of magnitude, so one call is enough to
        decide. If it comes out `anchored_on_vad_confirmation`,
        `MIN_SILENCE_DURATION` has to be subtracted from every mark and
        everything recomputed — and that would be a half-second correction
        over the whole comparison.
        """
        values = [
            m.end_of_turn_ms
            for m in self._samples
            if m.end_of_turn_ms is not None and not m.is_opening
        ]
        umbral_ms = MIN_SILENCE_DURATION * 1000.0 * _VERDICT_MARGIN

        if not values:
            return {
                "verdict": _ANCHOR_NO_DATA,
                "n_turns": 0,
                "umbral_ms": round(umbral_ms, 1),
                "detail": (
                    "no hubo turnos de régimen con end_of_turn_ms: sin ellos no "
                    "se puede decidir dónde ancla LiveKit"
                ),
            }

        median_value = float(median(values))
        corrected_value = median_value >= umbral_ms
        return {
            "verdict": _ANCHOR_CORRECTED if corrected_value else _ANCHOR_CONFIRMATION,
            "n_turns": len(values),
            "end_of_turn_median_ms": round(median_value, 1),
            "umbral_ms": round(umbral_ms, 1),
            "min_silence_duration_s": MIN_SILENCE_DURATION,
            "min_endpointing_delay_s": MIN_ENDPOINTING_DELAY,
            "detail": (
                "la mediana supera la espera de silencio del VAD, así que el cero "
                "ya está en el fin de habla real: NO hay que restar nada, y las "
                "latencias son comparables con PoC1 tal cual"
                if corrected_value
                else "la mediana NO alcanza la espera de silencio del VAD: el cero "
                "parece estar en la confirmación, hay que restar "
                "MIN_SILENCE_DURATION y recalcular la comparativa"
            ),
        }

    def detector_degradation(self) -> dict[str, Any] | None:
        """The detector fell back from remote to local mid-call, or started degraded.

        `_fall_back_to_local` replaces the remote transport with the local one
        when the inference service fails, and it does so **hot**: the call
        would start measuring one architecture and end measuring another.

        Such a call cannot be pooled with the others in the percentile pool
        —it would mix two turn-taking architectures within the same sample—
        but it is not thrown away either: it is reported separately, because
        *the fallback happening* is itself a LiveKit reliability data point.

        Returns `None` if there was no degradation.
        """
        requested = self.requested_detector
        initial = self.initial_detector
        final = self.final_detector
        expected = (
            "turn-detector-v1" if requested == "v1" else "turn-detector-v1-mini"
        )

        if initial and final and initial != final:
            return {
                "motivo": "el turn detector cambió de versión durante la llamada",
                "detail": (
                    f"arrancó como {initial} y terminó como {final}: la llamada "
                    "mezcla dos arquitecturas de turn-taking"
                ),
                "initial_detector": initial,
                "final_detector": final,
            }

        if requested and initial and initial != expected:
            return {
                "motivo": "el turn detector arrancó en una versión distinta de la pedida",
                "detail": (
                    f"se pidió {requested} ({expected}) y arrancó como {initial}"
                ),
                "initial_detector": initial,
                "final_detector": final,
            }

        return None

    def extra_block(self) -> dict[str, Any]:
        """Everything extra that LiveKit provides, for diagnostics.

        It goes in the JSON's `livekit_extra` key, **never** inside
        `latencies`: `TurnSample.from_dict` ignores keys it does not know, so
        putting them there would silently drop them.
        """
        return {
            "turn_detector": {
                "requested": self.requested_detector,
                "resolved_at_start": self.initial_detector,
                "resolved_at_end": self.final_detector,
                "cayo_a_local": (
                    self.initial_detector != self.final_detector
                    if self.initial_detector and self.final_detector
                    else None
                ),
                "note": (
                    "turn-detector-v1 corre REMOTO en LiveKit Inference; "
                    "turn-detector-v1-mini corre local en CPU. PoC2 fija v1-mini "
                    "porque corre entero en local, así que queda detector local "
                    "contra el Smart Turn v3 en proceso de PoC1 — sin un salto "
                    "de red que el resto de la arquitectura no paga. Si aquí "
                    "sale otra versión que la solicitada, la llamada no entra a "
                    "la poza de percentiles."
                ),
            },
            "zero_verdict": self.zero_verdict(),
            "end_of_turn": self.end_of_turn_summary(),
            **self._harness_block(),
            "per_turn": list(self._extras),
        }

    def _livekit_zeros(self) -> list[Any]:
        """The speech start and end that LiveKit estimated, turn by turn.

        `start` comes from the **first** anchored fragment and `end` from the
        **last**: a line can arrive split into several `ChatMessage`s, and
        taking the same fragment for both ends would measure the last piece
        instead of the whole turn.
        """
        from poc2_livekit.app.marks import LiveKitZero

        zeros: list[Any] = []
        for extra in self._extras:
            indice = extra.get("indice")
            if indice is None or extra.get("is_opening"):
                # The opening has no carrier turn before it: there is no
                # harness zero to compare it with.
                continue
            anchored = [
                f
                for f in (extra.get("user_fragments") or [])
                if isinstance(f, dict) and "stopped_speaking_at" in f
            ]
            if not anchored:
                continue
            start = anchored[0].get("started_speaking_at")
            end = anchored[-1].get("stopped_speaking_at")
            zeros.append(
                LiveKitZero(
                    indice=int(indice),
                    start=float(start) if isinstance(start, (int, float)) else None,
                    end=float(end) if isinstance(end, (int, float)) else None,
                )
            )
        return zeros

    def _harness_block(self) -> dict[str, Any]:
        """The comparison against the real zero, only if the harness sent marks.

        The key is omitted when none arrived: a real phone call has no
        harness, and publishing an empty block in the campaign's 15 would be
        noise that invites reading it as a failed capture.
        """
        if self._marks is None or not getattr(self._marks, "received", 0):
            return {}
        return {"harness": self._marks.block(self._livekit_zeros())}

    def end_of_turn_summary(self) -> dict[str, Any]:
        """How many turns closed by confidence and how many by timeout.

        It is the aggregate that makes the bimodality of `end_of_turn_ms`
        readable without having to open `per_turn` turn by turn.
        """
        from poc2_livekit.app.end_of_turn import UNKNOWN

        if self._end_of_turn_capture is None:
            return {"captured": False, "note": "la captura no está instalada"}

        count: dict[str, int] = {}
        for extra in self._extras:
            detail = extra.get("end_of_turn")
            if not isinstance(detail, dict):
                continue
            motivo = str(detail.get("motivo", UNKNOWN))
            count[motivo] = count.get(motivo, 0) + 1

        return {
            "captured": True,
            "by_reason": count,
            "predictions_seen": int(getattr(self._end_of_turn_capture, "total", 0)),
            "note": (
                "confianza = el detector acústico creyó que el turno terminó y "
                "se usó la espera corta; vencimiento = no se convenció y se "
                "agotó la espera larga, que es de dónde salen los ~2,500 ms. "
                "Diagnóstico de livekit_extra: NO entra en `latencias` ni en "
                "`end_of_turn_verdicts`, que sigue vacío por contrato."
            ),
        }
