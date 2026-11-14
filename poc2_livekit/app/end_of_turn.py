"""Why each turn ended: by detector confidence or by timeout.

## The finding that needs explaining

PoC2's `end_of_turn_ms` is **bimodal**: the vast majority of steady-state
turns fall at ~500-750 ms and about 10% go above 1 s, clustered at ~2,500 ms
(full breakdown in `docs/comparativo_results.md`).

These are not two populations of calls: they are **two different paths
through the same code**. `audio_recognition.py::_bounce_eou_task` starts with
the short wait and only lengthens it when the acoustic detector says the turn
probably did NOT end:

    endpointing_delay = self._endpointing.min_delay          # 0.3 s
    ...
    if (end_of_turn_probability is not None
        and unlikely_threshold is not None
        and end_of_turn_probability < unlikely_threshold):
        endpointing_delay = self._endpointing.max_delay      # the slow mode

With a `min_silence_duration` of 0.55 s that gives the fast mode; when the
probability does not reach the threshold, the turn waits the full `max_delay`.

In other words, **the slow mode is not measurement noise: it is the detector
saying "this person has not finished speaking" and getting it wrong.** Without
recording which path each turn took, a p50 mixes two different phenomena and
one cannot say which dominates. For the harness it is also what decides
whether a clip is usable: one trimmed abruptly that always falls into the slow
mode inflates all of its latencies.

## Why it is captured from the log and not from an API

LiveKit **does not expose the probability to code**: there is no event, no
callback, no public metric. `EOTInferenceMetrics` only carries durations and is
emitted by the remote path (LiveKit Inference), which PoC2 does not use because
it runs `v1-mini` locally — the same asymmetry that leaves
`end_of_turn_verdicts` empty in PoC2 (`docs/METRICS_CONTRACT.md`).

The only thing it does publish is a `logger.debug("eot prediction", extra={…})`
with the probability, the threshold and the chosen wait. Capturing it is
**observing**, not patching: the SDK is not touched, nor the decision, nor any
frozen constant. This goes to `livekit_extra`, not to `end_of_turn_verdicts`,
so the 15 calls already run remain comparable.

## The cost of turning on DEBUG

It requires setting the `livekit.agents` logger to DEBUG. The `voice` package
has 35 calls to `logger.debug` and **none on the per-audio-frame path**: they
are a handful per turn, not thousands per second. The filter discards every
DEBUG record other than the one of interest, so the worker's output stays
identical.

It can be turned off with `POC2_CAPTURE_END_OF_TURN=0`, in case the exact
environment of the 15 calls has to be reproduced.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Final

# The logger where LiveKit emits the prediction, and the exact message.
LIVEKIT_LOGGER: Final[str] = "livekit.agents"
PREDICTION_MESSAGE: Final[str] = "eot prediction"

SHUTOFF_VARIABLE: Final[str] = "POC2_CAPTURE_END_OF_TURN"

# The two paths of `_bounce_eou_task`.
BY_CONFIDENCE: Final[str] = "confianza"
BY_TIMEOUT: Final[str] = "vencimiento"
UNKNOWN: Final[str] = "desconocido"


@dataclass(frozen=True)
class Prediction:
    """One run of the acoustic detector, exactly as LiveKit published it.

    The detector runs **several times per turn** —it re-fires on every
    transcript update—, so the one that counts is the last: it is the one that
    set the wait with which the turn actually closed.
    """

    probability: float | None
    umbral: float | None
    wait_s: float | None
    trigger: str | None = None
    from_cache: bool | None = None

    @property
    def motivo(self) -> str:
        """`confianza` (confidence) if the detector believed the turn ended;
        otherwise `vencimiento` (timeout): the long wait ran out without it
        being convinced."""
        if self.probability is None or self.umbral is None:
            return UNKNOWN
        return BY_TIMEOUT if self.probability < self.umbral else BY_CONFIDENCE

    def as_json(self) -> dict[str, Any]:
        return {
            "motivo": self.motivo,
            "probability": self.probability,
            "umbral": self.umbral,
            "wait_s": self.wait_s,
            "trigger": self.trigger,
            "from_cache": self.from_cache,
        }


def _float(valor: Any) -> float | None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    return round(float(valor), 4)


@dataclass
class EndOfTurnCapture(logging.Filter):
    """Filter that keeps the detector's predictions and lets everything else through.

    It is a `logging.Filter` and not a `Handler` on purpose: the filter runs on
    the logger **before** the handlers, so it can discard the DEBUG records
    this module forces to be generated and leave the worker's output exactly as
    it was.
    """

    pending_items: list[Prediction] = field(default_factory=list)
    total: int = 0

    def __post_init__(self) -> None:
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() == PREDICTION_MESSAGE:
            self.pending_items.append(
                Prediction(
                    probability=_float(getattr(record, "probability", None)),
                    umbral=_float(getattr(record, "unlikely_threshold", None)),
                    wait_s=_float(getattr(record, "endpointing_delay", None)),
                    trigger=getattr(record, "trigger", None),
                    from_cache=getattr(record, "from_cache", None),
                )
            )
            self.total += 1
            return False

        # Everything else goes on its way, except the DEBUG that only exists
        # because this module raised the logger's level.
        return record.levelno > logging.DEBUG

    def take(self) -> list[Prediction]:
        """Empties and returns what has accumulated since the last time."""
        cumulatives, self.pending_items = self.pending_items, []
        return cumulatives

    def restart(self) -> None:
        """Starts from zero for a new call."""
        self.pending_items = []
        self.total = 0


def is_enabled() -> bool:
    return os.getenv(SHUTOFF_VARIABLE, "1").strip().lower() not in {"0", "false", "no"}


# The filter lives on LiveKit's logger, which is process-global. The worker
# can serve several jobs with the same process, and reinstalling would leave
# two filters counting the same prediction twice.
_installed: EndOfTurnCapture | None = None


def install(logger_name: str = LIVEKIT_LOGGER) -> EndOfTurnCapture | None:
    """Hooks up the capture, only once per process.

    Returns `None` if it is turned off via the environment. If it was already
    installed, returns the same one reset to zero: each call counts its own
    turns.
    """
    global _installed

    if not is_enabled():
        return None

    if _installed is not None:
        _installed.restart()
        return _installed

    _installed = EndOfTurnCapture()
    target = logging.getLogger(logger_name)
    # LiveKit's `logger.debug` does not even build the record if the logger is
    # not at DEBUG, so raising the level is mandatory. The filter makes sure
    # that does not change what is printed on screen.
    target.setLevel(logging.DEBUG)
    target.addFilter(_installed)
    return _installed


def summarize(predictions: list[Prediction]) -> dict[str, Any]:
    """What is stored per turn: the decisive one, and the others to audit it."""
    if not predictions:
        return {
            "motivo": UNKNOWN,
            "note": (
                "sin predicción del detector para este turno: o no corrió, o la "
                "captura está apagada (ver POC2_CAPTURAR_FIN_DE_TURNO)"
            ),
        }

    # The last one is the one that set the wait with which the turn closed; the
    # earlier ones are re-runs over partial transcripts.
    decisive = predictions[-1]
    summary = decisive.as_json()
    summary["predictions"] = len(predictions)
    if len(predictions) > 1:
        summary["priors"] = [p.as_json() for p in predictions[:-1]]
    return summary
