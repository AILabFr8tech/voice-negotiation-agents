"""The real zero of the turn in PoC3: the same packets, out of band.

The packet —the two zeros, their threshold, the `host` and the format version— is the
one from `shared/carrier_simulator/cero.py`, identical to the one PoC1 writes to disk and
the one PoC2 publishes over LiveKit's data channel. The only PoC3-specific parts are
the route it leaves by and what can be done with it on the other side.

## Why to disk, like PoC1

PoC3 runs over Twilio Media Streams, which carries audio and four events
(`connected`, `start`, `media`, `stop`). The bridge discards any `event`
it does not know (`telefonia._from_twilio_to_live`), and the only real side channel
would be `dtmf`, which would mean injecting a tone into the very conversation being
measured. There is no data channel. So the mark is written to a file and
the cross-check against the call JSON is done afterwards, by wall clock.

It is the pattern of `poc1_pipecat/app/marks.py`. **What follows is where it departs
and why**, which is what must be defensible in the report.

## Where it departs from PoC1

**1. The `call_id` is received, not inferred from the file name.** In PoC1 the
harness fabricates its own synthetic `callSid` and the file name is the
join key. In PoC3 the key has to be the Twilio SID
(`negociacion.call_id`) —the same one used to name `censo_<sid>.jsonl`,
`llamada_<sid>.json` and the audit recording— or the zero does not join against the
census, which is the only artifact holding the alpha's timeline. It is
passed explicitly to the constructor and the file name is derived from it: a single
source, not two that can diverge.

**2. This module does NOT include the pipeline side.** PoC1 has right here
`SondaDelCero` and `SondaDeSalidaDeAudio`, two `FrameProcessor`s that timestamp on the
wall clock the zero estimated by pipecat's VAD. PoC3 has no equivalent:
there is no end-of-turn event that can be timestamped. The model exposes no
`turn_detection`, no VAD, no threshold, no silence duration (see `CLAUDE.md` and
`telefonia.build_initial_session`). The zero PoC3 estimates is
`ColectorDeLatencia.inbound_speech()`, i.e. **local energy**, not a decision
of the framework.

Consequently the analogue of `zero_verdict()` is not here either: PoC1's
decides against the VAD's `stop_secs`, which is a knob with a known value.
PoC3 has no such knob —neither to set it nor to read it—, so its criterion
cannot be copied from PoC1's.

**3. The run state is published, not just counted.** PoC1 leaves its
failures in `emisor.fallas` and its caller copies them to the report. Here the
condition "this call is usable for the zero analysis" is computed
(`apto_para_analisis`) and written to a sibling file, because a call without
zeros has to be **explicitly excluded** and not simply shrink
the n without anyone noticing.

## Two clocks: why the acknowledgement is anchored on `time.time()` once

The mark's two zeros come from the harness, which timestamps them on the
`time.time()` axis: `Observado.inicio` is `time.time()` and the player advances from
there with `perf_counter` so as not to suffer the 15.625 ms resolution of
`GetSystemTimeAsFileTime` (§1 of the contract). This emitter stamps with **the same
function**, `time.time`, which is the default of `EmisorDeCeros`: the envelope stays
internally consistent.

The bridge's acknowledgement has to land on that same axis. `time.monotonic()` does not
work: on Windows it is `GetTickCount64()`, milliseconds since system
boot and not since the epoch, so subtracting an epoch zero from an uptime
acknowledgement does not give a latency, it gives the machine's age. The two processes
sharing a machine is **not enough** to fix it: it is a problem of origin, not
of synchronization.

**And there is a second reason, independent of the first: resolution.**
`time.time()` follows the global Windows timer resolution, which any
process can raise, so its real grain is around a millisecond and unstable.
`time.monotonic()` does not follow it: it advances in 15 ms steps, on the order of the
20 ms window on which `bridge.has_speech` decides. The **`time.time()` anchor
once + `perf_counter` from there** (`reloj.py`) fixes **origin and resolution
at the same time** —`perf_counter` is `QueryPerformanceCounter`, 100 ns— and it is the
combination already used by `ReproductorDeMediaStreams` (§1 of the contract) and
PoC1's `SondaDelCero`. The resolution measurements are in
`docs/campaign_poc3_estado_20260923.md` and in the appendix of
`docs/comparativo_results.md`.

**The timer is not pinned with `timeBeginPeriod` to achieve this.** It would change the
granularity of `asyncio` and with it the run conditions relative to the
PoC1 and PoC2 campaigns, which ran without touching it. Resolution is **measured** and
declared; it is not manipulated.

The August artifacts, which predate the anchor, do have the acknowledgement on another axis and
their bias cannot be computed: see `OLD_DIFFERENT_AXES_REASON`.

## Why JSON Lines and one line per turn

Same as PoC1 and for the same reason: it is written live, one envelope per line,
as soon as the clip finishes playing. If the run crashes mid-call, the
turns that did happen are already on disk; a single JSON built at the end would
be lost entirely.

No `fsync` is done. A `flush` leaves the line in the system buffer, which
survives the harness process dying —the realistic scenario— and costs
tens of microseconds. `fsync` would cost milliseconds of blocking the event
loop, and that would be noticeable to the player, which publishes frames every 20 ms
from another task.

## Why the readers are duplicated and not imported from PoC1

`poc1_pipecat/app/marks.py` imports `pipecat` on its first line, and PoC3 does not
install pipecat on purpose (see `pyproject.toml`): importing it from here
would bring another architecture's framework into this one's environment, which is
exactly what the comparison cannot afford. What is shared is what
decides the format —`codificar_*`, `decodificar`, `MarcaDeCero`— and it lives in
`shared/carrier_simulator/cero.py`, which is the only thing that has to match.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from shared.carrier_simulator.cero import (
    ZERO_PROTOCOL,
    SUMMARY_KIND,
    PROTOCOL_VERSION,
    ZeroEmitter,
    ZeroMark,
    Envelope,
    decode,
)

logger = logging.getLogger(__name__)

# Where the harness output lives. It is **not** `grabaciones/`, on purpose: that is
# where the censuses and the campaign JSONs live, and `shared.metrics.cli calcular`
# reads them all. Same criterion as `poc1_pipecat/diagnostico/arnes`.
DIAGNOSTICS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "harness"
)

# The name carries the Twilio SID, which is the same key used to name
# `grabaciones/censo_<sid>.jsonl` and `grabaciones/llamada_<sid>.json`. That way the
# cross-check is a `join` by file name and not a heuristic.
PREFIX: Final[str] = "zeros"
EXTENSION: Final[str] = ".jsonl"
STATE_EXTENSION: Final[str] = ".estado.json"

# The clock axis of each end, by name. They are published in the block so that
# whoever reads the JSON can verify the incompatibility instead of taking it on faith.
HARNESS_CLOCK: Final[str] = "time.time"
# The bridge anchors `time.time()` only once when the call opens and advances with
# `perf_counter`, so the acknowledgement lands on the epoch axis just like the harness
# marks.
ACKNOWLEDGEMENT_CLOCK: Final[str] = "time.time anclado una vez + perf_counter"

# --------------------------------------------------------------------------
# Step 3 requirement: the clock anchor at the bridge (covered by `reloj.py`)
# --------------------------------------------------------------------------
#
# It is declared under its own name and published in every block. It has two halves and
# both are necessary — fixing only one leaves the bias unusable because of the
# other:
#
# 1. ORIGIN. The harness timestamps the two zeros on `time.time` (epoch); an acknowledgement on
#    `time.monotonic` (uptime) would be on another axis, and subtracting across different
#    axes is not a latency.
# 2. RESOLUTION. `time.monotonic` advances in 15 ms steps (GetTickCount64),
#    on the order of the 20 ms window of `bridge.hay_voz`, and does not improve even if
#    the system timer is refined.
#
# Both are solved by the same mechanism, the one in `ReproductorDeMediaStreams`
# and `SondaDelCero`: anchor `time.time()` ONCE when the call starts and
# advance with `perf_counter` (100 ns grain), so that the acknowledgement lands on the
# harness axis and with fine resolution.
STEP_3_REQUIREMENT: Final[str] = "clock_anchor_at_bridge"

# The third piece of the requirement, which belongs not to the mechanism but to its audit:
# `time.time()` is adjustable and `perf_counter` is not. If NTP corrects the time
# mid-call, the anchor goes stale and everything timestamped afterwards is shifted with
# nothing to warn about it. That is why the drift between the two is recorded per call
# —`(time.time() - ancla_pared) - (perf_counter() - ancla_perf)`— and published.
# A jump there is a clock adjustment, and a call with an adjustment is declared non-
# interpretable instead of being corrected: correcting without knowing at what instant
# the jump happened would be distributing the error by eye.
DRIFT_TO_RECORD: Final[str] = "drift_time_time_vs_perf_counter_ms"

# And the threshold, whose **formula** was fixed before running the campaign. A criterion
# chosen after seeing the data stops being a criterion — but the number cannot
# be fixed in advance, because it depends on the clock's effective resolution,
# which varies with what is running on the machine (see the module header).
#
#     umbral_ms = max(DERIVA_PISO_MS, FACTOR_SOBRE_LA_RESOLUCION * R)
#
# with `R` = effective resolution of `time.time()` measured **within the call**.
#
# - **The factor 4.** The drift is a difference of two readings, each
#   quantized to ±1 tick, so with zero drift the apparent error reaches 2R.
#   The 4 leaves a factor-of-2 margin over that noise.
# - **The 2.0 ms floor.** With a very fine timer the threshold would become a
#   hair trigger and fire on scheduling jitter. It sits above the crystal's
#   natural drift, measured the same day: 0.23 ms per 1000 s, i.e.
#   0.046 ms in PoC3's longest call (201 s).
# - **There is no ceiling, on purpose.** A ceiling would break the argument: with a coarse R
#   the `clamp` would return less than 2R and the threshold would sit below the noise
#   it itself describes. If R comes out so coarse that the clock cannot resolve what
#   is needed, that **is declared in the call with R published** — the threshold is not
#   tightened to hide it.
#
# The statistic is the **maximum absolute value** observed during the call, not
# the final one: a jump that is corrected later leaves the marks timestamped while it
# lasted just as unusable. And that is why step 3 samples the drift
# periodically: with only two readings (on open and on hang-up) the criterion bounds
# a monotonic drift and **does not see** a jump that goes and comes back.
DRIFT_FLOOR_MS: Final[float] = 2.0
FACTOR_OVER_RESOLUTION: Final[int] = 4


def drift_threshold_ms(effective_resolution_ms: float) -> float:
    """This call's threshold, derived from the resolution measured in it.

    `effective_resolution_ms` is the smallest non-zero delta between two consecutive
    readings of `time.time()` in a tight loop — **not** what
    `time.get_clock_info` declares, which on Windows is the worst case and not what happens.
    """
    return max(DRIFT_FLOOR_MS, FACTOR_OVER_RESOLUTION * float(effective_resolution_ms))

# The reason the two axes could not be subtracted before the anchor. The
# old text is kept in `MOTIVO_VIEJO_DE_EJES_DISTINTOS` (kept in Spanish)
# because the August artifacts carry it written inside and it must be possible to
# recognize it when reading them: those ARE on different axes and their bias still
# cannot be computed.
OLD_DIFFERENT_AXES_REASON: Final[str] = (
    "el arnés fecha en `time.time()` (época) y el acuse del puente en "
    "`time.monotonic()` (uptime, GetTickCount64 en Windows). Son orígenes "
    "different_items: la resta no es una latencia. Falta fechar "
    "`LatencyCollector.inbound_speech/outbound_speech` también en `time.time()`, "
    "como hace `SondaDelCero` en PoC1"
)

COMMON_AXIS_REASON: Final[str] = (
    "the bridge anchors `time.time()` once when the call opens and advances with "
    "`perf_counter`, so the acknowledgement lands on the SAME epoch axis as the "
    "harness marks and with a 100 ns grain instead of 15 ms. Both halves "
    "of the requirement —origin and resolution— are covered by the same "
    "mechanism. The axis being common holds by construction; whether the call is "
    "interpretable depends on its drift, and that is declared by the `reloj` block of "
    "that call's JSON"
)

# The call that demonstrated it end to end. It is cited by `call_id` so that
# anyone can open the three files and redo the check, instead of
# having to trust this constant.
COMMON_AXIS_EVIDENCE: Final[dict[str, Any]] = {
    "call_id": "ARNES-call-20260922-132107",
    "when": "September 22, 2026",
    "what_was_checked": (
        "8 marks written and 8 declared, emitter summary closed and "
        "full coverage, with the bridge's `reloj` block at "
        "`interpretable: true`"
    ),
    "effective_resolution_ms": {"on_open": 0.3529, "on_hang_up": 0.504},
    "umbral_ms": 2.0161,
    "max_absolute_drift_ms": -1.4745,
    "drift_samples": 111,
}


def zeros_path(call_id: str, *, folder: Path | None = None) -> Path:
    """A call's `.jsonl`. `call_id` is the Twilio SID."""
    return (folder or DIAGNOSTICS_PATH) / f"{PREFIX}_{call_id}{EXTENSION}"


def state_path(call_id: str, *, folder: Path | None = None) -> Path:
    """The sibling file that says whether that call is usable for the analysis."""
    return (folder or DIAGNOSTICS_PATH) / f"{PREFIX}_{call_id}{STATE_EXTENSION}"


# --------------------------------------------------------------------------
# The emitter
# --------------------------------------------------------------------------


class DiskMarkEmitter(ZeroEmitter):
    """Writes the harness envelopes to a `.jsonl`, one per line.

    **It never propagates a write failure.** A diagnostics packet that did not
    go out is no reason to bring down a run that is already paying for a phone
    call. But it is not lost either: each failure is counted in `fallas`, and on
    close `ceros_<sid>.estado.json` is written with `apto_para_analisis`, so
    that the call is excluded **in writing** and does not silently lower the n.

    The event that defines the zero is **the end of the carrier clip**: the
    last frame left the harness (`clip_end`, a fact) and the last speech
    inside that clip (`measured_end_of_speech`, a measurement from the manifest). Both
    are computed by `shared/carrier_simulator/player.py`, which is what
    calls `enviar(...)` from `play_script`. So the interface of
    this module is exactly the one that player already knows how to fill, and the
    PoC3 player satisfies it by construction, with nothing to invent.
    """

    def __init__(
        self,
        *,
        call_id: str,
        publish_sample_rate: int,
        folder: Path | None = None,
        ruta: Path | str | None = None,
        clock: Any = None,
        host: str | None = None,
    ) -> None:
        # `reloj` defaults to `time.time`, the one in `EmisorDeCeros`: the same axis
        # on which the harness computed the two zeros this envelope carries.
        # Changing it would decouple the stamp from the content.
        super().__init__(publish_sample_rate=publish_sample_rate, clock=clock, host=host)
        if not call_id:
            # Without a SID no cross-check against the census is possible, and a file with
            # an invented name is worse than none: it looks like evidence.
            raise ValueError("the zero emitter needs the call SID")
        self.call_id = call_id
        self.ruta = Path(ruta) if ruta is not None else zeros_path(
            call_id, folder=folder
        )
        self.state_path = state_path(
            call_id, folder=self.ruta.parent if folder is None else folder
        )
        self.written_state: Path | None = None
        self._file: Any = None

    # -- writing ------------------------------------------------------------

    def _begin(self) -> Any:
        if self._file is None:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.ruta.open("a", encoding="utf-8", newline="\n")
        return self._file

    async def _deliver(self, carga: bytes) -> None:
        file = self._begin()
        # One line per envelope, no line breaks inside: `codificar_*` already produces compact
        # single-line JSON, and that way the file can be read in pieces
        # even if the run crashed halfway.
        file.write(carga.decode("utf-8") + "\n")
        file.flush()

    async def shut(
        self, *, turns_without_audio: Sequence[str] = (), sleep: Any = None
    ) -> bool:
        """Send the summary, release the descriptor and write the state.

        The state is **always** written, even if everything else failed: that is
        precisely the case where it is needed.
        """
        try:
            return await super().shut(
                turns_without_audio=turns_without_audio, sleep=sleep
            )
        finally:
            self.close_file()
            self.write_state()

    def close_file(self) -> None:
        """Release the descriptor. Idempotent: closing twice is not an error."""
        if self._file is not None:
            try:
                self._file.close()
            except OSError as error:
                self.faults.append(f"while closing the file: {error!r}")
                logger.warning("could not close %s: %r", self.ruta, error)
            finally:
                self._file = None

    # -- the run state ------------------------------------------------------

    def estado(self) -> dict[str, Any]:
        """Whether this call is usable for the zero analysis, and why not.

        It is computed by re-reading the file, not by counting in memory: what matters
        is **what remained**, which is what someone is going to compute on.
        """
        on_disk = marks_de(self.ruta)
        reasons: list[str] = []
        if not self.sent:
            reasons.append("the harness published no marks")
        if self.faults:
            reasons.append(f"{len(self.faults)} write(s) failed")
        if self.sent and not self.summary_sent:
            reasons.append("the summary did not go out: coverage cannot be verified")
        if len(on_disk) != self.sent:
            reasons.append(
                f"{len(on_disk)} marks left on disk out of {self.sent} "
                f"published"
            )
        return {
            "protocolo": ZERO_PROTOCOL,
            "version": PROTOCOL_VERSION,
            "call_id": self.call_id,
            "file": str(self.ruta),
            "marks_published_ones": self.sent,
            "marks_on_disk": len(on_disk),
            "summary_sent": self.summary_sent,
            "faults": list(self.faults),
            "eligible_for_analysis": not reasons,
            "why_not": reasons,
            "what_it_means": (
                "`apto_para_analisis: false` means this call does NOT "
                "enter the zero analysis and that its absence is known. Without "
                "this file, a run without marks would silently lower the n"
            ),
            "clock": clock_block(host=self._host),
        }

    def write_state(self) -> Path | None:
        """Dump `estado()` to the sibling file. Never raises.

        If this also fails —which is the realistic case when the disk is the
        problem— the log remains and `estado()` remains in memory, which is what the
        bridge process writes into the call JSON. Two independent
        carriers, so that a disk failure does not erase the record of the
        disk failure.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self.estado(), ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError as error:
            self.faults.append(f"while writing the state: {error!r}")
            logger.error(
                "could not write the zeros state to %s: %r — this "
                "call is left with no on-disk record that it is missing zeros",
                self.state_path,
                error,
            )
            return None
        self.written_state = self.state_path
        return self.state_path


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_zeros(ruta: Path | str) -> list[Envelope]:
    """The envelopes of a zeros file, in order.

    Tolerant of broken lines and **explicit** about them: a run that
    crashed mid-write leaves the last line incomplete, and that cannot
    prevent reading the earlier ones. What could not be read is logged,
    not silently discarded.
    """
    envelopes: list[Envelope] = []
    for numero, linea in enumerate(_lines(Path(ruta)), start=1):
        try:
            envelopes.append(decode(linea))
        except (ValueError, KeyError) as error:
            logger.warning("line %d of %s unreadable: %r", numero, ruta, error)
    return envelopes


def _lines(ruta: Path) -> Iterator[str]:
    if not ruta.is_file():
        return
    with ruta.open("r", encoding="utf-8") as file:
        for linea in file:
            if linea.strip():
                yield linea


def marks_de(ruta: Path | str) -> list[ZeroMark]:
    """The marks of a run, without duplicates and in sequence order.

    The closing summary repeats all marks on purpose —it is the second
    chance for one that got lost— so they must be deduplicated here or
    each turn would appear twice. The summary's copy wins when it exists: it is the
    one the harness confirmed at the end.
    """
    by_sequence: dict[int, ZeroMark] = {}
    for on_top in read_zeros(ruta):
        for mark in on_top.marks:
            if on_top.tipo == SUMMARY_KIND or mark.sequence not in by_sequence:
                by_sequence[mark.sequence] = mark
    return [by_sequence[k] for k in sorted(by_sequence)]


def coverage(ruta: Path | str) -> dict[str, Any]:
    """How many marks were expected and how many there are.

    Without the summary's declared total it is impossible to distinguish "all arrived"
    from "whatever arrived arrived", which is exactly the false reading
    this field exists to prevent.
    """
    envelopes = read_zeros(ruta)
    summaries = [s for s in envelopes if s.tipo == SUMMARY_KIND]
    marks = marks_de(ruta)
    declared = summaries[-1].declared_total if summaries else None
    return {
        "file": str(ruta),
        "marks": len(marks),
        "declared_total": declared,
        "had_summary": bool(summaries),
        "complete": declared is not None and declared == len(marks),
        "turns_without_audio": list(summaries[-1].turns_without_audio) if summaries else [],
    }


def clock_block(
    *,
    host: str | None = None,
    bridge_clock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The two axes, by name, and whether this call can be subtracted.

    It goes inside everything this module publishes. It is not documentation: it is the
    field that prevents anyone from reading a subtraction across different axes as if
    it were a latency.

    With the bridge anchor the axis is common **by construction**, so
    `mismo_eje` does not depend on the call. (Artifacts from before the anchor
    carry fixed `mismo_eje: false` and `sesgo_interpretable: false`, because their
    acknowledgement lived on `time.monotonic`.)

    What does depend on the call is `sesgo_interpretable`, and this module cannot
    know it on its own: it is declared by the `reloj` block the bridge writes into the
    JSON, which is the one that measured the drift. That is why it is received as
    `bridge_clock` and **without it the result is `None`**, which means "not
    checked" — never `true`, which would mean "checked and it came out fine".
    """
    interpretable = (
        bool(bridge_clock.get("interpretable"))
        if bridge_clock is not None
        else None
    )
    block: dict[str, Any] = {
        "harness_clock": HARNESS_CLOCK,
        "bridge_acknowledgement_clock": ACKNOWLEDGEMENT_CLOCK,
        "same_axis": True,
        "interpretable_bias": interpretable,
        "host_of_harness": host if host is not None else socket.gethostname(),
        "motivo": COMMON_AXIS_REASON,
        "requirement_met": STEP_3_REQUIREMENT,
        "evidence": COMMON_AXIS_EVIDENCE,
        "and_audit": DRIFT_TO_RECORD,
        "drift_threshold": (
            f"max({DRIFT_FLOOR_MS}, {FACTOR_OVER_RESOLUTION} x R), with R = "
            "effective resolution of time.time() MEASURED within the call"
        ),
        "drift_criterion": (
            "maximum absolute value during the call, not the final one. No ceiling: "
            "if R comes out so coarse that the clock cannot resolve what is needed, it is "
            "declared with R published instead of tightening the threshold. Formula fixed "
            "on 2026-09-22 BEFORE the campaign"
        ),
    }
    if interpretable is None:
        block["where_it_is_declared"] = (
            "in the `reloj` block of the call JSON, which is written by the "
            "bridge. Here it is `null` because this block is built by reading the "
            "harness marks and does not have that file in view"
        )
    else:
        block["max_absolute_drift_ms"] = (bridge_clock or {}).get(
            DRIFT_TO_RECORD, {}
        ).get("absolute_max_ms")
        block["umbral_ms"] = (bridge_clock or {}).get("umbral_ms")

    # The August artifacts carry the old reason inside and ARE on different
    # axes: it is named so that whoever reads them does not think this block
    # covers them retroactively.
    block["before_step_3"] = OLD_DIFFERENT_AXES_REASON

    return block


def zero_block(
    call_id: str,
    *,
    folder: Path | None = None,
    bridge_clock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the bridge process writes into the call JSON.

    It is built **by reading the disk**, not by receiving objects, because the caller
    is the other process: if the harness died without writing anything, this function
    still has to be able to say so. That is the case that matters — a call without
    zeros is marked `apto_para_analisis: false` with its reason, and does not
    vanish from the count without a trace.
    """
    ruta = zeros_path(call_id, folder=folder)
    ruta_estado = state_path(call_id, folder=folder)
    data = coverage(ruta)
    marks = marks_de(ruta)

    estado: dict[str, Any] | None = None
    try:
        if ruta_estado.is_file():
            read_result = json.loads(ruta_estado.read_text(encoding="utf-8"))
            estado = read_result if isinstance(read_result, dict) else None
    except (OSError, ValueError) as error:
        logger.warning("unreadable zeros state in %s: %r", ruta_estado, error)

    if estado is not None:
        eligible = bool(estado.get("eligible_for_analysis"))
        why = list(estado.get("why_not") or [])
    else:
        # Without a state file it cannot be claimed that the run was fine:
        # the absence of the file IS the reason, and it is stated in those words.
        eligible = False
        why = [
            "there is no emitter state file: either the harness did not run, or it died "
            "before closing"
        ]
        if not data["marks"]:
            why.append("and there are no marks on disk either")

    return {
        "protocolo": ZERO_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "call_id": call_id,
        "mark_path": "file on disk, out of band",
        "what_it_is": (
            "the two real zeros of each turn, measured by whoever played the "
            "clip. Harness diagnostics: it does not enter `latencias` and changes "
            "no sample"
        ),
        "eligible_for_analysis": eligible,
        "why_not": why,
        "cobertura": data,
        "sender_state": estado,
        "marks_detail": [m.a_dict() for m in marks],
        "clock": clock_block(
            host=next((m.host for m in marks if m.host), None),
            bridge_clock=bridge_clock,
        ),
        "how_it_is_subtracted": (
            "this block publishes the REAL zero. Since step 3 the bridge's "
            "acknowledgement lives on the same epoch axis (`eje_de_pared` of the call "
            "JSON), so the bias CAN be computed — but only for the "
            "calls whose `reloj` block comes out `interpretable`. The subtraction is not "
            "done here: this module publishes the input and the two-stage "
            "discriminant consumes it. See `reloj.sesgo_interpretable`"
        ),
    }


__all__ = [
    "DRIFT_TO_RECORD",
    "DRIFT_FLOOR_MS",
    "EXTENSION",
    "STATE_EXTENSION",
    "FACTOR_OVER_RESOLUTION",
    "COMMON_AXIS_EVIDENCE",
    "COMMON_AXIS_REASON",
    "OLD_DIFFERENT_AXES_REASON",
    "PREFIX",
    "ACKNOWLEDGEMENT_CLOCK",
    "HARNESS_CLOCK",
    "STEP_3_REQUIREMENT",
    "DIAGNOSTICS_PATH",
    "DiskMarkEmitter",
    "zero_block",
    "clock_block",
    "cobertura",
    "read_zeros",
    "marks_de",
    "zeros_path",
    "state_path",
    "drift_threshold_ms",
]
