"""The turn's real zero: what gets published, and how any harness builds it.

The harness is the only one that knows **for certain** when the carrier's audio
went out: it pushed the frames itself. Each PoC's worker does not know it; it
*reconstructs* it from its VAD, which means that **the zero on which all the
reported latency hangs is an estimate**. This module defines the packet with which
the harness publishes the real instant, so that estimate can be measured head-on
instead of argued about by magnitude.

## The TWO zeros, and why one is not enough

- **`clip_end`**: the last frame went out. It is a **physical fact**; it does not
  depend on any criterion.
- **`measured_end_of_speech`**: where speech stopped *inside* the clip. It is a
  **measurement** and depends on the threshold, so it travels with its `drop_db` and
  cannot be read without it. It is measured when the clip is prepared
  (`shared/audio/level.py::end_of_speech`) and lives in the manifest: it is a property
  of the clip, and the 3 PoCs play the same clip.

        clip_bias_ms   = worker_zero - clip_end
        speech_bias_ms = worker_zero - measured_end_of_speech

The second one is the one that means something. Publishing only the file one was the
first design and it produced impossible negative biases: a clip drags up to 1.3 s
of breathing and room tail after the last syllable, and the zero was
timestamped at the end of the *file*.

## Why the packet lives in `shared/` and the transport does not

The format has to be **the same in the 3 PoCs** or the biases cannot be
read together: a key with another name or a different version would produce a
number that looks like data. What changes per PoC is the route it travels:

- **PoC2** publishes it in-band, over the LiveKit data channel
  (`poc2_livekit/app/marks.py`).
- **PoC1** has no data channel (Media Streams carries audio and four
  events, and pipecat's serializer discards what it does not know), so it
  writes it **out of band**, to disk, and the cross-match is done by wall clock
  (`poc1_pipecat/app/marks.py`).

Both routes carry the same data with the same timestamp, set by the
harness **before** publishing. That is what makes them comparable; what changes
is the cost of the route, and that is declared separately.

## Two clocks

Harness and worker are different processes: subtracting their marks only makes sense if
they share a clock. Each mark travels with the harness's `host`; if it does not match the
worker's, the bias is flagged as **not interpretable** and is not corrected: a
clock correction without real synchronization would be making things up.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)

# Protocol identifier, which in PoC2 is also the data channel's topic.
# The `arnes.` prefix separates it from LiveKit's `lk.*` ones.
#
# It is called `cero_del_turno` and not `fin_de_voz` because it carries **both** zeros
# and neither of the two names would describe the packet.
ZERO_PROTOCOL: Final[str] = "arnes.cero_del_turno"

# Format version. Readers reject what they do not understand instead of reading it
# halfway: a misinterpreted mark gives a bias that looks like data.
#
# v2: `end_of_speech` (which measured the end of the CLIP) is renamed `clip_end`,
# `measured_end_of_speech` is added with its `drop_db`, and
# `margen_de_recorte_ms`, which corrected nothing, is removed.
#
# v3: the summary carries `turns_without_audio`. It bumps the version instead of being an
# optional key on purpose: a worker that does not read it would publish percentiles
# that **include** turns where the TTS did not sound, and would do so silently.
PROTOCOL_VERSION: Final[int] = 3

MARK_KIND: Final[str] = "mark"
SUMMARY_KIND: Final[str] = "summary"


# --------------------------------------------------------------------------
# The packet
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroMark:
    """The two zeros of a turn, as seen by whoever played the clip.

    All instants are **absolute** (`time.time()`), not relative to the
    harness's zero: whoever reads them does not have that zero and could not reconstruct them.

    `clip_end` is a fact (the last frame went out). `measured_end_of_speech` is
    a measurement against a threshold, and that is why `drop_db` goes **in this same
    structure**: separating them would make the number irreproducible.
    """

    sequence: int
    turn: str
    clip_id: str
    start_of_clip: float
    clip_end: float
    publish_sample_rate: int
    # The measured end of speech, and its criterion. `None` when the clip does not carry the
    # measurement (for example a loose clip that did not come out of the prepare step): then
    # there is only the file zero, and the block says so instead of making it up.
    measured_end_of_speech: float | None = None
    drop_db: float | None = None
    speech_threshold_dbfs: float | None = None
    host: str = ""
    sent_at: float = 0.0

    @property
    def duration_ms(self) -> float:
        return round((self.clip_end - self.start_of_clip) * 1000.0, 1)

    @property
    def cola_ms(self) -> float | None:
        """How long the file continues after the last speech."""
        if self.measured_end_of_speech is None:
            return None
        return round((self.clip_end - self.measured_end_of_speech) * 1000.0, 1)

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroMark":
        """Rebuilds a mark from a received packet.

        Strict about what it needs (without `clip_end` there is nothing to
        pair) and tolerant about the rest: a new field from a future
        version must not knock the worker over mid-call.

        The end of speech is **not** filled with a default: if it is missing, it was not
        measured, and an invented value there would be exactly the pretty, false number
        this design exists to avoid.
        """
        if "clip_end" not in data:
            raise ValueError("la marca no trae clip_end")

        def _optional(clave: str) -> float | None:
            valor = data.get(clave)
            return None if valor is None else float(valor)

        return cls(
            sequence=int(data.get("sequence", 0)),
            turn=str(data.get("turn", "")),
            clip_id=str(data.get("clip_id", "")),
            start_of_clip=float(data.get("start_of_clip", 0.0)),
            clip_end=float(data["clip_end"]),
            publish_sample_rate=int(data.get("publish_sample_rate", 0)),
            measured_end_of_speech=_optional("measured_end_of_speech"),
            drop_db=_optional("drop_db"),
            speech_threshold_dbfs=_optional("speech_threshold_dbfs"),
            host=str(data.get("host", "")),
            sent_at=float(data.get("sent_at", 0.0)),
        )


@dataclass(frozen=True)
class Envelope:
    """What came in a harness packet."""

    tipo: str
    marks: tuple[ZeroMark, ...]
    declared_total: int | None = None
    # Turns where the agent produced text and not a single frame with speech sounded.
    # It only travels in the summary: when the mark is sent live it is not yet
    # known, because the agent has not even started answering.
    turns_without_audio: tuple[str, ...] = ()


def encode_mark(mark: ZeroMark) -> bytes:
    return json.dumps(
        {
            "protocolo": ZERO_PROTOCOL,
            "version": PROTOCOL_VERSION,
            "tipo": MARK_KIND,
            "mark": mark.a_dict(),
        },
        ensure_ascii=False,
    ).encode("utf-8")


def encode_summary(
    marks: Sequence[ZeroMark], *, turns_without_audio: Sequence[str] = ()
) -> bytes:
    """All the marks again, plus how many the harness says it sent.

    `total` is what makes coverage verifiable: without it, the reader cannot
    tell "all of them arrived" from "the ones that arrived, arrived".

    `turns_without_audio` goes here and not in each mark because when the mark is published
    it is not yet known (the clip has just ended and the agent has not
    answered), and because it is precisely what is needed to keep those turns out of the
    percentile.
    """
    return json.dumps(
        {
            "protocolo": ZERO_PROTOCOL,
            "version": PROTOCOL_VERSION,
            "tipo": SUMMARY_KIND,
            "total": len(marks),
            "turns_without_audio": list(turns_without_audio),
            "marks": [m.a_dict() for m in marks],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def decode(carga: bytes | str) -> Envelope:
    """Reads a harness packet. `ValueError` if it is not ours or cannot be understood."""
    if isinstance(carga, bytes):
        carga = carga.decode("utf-8")
    data = json.loads(carga)
    if not isinstance(data, dict):
        raise ValueError("el paquete no es un objeto JSON")
    if data.get("protocolo") != ZERO_PROTOCOL:
        raise ValueError(f"protocolo ajeno: {data.get('protocolo')!r}")
    version = int(data.get("version", 0))
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"versión {version} del protocolo de marcas; este worker entiende "
            f"la {PROTOCOL_VERSION}"
        )

    tipo = str(data.get("tipo", ""))
    if tipo == MARK_KIND:
        return Envelope(tipo, (ZeroMark.from_dict(dict(data["mark"])),))
    if tipo == SUMMARY_KIND:
        raw_ones = list(data.get("marks") or [])
        return Envelope(
            tipo,
            tuple(ZeroMark.from_dict(dict(m)) for m in raw_ones),
            declared_total=int(data.get("total", len(raw_ones))),
            turns_without_audio=tuple(str(t) for t in (data.get("turns_without_audio") or [])),
        )
    raise ValueError(f"tipo de paquete desconocido: {tipo!r}")


# --------------------------------------------------------------------------
# The emitter, without its transport
# --------------------------------------------------------------------------


class ZeroEmitter:
    """Builds the marks and decides when they go out. Each PoC provides the route.

    Subclass it and write `_deliver`. Everything else (converting `end_of_speech_s` to
    an absolute instant, numbering, timestamping, counting failures, sending the
    summary) lives here, so that the 3 PoCs produce the same packet with the
    same fields and the same version. Whether the format matches cannot depend
    on someone remembering to copy a change.
    """

    # How long to wait after publishing the summary. It is zero unless the transport
    # returns before having really delivered: PoC2 raises it because
    # `publish_data` returns when the Rust side accepted the packet, not when the
    # SFU delivered it, and disconnecting at that instant can drop it.
    flush_wait_s: float = 0.0

    def __init__(
        self,
        *,
        publish_sample_rate: int,
        clock: Any = None,
        host: str | None = None,
    ) -> None:
        self._sample_rate = int(publish_sample_rate)
        self._clock = clock or time.time
        self._host = host if host is not None else socket.gethostname()
        self._marks: list[ZeroMark] = []
        self.faults: list[str] = []
        self.summary_sent: bool = False

    @property
    def marks(self) -> list[ZeroMark]:
        return list(self._marks)

    @property
    def sent(self) -> int:
        return len(self._marks)

    async def _deliver(self, carga: bytes) -> None:
        """Sends the packet out by whatever route. The only thing each PoC writes."""
        raise NotImplementedError

    async def _publish(self, carga: bytes, what: str) -> bool:
        """Delivers and records the failure if there is one. **Diagnostics never knock
        over the run**: losing a mark costs a bias, losing the call costs
        the call."""
        try:
            await self._deliver(carga)
            return True
        except Exception as error:  # noqa: BLE001
            self.faults.append(f"{what}: {error!r}")
            logger.warning("no se pudo publicar %s: %r", what, error)
            return False

    async def send(
        self,
        *,
        turn: str,
        clip_id: str,
        start_of_clip: float,
        clip_end: float,
        end_of_speech_s: float | None = None,
        drop_db: float | None = None,
        speech_threshold_dbfs: float | None = None,
    ) -> ZeroMark:
        """Publishes a turn's two zeros, as soon as the clip finishes sounding.

        `end_of_speech_s` is the offset **from the start of the clip** to
        the last speech, measured when the clip was prepared and read from the manifest; here
        it is converted to an absolute instant. If it is missing, the mark goes out with only the
        file zero: an end of speech is never made up.

        The timestamp (`sent_at`) and the two zeros are computed **before**
        delivering, so however long the transport takes does not move them. That is what
        allows comparing PoC1's bias with PoC2's even though their routes
        cost different things.

        It is sent live and not only at the end so that the packet travels
        while the agent is still processing that turn: if the run
        drops mid-call, the marks of the turns that did happen have already
        gone out.
        """
        start = round(float(start_of_clip), 6)
        mark = ZeroMark(
            sequence=len(self._marks) + 1,
            turn=turn,
            clip_id=clip_id,
            start_of_clip=start,
            clip_end=round(float(clip_end), 6),
            publish_sample_rate=self._sample_rate,
            measured_end_of_speech=(
                None if end_of_speech_s is None else round(start + float(end_of_speech_s), 6)
            ),
            drop_db=None if drop_db is None else float(drop_db),
            speech_threshold_dbfs=(
                None if speech_threshold_dbfs is None else round(float(speech_threshold_dbfs), 2)
            ),
            host=self._host,
            sent_at=round(float(self._clock()), 6),
        )
        self._marks.append(mark)
        await self._publish(encode_mark(mark), f"la marca {mark.sequence}")
        return mark

    async def shut(
        self, *, turns_without_audio: Sequence[str] = (), sleep: Any = None
    ) -> bool:
        """Sends the summary and waits for the transport to flush it.

        Returns whether the summary went out. `sleep` is injected so the
        wait can be tested without paying for it.

        `turns_without_audio` is the only thing the harness knows and the worker cannot
        deduce: the worker sees its own TTS request audio, not whether that audio actually
        sounded on the other side.
        """
        if not self._marks:
            return False
        ok = await self._publish(
            encode_summary(self._marks, turns_without_audio=turns_without_audio),
            "el resumen",
        )
        self.summary_sent = ok
        if ok and self.flush_wait_s > 0:
            import asyncio

            await (sleep or asyncio.sleep)(self.flush_wait_s)
        return ok
