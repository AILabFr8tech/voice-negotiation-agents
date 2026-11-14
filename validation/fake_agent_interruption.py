"""A fake agent that falls silent a known time after it is interrupted.

It is the fake agent from the latency validation (`validation.fake_agent`), with
three changes to validate the measurement of the interruption scenario:

1. **Answers at length.** To each script line it replies with `LONG_DURATION_S`
   of continuous speech, so that the harness has time to fire an event over it.
2. **Recognizes the event clips** the same way as the script clips: by the
   bytes of their first frame. It timestamps the onset and end of their speech
   with its own clock, when the segment containing each edge arrives.
3. **On an interruption it falls silent exactly `S` ms after** the carrier's
   speech starts, and replies `R` ms after it ends. On a backchannel it keeps
   talking.

The ground truth is taken from what the agent itself emitted: the instant its
last segment with speech (RMS ≥ 200, the harness threshold) went out before
falling silent, and that of its first segment with speech when replying.
Nothing is estimated.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw
from shared.carrier_simulator.script import load_clip_to_publish
from shared.carrier_simulator.observed import STREAK_HANGOVER_S, RMS_WITH_SPEECH
from validation.fake_agent import (
    SAMPLES_PER_FRAME,
    S_PER_FRAME,
    AGENT_TEXT,
    FakeAgent,
    Response,
    to_level,
    build_response,
    choose_voice,
    clip_fingerprints,
)

LONG_DURATION_S: Final[float] = 8.0


@dataclass(frozen=True)
class EventFingerprint:
    clip_id: str
    tipo: str
    speech_onset_sample: int
    end_of_speech_sample: int


@dataclass
class OngoingEvent:
    fingerprint: EventFingerprint
    initial_frame: int
    t_speech_onset: float | None = None


def event_fingerprints(folder: Path) -> dict[bytes, EventFingerprint]:
    """First published frame of each interruption clip → what it is and its edges."""
    from shared.audio.interruptions import load_manifest

    fingerprints: dict[bytes, EventFingerprint] = {}
    for clip_id, e in load_manifest(folder).items():
        ruta = folder / f"{clip_id}.ulaw"
        if not ruta.is_file():
            continue
        _, published, _, _ = load_clip_to_publish(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        fingerprints[published[:SAMPLES_PER_FRAME]] = EventFingerprint(
            clip_id, e["tipo"],
            round(float(e["speech_onset_s"]) * TELEPHONY_SAMPLE_RATE),
            min(len(published), round(float(e["end_of_speech_s"]) * TELEPHONY_SAMPLE_RATE)),
        )
    return fingerprints


def continuous_speech(voice: np.ndarray, seconds: float, level_dbfs: float) -> np.ndarray:
    """The longest continuous streak of `voice`, repeated up to `seconds`.

    Continuous by the harness rule (segments with RMS ≥ 200, gaps shorter than
    0.3 s), so that the response carries no pauses the harness could mistake
    for a stop.
    """
    voice = to_level(voice, level_dbfs)
    n = len(voice) // SAMPLES_PER_FRAME
    frames = voice[: n * SAMPLES_PER_FRAME].reshape(n, SAMPLES_PER_FRAME).astype(np.float64)
    active = np.flatnonzero(np.sqrt((frames ** 2).mean(axis=1)) >= RMS_WITH_SPEECH)
    best = (0, 0)
    start = end = int(active[0])
    for i in active[1:]:
        if (i - end) * S_PER_FRAME > STREAK_HANGOVER_S:
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = int(i)
        end = int(i)
    if end - start > best[1] - best[0]:
        best = (start, end)
    piece = voice[best[0] * SAMPLES_PER_FRAME:(best[1] + 1) * SAMPLES_PER_FRAME]
    times = int(np.ceil(seconds * TELEPHONY_SAMPLE_RATE / len(piece)))
    return np.tile(piece, times)[: int(seconds * TELEPHONY_SAMPLE_RATE)]


def _with_voice(trama: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(trama.astype(np.float64) ** 2))) >= RMS_WITH_SPEECH


class AgentThatStops(FakeAgent):
    def __init__(
        self, *, events_fingerprints: dict[bytes, EventFingerprint], long: Response,
        stop_s: float, replica_seg: float, **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.events_fingerprints = events_fingerprints
        self.long = long
        self.stop_s = stop_s
        self.replica_seg = replica_seg
        self.event: OngoingEvent | None = None
        self.halt_at: float | None = None
        self.event_row: dict[str, Any] | None = None
        # The interruption row waiting to learn when the agent fell silent.
        self.stop_row: dict[str, Any] | None = None

    async def speak(self, ws: Any, *, target: float | None, row: dict[str, Any]) -> None:
        """Like the original, but with a long, interruptible response."""
        response = self.long if row.get("tipo") == "turn" else self.response
        pcm = response.pcm
        rest = (-len(pcm)) % SAMPLES_PER_FRAME
        pcm = np.concatenate([pcm, np.zeros(rest, dtype=np.int16)])
        pcm_frames = pcm.reshape(-1, SAMPLES_PER_FRAME)
        primero = response.phrase_start
        onset = time.perf_counter() if target is None else target - primero / TELEPHONY_SAMPLE_RATE
        sends: list[float] = []
        last_with_speech: float | None = None
        for i, trama in enumerate(pcm_frames):
            deadline = onset + i * S_PER_FRAME
            wait = deadline - time.perf_counter()
            if wait > 0.002:
                await asyncio.sleep(wait - 0.002)
            while time.perf_counter() < deadline:
                pass
            if row.get("tipo") == "turn" and self.halt_at is not None and time.perf_counter() >= self.halt_at:
                # Falls silent: stops sending audio. The ground truth is its last segment with speech.
                self._close_event(last_with_speech)
                break
            t = time.perf_counter()
            sends.append(t)
            if _with_voice(trama):
                last_with_speech = t
                row.setdefault("t_first_segment_with_speech", t)
            if i == 0:
                await self._send(ws, {
                    "event": "harness.text", "de": "agent", "text": AGENT_TEXT,
                    "final": True, "has_spaces": False,
                })
            await self._send(ws, {
                "event": "mean", "streamSid": self.stream_sid,
                "mean": {"payload": base64.b64encode(pcm16_a_ulaw(trama)).decode("ascii")},
            })
        if sends:
            row["t_phrase_start"] = sends[primero // SAMPLES_PER_FRAME] + (primero % SAMPLES_PER_FRAME) / TELEPHONY_SAMPLE_RATE
        self.annotate(row)

    def _close_event(self, last_with_speech: float | None) -> None:
        f = self.stop_row
        self.stop_row = None
        self.halt_at = None
        if f is None or last_with_speech is None:
            return
        f["t_last_speech"] = last_with_speech
        f["actual_stop_ms"] = round((last_with_speech - f["t_speech_onset"]) * 1000.0, 3)

    def _on_receive(self, ws: Any, n: int, data: bytes, arrival: float) -> None:
        fingerprint = self.events_fingerprints.get(data)
        if fingerprint is not None:
            self.event = OngoingEvent(fingerprint, n)
        ev = self.event
        if ev is not None:
            h = ev.fingerprint
            start_frame, inside_start = divmod(h.speech_onset_sample, SAMPLES_PER_FRAME)
            end_frame, inside_end = divmod(h.end_of_speech_sample, SAMPLES_PER_FRAME)
            if n == ev.initial_frame + start_frame:
                ev.t_speech_onset = arrival + inside_start / TELEPHONY_SAMPLE_RATE
                self.event_row = {
                    "tipo": "event", "clip_id": h.clip_id, "kind": h.tipo,
                    "t_speech_onset": ev.t_speech_onset,
                    "target_stop_ms": round(self.stop_s * 1000.0, 3)
                    if h.tipo == "interrupcion" else None,
                }
                if h.tipo == "interrupcion":
                    self.halt_at = ev.t_speech_onset + self.stop_s
                    self.stop_row = self.event_row
            if n == ev.initial_frame + end_frame:
                t_end = arrival + inside_end / TELEPHONY_SAMPLE_RATE
                self.event = None
                row = self.event_row or {}
                row["t_end_of_speech"] = t_end
                self.event_row = None
                if h.tipo == "interrupcion":
                    replica = {"tipo": "replica", "clip_id": h.clip_id, "t_end_of_speech": t_end,
                               "target_reply_ms": round(self.replica_seg * 1000.0, 3)}
                    self._launch(self._reply(ws, row, replica, t_end + self.replica_seg))
                else:
                    self.annotate(row)
            return
        super()._on_receive(ws, n, data, arrival)

    async def _reply(self, ws: Any, event_row: dict[str, Any], replica: dict[str, Any], target: float) -> None:
        # The event row is written once its last segment with speech is known.
        while self.halt_at is not None and time.perf_counter() < target:
            await asyncio.sleep(0.01)
        if self.halt_at is not None:
            # The interruption arrived when it was no longer speaking: there was nothing to cut.
            self.halt_at = None
            self.stop_row = None
            event_row["was_not_speaking"] = True
        self.annotate(event_row)
        await self.speak(ws, target=target, row=replica)


async def run_server(args: argparse.Namespace) -> None:
    import logging

    from websockets.asyncio.server import serve

    from shared.carrier_simulator.player import high_clock_resolution

    logging.getLogger("websockets").setLevel(logging.CRITICAL)
    folder = Path(args.clips)
    voice = choose_voice(folder, args.voice)
    corta = build_response(voice, level_dbfs=args.level_dbfs, soft_onset=False)
    long_pcm = continuous_speech(voice, LONG_DURATION_S, args.level_dbfs)
    long = Response(long_pcm, 0, None)
    agent = AgentThatStops(
        events_fingerprints=event_fingerprints(Path(args.interruption_clips)),
        long=long,
        stop_s=args.stop_ms / 1000.0,
        replica_seg=args.replica_ms / 1000.0,
        fingerprints=clip_fingerprints(folder),
        response=corta,
        delay_s=args.delay_ms / 1000.0,
        truth=Path(args.truth),
        seed=args.seed,
    )
    Path(args.truth).parent.mkdir(parents=True, exist_ok=True)
    Path(args.truth).write_text("", encoding="utf-8")
    with high_clock_resolution():
        async with serve(agent.handle, "127.0.0.1", args.port, max_size=None):
            print(f"fake agent (interruptions) listening on {args.port}", flush=True)
            await agent.done.wait()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--interruption-clips", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--delay-ms", type=float, default=1000.0)
    p.add_argument("--stop-ms", type=float, required=True)
    p.add_argument("--replica-ms", type=float, default=800.0)
    p.add_argument("--level-dbfs", type=float, default=-20.0)
    p.add_argument("--truth", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
