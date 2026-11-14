"""A fake agent that answers with a known delay, over PoC1's WebSocket.

It is used to measure the harness against something whose response is known in
advance. It speaks the same protocol as PoC1's `/ws` (Twilio Media Streams:
`connected`, `start`, `media`, `stop`) and, instead of STT, LLM and TTS, does this:

1. **Recognizes each clip by its first frame.** The bytes the harness publishes
   are deterministic (the clip on disk plus a noise floor whose seed comes from
   the clip's sha256), so the first 20 ms frame of each clip is an exact
   fingerprint. There is no need to estimate where the clip started: it is known.
2. **Timestamps the real end of speech** with its own clock: the instant the
   frame containing the last sample with speech arrived (the manifest's
   `end_of_speech_s`) plus the position of that sample within the frame.
3. **Answers exactly D ms later**, with speech at a known level. The onset of
   the response falls at a random position within the frame, so that rounding
   to 20 ms frames also enters the measurement.

Everything is timestamped with `time.perf_counter()`, which is the same clock in
both processes (QueryPerformanceCounter on Windows, CLOCK_MONOTONIC on Linux).
The ground truth of each turn is written to a `.jsonl` to be compared later with
what the harness reported.

It does not touch the code of the PoCs or of the harness: it only imports their
clip-loading and encoding functions.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw
from shared.audio.manifest import MANIFEST_NAME, load_manifest
from shared.audio.level import rms_of_speech, windows_with_speech
from shared.carrier_simulator.script import load_clip, load_clip_to_publish

SAMPLES_PER_FRAME: Final[int] = TELEPHONY_SAMPLE_RATE * 20 // 1000
S_PER_FRAME: Final[float] = SAMPLES_PER_FRAME / TELEPHONY_SAMPLE_RATE

# Duration of the response. Longer than 2 s on purpose: the harness accepts,
# without consulting the text, any speech streak longer than 2 s, so the
# measurement does not depend on the transcript arriving.
RESPONSE_DURATION_S: Final[float] = 2.6

# The "mm" of the soft onset: a short nasal hum, 12 dB below the speech of the
# phrase, followed by a brief pause before the phrase.
MM_DURATION_S: Final[float] = 0.30
MM_PAUSE_S: Final[float] = 0.15
MM_BELOW_SPEECH_DB: Final[float] = 12.0
MM_F0_HZ: Final[float] = 120.0

AGENT_TEXT: Final[str] = "Claro, le platico cómo quedaría el viaje."


# --------------------------------------------------------------------------
# The response signal
# --------------------------------------------------------------------------


def to_level(pcm: np.ndarray, level_dbfs: float) -> np.ndarray:
    """Scale so that the RMS of the windows with speech ends up at `level_dbfs`.

    It is the same criterion the clips were normalized with (RMS of the 10 ms
    windows within 25 dB of the peak). Clips to int16 if needed: at −6 dBFS RMS
    the speech peaks exceed full scale.
    """
    actual = rms_of_speech(pcm, TELEPHONY_SAMPLE_RATE)
    gain = 10.0 ** ((level_dbfs - actual) / 20.0)
    return np.clip(np.round(pcm.astype(np.float64) * gain), -32768, 32767).astype(np.int16)


def speech_onset(pcm: np.ndarray) -> int:
    """First sample of the first window with speech (25 dB below the peak).

    It is the symmetric counterpart of the clips' end-of-speech rule: the real
    latency goes from one edge to the other, both measured with the same criterion.
    """
    with_voice, per_window, _ = windows_with_speech(pcm, TELEPHONY_SAMPLE_RATE)
    return int(with_voice[0]) * per_window if with_voice.size else 0


def hum_mm(level_dbfs: float) -> np.ndarray:
    """A synthetic "mm": 120 Hz harmonics with 30 ms soft edges."""
    n = int(MM_DURATION_S * TELEPHONY_SAMPLE_RATE)
    t = np.arange(n) / TELEPHONY_SAMPLE_RATE
    sig = sum((0.6**k) * np.sin(2 * np.pi * MM_F0_HZ * (k + 1) * t) for k in range(4))
    edge = int(0.03 * TELEPHONY_SAMPLE_RATE)
    envelope = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge))
    envelope[:edge] = ramp
    envelope[-edge:] = ramp[::-1]
    sig = sig * envelope
    rms = np.sqrt(np.mean(sig**2))
    target = 32768.0 * 10.0 ** (level_dbfs / 20.0)
    return np.round(sig * target / rms).astype(np.int16)


@dataclass(frozen=True)
class Response:
    """The built response, with its edges known in samples."""

    pcm: np.ndarray
    # First sample of the phrase according to the −25 dB rule.
    phrase_start: int
    # First sample of the "mm", or `None` if there is no soft onset.
    start_mm: int | None


def build_response(voice: np.ndarray, *, level_dbfs: float, soft_onset: bool) -> Response:
    frase = to_level(voice[: int(RESPONSE_DURATION_S * TELEPHONY_SAMPLE_RATE)], level_dbfs)
    phrase_start = speech_onset(frase)
    if not soft_onset:
        return Response(frase, phrase_start, None)
    mm = hum_mm(level_dbfs - MM_BELOW_SPEECH_DB)
    pause = np.zeros(int(MM_PAUSE_S * TELEPHONY_SAMPLE_RATE), dtype=np.int16)
    # The phrase is trimmed to its speech onset so that the pause is exactly the pause.
    frase = frase[phrase_start:]
    pcm = np.concatenate([mm, pause, frase])
    return Response(pcm, len(mm) + len(pause), 0)


# --------------------------------------------------------------------------
# The clip fingerprints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    clip_id: str
    text: str
    end_of_speech_sample: int


def clip_fingerprints(folder: Path) -> dict[bytes, Fingerprint]:
    """First published frame of each clip → which clip it is and where its speech ends."""
    manifest = load_manifest(folder / MANIFEST_NAME)
    fingerprints: dict[bytes, Fingerprint] = {}
    for entry in manifest.entries:
        ruta = folder / f"{entry.id}.ulaw"
        if not ruta.is_file() or entry.end_of_speech_s is None:
            continue
        _, published, _, _ = load_clip_to_publish(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        first = published[:SAMPLES_PER_FRAME]
        if first in fingerprints:
            raise SystemExit(f"{entry.id} and {fingerprints[first].clip_id} start with the same frame")
        fingerprints[first] = Fingerprint(
            entry.id, entry.text,
            min(len(published), round(float(entry.end_of_speech_s) * TELEPHONY_SAMPLE_RATE)),
        )
    return fingerprints


# --------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------


@dataclass
class CurrentClip:
    fingerprint: Fingerprint
    initial_frame: int
    t_first_frame: float


class FakeAgent:
    def __init__(
        self,
        *,
        fingerprints: dict[bytes, Fingerprint],
        response: Response,
        delay_s: float,
        truth: Path,
        seed: int,
    ) -> None:
        self.fingerprints = fingerprints
        self.response = response
        self.delay_s = delay_s
        self.truth = truth
        self.chance = random.Random(seed)
        self.stream_sid = ""
        self.frames = 0
        self.in_progress: CurrentClip | None = None
        self.tasks: set[asyncio.Task[None]] = set()
        self.done = asyncio.Event()
        self._lock = asyncio.Lock()

    def annotate(self, row: dict[str, Any]) -> None:
        with self.truth.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _send(self, ws: Any, message: dict[str, Any]) -> None:
        async with self._lock:
            await ws.send(json.dumps(message))

    async def speak(
        self, ws: Any, *, target: float | None, row: dict[str, Any]
    ) -> None:
        """Send the response with its onset at `target` (perf_counter).

        `target=None` is the greeting: it goes out as soon as possible.
        """
        offset = self.chance.randrange(SAMPLES_PER_FRAME)
        pcm = np.concatenate([np.zeros(offset, dtype=np.int16), self.response.pcm])
        rest = (-len(pcm)) % SAMPLES_PER_FRAME
        pcm = np.concatenate([pcm, np.zeros(rest, dtype=np.int16)])
        ulaw = pcm16_a_ulaw(pcm)
        frames = [ulaw[i:i + SAMPLES_PER_FRAME] for i in range(0, len(ulaw), SAMPLES_PER_FRAME)]

        phrase_start = offset + self.response.phrase_start
        start_mm = None if self.response.start_mm is None else offset + self.response.start_mm
        primero = start_mm if start_mm is not None else phrase_start
        onset = (
            time.perf_counter() if target is None
            else target - primero / TELEPHONY_SAMPLE_RATE
        )
        sends: list[float] = []
        for i, trama in enumerate(frames):
            deadline = onset + i * S_PER_FRAME
            wait = deadline - time.perf_counter()
            if wait > 0.002:
                await asyncio.sleep(wait - 0.002)
            while time.perf_counter() < deadline:
                pass
            sends.append(time.perf_counter())
            if i == 0:
                await self._send(ws, {
                    "event": "harness.text", "de": "agent", "text": AGENT_TEXT,
                    "final": True, "has_spaces": False,
                })
            await self._send(ws, {
                "event": "mean", "streamSid": self.stream_sid,
                "mean": {"payload": base64.b64encode(trama).decode("ascii")},
            })

        def moment(sample: int) -> float:
            j, inside = divmod(sample, SAMPLES_PER_FRAME)
            return sends[j] + inside / TELEPHONY_SAMPLE_RATE

        row.update({
            "offset_samples": offset,
            "t_phrase_start": moment(phrase_start),
            "t_mm_start": None if start_mm is None else moment(start_mm),
            "t_first_frame_response": sends[0],
        })
        if row.get("tipo") == "turn":
            row["real_latency_ms"] = round((moment(primero) - row["t_end_of_speech"]) * 1000.0, 3)
            row["real_phrase_latency_ms"] = round((row["t_phrase_start"] - row["t_end_of_speech"]) * 1000.0, 3)
        self.annotate(row)

    def _launch(self, coroutine: Any) -> None:
        tarea = asyncio.create_task(coroutine)
        self.tasks.add(tarea)
        tarea.add_done_callback(self.tasks.discard)

    async def handle(self, ws: Any) -> None:
        async for raw_value in ws:
            arrival = time.perf_counter()
            message = json.loads(raw_value)
            event = str(message.get("event", ""))
            if event == "start":
                self.stream_sid = str(message.get("streamSid", ""))
                self._launch(self.speak(ws, target=None, row={"tipo": "greeting"}))
            elif event == "mean":
                data = base64.b64decode((message.get("mean") or {}).get("payload") or "")
                n = self.frames
                self.frames += 1
                self._on_receive(ws, n, data, arrival)
            elif event == "stop":
                break
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.done.set()

    def _on_receive(self, ws: Any, n: int, data: bytes, arrival: float) -> None:
        fingerprint = self.fingerprints.get(data)
        if fingerprint is not None:
            self.in_progress = CurrentClip(fingerprint, n, arrival)
            return
        actual = self.in_progress
        if actual is None:
            return
        end_frame, inside = divmod(actual.fingerprint.end_of_speech_sample, SAMPLES_PER_FRAME)
        if n != actual.initial_frame + end_frame:
            return
        self.in_progress = None
        end_of_speech = arrival + inside / TELEPHONY_SAMPLE_RATE
        self._launch(self._send(ws, {
            "event": "harness.text", "de": "carrier", "text": actual.fingerprint.text,
            "final": True, "has_spaces": False,
        }))
        row = {
            "tipo": "turn",
            "clip_id": actual.fingerprint.clip_id,
            "t_first_frame_clip": actual.t_first_frame,
            "t_end_of_speech": end_of_speech,
            "target_delay_ms": round(self.delay_s * 1000.0, 3),
        }
        self._launch(self.speak(ws, target=end_of_speech + self.delay_s, row=row))


def choose_voice(folder: Path, clip_id: str) -> np.ndarray:
    samples, _ = load_clip(folder / f"{clip_id}.ulaw", publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    return samples


async def run_server(args: argparse.Namespace) -> None:
    from websockets.asyncio.server import serve

    from shared.carrier_simulator.player import high_clock_resolution

    # The driver's port probe opens and closes a TCP connection without a
    # handshake; it is not an agent error.
    logging.getLogger("websockets").setLevel(logging.CRITICAL)
    folder = Path(args.clips)
    response = build_response(
        choose_voice(folder, args.voice), level_dbfs=args.level_dbfs, soft_onset=args.soft_onset
    )
    agent = FakeAgent(
        fingerprints=clip_fingerprints(folder),
        response=response,
        delay_s=args.delay_ms / 1000.0,
        truth=Path(args.truth),
        seed=args.seed,
    )
    Path(args.truth).parent.mkdir(parents=True, exist_ok=True)
    Path(args.truth).write_text("", encoding="utf-8")
    with high_clock_resolution():
        async with serve(agent.handle, "127.0.0.1", args.port, max_size=None):
            print(f"fake agent listening on {args.port}", flush=True)
            await agent.done.wait()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--voice", required=True, help="clip used as the agent's voice")
    p.add_argument("--delay-ms", type=float, required=True)
    p.add_argument("--level-dbfs", type=float, default=-20.0)
    p.add_argument("--soft-onset", action="store_true")
    p.add_argument("--truth", required=True, help="jsonl where what actually happened is written")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
