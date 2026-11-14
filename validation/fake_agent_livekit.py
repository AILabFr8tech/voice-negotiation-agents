"""A fake agent that answers with a known delay, inside the LiveKit room.

It is the twin of `validation.fake_agent` for PoC2's path. It joins the same
room as the harness, on LiveKit Cloud, as another participant, in place of the
PoC2 worker. The carrier's audio reaches it exactly the way it would reach the
worker (published by the harness, compressed with Opus, forwarded by the SFU)
and its response returns to the harness along the same path. The only thing
missing is PoC2's pipeline (STT, LLM, TTS): in its place there is a known delay.

The fake agent:

1. **Recognizes each clip by its waveform** (`validation.recognizer`): Opus
   changes the bytes, so there is no exact fingerprint as in PoC1.
2. **Timestamps the real end of speech**: the instant the frame containing the
   clip's last sample with speech reached it.
3. **Answers exactly D ms later**, over a continuous 24 kHz audio track in
   20 ms frames at real-time pacing, the way the worker publishes.
4. **Receives the harness marks** over the data channel, like the worker, and
   at the end leaves them in a JSON shaped like the worker's
   (`livekit_extra.harness`), so that latency is computed with the same function
   as in the campaign.

Everything is timestamped with `time.perf_counter()`, the same clock the harness
uses in the other process. It needs the LiveKit keys from `.env`; it does not
call Deepgram, OpenAI or ElevenLabs, and it does not dispatch the worker.

    python -m validation.fake_agent_livekit --room ... --clips ... --scenario 1 ...
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from validation.fake_agent import build_response, choose_voice
from validation.recognizer import (
    EndOfSpeech,
    NotRecognized,
    Recognizer,
    templates,
    resample_audio,
)

# Identity of the fake agent in the room. Different from the worker's on
# purpose: in the room it must not be mistakable for it.
FAKE_AGENT_IDENTITY: Final[str] = "agente-falso-validacion"
NAME_OF_TRACK: Final[str] = "fake_agent"

# At the rate the harness hears (`poc2_livekit.app.harness.LISTEN_SAMPLE_RATE`) and
# that PoC3's model uses, so that recognition is done the same way on both paths.
SAMPLE_RATE: Final[int] = 24_000
SAMPLES_PER_FRAME: Final[int] = SAMPLE_RATE // 50
S_PER_FRAME: Final[float] = SAMPLES_PER_FRAME / SAMPLE_RATE

# Through Opus the waveform does not arrive intact: the peak drops. What is
# also required is a margin over the second candidate line (`Recognizer`).
MIN_CORRELATION: Final[float] = 0.50

logger = logging.getLogger(__name__)


@dataclass
class Scheduled:
    """A response placed on the track, with its edges in absolute samples."""

    pcm: np.ndarray
    start: int
    phrase_start: int
    start_mm: int | None
    row: dict[str, Any]


@dataclass
class ContinuousTrack:
    """The fake agent's voice: one 20 ms frame every 20 ms, whether there is speech or not.

    Each frame is timestamped just before handing it to the `AudioSource`, the
    same convention the harness uses to timestamp its own (`shared/carrier_simulator/pacing.py`).
    """

    source: Any
    rtc: Any
    sends: list[float] = field(default_factory=list)
    replies: list[Scheduled] = field(default_factory=list)
    t0: float = 0.0

    def sample_at(self, t: float) -> int:
        """The track sample that goes out at instant `t` (perf_counter)."""
        return round((t - self.t0) * SAMPLE_RATE)

    def moment(self, sample: int) -> float | None:
        j, inside = divmod(sample, SAMPLES_PER_FRAME)
        if j >= len(self.sends):
            return None
        return self.sends[j] + inside / SAMPLE_RATE

    def schedule(self, pcm: np.ndarray, *, onset: int, primero: int,
                  phrase_start: int, start_mm: int | None, row: dict[str, Any]) -> None:
        """Place `pcm` so that its sample `primero` goes out at sample `onset`."""
        start = max(onset - primero, len(self.sends) * SAMPLES_PER_FRAME + SAMPLES_PER_FRAME)
        self.replies.append(Scheduled(
            pcm, start, start + phrase_start,
            None if start_mm is None else start + start_mm, row,
        ))

    def _trama(self, j: int) -> np.ndarray:
        since = j * SAMPLES_PER_FRAME
        until = since + SAMPLES_PER_FRAME
        trama = np.zeros(SAMPLES_PER_FRAME, dtype=np.int32)
        for r in self.replies:
            a, b = max(since, r.start), min(until, r.start + len(r.pcm))
            if a < b:
                trama[a - since:b - since] += r.pcm[a - r.start:b - r.start]
        return np.clip(trama, -32768, 32767).astype(np.int16)

    async def launch(self, halt: asyncio.Event) -> None:
        self.t0 = time.perf_counter()
        j = 0
        while not halt.is_set():
            deadline = self.t0 + j * S_PER_FRAME
            wait = deadline - time.perf_counter()
            if wait > 0.002:
                await asyncio.sleep(wait - 0.002)
            while time.perf_counter() < deadline:
                pass
            trama = self._trama(j)
            self.sends.append(time.perf_counter())
            await self.source.capture_frame(self.rtc.AudioFrame(
                data=trama.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1,
                samples_per_channel=SAMPLES_PER_FRAME,
            ))
            j += 1


def token(settings: Any, room: str) -> str:
    from livekit.api import AccessToken, VideoGrants

    return (
        AccessToken(api_key=settings.livekit_api_key, api_secret=settings.livekit_api_secret)
        .with_identity(FAKE_AGENT_IDENTITY)
        .with_name("Agente falso (validación del arnés)")
        .with_grants(VideoGrants(
            room_join=True, room=room, can_publish=True, can_subscribe=True,
            can_publish_data=True,
        ))
        .to_jwt()
    )


async def run_server(args: argparse.Namespace) -> None:
    from livekit import rtc

    from poc2_livekit.app.config import CARRIER_IDENTITY, load_settings
    from poc2_livekit.app.marks import MarksRecord
    from validation.validate_harness import reproducible_ones

    settings = load_settings()
    folder = Path(args.clips)
    of_scenario = [c for c in reproducible_ones() if c.startswith(f"e{args.scenario}_")]
    ear = Recognizer(
        templates(folder, of_scenario, SAMPLE_RATE), SAMPLE_RATE, min_correlation=MIN_CORRELATION
    )
    response = build_response(
        choose_voice(folder, args.voice), level_dbfs=args.level_dbfs,
        soft_onset=args.soft_onset,
    )
    reason = SAMPLE_RATE // TELEPHONY_SAMPLE_RATE
    pcm = resample_audio(response.pcm, TELEPHONY_SAMPLE_RATE, SAMPLE_RATE)
    phrase_start = reason * response.phrase_start
    start_mm = None if response.start_mm is None else reason * response.start_mm
    primero = start_mm if start_mm is not None else phrase_start
    delay = args.delay_ms / 1000.0

    truth = Path(args.truth)
    truth.parent.mkdir(parents=True, exist_ok=True)
    other_ones: list[dict[str, Any]] = []
    marks = MarksRecord()
    room = rtc.Room()
    audio_track: ContinuousTrack | None = None
    halt = asyncio.Event()
    left_early = asyncio.Event()
    tasks: set[asyncio.Task[Any]] = set()
    greeting_done = False

    def _adopt(coroutine: Any) -> None:
        tarea = asyncio.create_task(coroutine)
        tasks.add(tarea)
        tarea.add_done_callback(tasks.discard)

    def _greet() -> None:
        nonlocal greeting_done
        if greeting_done or audio_track is None:
            return
        greeting_done = True
        now = audio_track.sample_at(time.perf_counter()) + 2 * SAMPLES_PER_FRAME
        audio_track.schedule(pcm, onset=now + primero, primero=primero,
                        phrase_start=phrase_start, start_mm=start_mm, row={"tipo": "greeting"})

    async def _listen(track: Any) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for event in stream:
                arrival = time.perf_counter()
                trama = getattr(event, "frame", None)
                if trama is None:
                    continue
                ear_says = ear.receive(np.frombuffer(bytes(trama.data), dtype="<i2"), arrival)
                if isinstance(ear_says, NotRecognized):
                    logger.warning("clip not recognized (correlation %.3f)", ear_says.correlation)
                    other_ones.append({"tipo": "not_recognized", **ear_says.__dict__})
                elif isinstance(ear_says, EndOfSpeech) and audio_track is not None:
                    row = {
                        "tipo": "turn",
                        "clip_id": ear_says.template.clip_id,
                        "correlation": round(ear_says.correlation, 4),
                        "second_correlation": round(ear_says.second_one, 4),
                        "t_first_sample_clip": ear_says.t_first_sample,
                        "t_end_of_speech": ear_says.t_end_of_speech,
                        "target_delay_ms": round(delay * 1000.0, 3),
                    }
                    target = audio_track.sample_at(ear_says.t_end_of_speech + delay)
                    audio_track.schedule(pcm, onset=target, primero=primero,
                                    phrase_start=phrase_start, start_mm=start_mm, row=row)
        finally:
            await stream.aclose()

    @room.on("track_subscribed")
    def _on_subscribe(track: Any, publishing: Any, participant: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        if getattr(participant, "identity", "") != CARRIER_IDENTITY:
            return
        _adopt(_listen(track))
        # The worker greets when the carrier picks up; here, when the carrier's
        # audio starts arriving.
        _greet()

    @room.on("participant_disconnected")
    def _on_leave(participant: Any) -> None:
        if getattr(participant, "identity", "") == CARRIER_IDENTITY:
            left_early.set()

    @room.on("data_received")
    def _on_data_received(packet: Any) -> None:
        participant = getattr(packet, "participant", None)
        marks.receive(
            getattr(packet, "data", b"") or b"",
            topic=str(getattr(packet, "topic", "") or ""),
            identity=str(getattr(participant, "identity", "") or ""),
        )

    await room.connect(settings.livekit_url, token(settings, args.room))
    source = rtc.AudioSource(SAMPLE_RATE, 1)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track(NAME_OF_TRACK, source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE, dtx=False),
    )
    audio_track = ContinuousTrack(source=source, rtc=rtc)
    _adopt(audio_track.launch(halt))
    Path(args.ready).write_text("ready", encoding="utf-8")
    print(f"fake agent in room {args.room}", flush=True)

    # If the harness was already in the room on joining, its track does not fire the event.
    for participant in room.remote_participants.values():
        for publishing in participant.track_publications.values():
            if publishing.track is not None:
                _on_subscribe(publishing.track, publishing, participant)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(left_early.wait(), timeout=args.tope_seg)
    # Let any response still playing go out before cutting the track.
    await asyncio.sleep(1.0)
    halt.set()
    for tarea in list(tasks):
        tarea.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # The ground truth: each response with the instants its edges went out.
    with truth.open("w", encoding="utf-8") as f:
        for r in audio_track.replies:
            row = dict(r.row)
            t_frase = audio_track.moment(r.phrase_start)
            t_mm = None if r.start_mm is None else audio_track.moment(r.start_mm)
            row.update({"t_phrase_start": t_frase, "t_mm_start": t_mm})
            t_primero = t_mm if t_mm is not None else t_frase
            if row.get("tipo") == "turn" and t_primero is not None:
                row["real_latency_ms"] = round((t_primero - row["t_end_of_speech"]) * 1000.0, 3)
                row["real_phrase_latency_ms"] = round(
                    (t_frase - row["t_end_of_speech"]) * 1000.0, 3
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in other_ones:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # The harness marks, shaped like the worker's JSON.
    Path(args.marks).write_text(
        json.dumps({"latencies": [], "livekit_extra": {"harness": marks.block([])}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await room.disconnect()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--room", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--scenario", type=int, required=True)
    p.add_argument("--voice", required=True, help="clip used as the agent's voice")
    p.add_argument("--delay-ms", type=float, required=True)
    p.add_argument("--level-dbfs", type=float, default=-20.0)
    p.add_argument("--soft-onset", action="store_true")
    p.add_argument("--truth", required=True, help="jsonl where what actually happened is written")
    p.add_argument("--marks", required=True, help="json where the harness marks are stored")
    p.add_argument("--ready", required=True, help="file written once in the room")
    p.add_argument("--tope-seg", type=float, default=600.0)
    args = p.parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
