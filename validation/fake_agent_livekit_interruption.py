"""A fake agent that falls silent a known time after being interrupted, inside the LiveKit room.

It is `validation.fake_agent_livekit` (another participant in the same LiveKit
Cloud room as the harness, in place of the PoC2 worker) with the changes of
`validation.fake_model_poc3_interruption`, which is its twin on PoC3's path:

1. **Answers at length.** To each script line it replies with `LONG_DURATION_S`
   of continuous speech, so that the harness has time to fire an event over it.
2. **Knows in advance what happens in each turn** (`--plan`, the same one the
   harness builds with `attach_events`). Through Opus the clip can only be
   recognized by waveform, ~0.8 s after it starts; if the turn carries an
   interruption, the first speech that reaches it while it is talking is taken
   as the onset of the interruption and it falls silent `S` ms later: the
   response currently playing is cut on the track. On a backchannel or in a
   control it keeps talking.
3. **The ground truth is timestamped with the clip already recognized and with
   the track's sends.** The speech onset and end of the interruption come from
   the recognizer and from the interruption clips' manifest; the instant it fell
   silent is that of the last 20 ms segment with speech (RMS ≥ 200) that went
   out on the track, and that of the reply, its first segment with speech. It
   replies `R` ms after the interruption's end of speech.

It needs the LiveKit keys from `.env`; it does not call Deepgram, OpenAI or
ElevenLabs, and it does not dispatch the worker.

    python -m validation.fake_agent_livekit_interruption --room ... --plan plan.json ...
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.carrier_simulator.observed import RMS_WITH_SPEECH
from validation.fake_agent import build_response, choose_voice
from validation.fake_agent_interruption import LONG_DURATION_S, continuous_speech
from validation.fake_agent_livekit import (
    MIN_CORRELATION,
    SAMPLES_PER_FRAME,
    NAME_OF_TRACK,
    SAMPLE_RATE,
    ContinuousTrack,
    Scheduled,
    token,
)
from validation.fake_model_poc3_interruption import event_templates
from validation.recognizer import RMS_OF_SPEECH, EndOfSpeech, NotRecognized, Recognizer, templates, resample_audio

logger = logging.getLogger(__name__)


def _rms(piece: np.ndarray) -> float:
    return float(np.sqrt(np.mean(piece.astype(np.float64) ** 2))) if len(piece) else 0.0


def speech_segments(r: Scheduled) -> list[int]:
    """Absolute track sample where each 20 ms segment of `r` with speech starts."""
    return [
        r.start + k for k in range(0, len(r.pcm), SAMPLES_PER_FRAME)
        if _rms(r.pcm[k:k + SAMPLES_PER_FRAME]) >= RMS_WITH_SPEECH
    ]


def cut(r: Scheduled, sample: int) -> None:
    """Trim the response so that it stops playing at `sample` (absolute)."""
    r.pcm = r.pcm[: max(0, sample - r.start)]


async def run_server(args: argparse.Namespace) -> None:
    from livekit import rtc

    from poc2_livekit.app.config import CARRIER_IDENTITY, load_settings
    from validation.validate_harness import reproducible_ones

    settings = load_settings()
    folder = Path(args.clips)
    of_scenario = [c for c in reproducible_ones() if c.startswith(f"e{args.scenario}_")]
    of_events, events = event_templates(Path(args.interruption_clips))
    ear = Recognizer(
        templates(folder, of_scenario, SAMPLE_RATE) + of_events, SAMPLE_RATE,
        min_correlation=MIN_CORRELATION,
    )
    plan: dict[str, str] = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    voice = choose_voice(folder, args.voice)
    corta = build_response(voice, level_dbfs=args.level_dbfs, soft_onset=False)
    reason = SAMPLE_RATE // TELEPHONY_SAMPLE_RATE
    pcm_corta = resample_audio(corta.pcm, TELEPHONY_SAMPLE_RATE, SAMPLE_RATE)
    primero_corta = reason * (corta.start_mm if corta.start_mm is not None else corta.phrase_start)
    long_pcm = resample_audio(continuous_speech(voice, LONG_DURATION_S, args.level_dbfs), TELEPHONY_SAMPLE_RATE, SAMPLE_RATE)
    delay = args.delay_ms / 1000.0
    stop = args.stop_ms / 1000.0
    replica_seg = args.replica_ms / 1000.0

    truth = Path(args.truth)
    truth.parent.mkdir(parents=True, exist_ok=True)
    other_ones: list[dict[str, Any]] = []
    seen_events: list[dict[str, Any]] = []
    room = rtc.Room()
    audio_track: ContinuousTrack | None = None
    halt = asyncio.Event()
    left_early = asyncio.Event()
    tasks: set[asyncio.Task[Any]] = set()
    greeting_done = False
    # The turn currently playing, which event it carries and the cut in progress.
    estado: dict[str, Any] = {"turn": None, "tipo": None, "cutoff": None}

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
        audio_track.schedule(pcm_corta, onset=now + primero_corta, primero=primero_corta,
                        phrase_start=primero_corta, start_mm=None, row={"tipo": "greeting"})

    def _playing(r: Scheduled | None, now: int) -> bool:
        return r is not None and r.start <= now < r.start + len(r.pcm)

    def _build_cut(arrival: float) -> None:
        """First carrier speech over a turn that carries an interruption."""
        assert audio_track is not None
        r = estado["turn"]
        sample = audio_track.sample_at(arrival + stop)
        sample += (-sample) % SAMPLES_PER_FRAME
        estado["cutoff"] = {"response": r, "built_at": arrival, "sample": sample}
        cut(r, sample)

    def _on_event_end(end: EndOfSpeech) -> None:
        assert audio_track is not None
        clip_id = end.template.clip_id
        kind, speech_onset = events[clip_id]
        row: dict[str, Any] = {
            "tipo": "event", "clip_id": clip_id, "kind": kind,
            "correlation": round(end.correlation, 4),
            "t_speech_onset": _arrival_moment(end.start_sample + speech_onset),
            "t_end_of_speech": _arrival_moment(end.start_sample + end.template.end_of_speech),
            "target_stop_ms": round(stop * 1000.0, 3) if kind == "interrupcion" else None,
        }
        cutoff = estado["cutoff"] if kind == "interrupcion" else None
        estado["cutoff"] = None if kind == "interrupcion" else estado["cutoff"]
        seen_events.append({"row": row, "cutoff": cutoff})
        if kind != "interrupcion":
            return
        if cutoff is None:
            row["was_not_speaking"] = True
        target = audio_track.sample_at(row["t_end_of_speech"] + replica_seg)
        if cutoff is not None:
            target = max(target, cutoff["sample"] + primero_corta + SAMPLES_PER_FRAME)
        audio_track.schedule(
            pcm_corta, onset=target, primero=primero_corta,
            phrase_start=primero_corta, start_mm=None,
            row={"tipo": "replica", "clip_id": clip_id, "t_end_of_speech": row["t_end_of_speech"],
                  "target_reply_ms": round(replica_seg * 1000.0, 3)},
        )

    def _arrival_moment(sample: int) -> float:
        import bisect

        j = bisect.bisect_right(ear._end_arrivals, sample)
        chunk_start = ear._end_arrivals[j - 1] if j > 0 else 0
        return ear._arrivals_t[j] + (sample - chunk_start) / SAMPLE_RATE

    async def _listen(track: Any) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for event in stream:
                arrival = time.perf_counter()
                trama = getattr(event, "frame", None)
                if trama is None or audio_track is None:
                    continue
                data = np.frombuffer(bytes(trama.data), dtype="<i2")
                if (
                    estado["tipo"] == "interrupcion" and estado["cutoff"] is None
                    and _playing(estado["turn"], audio_track.sample_at(arrival))
                    and _rms(data) >= RMS_OF_SPEECH
                ):
                    _build_cut(arrival)
                ear_says = ear.receive(data, arrival)
                if isinstance(ear_says, NotRecognized):
                    logger.warning("clip not recognized (correlation %.3f)", ear_says.correlation)
                    other_ones.append({"tipo": "not_recognized", **ear_says.__dict__})
                elif isinstance(ear_says, EndOfSpeech):
                    clip_id = ear_says.template.clip_id
                    if clip_id in events:
                        _on_event_end(ear_says)
                        continue
                    row = {
                        "tipo": "turn",
                        "clip_id": clip_id,
                        "correlation": round(ear_says.correlation, 4),
                        "second_correlation": round(ear_says.second_one, 4),
                        "t_first_sample_clip": ear_says.t_first_sample,
                        "t_end_of_speech": ear_says.t_end_of_speech,
                        "target_delay_ms": round(delay * 1000.0, 3),
                    }
                    target = audio_track.sample_at(ear_says.t_end_of_speech + delay)
                    audio_track.schedule(long_pcm.copy(), onset=target, primero=0,
                                    phrase_start=0, start_mm=None, row=row)
                    estado["turn"] = audio_track.replies[-1]
                    estado["tipo"] = plan.get(clip_id)
                    estado["cutoff"] = None
        finally:
            await stream.aclose()

    @room.on("track_subscribed")
    def _on_subscribe(track: Any, publishing: Any, participant: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        if getattr(participant, "identity", "") != CARRIER_IDENTITY:
            return
        _adopt(_listen(track))
        _greet()

    @room.on("participant_disconnected")
    def _on_leave(participant: Any) -> None:
        if getattr(participant, "identity", "") == CARRIER_IDENTITY:
            left_early.set()

    await room.connect(settings.livekit_url, token(settings, args.room))
    source = rtc.AudioSource(SAMPLE_RATE, 1)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track(NAME_OF_TRACK, source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE, dtx=False),
    )
    audio_track = ContinuousTrack(source=source, rtc=rtc)
    _adopt(audio_track.launch(halt))
    Path(args.ready).write_text("ready", encoding="utf-8")
    print(f"fake agent (interruptions) in room {args.room}", flush=True)

    for participant in room.remote_participants.values():
        for publishing in participant.track_publications.values():
            if publishing.track is not None:
                _on_subscribe(publishing.track, publishing, participant)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(left_early.wait(), timeout=args.tope_seg)
    await asyncio.sleep(1.0)
    halt.set()
    for tarea in list(tasks):
        tarea.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # The ground truth, with the instants each segment actually went out on the track.
    with truth.open("w", encoding="utf-8") as f:
        for visto in seen_events:
            row, cutoff = dict(visto["row"]), visto["cutoff"]
            if cutoff is not None:
                segments = speech_segments(cutoff["response"])
                t_ultima = audio_track.moment(segments[-1]) if segments else None
                row["t_armado"] = cutoff["built_at"]
                if t_ultima is None:
                    row["was_not_speaking"] = True
                else:
                    row["t_last_speech"] = t_ultima
                    row["actual_stop_ms"] = round((t_ultima - row["t_speech_onset"]) * 1000.0, 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for r in audio_track.replies:
            if r.row.get("tipo") != "replica":
                continue
            row = dict(r.row)
            segments = speech_segments(r)
            row["t_first_segment_with_speech"] = audio_track.moment(segments[0]) if segments else None
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in other_ones:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    await room.disconnect()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--room", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--interruption-clips", required=True)
    p.add_argument("--scenario", type=int, required=True)
    p.add_argument("--plan", required=True, help="json: script clip → kind of the event in its turn")
    p.add_argument("--voice", required=True, help="clip used as the agent's voice")
    p.add_argument("--delay-ms", type=float, default=1000.0)
    p.add_argument("--stop-ms", type=float, required=True)
    p.add_argument("--replica-ms", type=float, default=800.0)
    p.add_argument("--level-dbfs", type=float, default=-20.0)
    p.add_argument("--truth", required=True, help="jsonl where what actually happened is written")
    p.add_argument("--ready", required=True, help="file written once in the room")
    p.add_argument("--tope-seg", type=float, default=900.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
