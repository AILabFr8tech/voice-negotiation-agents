"""A fake GPT-Live that falls silent a known time after being interrupted, behind PoC3's bridge.

It is `validation.fake_model_poc3` (PoC3's production bridge with the model
replaced) with the changes of `validation.fake_agent_interruption`, which is
its twin on PoC1's path:

1. **Answers at length.** To each script line it replies with `LONG_DURATION_S`
   of continuous speech, in 100 ms deltas at real-time pacing, so that the
   harness has time to fire an event over it.
2. **Knows in advance what happens in each turn.** Through the bridge the audio
   arrives resampled and the clip can only be recognized by waveform, ~0.8 s
   after it starts: too late to fall silent at 200 ms. So the validator passes
   it the event plan (`--plan`, the same one the harness builds with
   `attach_events`) and, if the turn carries an interruption, the first speech
   that reaches it while it is talking is taken as the onset of the
   interruption and it falls silent `S` ms later. On a backchannel or in a
   control it keeps talking.
3. **The ground truth is timestamped afterwards, with the clip already
   recognized.** The speech onset and end of the interruption clip come from
   the waveform recognition and from the interruption clips' manifest, just like
   the end of speech of the script lines. The instant it fell silent is that of
   the last 20 ms segment with speech (RMS ≥ 200) it managed to send, **in
   playback time**: the delta goes out on schedule and each segment within it
   plays 20 ms after the previous one. That is what the person on the phone
   would hear. It replies `R` ms after the interruption's end of speech.

`S` does not have to match the real stop: what is compared is what the harness
reports against what the model actually did.

    python -m validation.fake_model_poc3_interruption --port 8765 --plan plan.json ...
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import bisect
import contextlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.carrier_simulator.observed import RMS_WITH_SPEECH
from validation import fake_model_poc3 as base
from validation.fake_agent import AGENT_TEXT, Response, build_response, choose_voice
from validation.fake_agent_interruption import LONG_DURATION_S, continuous_speech
from validation.recognizer import (
    TEMPLATE_LENGTH_S,
    RMS_OF_SPEECH,
    EndOfSpeech,
    NotRecognized,
    Template,
    templates,
    resample_audio,
)

# The session sample rate. `run_server` changes it with `--format-kind pcmu`
# (8 kHz μ-law, no resampling); by default it is 24 kHz PCM16, the September one.
SAMPLE_RATE: int = base.MODEL_SAMPLE_RATE
ULAW: bool = False
# 20 ms segment at the model's rate: the unit with which the harness decides whether there is speech.
SAMPLES_PER_SEGMENT: int = SAMPLE_RATE // 50
# One 100 ms delta at the model's rate, and how many model samples per 8 kHz
# sample.
BY_DELTA: int = SAMPLE_RATE // 10
RAZON: int = SAMPLE_RATE // TELEPHONY_SAMPLE_RATE


def _use_format(format_kind: str) -> None:
    global SAMPLE_RATE, ULAW, SAMPLES_PER_SEGMENT, BY_DELTA, RAZON
    ULAW = format_kind == "pcmu"
    SAMPLE_RATE = TELEPHONY_SAMPLE_RATE if ULAW else base.MODEL_SAMPLE_RATE
    SAMPLES_PER_SEGMENT = SAMPLE_RATE // 50
    BY_DELTA = SAMPLE_RATE // 10
    RAZON = SAMPLE_RATE // TELEPHONY_SAMPLE_RATE


@dataclass
class EventConfiguration:
    # Script clip → kind of the event the harness fires during the response to that clip.
    plan: dict[str, str]
    # Interruption or backchannel clip → (kind, speech-onset sample at 24 kHz).
    events: dict[str, tuple[str, int]]
    long: Response
    stop_s: float
    replica_seg: float


EVENTS: EventConfiguration | None = None


def event_templates(folder: Path) -> tuple[list[Template], dict[str, tuple[str, int]]]:
    """The interruption and backchannel clips, as candidates for the recognizer."""
    from shared.audio import ulaw_a_pcm16
    from shared.audio.interruptions import load_manifest
    from shared.carrier_simulator.script import load_clip_to_publish

    output: list[Template] = []
    events: dict[str, tuple[str, int]] = {}
    for clip_id, e in load_manifest(folder).items():
        ruta = folder / f"{clip_id}.ulaw"
        if not ruta.is_file():
            continue
        _, published, _, _ = load_clip_to_publish(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        pcm = resample_audio(ulaw_a_pcm16(published), TELEPHONY_SAMPLE_RATE, SAMPLE_RATE)
        output.append(Template(
            clip_id=clip_id,
            text=e["text"],
            start=pcm[: int(TEMPLATE_LENGTH_S * SAMPLE_RATE)].astype(np.float64),
            end_of_speech=min(len(pcm), round(float(e["end_of_speech_s"]) * SAMPLE_RATE)),
            largo=len(pcm),
        ))
        events[clip_id] = (e["tipo"], round(float(e["speech_onset_s"]) * SAMPLE_RATE))
    return output, events


def _rms(piece: np.ndarray) -> float:
    return float(np.sqrt(np.mean(piece.astype(np.float64) ** 2))) if len(piece) else 0.0


class ModelThatStops(base.FakeModel):
    """`FakeModel` with a long, interruptible response."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        if EVENTS is None:
            raise RuntimeError("fake model event configuration is missing")
        self.ev = EVENTS
        # Kind of the event of the turn whose response is playing (or about to).
        self.turn_kind: str | None = None
        # True only while the response to a script line is playing: from when its
        # first delta went out, not from when it was scheduled. During the second
        # of waiting before answering, the tail of the script clip is still
        # arriving (the real clips carry soft speech after the manifest's end of
        # speech), and mistaking it for the interruption would silence the model
        # before it speaks.
        self.speaking_turn = False
        # The cut in progress: when it was set up, when it falls silent and when it fell silent.
        self.cutoff: dict[str, Any] | None = None

    # -- What it hears --

    def _moment(self, sample: int) -> float:
        """Arrival of the chunk containing `sample`, plus its position within the chunk."""
        ear = self._ear
        j = bisect.bisect_right(ear._end_arrivals, sample)
        chunk_start = ear._end_arrivals[j - 1] if j > 0 else 0
        return ear._arrivals_t[j] + (sample - chunk_start) / SAMPLE_RATE

    async def send_audio(self, pcm24: bytes) -> None:
        arrival = time.perf_counter()
        if ULAW:
            from poc3_gpt_live.app.bridge import ulaw_a_pcm16 as _decode

            piece = _decode(pcm24)
        else:
            piece = np.frombuffer(pcm24, dtype="<i2")
        if (
            self.speaking_turn and self.turn_kind == "interrupcion"
            and self.cutoff is None and _rms(piece) >= RMS_OF_SPEECH
        ):
            self.cutoff = {"built_at": arrival, "halt_at": arrival + self.ev.stop_s,
                          "t_last_speech": None, "silent": False}
        event = self._ear.receive(piece, arrival)
        if isinstance(event, NotRecognized):
            base.logger.warning("clip not recognized (correlation %.3f)", event.correlation)
            self._annotate({"tipo": "not_recognized", **event.__dict__})
        elif isinstance(event, EndOfSpeech):
            if event.template.clip_id in self.ev.events:
                self._on_event_end(event)
            else:
                self.turn_kind = self.ev.plan.get(event.template.clip_id)
                # A new script line closes the previous turn: a cut left set up
                # there (with no event claiming it) does not carry over to this one.
                self.cutoff = None
                self._on_speech_end(event)

    def _on_event_end(self, event: EndOfSpeech) -> None:
        clip_id = event.template.clip_id
        kind, speech_onset = self.ev.events[clip_id]
        row: dict[str, Any] = {
            "tipo": "event", "clip_id": clip_id, "kind": kind,
            "correlation": round(event.correlation, 4),
            "t_speech_onset": self._moment(event.start_sample + speech_onset),
            "t_end_of_speech": self._moment(event.start_sample + event.template.end_of_speech),
            "target_stop_ms": (
                round(self.ev.stop_s * 1000.0, 3) if kind == "interrupcion" else None
            ),
        }
        self._cola.put_nowait({"type": "session.input_transcript.delta",
                               "delta": " " + event.template.text})
        if kind == "interrupcion":
            self._launch(self._reply(row, row["t_end_of_speech"] + self.ev.replica_seg))
        else:
            self._annotate(row)

    async def _reply(self, row: dict[str, Any], target: float) -> None:
        # The event row is written once its last segment with speech is known.
        while (
            self.cutoff is not None and not self.cutoff["silent"]
            and time.perf_counter() < target
        ):
            await asyncio.sleep(0.01)
        cutoff, self.cutoff = self.cutoff, None
        if cutoff is None or not cutoff["silent"] or cutoff["t_last_speech"] is None:
            # The interruption arrived when it was no longer speaking: there was nothing to cut.
            row["was_not_speaking"] = True
        else:
            row["t_armado"] = cutoff["built_at"]
            row["t_last_speech"] = cutoff["t_last_speech"]
            row["actual_stop_ms"] = round(
                (cutoff["t_last_speech"] - row["t_speech_onset"]) * 1000.0, 3
            )
        self._annotate(row)
        replica = {"tipo": "replica", "clip_id": row["clip_id"], "t_end_of_speech": row["t_end_of_speech"],
                   "target_reply_ms": round(self.ev.replica_seg * 1000.0, 3)}
        await self.speak(target=target, row=replica)

    # -- What it says --

    async def speak(self, *, target: float | None, row: dict[str, Any]) -> None:
        """Like the original, with the long, interruptible response on turns."""
        is_turn = row.get("tipo") == "turn"
        r = self.ev.long if is_turn else self.cfg.response
        offset = self.chance.randrange(BY_DELTA)
        voice = r.pcm.astype(np.int16) if ULAW else base.a_24k(r.pcm)
        pcm = np.concatenate([np.zeros(offset, dtype=np.int16), voice])
        rest = (-len(pcm)) % BY_DELTA
        pcm = np.concatenate([pcm, np.zeros(rest, dtype=np.int16)])
        deltas = [pcm[i:i + BY_DELTA] for i in range(0, len(pcm), BY_DELTA)]
        if ULAW:
            from poc3_gpt_live.app.bridge import pcm16_a_ulaw as _encode

        phrase_start = offset + RAZON * r.phrase_start
        start_mm = None if r.start_mm is None else offset + RAZON * r.start_mm
        primero = start_mm if start_mm is not None else phrase_start
        onset = time.perf_counter() if target is None else target - primero / SAMPLE_RATE
        sends: list[float] = []
        try:
            for i, delta in enumerate(deltas):
                deadline = onset + i * base.S_PER_DELTA
                wait = deadline - time.perf_counter()
                if wait > 0.016:
                    await asyncio.sleep(wait - 0.016)
                while time.perf_counter() < deadline:
                    pass
                cutoff = self.cutoff
                if (
                    is_turn and cutoff is not None and not cutoff["silent"]
                    and time.perf_counter() >= cutoff["halt_at"]
                ):
                    # Falls silent: sends no more deltas.
                    cutoff["t_last_speech"] = self._last_speech_segment(pcm, sends)
                    cutoff["silent"] = True
                    break
                t = time.perf_counter()
                sends.append(t)
                self._cola.put_nowait({
                    "type": "session.output_audio.delta",
                    "delta": base64.b64encode(
                        _encode(delta) if ULAW else delta.astype("<i2").tobytes()
                    ).decode("ascii"),
                })
                if i == 0:
                    if is_turn:
                        self.speaking_turn = True
                    self._cola.put_nowait({
                        "type": "session.output_transcript.delta",
                        "delta": " " + AGENT_TEXT,
                    })
        finally:
            if is_turn:
                self.speaking_turn = False

        def moment(sample: int) -> float:
            j, inside = divmod(sample, BY_DELTA)
            return sends[j] + inside / SAMPLE_RATE

        if sends:
            row["t_primer_delta"] = sends[0]
            row["t_first_segment_with_speech"] = self._first_segment_with_speech(pcm, sends)
            if primero // BY_DELTA < len(sends):
                row["t_phrase_start"] = moment(primero)
        if row.get("tipo") == "turn" and row.get("t_phrase_start") is not None:
            row["real_latency_ms"] = round((row["t_phrase_start"] - row["t_end_of_speech"]) * 1000.0, 3)
        self._annotate(row)

    @staticmethod
    def _speech_segments(pcm: np.ndarray, sends: list[float]) -> list[float]:
        """Playback instant of each 20 ms segment with speech already sent."""
        n = len(sends) * BY_DELTA
        output = []
        for k in range(0, n, SAMPLES_PER_SEGMENT):
            if _rms(pcm[k:k + SAMPLES_PER_SEGMENT]) >= RMS_WITH_SPEECH:
                j, inside = divmod(k, BY_DELTA)
                output.append(sends[j] + inside / SAMPLE_RATE)
        return output

    def _last_speech_segment(self, pcm: np.ndarray, sends: list[float]) -> float | None:
        segments = self._speech_segments(pcm, sends)
        return segments[-1] if segments else None

    def _first_segment_with_speech(self, pcm: np.ndarray, sends: list[float]) -> float | None:
        segments = self._speech_segments(pcm, sends)
        return segments[0] if segments else None


async def run_server(args: argparse.Namespace) -> None:
    global EVENTS

    import uvicorn

    from poc3_gpt_live.app import bridge, telephony
    from validation.validate_harness import reproducible_ones

    _use_format(args.format_kind)
    folder = Path(args.clips)
    of_scenario = [c for c in reproducible_ones() if c.startswith(f"e{args.scenario}_")]
    of_events, events = event_templates(Path(args.interruption_clips))
    truth = Path(args.truth)
    truth.parent.mkdir(parents=True, exist_ok=True)
    truth.write_text("", encoding="utf-8")
    voice = choose_voice(folder, args.voice)
    base.CONFIGURATION = base.Configuration(
        candidate_ones=templates(folder, of_scenario, SAMPLE_RATE) + of_events,
        response=build_response(voice, level_dbfs=args.level_dbfs, soft_onset=False),
        delay_s=args.delay_ms / 1000.0,
        truth=truth,
        seed=args.seed,
        format_kind=args.format_kind,
    )
    EVENTS = EventConfiguration(
        plan=json.loads(Path(args.plan).read_text(encoding="utf-8")),
        events=events,
        long=Response(continuous_speech(voice, LONG_DURATION_S, args.level_dbfs), 0, None),
        stop_s=args.stop_ms / 1000.0,
        replica_seg=args.replica_ms / 1000.0,
    )
    recordings = Path(args.recordings)
    recordings.mkdir(parents=True, exist_ok=True)
    # The only things changed in the bridge: the model and the folder where it
    # leaves its files. Not a single line of PoC3's code.
    bridge.GptLiveClient = ModelThatStops  # type: ignore[misc]
    telephony.RECORDINGS_PATH = recordings  # type: ignore[attr-defined]

    servidor: Any = None

    def _on_finish(call_sid: str, destination: Path) -> None:
        if servidor is not None:
            servidor.should_exit = True

    app = telephony.create_app(base.settings_without_credentials(args.format_kind), text_echo=True, on_finish=_on_finish)
    servidor = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    )
    await servidor.serve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
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
    p.add_argument("--recordings", required=True, help="folder for the bridge's files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format-kind", choices=("pcm24", "pcmu"), default="pcm24")
    args = p.parse_args(argv)
    logging.getLogger("websockets").setLevel(logging.CRITICAL)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
