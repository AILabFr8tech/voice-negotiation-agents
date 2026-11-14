"""A fake GPT-Live that answers with a known delay, behind PoC3's bridge.

It is the twin of `validation.fake_agent` for PoC3's path. There the fake agent
takes the place of all of PoC1 on the other side of the WebSocket; here the
bridge is the production one (`telephony.create_app`, with its Media Stream, its
resampling from 8 kHz to 24 kHz and back, its latency collector and its text
echo) and the only fake thing is the model: `bridge.GptLiveClient` is replaced
by `FakeModel`, just as PoC3's tests do (`tests/conftest.py`). What is not
validated is the internet segment between the bridge and OpenAI: with a fake
model that segment does not exist.

The fake model:

1. **Recognizes each clip by its waveform.** The audio reaches the bridge
   already resampled to 24 kHz, so the bytes cannot serve as a fingerprint as in
   PoC1. The onset of the clip is correlated against that of each candidate line
   of the scenario, resampled the same way; the peak gives which clip it is and
   at which sample it started.
2. **Timestamps the real end of speech**: the instant the chunk of audio
   containing the clip's last sample with speech (its manifest
   `end_of_speech_s`) reached it. That is when the real model would have that
   sample.
3. **Answers exactly D ms later** with 100 ms `session.output_audio.delta`
   events at real-time pacing, which is the cadence of the model in its general
   availability version. The onset of the response falls at a random position
   within the delta.

It speaks GPT-Live's **public** protocol (the one the bridge speaks):
`session.output_audio.delta` with `delta` and no timings,
`session.input/output_transcript.delta` with `delta`, and no `turn.*`.

With `--format-kind pcmu` the session is 8 kHz μ-law, like the October campaign:
the bridge does not resample, the fake model recognizes the clip at 8 kHz and
answers in 100 ms μ-law deltas (800 bytes), the same size the public API sends
in that format.

Everything is timestamped with `time.perf_counter()`, the same clock the harness
uses in the other process. The ground truth of each turn goes to a `.jsonl`.

It is launched as a separate process, just like the bridge in the campaign:

    python -m validation.fake_model_poc3 --port 8765 --clips ... --scenario 1 ...
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
from typing import Any, AsyncIterator, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from validation.fake_agent import AGENT_TEXT, Response, build_response, choose_voice
from validation.recognizer import (
    EndOfSpeech,
    NotRecognized,
    Template,
    Recognizer,
    templates,
    resample_audio,
)

MODEL_SAMPLE_RATE: Final[int] = 24_000
RAZON: Final[int] = MODEL_SAMPLE_RATE // TELEPHONY_SAMPLE_RATE

# The general availability model delivers 100 ms deltas (`docs/hallazgo_gpt_live_ga.md`).
SAMPLES_PER_DELTA: Final[int] = MODEL_SAMPLE_RATE // 10
S_PER_DELTA: Final[float] = SAMPLES_PER_DELTA / MODEL_SAMPLE_RATE

# Through the bridge the clip arrives almost intact: the correlation peak comes out ≈1.
MIN_CORRELATION: Final[float] = 0.90

logger = logging.getLogger(__name__)


def a_24k(pcm8: np.ndarray) -> np.ndarray:
    """From 8 kHz to 24 kHz with the same library and quality as the bridge."""
    return resample_audio(pcm8, TELEPHONY_SAMPLE_RATE, MODEL_SAMPLE_RATE)


# --------------------------------------------------------------------------
# The fake model
# --------------------------------------------------------------------------


@dataclass
class Configuration:
    candidate_ones: list[Template]
    response: Response
    delay_s: float
    truth: Path
    seed: int
    # `pcm24` (PCM16 24 kHz, the September one) or `pcmu` (μ-law 8 kHz).
    format_kind: str = "pcm24"

    @property
    def sample_rate(self) -> int:
        return TELEPHONY_SAMPLE_RATE if self.format_kind == "pcmu" else MODEL_SAMPLE_RATE

    @property
    def samples_per_delta(self) -> int:
        return self.sample_rate // 10


# Set by `run_server` before starting the bridge. `FakeModel` reads it when
# constructed, because the bridge constructs it with `GptLiveClient`'s signature.
CONFIGURATION: Configuration | None = None


class FakeModel:
    """Impersonates `bridge.GptLiveClient`: same interface, no network."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        initial_session: dict[str, Any],
        *,
        census: Any = None,
    ) -> None:
        if CONFIGURATION is None:
            raise RuntimeError("fake model configuration is missing")
        self.cfg = CONFIGURATION
        self.census = census
        self.usage: dict[str, Any] = {}
        self.closed = False
        self.session_confirmed: dict[str, Any] = {}
        self.chance = random.Random(self.cfg.seed)
        self._cola: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._t0 = time.perf_counter()
        self._ear = Recognizer(
            self.cfg.candidate_ones, self.cfg.sample_rate, min_correlation=MIN_CORRELATION
        )
        self._tasks: set[asyncio.Task[None]] = set()

    # -- Lifecycle, with GptLiveClient's signature --

    async def connect(self) -> dict[str, Any]:
        self.session_confirmed = {"id": "fake_sess", "status": "started"}
        return self.session_confirmed

    async def add_context(self, text: str, **_: Any) -> None:
        """The opening context: the model greets, like the real one."""
        self._launch(self.speak(target=None, row={"tipo": "greeting"}))

    async def answer_function(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def proceed_delegation(self) -> None:
        return None

    async def shut(self) -> None:
        if self.closed:
            return
        self.closed = True
        for tarea in list(self._tasks):
            tarea.cancel()
        self.usage = {"seconds": round(time.perf_counter() - self._t0, 1)}
        await self._cola.put(None)

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._cola.get()
            if event is None:
                return
            yield event

    def _launch(self, coroutine: Any) -> None:
        tarea = asyncio.create_task(coroutine)
        self._tasks.add(tarea)
        tarea.add_done_callback(self._tasks.discard)

    # -- What it hears --

    async def send_audio(self, pcm24: bytes) -> None:
        arrival = time.perf_counter()
        if self.cfg.format_kind == "pcmu":
            from poc3_gpt_live.app.bridge import ulaw_a_pcm16

            samples = ulaw_a_pcm16(pcm24)
        else:
            samples = np.frombuffer(pcm24, dtype="<i2")
        event = self._ear.receive(samples, arrival)
        if isinstance(event, NotRecognized):
            logger.warning("clip not recognized (correlation %.3f)", event.correlation)
            self._annotate({"tipo": "not_recognized", **event.__dict__})
        elif isinstance(event, EndOfSpeech):
            self._on_speech_end(event)

    def _on_speech_end(self, event: EndOfSpeech) -> None:
        template = event.template
        row = {
            "tipo": "turn",
            "clip_id": template.clip_id,
            "correlation": round(event.correlation, 4),
            "second_correlation": round(event.second_one, 4),
            "t_first_sample_clip": event.t_first_sample,
            "t_end_of_speech": event.t_end_of_speech,
            "target_delay_ms": round(self.cfg.delay_s * 1000.0, 3),
        }
        self._cola.put_nowait({
            "type": "session.input_transcript.delta",
            "delta": " " + template.text,
        })
        self._launch(self.speak(target=event.t_end_of_speech + self.cfg.delay_s, row=row))

    # -- What it says --

    async def speak(self, *, target: float | None, row: dict[str, Any]) -> None:
        """Emit the response with its onset at `target` (perf_counter).

        `target=None` is the greeting: it goes out as soon as possible.
        """
        r = self.cfg.response
        ulaw = self.cfg.format_kind == "pcmu"
        sample_rate, by_delta = self.cfg.sample_rate, self.cfg.samples_per_delta
        reason = sample_rate // TELEPHONY_SAMPLE_RATE
        offset = self.chance.randrange(by_delta)
        voice = r.pcm.astype(np.int16) if ulaw else a_24k(r.pcm)
        pcm = np.concatenate([np.zeros(offset, dtype=np.int16), voice])
        rest = (-len(pcm)) % by_delta
        pcm = np.concatenate([pcm, np.zeros(rest, dtype=np.int16)])
        deltas = [pcm[i:i + by_delta] for i in range(0, len(pcm), by_delta)]

        phrase_start = offset + reason * r.phrase_start
        start_mm = None if r.start_mm is None else offset + reason * r.start_mm
        primero = start_mm if start_mm is not None else phrase_start
        onset = (
            time.perf_counter() if target is None
            else target - primero / sample_rate
        )
        if ulaw:
            from poc3_gpt_live.app.bridge import pcm16_a_ulaw
        sends: list[float] = []
        for i, delta in enumerate(deltas):
            deadline = onset + i * S_PER_DELTA
            wait = deadline - time.perf_counter()
            if wait > 0.016:
                await asyncio.sleep(wait - 0.016)
            while time.perf_counter() < deadline:
                pass
            t = time.perf_counter()
            sends.append(t)
            self._cola.put_nowait({
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(
                    pcm16_a_ulaw(delta) if ulaw else delta.astype("<i2").tobytes()
                ).decode("ascii"),
            })
            if i == 0:
                self._cola.put_nowait({
                    "type": "session.output_transcript.delta",
                    "delta": " " + AGENT_TEXT,
                })

        def moment(sample: int) -> float:
            j, inside = divmod(sample, by_delta)
            return sends[j] + inside / sample_rate

        row.update({
            "offset_samples": offset,
            "t_phrase_start": moment(phrase_start),
            "t_mm_start": None if start_mm is None else moment(start_mm),
            "t_primer_delta": sends[0],
        })
        if row.get("tipo") == "turn":
            row["real_latency_ms"] = round((moment(primero) - row["t_end_of_speech"]) * 1000.0, 3)
            row["real_phrase_latency_ms"] = round(
                (row["t_phrase_start"] - row["t_end_of_speech"]) * 1000.0, 3
            )
        self._annotate(row)

    def _annotate(self, row: dict[str, Any]) -> None:
        with self.cfg.truth.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# The production bridge, with the fake model
# --------------------------------------------------------------------------


def settings_without_credentials(format_kind: str = "pcm24") -> Any:
    """The bridge settings. Nothing goes out to the network, so the key is fake."""
    from poc3_gpt_live.app.config import (
        DEFAULT_DELEGATION_MODEL,
        DEFAULT_LIVE_MODEL,
        Settings,
    )

    return Settings(
        openai_api_key="sk-validacion",
        live_model=DEFAULT_LIVE_MODEL,
        voice="marin",
        delegation_model=DEFAULT_DELEGATION_MODEL,
        audio_format=format_kind,
    )


async def run_server(args: argparse.Namespace) -> None:
    global CONFIGURATION

    import uvicorn

    from poc3_gpt_live.app import bridge, telephony
    from validation.validate_harness import reproducible_ones

    folder = Path(args.clips)
    of_scenario = [c for c in reproducible_ones() if c.startswith(f"e{args.scenario}_")]
    truth = Path(args.truth)
    truth.parent.mkdir(parents=True, exist_ok=True)
    truth.write_text("", encoding="utf-8")
    CONFIGURATION = Configuration(
        candidate_ones=templates(
            folder, of_scenario, TELEPHONY_SAMPLE_RATE if args.format_kind == "pcmu" else MODEL_SAMPLE_RATE
        ),
        response=build_response(
            choose_voice(folder, args.voice), level_dbfs=args.level_dbfs,
            soft_onset=args.soft_onset,
        ),
        delay_s=args.delay_ms / 1000.0,
        truth=truth,
        seed=args.seed,
        format_kind=args.format_kind,
    )
    recordings = Path(args.recordings)
    recordings.mkdir(parents=True, exist_ok=True)
    # The only things changed in the bridge: the model and the folder where it
    # leaves its files. Not a single line of PoC3's code.
    bridge.GptLiveClient = FakeModel  # type: ignore[misc]
    telephony.RECORDINGS_PATH = recordings  # type: ignore[attr-defined]

    servidor: Any = None

    def _on_finish(call_sid: str, destination: Path) -> None:
        if servidor is not None:
            servidor.should_exit = True

    app = telephony.create_app(settings_without_credentials(args.format_kind), text_echo=True, on_finish=_on_finish)
    servidor = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    )
    await servidor.serve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--scenario", type=int, required=True)
    p.add_argument("--voice", required=True, help="clip used as the agent's voice")
    p.add_argument("--delay-ms", type=float, required=True)
    p.add_argument("--level-dbfs", type=float, default=-20.0)
    p.add_argument("--soft-onset", action="store_true")
    p.add_argument("--format-kind", choices=("pcm24", "pcmu"), default="pcm24",
                   help="session format: pcm24 (September) or pcmu (October, no resampling)")
    p.add_argument("--truth", required=True, help="jsonl where what actually happened is written")
    p.add_argument("--recordings", required=True, help="folder for the bridge's files")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.getLogger("websockets").setLevel(logging.CRITICAL)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
