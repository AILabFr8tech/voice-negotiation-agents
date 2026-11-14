"""The expensive scaffolding of PoC3's tests: a whole run, at $0.

Starting the production bridge, acting out a script against it and waiting for
it to write its artifacts takes ~16 s. Doing it once per test file multiplies
that without buying anything, so the run lives here, with session scope, and it
is shared by `test_harness.py` (which looks at what the harness observed) and
`test_verification.py` (which looks at what was left on disk).

The alpha is fake. Everything else —the WebSocket, the server, the whole
bridge, the resampling from 8k to 24k and back, the negotiation engine, the
zeros written to disk and the call's JSON— is the production one. The model is
fake because each run against the real one costs ~$0.13, and a test that costs
money does not get run.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import Any, AsyncIterator
from unittest import mock

import numpy as np
import pytest

from poc3_gpt_live.app import harness, bridge, media_streams as ms, telephony
from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    DEFAULT_LIVE_MODEL,
    Settings,
)
from poc3_gpt_live.app.marks import marks_de, zeros_path
from shared.carrier_simulator import ScriptStep

ALPHA_SAMPLE_RATE = 24_000

# What the fake agent answers. Long on purpose: above `BACKCHANNEL_CAP_S`, so
# that the burst classifier accepts it without consulting the text and the test
# does not depend on the backchannel word list.
RESPONSE_SECONDS = 2.6

# Fragments in the alpha's real format: the separator goes at the START of the
# fragment, not at the end. Taken from the census of `CA221584…`.
FRAGMENTED_RESPONSE = (" Claro,", " con", " mucho", " gusto,", " le", " explico.")
ASSEMBLED_RESPONSE = "Claro, con mucho gusto, le explico."


def ulaw(audio_ms: int, amplitude: int = 6_000) -> bytes:
    """An audible tone in 8 kHz μ-law, of the requested length."""
    n = int(8_000 * audio_ms / 1000)
    t = np.arange(n) / 8_000.0
    return ms.pcm16_a_ulaw((np.sin(2 * np.pi * 300 * t) * amplitude).astype(np.int16))


def pcm24(seconds: float, amplitude: int = 8_000) -> bytes:
    """The agent's voice, in the alpha's domain: PCM16 at 24 kHz."""
    n = int(ALPHA_SAMPLE_RATE * seconds)
    t = np.arange(n) / float(ALPHA_SAMPLE_RATE)
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype("<i2").tobytes()


class FakeAlpha:
    """A `GptLiveClient` that answers once it stops being interrupted.

    It does not imitate the model: it imitates the **protocol**. It hears the
    audio the bridge sends it, and when it detects that the carrier spoke and
    went quiet, it emits the response as GPT-Live would emit it over the public
    API — chunked `session.output_audio` deltas and `session.output_transcript`
    fragments with the space at the start, without `turn.*`.
    """

    # How much speech it has to hear before considering that there was a line,
    # and how much silence afterwards before answering it.
    MIN_SPEECH_S = 0.2
    SILENCE_TO_ANSWER_S = 0.5

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        initial_session: dict[str, Any],
        *,
        census: Any = None,
    ) -> None:
        self.census = census
        self.usage: dict[str, Any] = {}
        self.closed = False
        self.replies = 0
        self.contexts: list[str] = []
        self._cola: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._speech_s = 0.0
        self._silence_s = 0.0

    # -- Lifecycle --

    async def connect(self) -> dict[str, Any]:
        return {"id": "fake_sess", "status": "started"}

    async def add_context(self, text: str, **_: Any) -> None:
        """The opening context. The agent answers with a greeting, like the real one."""
        self.contexts.append(text)
        self._answer(" Buenos días, le llamo de parte de Freight Technologies.")

    async def answer_function(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def proceed_delegation(self) -> None:
        return None

    async def shut(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.usage = {"seconds": 12.0}
        await self._cola.put(None)

    # -- The two directions --

    async def send_audio(self, pcm24: bytes) -> None:
        seconds = len(pcm24) / 2.0 / ALPHA_SAMPLE_RATE
        if bridge.has_speech(pcm24):
            self._speech_s += seconds
            self._silence_s = 0.0
            return

        self._silence_s += seconds
        if (
            self._speech_s >= self.MIN_SPEECH_S
            and self._silence_s >= self.SILENCE_TO_ANSWER_S
        ):
            self._speech_s = 0.0
            self._silence_s = 0.0
            self._answer()

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._cola.get()
            if event is None:
                return
            yield event

    # -- What a response sounds like --

    def _answer(self, greeting: str = "") -> None:
        self.replies += 1
        fragments = (greeting,) if greeting else FRAGMENTED_RESPONSE

        # The deltas are chunked, like the alpha's: ~100 ms each.
        chunked = 0.1
        remaining = RESPONSE_SECONDS
        while remaining > 0:
            piece = min(chunked, remaining)
            self._cola.put_nowait(
                {
                    "type": bridge.EV_OUTPUT_AUDIO_DELTA,
                    "delta": base64.b64encode(pcm24(piece)).decode("ascii"),
                }
            )
            remaining -= piece

        for fragment in fragments:
            # The text travels in `delta`, at the top level: that is what
            # `bridge.transcript_text` reads.
            self._cola.put_nowait(
                {"type": bridge.EV_OUTPUT_TRANSCRIPT, "delta": fragment}
            )


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-prueba",
        live_model=DEFAULT_LIVE_MODEL,
        voice="marin",
        delegation_model=DEFAULT_DELEGATION_MODEL,
    )


async def start_bridge(port: int) -> tuple[Any, Any]:
    """The production bridge, in this same process and on a free port."""
    import uvicorn

    app = telephony.create_app(_settings(), text_echo=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    servidor = uvicorn.Server(config)
    tarea = asyncio.create_task(servidor.serve())
    for _ in range(200):
        if servidor.started:
            return servidor, tarea
        await asyncio.sleep(0.05)
    raise AssertionError("el puente no levantó")


SCRIPT = [
    ScriptStep(turn="T1", text="Sí, soy Carlos.", clip_id="c1", ulaw=ulaw(500)),
    ScriptStep(turn="T2", text="Va, cuénteme.", clip_id="c2", ulaw=ulaw(500)),
]


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """A single run for the whole test file: starting it takes ~16 s."""
    folder = tmp_path_factory.mktemp("harness")
    recordings = tmp_path_factory.mktemp("recordings")
    port = harness._free_port()

    async def _launch():
        servidor, tarea = await start_bridge(port)
        try:
            return await harness.act_out_script(
                f"ws://127.0.0.1:{port}/{telephony.RUTA_WS}",
                script=SCRIPT,
                folder=folder,
                label="prueba",
                wait_after_clip=6.0,
            )
        finally:
            servidor.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(tarea, timeout=20)

    with mock.patch.object(bridge, "GptLiveClient", FakeAlpha), mock.patch.object(
        telephony, "RECORDINGS_PATH", recordings
    ):
        observed, call_sid = asyncio.run(_launch())

    return observed, call_sid, folder, recordings
