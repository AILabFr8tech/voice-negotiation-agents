"""GPT-Live over WebSocket and over WebRTC: the same recorded lines, the same measurement.

Tests whether the connection path explains PoC3 latency. The campaign used the
server WebSocket (`wss://api.openai.com/v1/live/sessions`); ChatGPT in the
browser uses WebRTC. Each session opens GPT-Live with one-line instructions,
no tools, sends it four recorded carrier lines in real time (μ-law at 8 kHz;
over WebRTC, the same samples upsampled to 48 kHz) and measures, for each line:

- `primer_audio_ms`: from the end of its speech to the first agent audio block
  with speech (−44 dBFS, the harness threshold) that reaches this process. Over
  WebRTC the audio has already gone through the receive buffer.
- `last_word_ms`: from the end of its speech to the arrival of the transcript
  of its last word.
- `model_ms`: on the model's clock, from the end of the line's last word to
  the start of the agent's first word after the line started.

**It costs money**: it opens paid GPT-Live sessions (about 50 s each). It takes
the key from `.env` with `load_settings`, like the bridge, and does not print
it. WebRTC needs `aiortc`. The WebRTC session is created with
`POST /v1/live/sessions` and a JSON body
`{"session": …, "transport": {"type": "webrtc", "sdp": <offer>}}`; the response
carries the answer SDP in `transport.sdp`. Events arrive over the
`oai-events` data channel.

Appends rows to `transport_gpt_live.csv` in `--output`. The ones from
October 21, 2026 are three sessions per path, alternated.

    uv run --project poc3_gpt_live python -m validation.transport_gpt_live ws
    uv run --project poc3_gpt_live python -m validation.transport_gpt_live rtc
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import datetime as dt
import fractions
import json
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
CLIPS: Final[Path] = RAIZ / "shared" / "test_scenarios" / "clips"
LINES: Final[tuple[str, ...]] = ("e1_t2", "e1_t3", "e2_t2", "e1_t4")
INITIAL_SILENCE_S: Final[float] = 2.0
WAIT_AFTER_LINE_S: Final[float] = 9.0
SPEECH_THRESHOLD: Final[float] = 10 ** (-44 / 20) * 32768
INSTRUCTIONS: Final[str] = (
    "Eres un agente de fletes de Freight Technologies que habla por teléfono con un "
    "transportista llamado Carlos. Contesta en español, en una o dos frases cortas."
)
COLUMNS: Final[list[str]] = ["date", "via", "session", "clip", "primer_audio_ms", "last_word_ms", "model_ms"]


def _bridge():
    from poc3_gpt_live.app import bridge
    return bridge


def end_of_speech(pcm: np.ndarray, sample_rate: int) -> float:
    """End of the last 20 ms window with speech (−40 dBFS) in the line's audio."""
    v = int(sample_rate * 0.02)
    n = len(pcm) // v
    rms = np.sqrt((pcm[: n * v].astype(float).reshape(n, v) ** 2).mean(axis=1))
    return (np.nonzero(rms >= 10 ** (-40 / 20) * 32768)[0][-1] + 1) * 0.02


def script() -> list[tuple[str, np.ndarray, float]]:
    output = []
    for clip in LINES:
        pcm = np.asarray(_bridge().ulaw_a_pcm16((CLIPS / f"{clip}.ulaw").read_bytes()), dtype="<i2")
        output.append((clip, pcm, end_of_speech(pcm, 8000)))
    return output


class Record:
    def __init__(self) -> None:
        self.voice: list[float] = []
        self.ends: list[tuple[str, float]] = []
        self.events: list[tuple[float, dict[str, Any]]] = []

    def audio(self, t: float, pcm: np.ndarray) -> None:
        if pcm.size and np.sqrt((pcm.astype(float) ** 2).mean()) >= SPEECH_THRESHOLD:
            self.voice.append(t)

    def event(self, t: float, ev: dict[str, Any]) -> None:
        if ev.get("type", "").endswith("transcript.delta"):
            self.events.append((t, ev))
        elif ev.get("type") == "error":
            print("ERROR", ev)

    def results(self) -> list[dict[str, Any]]:
        entry = [(t, e) for t, e in self.events if e["type"] == "session.input_transcript.delta"]
        output = [(t, e) for t, e in self.events if e["type"] == "session.output_transcript.delta"]
        rows = []
        for i, (clip, end) in enumerate(self.ends):
            tope = self.ends[i + 1][1] - 1.0 if i + 1 < len(self.ends) else float("inf")
            primero = next((t for t in self.voice if end < t < tope), None)
            mias = [(t, e) for t, e in entry if end - 4 < t < end + 2.5]
            model_name = ultima = None
            if mias:
                ultima = round((max(t for t, _ in mias) - end) * 1000)
                s_end = max(e["end_ms"] for _, e in mias)
                since = min(e["start_ms"] for _, e in mias)
                agent = [e["start_ms"] for t, e in output if t < tope and e["start_ms"] >= since]
                model_name = min(agent) - s_end if agent else None
            rows.append({"clip": clip, "primer_audio_ms": None if primero is None else round((primero - end) * 1000),
                          "last_word_ms": ultima, "model_ms": model_name})
        return rows


async def via_websocket(settings, reg: Record) -> None:
    import websockets

    bridge = _bridge()
    session = {"model": settings.live_model, "instructions": INSTRUCTIONS,
              "audio": {"output": {"voice": settings.voice}, "format": {"type": "audio/pcmu", "rate": 8000}}}
    async with websockets.connect(settings.url_live(), additional_headers=settings.headers_live(), max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "event_id": "start_1", "session": session}))

        async def leer() -> None:
            async for m in ws:
                t = time.time()
                ev = json.loads(m)
                if ev.get("type") == "session.output_audio.delta":
                    reg.audio(t, np.asarray(bridge.ulaw_a_pcm16(base64.b64decode(ev["delta"]))))
                else:
                    reg.event(t, ev)

        lector = asyncio.create_task(leer())
        await asyncio.sleep(1.0)
        t0 = time.perf_counter()
        sent_one = 0.0

        async def send(pcm: np.ndarray) -> None:
            nonlocal sent_one
            for i in range(0, len(pcm), 160):
                piece = pcm[i:i + 160]
                audio = base64.b64encode(bridge.pcm16_a_ulaw(piece.astype("<i2"))).decode()
                await ws.send(json.dumps({"type": "session.input_audio.append", "audio": audio}))
                sent_one += len(piece) / 8000
                wait = t0 + sent_one - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)

        await send(np.zeros(int(8000 * INITIAL_SILENCE_S), dtype="<i2"))
        for clip, pcm, end in script():
            start = time.time()
            await send(pcm)
            reg.ends.append((clip, start + end))
            await send(np.zeros(int(8000 * WAIT_AFTER_LINE_S), dtype="<i2"))
        await ws.send(json.dumps({"type": "session.close", "event_id": "cierre"}))
        await asyncio.sleep(1.0)
        lector.cancel()


async def via_webrtc(settings, reg: Record) -> None:
    import aiohttp
    import av
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

    text_lines = script()
    upsample_factor = 6  # 8 kHz -> 48 kHz
    plan: list[tuple[str | None, np.ndarray, float]] = [(None, np.zeros(int(48000 * INITIAL_SILENCE_S)), 0.0)]
    for clip, pcm, end in text_lines:
        plan.append((clip, np.interp(np.arange(len(pcm) * upsample_factor) / upsample_factor, np.arange(len(pcm)), pcm.astype(float)), end))
        plan.append((None, np.zeros(int(48000 * WAIT_AFTER_LINE_S)), 0.0))
    total = np.concatenate([p for _, p, _ in plan]).astype("<i2")
    marks, pos = [], 0
    for clip, p, end in plan:
        if clip is not None:
            marks.append((pos / 48000 + end, clip))
        pos += len(p)

    class Track(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.i = 0
            self.t0: float | None = None

        async def recv(self):
            if self.t0 is None:
                self.t0 = time.time()
                reg.ends.extend((clip, self.t0 + since) for since, clip in marks)
            wait = self.t0 + self.i * 0.02 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            piece = total[self.i * 960:(self.i + 1) * 960]
            if piece.size < 960:
                piece = np.zeros(960, dtype="<i2")
            frame = av.AudioFrame.from_ndarray(piece.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = 48000
            frame.pts = self.i * 960
            frame.time_base = fractions.Fraction(1, 48000)
            self.i += 1
            return frame

    pc = RTCPeerConnection()
    pc.addTrack(Track())
    canal = pc.createDataChannel("oai-events")

    @canal.on("message")
    def _(m):
        reg.event(time.time(), json.loads(m))

    async def listen(track) -> None:
        while True:
            try:
                fr = await track.recv()
            except Exception:
                return
            reg.audio(time.time(), fr.to_ndarray().astype("<i2").reshape(-1))

    @pc.on("track")
    def _(track):
        if track.kind == "audio":
            asyncio.ensure_future(listen(track))

    await pc.setLocalDescription(await pc.createOffer())
    session = {"model": settings.live_model, "instructions": INSTRUCTIONS, "audio": {"output": {"voice": settings.voice}}}
    async with aiohttp.ClientSession() as http:
        async with http.post("https://api.openai.com/v1/live/sessions",
                             json={"session": session, "transport": {"type": "webrtc", "sdp": pc.localDescription.sdp}},
                             headers=settings.headers_live()) as r:
            cuerpo = await r.text()
            if r.status >= 300:
                raise SystemExit(f"WebRTC: HTTP {r.status}: {cuerpo[:500]}")
    response = json.loads(cuerpo)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=response["transport"]["sdp"], type="answer"))
    await asyncio.sleep(len(total) / 48000 + 2.0)
    await pc.close()


async def launch(via: str, sessions: int, output: Path) -> None:
    from poc3_gpt_live.app.config import load_settings

    settings = load_settings()
    ruta = output / "transport_gpt_live.csv"
    new = not ruta.is_file()
    output.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for s in range(sessions):
            reg = Record()
            await (via_websocket if via == "ws" else via_webrtc)(settings, reg)
            for row in reg.results():
                row.update(date=dt.date.today().isoformat(), via=via, session=s + 1)
                w.writerow(row)
                print(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("via", choices=("ws", "rtc"))
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--output", type=Path, default=DATA)
    args = ap.parse_args(argv)
    asyncio.run(launch(args.via, args.sessions, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
