# PoC1 — Pipecat

The first of the 3 PoCs in the comparison: Pipecat's voice pipeline on top of
Twilio Media Streams, with negotiation done through tool calling. The
negotiation logic lives in [`shared/negotiation_engine/`](../shared/negotiation_engine/);
this package only wraps it. The measurements in the report were run with the
harness ([`app/harness.py`](app/harness.py)), which acts out the script against
this same pipeline without a phone.

## Modules

| Module | What it does |
|---|---|
| [`app/config.py`](app/config.py) | Reads the repo-root `.env`. Validates separately what the smoke test, voice and telephony each require. |
| [`app/tools.py`](app/tools.py) | The 4 schemas the LLM sees and the handlers that delegate to the engine. |
| [`app/pipeline.py`](app/pipeline.py) | Services (STT/LLM/TTS), VAD + end of turn, context and the smoke test's text pipeline. |
| [`app/latency.py`](app/latency.py) | Per-stage latency capture and capture of what the agent spoke. |
| [`app/marks.py`](app/marks.py) | The real zero of each turn, received from the harness out of band. |
| [`app/telephony.py`](app/telephony.py) | TwiML, Media Stream and the voice pipeline. FastAPI server. Does not dial. |
| [`app/media_streams.py`](app/media_streams.py) | Media Streams client: talks to `/ws` the way Twilio would. Used by the harness. |
| [`app/harness.py`](app/harness.py) | Harness: acts out the script against the full pipeline, without a phone. |
| [`app/zero_diagnostics.py`](app/zero_diagnostics.py) | How far PoC1's zero drifts from the harness's real instant. |
| [`app/smoke.py`](app/smoke.py) | Smoke test, levels 0 and 1. |
| [`app/call.py`](app/call.py) | Dials the outbound call. The only module that spends on telephony. |

The trip data, the call state and the prompt interpolation are shared by all
3 PoCs: [`shared/negotiation_engine/trip.py`](../shared/negotiation_engine/trip.py),
[`shared/negotiation_engine/session.py`](../shared/negotiation_engine/session.py) and
[`shared/system_prompt/prompt.py`](../shared/system_prompt/prompt.py).

## Architecture of this PoC

| Layer | Technology |
|---|---|
| Telephony | Twilio, Media Streams |
| Media | WebSocket (FastAPI) |
| STT | Deepgram `nova-3` / `multi` (the bake-off against ElevenLabs Scribe was not run) |
| LLM | gpt-4.1-mini (fixed for all 3 PoCs on August 3, 2026; `OPENAI_MODEL` in `.env`) |
| TTS | ElevenLabs Flash |
| End of turn | Silero VAD + `LocalSmartTurnAnalyzerV3` |
| Negotiation | `shared/negotiation_engine` via tool calling |

## Environment

Managed with [uv](https://docs.astral.sh/uv/). Python 3.12 (not 3.13/3.14: it
is the version the lock was resolved with).

```bash
uv sync                 # installs from uv.lock
uv run python -V        # 3.12.13
```

`pyproject.toml` + `uv.lock` are committed, so that the 4 PoCs are
reproducible and their metrics comparable.

### Notes on the pipecat-ai extras

- **There is no `twilio` extra** in pipecat-ai 1.6.0. The Media Streams
  transport is the `websocket` extra; `TwilioFrameSerializer` already ships in
  the core.
- **`LocalSmartTurnAnalyzerV3` does not need its own extra.** It only imports
  `onnxruntime`, `numpy` and `soxr`, which come with the `silero` extra, and
  the `smart-turn-v3.2-cpu.onnx` model is bundled inside pipecat. The
  `local-smart-turn` extra is for the **v2/CoreML** analyzers and pulls in
  `torch`, `torchaudio` and `coremltools` — the last one has no Windows support.

## Tests and smoke test

Both commands run from the repo **root** (that is where `pytest.ini` is, and
`poc1_pipecat` is a uv virtual project: it is not installed, it is imported from
the root).

```bash
uv run --project poc1_pipecat pytest
uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke
```

The **level 0** smoke test does not touch the network or spend credits. It
checks that the `.env` loads, that the system prompt interpolates without
leaving placeholders, that the maximum price appears neither in the prompt nor
in the persisted log (and that the leak detector does fire when there is one),
that the 4 schemas come from the real catalog and enum, that the handlers give
the correct verdicts in a full negotiation against the engine, and that the
whole pipeline assembles —VAD and end-of-turn ONNX models included— with the 4
functions registered on the LLM.

### Level 1: real APIs, no phone

```bash
uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke --online --accept-cost
uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke --online --accept-cost --loopback-stt
```

Negotiates a fixed 4-turn script against the real gpt-4.1-mini and ElevenLabs.
There is no telephony: no number is dialed. At the end it reports the history
of validated offers, the compensations offered, the outcome, the TTFB per
service and a WAV of the agent's voice in `recordings/` (gitignored).

The most important thing it reports is **compliance with the financial
limit**: it searches what the agent said for amounts written as digits and
checks that every one of them came from an offer that `proponer_precio`
accepted.

With `--loopback-stt` it feeds that same audio to Deepgram, which is the way to
check the STT without a phone call. It feeds the audio in real time, so the run
takes as long as the audio lasts.

**It costs money** (~$0.10 per run): that is why `--accept-cost` is
mandatory. Without that flag, `--online` only prints the estimate and exits.
Breakdown and assumptions in [`docs/costos_poc1.md`](../docs/costos_poc1.md).

## Level 2: the real phone call

Three pieces, two terminals. The tunnel and the server stay up between calls;
only the dialer is rerun for each call.

```bash
# Terminal 1 — the tunnel to the static PUBLIC_URL domain in the .env
ngrok http 8000 --domain=<your-domain>.ngrok-free.dev

# Terminal 2 — the TwiML and Media Stream server (does not dial, does not spend)
uv run --project poc1_pipecat python -m poc1_pipecat.app.telephony

# Terminal 2, each time you want a call (no need to Ctrl+C the server)
uv run --project poc1_pipecat python -m poc1_pipecat.app.call --accept-cost
```

The dialer checks `/health` on the public domain **before** dialing: if the
tunnel or the server is not up, it does not waste the call. Without
`--accept-cost` it only prints the estimate and exits.

| Flag | What for |
|---|---|
| `--a +52...` | dial a different number (default: `TWILIO_DESTINATION_NUMBER`) |
| `--no-record` | turns off Twilio's native recording (records by default) |
| `--limit 300` | lowers the hard duration cap (default 600 s) |

You answer, and the agent speaks first. On hang-up, the server terminal prints
the timed transcript, the history of validated offers and the outcome;
everything is also saved to
`poc1_pipecat/recordings/call_<call_sid>.json` (gitignored).

The Twilio recording comes out in **2 channels**, one per party: that is what
makes the evidence usable for computing WER and scoring turn-taking, because in
a mixed channel you cannot tell who talked over whom.

## Credentials

Copy [`.env.example`](../.env.example) from the root to `.env` and fill it in.
At the end, the smoke test says what is missing for the next level.

## Twilio: how the account is being used

Decision made for this increment: the existing number is used **for outbound
calls only**. An outbound call carries its own TwiML `url` per call, so it does
not modify the webhook configured on the number — and no code in this PoC
touches the configuration of the dispatcher agent's number.

What it does share with the dispatcher: caller ID, call log, recordings and
billing. **Still pending**: a dedicated subaccount for SPOTS or an additional,
exclusive number, before the formal runs of the 5 scenarios.

## BEFORE the first real call

Guardrails from [CLAUDE.md](../CLAUDE.md), non-negotiable:

1. **Approved cost table**: [`docs/costos_poc1.md`](../docs/costos_poc1.md).
   A 4-minute call comes to ~$0.25–0.28 USD, against ~$0.10 for smoke test
   level 1. PoC1 dials via `outbound-api` at $0.016/min, not via the
   dispatcher's trunk at $0.012/min: Pipecat cannot originate through a trunk,
   and the difference is a matter of cents. Cost is not the reason not to dial.
2. **Recording notice and consent** (REPEP, LFPDPPP, CRT). The v0 prompt does
   **not** include a recording notice. Decision of August 3, 2026: we proceed
   without it because the 15 calls in the comparison go to our own cell phone
   and the goal is to measure technology. It is **blocking** before calling
   any real carrier.

## Status

- [x] JSON tool-calling schemas for the 4 negotiation functions
- [x] System prompt interpolation (explicit replacement by key, not
      `str.format()` — see why in
      [`shared/system_prompt/prompt.py`](../shared/system_prompt/prompt.py))
- [x] Level 0 smoke test, no network and no cost
- [x] Pipeline: services, local VAD + end of turn, context and tool calling
- [x] Level 1 smoke test: real LLM, TTS and STT, no phone
- [x] Approved cost table
- [x] Transport: Twilio Media Streams WebSocket + `TwilioFrameSerializer`
- [x] FastAPI server: outbound call, TwiML, recording and persistence
- [x] Per-stage latency instrumentation for `shared/metrics/`
- [x] Harness on top of `shared/carrier_simulator/` to measure without telephony
- [ ] Dedicated Twilio subaccount
