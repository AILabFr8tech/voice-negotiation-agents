# PoC3 — GPT-Live alpha

Third of the comparison's 3 PoCs. **It replaces the original OpenAI Realtime API
slot** (change made on August 12, 2026): the model is
`gpt-live-1-boulder-alpha`, full-duplex —it listens and speaks at the same
time— and it delegates the work that needs tools to a Responses API model while
the conversation goes on.

> ## Public protocol since September 30, 2026
>
> Access to the alpha protocol (`/v1/live?model=` + `OpenAI-Alpha:
> quicksilver=v2`) was withdrawn: the server answers `quicksilver_v2_access_denied`.
> The bridge now speaks the **public** `gpt-live-1` protocol:
> `wss://api.openai.com/v1/live/sessions`, `Authorization` only, `session.start`
> with the model inside it, `session.*` events, and delegation wrapped in
> `response.event`. The mapping table is in the docstring of
> [`app/bridge.py`](app/bridge.py). The audio is still 24 kHz PCM16 (requested
> explicitly), so the resampling and the measurement instrument do not change.
> What is lost: `turn.*` and the `start_ms`/`end_ms` times of the output audio no
> longer exist, so the diagnostics that used the server's timeline over the audio
> (`delivery_lag`, `per_turn`) come out empty. The main latency and the
> interruption measurement did not depend on them.
>
> What follows describes the August and September alpha, and is kept as a
> record of how those campaigns ran.

> ## ⚠ This runs on an alpha, not on a GA release
>
> The alpha terms **forbid production traffic**, **forbid exposing the generated
> audio outside the organization**, and OpenAI states that the model and the
> shape of the API **will change** before GA — for which there is no date.
> Nor does it promise production-grade reliability or latency.
>
> PoC3 is valid as an **exploratory probe**, not as a candidate eligible on equal
> terms with PoC1 and PoC2. The final report has to cite that asymmetry every
> time it presents a number from this PoC. Details in
> [`docs/METRICS_CONTRACT.md`](../docs/METRICS_CONTRACT.md) §2.5 and in
> [`docs_alpha/README.md`](docs_alpha/README.md).
>
> The alpha documentation in [`docs_alpha/`](docs_alpha/) is **confidential**.

## Modules

| Module | What it does |
|---|---|
| [`app/config.py`](app/config.py) | Reads the root `.env`. Validates separately what the smoke test, the alpha session and the telephony require. |
| [`app/tools.py`](app/tools.py) | The 4 schemas the model sees, the `delegation` object, and the bridge between the alpha's events and the negotiation engine. |
| [`app/bridge.py`](app/bridge.py) | **The core.** G.711 μ-law codec, 8 kHz ↔ 24 kHz resampling, and the `/v1/live` WebSocket client. |
| [`app/latency.py`](app/latency.py) | Latency capture under PoC3's reduced contract, and the verdict on the clock anchoring. |
| [`app/telephony.py`](app/telephony.py) | TwiML, Media Stream and call orchestration. FastAPI server. Does not dial. |
| [`app/call.py`](app/call.py) | Dials the outgoing call. The only module that spends on telephony. |
| [`app/smoke.py`](app/smoke.py) | Smoke test, levels 0 and 1. |

**There is no `trip.py`, `prompt.py` or `session.py`.** The trip data, the
interpolation of the v0 prompt and the negotiation state live in
[`shared/`](../shared/) and all 3 PoCs import them. Duplicating them would let
the financial configuration diverge between architectures without the
comparison noticing.

## Architecture of this PoC

| Layer | Technology |
|---|---|
| Telephony | Twilio, Media Streams — **identical to PoC1, on purpose** |
| Media | WebSocket (FastAPI) + 8 kHz μ-law ↔ 24 kHz PCM16 bridge |
| Conversational engine | `gpt-live-1-boulder-alpha` (native STT, LLM and TTS, not separable) |
| Tool calling | `responses` delegation → `gpt-4.1-mini` (the same as PoC1 and PoC2) |
| End of turn | internal to the server, not exposed |
| Negotiation | `shared/negotiation_engine` via tool calling |

```
Twilio  --μ-law 8k-->  bridge  --PCM16 24k-->  wss://api.openai.com/v1/live
Twilio  <--μ-law 8k--  bridge  <--PCM16 24k--  (output_audio.delta)
```

The telephony is kept the same as PoC1's so that the comparison measures the
conversational engine and not the media bus. **Neither LiveKit nor WebRTC is
used here.**

## Five decisions to know before touching the code

**1. Tool calling goes through `responses` delegation, not `client`.** The alpha
offers two modes and only one allows tool calling. With `client` the model sends
free text and is answered with text: without a `call_id` or structured arguments
it would be **impossible** to guarantee the non-negotiable rule in CLAUDE.md —the
agent never utters an amount unless `proponer_precio` validates it—. With
`responses` there are `response.function_call_arguments.done` and
`delegation.function_call_output.create`, which is the contract the comparison
measures. The delegate model is the same `gpt-4.1-mini` frozen for all 3.

**2. There is no documented way to tell the agent "speak now".** The alpha has
six client events and none of them triggers a response; `response.create` is
explicitly forbidden on this stream. But PoC3 places **outgoing** calls, where
the agent opens, and the 5 scenarios assume that. The only available channel is
`session.context.append`, so the agent is sent **a fact, not an instruction**.
The behavior still comes entirely from `v0.md`. This is a real deviation from
PoC1 and PoC2, and it is declared.

> ⚠ **And the fact had to be marked as a system event.** With the original text
> ("La llamada telefónica saliente acaba de ser contestada.", i.e. "The outgoing
> phone call has just been answered.") the model **read its own plan aloud**
> before greeting: 10.4 s of audio saying
> *"Ya entró la llamada, te están oyendo. Arranca con un saludo breve, di que
> llamas de parte de Freight Technologies y confirma si hablas con Carlos. Y haz
> una pausa ahí."* ("The call is through, they can hear you. Start with a brief
> greeting, say you are calling on behalf of Freight Technologies and confirm
> whether you are speaking with Carlos. And pause there."), and only then
> *"Hola, buenos días…"*. That text is not in any file in the repo and the first
> delegation did not happen until second 53: the `gpt-live` model generated it.
> In full-duplex there is no separate reasoning channel — the only output
> channel is the voice, and that is where the plan went.
>
> `OPENING_CONTEXT` now carries a prefix `[evento del sistema — no leas
> esta línea en voz alta…]` ("system event — do not read this line aloud…"). It
> is **channel hygiene, not negotiation behavior**: it says nothing about prices,
> concessions, closing or style, which still come entirely from `v0.md`. Even so,
> it is a deviation specific to PoC3 (PoC1 and PoC2 do not need it) and it counts
> as such in the report. Whether the mitigation works **is unverified against
> the live API**: level-1 smoke test closes that.

**3. There is no `output_audio.done`.** The agent has finished speaking when
deltas stop arriving, and no event says so. Hanging up after the negotiation
closes —pipecat's `stop_when_done()`, LiveKit's `drain()`— is solved with a 2 s
silence timer. PoC3 systematically hangs up later than the other two, by
construction.

**4. `{"event":"clear"}` is not sent to Twilio on barge-ins.** GPT-Live generates
faster than real time, so Twilio builds up a queue and what was already sent is
still heard even after the model goes quiet. `clear` would drop it — and that
would impose PoC1's interruption policy on a full-duplex architecture, so the
comparison would be measuring our policy. The model governs its own voice and
the effect is noted when scoring turn-taking.

**5. `interrupted` means "barge-in that cuts off", not "overlap".** In
full-duplex, overlap is the normal mode of operation; with the contract's
literal criterion almost every turn would land in the `interrupted` bucket and
PoC3 would be left without a steady-state p50. Here it marks the turn only if
the carrier talked over the agent **and** the agent went silent within half a
second. Overlaps that do not cut off are counted separately, in
`gpt_live_extra.turn_taking`.

## What is measured and what is not

`shared/metrics/` **computes**; `app/latency.py` **measures**. PoC3's contract is
reduced, and that is the result, not an instrumentation gap:

| Field | Captured? | Why |
|---|---|---|
| `full_turn_ms` | ✅ | end of the carrier's turn → first `output_audio.delta` |
| `end_of_turn_ms` | ❌ | VAD and turn-taking run inside the server |
| `stt_ms` | ❌ | there is no discrete STT: the model hears directly |
| `llm_ttft_ms` | ❌ | there is no LLM that starts at an observable instant |
| `tts_ttfb_ms` | ❌ | there is no discrete TTS |
| `interrupted` | ✅ | barge-in that cuts off (see above) |
| `is_opening` | ✅ | cold-start turn |
| `end_of_turn_verdicts` | empty | the alpha does not expose the probability |

The four missing stages are **omitted** from the JSON instead of being written as
`null`: "not applicable" and "not measured" are different things and the report
reads them differently.

**The zero is resolved with two clocks and verified.** The alpha only gives times
on its own timeline; the first audio is stamped with the local clock when it
reaches the bridge. They are anchored with `session.started` and **checked
against the call's own data** (`clock_verdict`): if the residual drifts, the run
comes out flagged with `exclude_from_pooled` and does not count toward the
percentiles. It is the same pattern PoC2 used to verify its zero.

**The resampling is inside the latency and is not subtracted.** Measured: 24.2 ms
per direction in steady state, peaks of 73 ms. It is part of what this
architecture costs over telephony.

**PoC3's acknowledgement has no independent timing validation, and it will not
have one.** In PoC1 and PoC2, channel 0 of the WAV is exact because the player
knows its deadline; in PoC3 both channels are timestamped on arrival and the
agent's channel shares the `t_local` the acknowledgement hangs from, so the WAV
inherits the clock and does not validate the timestamp independently. It
validates the content at the mark, which is the failure mode of concern, and
nothing more. PoC3's acknowledgement will not have independent timing
validation in any run. Declared asymmetry.

## Environment

Managed with [uv](https://docs.astral.sh/uv/). Python 3.12, same as PoC1 and
PoC2 — no dependency requires it; it is pinned so that the comparison does not
change Python version between architectures.

```bash
uv sync --project poc3_gpt_live
```

`pyproject.toml` + `uv.lock` are committed, so that the 3 PoCs are reproducible
and the metrics comparable.

What PoC3 does **not** install, which is in itself a result: no Deepgram, no
ElevenLabs, no VAD or turn detector, and no orchestration framework. The
"pipeline" is a bridge between two WebSockets. It reads both ways: fewer pieces
to integrate, and zero visibility into what happens inside.

## Tests and smoke test

The commands run from the repo **root** (that is where `pytest.ini` is, and
`poc3_gpt_live` is a uv virtual project: it is not installed, it is imported
from the root).

```bash
uv run --project poc3_gpt_live pytest poc3_gpt_live
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.config
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke
```

**Level 0** does not touch the network or consume credits. 15 checks: the
`.env`, the prompt without placeholders, the max price absent (with a negative
control of the leak detector), the 4 schemas against the real catalog, the
delegation in `responses` mode with the frozen model, the initial session
without a repeated `model`, the handlers against the engine, **the complete
tool-calling contract with a synthetic alpha event**, the μ-law codec and the
audio bridge, the latency capture contract as read by `shared/metrics/`, and the
barge-in criterion.

### Level 1: real alpha session, no phone

```bash
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke --online --accept-cost
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.smoke --online --accept-cost \
    --wav path/to/other.wav
```

Opens a real session against the alpha and feeds it audio from a WAV **through
the same bridge Twilio would use** —converted to 8 kHz μ-law first, in 20 ms
frames and at real-time pacing—, so that the complete path is exercised and not
a lab version with wideband audio. It checks that `output_audio.delta` arrives,
that the transcript comes out for both parties and that `full_turn_ms` is not
`None`.

By default it uses `poc1_pipecat/recordings/humo_nivel1.wav` (PoC1's agent voice,
speaking domain Spanish). It does not act out the script, so tool calling may
not fire: that is already verified at level 0 with the stub.

**It costs money** (~$0.01 per run): that is why `--accept-cost` is mandatory.
The `gpt-live` model is not billed during the alpha; what costs is the
delegation. Breakdown in [`docs/costos_poc3.md`](../docs/costos_poc3.md).

## Level 2: the real phone call

Three pieces, two terminals. The tunnel and the server stay up between calls;
only the dialer is re-run for each call.

```bash
# Terminal 1 — the tunnel to the static domain of PUBLIC_URL in the .env
ngrok http 8000 --domain=<your-domain>.ngrok-free.dev

# Terminal 2 — the TwiML and Media Stream server (does not dial, does not spend)
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.telephony

# Terminal 2, each time you want a call
uv run --project poc3_gpt_live python -m poc3_gpt_live.app.call \
    --scenario 1 --repetition 1 --accept-cost
```

The dialer checks `/health` on the public domain **before** dialing: if the
tunnel or the server is not up, it does not spend the call. Without
`--accept-cost` it only prints the estimate and exits.

| Flag | What for |
|---|---|
| `--a +52...` | dial another number (default: `TWILIO_DESTINATION_NUMBER`) |
| `--scenario N --repetition N` | labels the run; without them it has to be identified by hand afterwards |
| `--limit 300` | lowers the hard duration cap (default 600 s) |

**There is no flag to record.** PoC3 does not record: the WER is computed
against the acted script (metrics contract §1), and on top of that the alpha
terms forbid exposing the generated audio outside the organization.

You answer, and the agent speaks first. On hang-up, the server terminal prints
the transcript with timings, the clock verdict, the count of overlaps and
cut-offs, and the outcome; everything is also saved in
`poc3_gpt_live/recordings/call_<call_sid>.json` (gitignored).

## Credentials

Copy the root [`.env.example`](../.env.example) to `.env` and fill it in. PoC3
needs `OPENAI_API_KEY` and `OPENAI_LIVE_MODEL` (default `gpt-live-1`); it does
**not** use `DEEPGRAM_API_KEY` or `ELEVEN_API_KEY`.

`OPENAI_LIVE_ALPHA_HEADER` **is no longer read** (public protocol, see above). If
it is still in the `.env` it does no harm.

## BEFORE the first real call

Guardrails from [CLAUDE.md](../CLAUDE.md), non-negotiable:

1. **Approved cost table**: [`docs/costos_poc3.md`](../docs/costos_poc3.md).
2. **Recording notice and consent** (REPEP, LFPDPPP, CRT). The v0 prompt does not
   include a notice. The comparison's 15 calls go to our own cell phone; it is
   **blocking** before dialing any real carrier.
3. **The alpha terms.** Low-volume use, no production, and the generated audio
   does not leave the organization. The 15 calls to our own cell phone comply; a
   demo with a real carrier does not.

## Status

- [x] Structure, uv environment and alpha documentation in the repo
- [x] `config.py` with the three validations
- [x] The 4 schemas and the tool-calling contract through `responses` delegation
- [x] `bridge.py`: G.711 codec verified against the reference, resampling with
      anti-aliasing, and the alpha WebSocket client
- [x] `telephony.py`: TwiML, Media Stream and call orchestration
- [x] `latency.py` with the reduced contract and the clock verdict
- [x] Level-0 smoke test — 15 checks, no network and no cost
- [x] Tests (129) green
- [x] **Call 1.1 of August 12, 2026 — a warm-up, it does NOT count as one of
      the 15.** It negotiated and closed in `acuerdo`, and uncovered three
      instrumentation defects that were fixed with its data (see below). Its
      `latencies` came from the old collector, so **they do not enter the
      percentile pool**
- [x] **Call 1.2 of August 12, 2026 — also a warm-up.**
      It confirmed the three fixes from 1.1 (readable transcript and a realistic
      **WER of 5.45%**, an opening without narrating the plan, `is_opening`
      flagged, 0 cut-offs, matching the hand annotation) and uncovered three
      other things, already fixed: the false 8,266 ms maximum caused by a
      backchannel, the hang-up that swallowed the goodbye, and that
      `delivery_lag` was being read as latency when it is not
- [ ] **Level-1 smoke test not run**: requires real spending against the alpha
- [ ] Verify the event flow against the real service — everything about the
      alpha comes from `docs_alpha/` and **is not confirmed against the live API**

### What call 1.2 uncovered

1. **A false maximum of 8,266 ms.** The carrier slipped in an "ajá" ("uh-huh")
   while the agent was speaking; that burst set the zero, the agent kept talking
   over it, and its next sentence consumed it eight seconds later. The pending
   zero now requires the agent to have been silent since the burst ended
   (`zeros_vencidos_por_apoyo_verbal`, zeros expired by a backchannel).
2. **The hang-up swallowed the goodbye.** `registrar_acuerdo` set up the
   closing, the agent said "Un segundo..." ("One second...") and went silent
   while the delegate composed the goodbye; the 2 s of silence elapsed in the
   middle of the delegation. The census shows it in full: the complete goodbye
   text at second 132.5, `turn.done` with `" Un segundo..."` at 134.95,
   `session.closed` with `reason: "client_request"` at 135.14. **The call
   dropped on the carrier without a goodbye.** The monitor now also waits until
   no delegation is in flight, with `SILENCE_TO_HANG_UP_S` at 3 s and a hard cap
   of 20 s so that a stuck delegation does not bill indefinitely.
3. **`delivery_lag` was being read as latency, and it is not.** It published
   `final_delay_ms: 31312` and `drift_ms: 26996` next to turns of
   1,300–1,700 ms. The alpha's `start_ms` timeline advances more slowly than the
   wall clock (959 contiguous deltas, 2.6 s of gaps, 97.6 s of timeline in
   128.8 s of call), so the distance grows on its own. It is evidence of why
   the measurement is taken from outside, not a delay. The block now says so
   and sets `thresholds_apply: false`.

### What call 1.1 uncovered

All three were in that call's JSON, and all three are fixed:

1. **The transcript came out with words stuck together** (`"teestán"`,
   `"soyCarlos"`). `bridge.transcript_text` did a `strip()` of each fragment, and
   the alpha sends the separating space **at the start** of the fragment. The
   call's WER was reported as **84.5%** on a transcript that, with the fix,
   measures **0.0%** (58/58 words, 309/309 characters). The CER's 32 deletions
   were exactly the swallowed spaces.
2. **The latency counted 19 samples for 8 turns**, 7 without a number, and none
   flagged as the opening. Speech bursts instead of loose frames, one sample per
   answered burst, and `is_opening` = the call's first sample. Details in
   `docs/METRICS_CONTRACT.md` §2.7 and in the header of `app/latency.py`.
3. **`interrupted` flagged 8 of 19 turns** with `interrupciones_no_deseadas: 0`
   (unwanted interruptions) annotated by hand: the cut-off criterion was missing
   its lower bound.

And two findings that are **not** instrumentation bugs and go into the report:

- the model read its plan aloud before greeting (decision 2 above);
- `delivery_pacing` 0.963: the alpha delivered 157.6 s of audio in 163.6 s of
  call. It does not affect `full_turn_ms` —local energy does not translate the
  server's timeline— but it is an audio deficit that has to be watched as a
  reliability issue.
