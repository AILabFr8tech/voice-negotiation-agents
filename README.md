# Voice Negotiation Agents: Pipecat, LiveKit and GPT-Live negotiating freight rates in Spanish

[![reproduce](https://github.com/AILabFr8tech/voice-negotiation-agents/actions/workflows/reproduce.yml/badge.svg)](https://github.com/AILabFr8tech/voice-negotiation-agents/actions/workflows/reproduce.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository holds the code and data behind the paper:

> **Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone
> Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live.**
> Jovan Popoca Moran, Eduardo de Avila-Armenta, Raúl Alejandro Aquino-Ibarra,
> Jorge Octavio Nicolás-Díaz, Umberto Leon-Dominguez. *(Citation below.)*

A voice agent calls a carrier and negotiates a freight rate in Mexican Spanish. We
built the same agent three ways:

| | Architecture | Speech in | Language model | Speech out |
|---|---|---|---|---|
| **PoC1** | cascade orchestrated by [Pipecat](https://github.com/pipecat-ai/pipecat) | Deepgram | OpenAI | ElevenLabs Flash |
| **PoC2** | cascade orchestrated by [LiveKit Agents](https://github.com/livekit/agents) | Deepgram | OpenAI | ElevenLabs Flash |
| **PoC3** | OpenAI GPT-Live, a speech model that listens and speaks itself; it delegates tool calls to the same LLM | — | OpenAI | — |

All three share the same negotiation engine, system prompt and tools. A program
(the **harness**) plays the carrier: it plays the same recorded lines to every
agent, without the telephone network, and measures latency on the listener's side.
The harness was validated against simulated agents with a known delay over the
audio path of each architecture.

**Every number in the paper is regenerated from the tables in this repository.** The
paper's figures are LaTeX macros (`\RObsMedA`, `\RWerB`, …) written by scripts, not
by hand. CI regenerates all 1,593 of them from the committed tables and checks that
they match the paper on every push.

## Reproduce the paper's numbers

Requires [uv](https://docs.astral.sh/uv/) (it installs the pinned Python 3.12 and
the locked dependencies). No API keys are needed.

```bash
OUT=regenerated && mkdir -p $OUT
uv run --project analysis --locked python -m shared.metrics.analysis_report        --output $OUT
uv run --project analysis --locked python -m shared.metrics.interruptions_report   --output $OUT
uv run --project analysis --locked python -m shared.metrics.gpt_live_clock_report  --output $OUT/gpt_live_clock.tex
uv run --project analysis --locked python -m shared.metrics.timeline_report        --output $OUT/backchannel_timeline.tex
uv run --project analysis --locked python -m shared.metrics.cascade_stages_report  --output $OUT/cascade_stages.tex

# compare with the macros the paper uses
uv run --project analysis --locked python .github/ci/compare_generated.py docs/report/latex/generated $OUT
```

The comparison checks every macro, every table line and every value of
`results.json`. One macro, `\RClipsIntShaOk`, checks the audio clips against their
hashes, so it can only be computed with the recordings (see *What is not included*).
The macro `\RAnalisisPython` records the Python version; to match it exactly, use
Python 3.12.13 (`uv run --python 3.12.13 ...`), as CI does.

The paper itself is [docs/paper/en/sn-article.tex](docs/paper/en/sn-article.tex), with
the macros already embedded (Springer Nature asks for a single `.tex`).
[docs/paper/en/embed_figures.py](docs/paper/en/embed_figures.py) rewrites them from
[docs/report/latex/generated/](docs/report/latex/generated/).

## Run the tests

```bash
uv run --project poc1_pipecat  --locked pytest -q poc1_pipecat shared validation
uv run --project poc2_livekit  --locked pytest -q poc2_livekit
uv run --project poc3_gpt_live --locked pytest -q poc3_gpt_live
```

About 1,450 tests, all offline. Two PoC3 tests are Windows-only (the batch
runner's Windows virtual-environment layout and a workaround for the 15 ms Windows
clock tick) and are skipped elsewhere.

## Run the agents

Running a call needs your own accounts: OpenAI, Deepgram and ElevenLabs, plus
Twilio (PoC1 and PoC3, Media Streams) or LiveKit with a SIP trunk (PoC2). Copy
[.env.example](.env.example) to `.env` and fill it in. Each PoC's README explains
how to start the agent, run the harness against it and run a batch of calls:
[poc1_pipecat/](poc1_pipecat/README.md), [poc2_livekit/](poc2_livekit/README.md),
[poc3_gpt_live/](poc3_gpt_live/README.md). The harness and the batch runner print
an estimated cost and refuse to place calls without `--accept-cost`.

## Language

The code is in English. Most comments and documentation are too; the translation
of the remaining Spanish comments and module documentation is in progress. **What
the agent says and hears stays in Spanish**, because the experiment is about Mexican Spanish speech:
the system prompt ([shared/system_prompt/v0.md](shared/system_prompt/v0.md)), the
tool names, descriptions and replies (`proponer_precio`, `registrar_acuerdo`, …),
the scenario lines the carrier speaks, the compensation texts, and the word lists
that match spoken Spanish (number words, backchannels such as "ajá" or "ok").
Some data values and keys also stay in Spanish because they come from the
committed data or feed the paper's macro names (outcome codes such as
`sin_respuesta`, condition names such as `retardo_300`, `\R...` macros).
[shared/test_scenarios/translation_en.yaml](shared/test_scenarios/translation_en.yaml)
has English translations of the scenario lines.

## What is not included

- **The audio recordings of the carrier's lines, and the raw per-call logs and
  transcripts.** They contain the voice of the person who recorded the lines. The
  clip manifests
  ([shared/test_scenarios/clips/manifest.json](shared/test_scenarios/clips/manifest.json),
  [interruption_clips/manifest.json](shared/test_scenarios/interruption_clips/manifest.json))
  list the SHA-256 hash of every clip, which shows that the three architectures heard
  the same audio. Everything the paper computes from them is in the exported tables.
- **API keys** for the commercial services.

## Repository layout

```
poc1_pipecat/        PoC1: the Pipecat cascade (app/, tests/)
poc2_livekit/        PoC2: the LiveKit Agents cascade
poc3_gpt_live/       PoC3: GPT-Live, plus diagnostics/ used in the paper's PoC3 analyses
shared/
  negotiation_engine/  pricing, compensations, session and outcomes (same for all PoCs)
  system_prompt/       the system prompt (Spanish) and its rendering
  test_scenarios/      the five negotiation scenarios, interruption clips, clip manifests
  carrier_simulator/   the harness: plays the carrier and timestamps both sides
  audio/               clip preparation, audio utilities
  lote/                the batch runner ("lote" = batch) for campaigns of calls
  llm_transport/       LLM timeout and retry policy, identical in the three PoCs
  metrics/             latency, WER, overlaps; the *_report.py scripts write the paper's macros
validation/          simulated agents with a known delay, instrument validations
analysis/            the pinned environment (numpy, scipy) used to compute the numbers
docs/
  report/data/         exported per-call and per-turn tables: the input of every number
  report/latex/generated/  the macros and tables the paper uses
  paper/en/            the paper and its supplementary material
  METRICS_CONTRACT.md  how each metric is defined and measured
  populations_and_preregistrations.md  call populations, exclusions and preregistered hypotheses
```

## Citing

Please cite the paper, and the software if you use the code or data:

```bibtex
@unpublished{popocamoran2026voice,
  title  = {Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live},
  author = {Popoca Moran, Jovan and de Avila-Armenta, Eduardo and Aquino-Ibarra, Ra{\'u}l Alejandro and
            Nicol{\'a}s-D{\'i}az, Jorge Octavio and Leon-Dominguez, Umberto},
  note   = {Manuscript},
  year   = {2026}
}

@software{voice_negotiation_agents,
  title  = {Voice Negotiation Agents: Pipecat, LiveKit and GPT-Live negotiating freight rates in Spanish},
  author = {Popoca Moran, Jovan and de Avila-Armenta, Eduardo and Aquino-Ibarra, Ra{\'u}l Alejandro and
            Nicol{\'a}s-D{\'i}az, Jorge Octavio and Leon-Dominguez, Umberto},
  organization = {AI Lab, Freight Technologies Inc.},
  url    = {https://github.com/AILabFr8tech/voice-negotiation-agents},
  license = {Apache-2.0},
  year   = {2026}
}
```

GitHub's **Cite this repository** button (from [CITATION.cff](CITATION.cff)) gives
the same citation in other formats.

## License

Copyright 2026 Freight Technologies, Inc. (Fr8Tech). Licensed under the
[Apache License, Version 2.0](LICENSE): code and data. You may use, modify and
redistribute it, including commercially, provided you keep the license and the
[NOTICE](NOTICE) file. The license grants no rights to the Freight Technologies or
Fr8Tech names and marks. Pipecat, LiveKit, OpenAI, GPT-Live, Deepgram, ElevenLabs
and Twilio are trademarks of their respective owners.
