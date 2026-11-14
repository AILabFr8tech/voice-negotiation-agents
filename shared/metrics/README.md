# metrics/ — automatic computation of the comparison metrics

Everything numeric is computed automatically. **Identical in all 3 PoCs**: if
each PoC measured in its own way, the numbers would not be comparable and the
technical decision would have no backing. That is why the computation lives
here and not inside each PoC — the only thing that changes between
architectures is the **capture** of the samples.

## The only manual part

Two things, and nothing else:

1. **Correct your reference transcript** (2-3 min per call). The file is
   generated pre-filled with the script lines; you only correct where you
   deviated.
2. **Enter the two 1-5 ratings** for naturalness and turn-taking, using the
   script's rubric.

Percentiles, quartiles, WER, CER, amount error and aggregates: automatic.

Only what distinguishes one voice technology from another is measured: latency
per stage, STT quality, turn-taking and reliability. What the agent says is the
vehicle of the conversation —it gives the WER domain vocabulary and exercises
tool calling— but it is not an object of measurement and does not appear in the
report.

## The workflow, in four commands

```bash
# 1. After each call: generate the reference file seeded with the script
python -m shared.metrics.cli reference \
    --json poc1_pipecat/grabaciones/call_CAxxx.json \
    --scenario 2 --repetition 1 --poc 1

# 2. Open that .txt, listen to YOUR channel of the audio, correct the `yo:` lines
#    where you improvised, and enter `naturalness` and `turn_taking`. (2-3 min.)

# 3. Compute and write the `metrics` block into the same call JSON
python -m shared.metrics.cli compute --folder poc1_pipecat/grabaciones

# 4. Consolidate the PoC into Markdown + CSV, ready for the comparison
python -m shared.metrics.cli report --folder poc1_pipecat/grabaciones \
    --poc "PoC1 — Pipecat" --output docs/metrics
```

And to check that the computation works **without spending a single call**:

```bash
python -m shared.metrics.cli demo
```

It generates 15 synthetic calls (5 scenarios × 3 repetitions), computes them and
produces the full report. It is what has to be run before trusting the module
with real data.

## The modules

| Module | What it does |
|---|---|
| `text.py` | Spanish normalization and numeral parser. It is what makes `"dieciocho mil" == "18,000"`. |
| `wer.py` | Edit-distance alignment: WER, CER and amount extraction error, with S/D/I breakdown. |
| `latency.py` | Per-turn sample contract, percentiles and quartiles per bucket. |
| `scenarios.py` | Reads the script as data (`shared/test_scenarios/scenarios.yaml`). |
| `reference.py` | Generates and parses the editable reference file. |
| `calls.py` | Joins JSON + reference; writes the `metrics` block back into the JSON. |
| `report.py` | Per-PoC consolidation in Markdown and CSV. |
| `synthetic.py` | Synthetic corpus: validates the computation without spending calls. |
| `cli.py` | The four verbs above. |

No external dependencies beyond `pyyaml` (which the compensation catalog already
uses): the 3 PoCs have different `uv` environments, so only the standard library
goes in here. The edit distance and the percentiles are implemented by hand, on
purpose.

## Decisions that affect the numbers

These are fixed and documented because moving them changes the result of the
comparison:

- **Percentiles by linear interpolation** (Hyndman-Fan method 7, the default
  of `numpy.percentile`). Fixed and visible so that a library change does not
  move the report without anyone noticing.
- **Three latency buckets, never mixed:** `apertura` (opening: turn 1, cold
  start), `regimen` (steady state: turns 2..n) and `interrupted` (barge-in). An
  interrupted turn goes to `interrupted` even if it is the first one.
- **Aggregation across calls:** latency samples are **pooled** and the
  percentile comes from the pool; WER counts are **summed** and the rate is
  computed at the end. Percentiles and rates are never averaged — the mean of
  percentages would give the same weight to a 3-word turn and to a 40-word one.
- **Percentiles with insufficient sample are flagged, not hidden:** with 8
  turns the p99 is the maximum in disguise. The number is reported with `⚠` and
  a warning; hiding it would force recomputing it by hand.
- **Normalization:** lowercase, folded accents (ñ is kept: "año" ≠ "ano"),
  punctuation removed, numerals canonicalized. Filler words are **not**
  deleted: they are speech the STT has to recognize.
- **A number counts as an amount** if it reaches 1,000. That is what keeps
  "dos horas de espera" or "53 pies" out of the amount metric.

## Open dependency: latency capture

The telephony pipeline **does not write latency samples yet**. This module
already defines the contract: a list under the `latencies` key of the per-call
JSON, with one object per turn.

```json
"latencies": [
  {"indice": 0, "full_turn_ms": 2100.5, "end_of_turn_ms": 210.0,
   "stt_ms": 180.0, "llm_ttft_ms": 900.0, "tts_ttfb_ms": 810.5,
   "interrupted": false, "is_opening": true}
]
```

`TurnSample.from_events()` builds the sample from absolute instants, so that
capture does not reimplement the subtractions. While that key does not exist,
the report comes out with an empty latency section and a warning — it does not
crash, but there is no latency decision criterion either.

## Tests

```bash
python -m pytest shared/metrics/ -q
```

Everything is validated with synthetic data: no test needs a call, an API key
or a phone. It includes a drift test (`test_scenarios_file.py::TestNoDrift`)
that checks that the lines of `scenarios.yaml` still exist in the prose script
— the script lives in two formats and that test is what keeps them from
drifting apart.
