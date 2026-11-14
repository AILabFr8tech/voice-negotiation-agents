# Batch mode

Many harness calls, alternating PoCs, with nobody watching.

Why the PoCs alternate and what the batch does **not** do: `docs/METRICS_CONTRACT.md` §2.8.
This is the operating manual.

## Running

With the interpreter of either PoC (this package is standard library only, and
what imports pipecat or LiveKit is each call's subprocess):

```powershell
# 1. Quote. Spends nothing: it asks each harness what each scenario would cost.
poc1_pipecat\.venv\Scripts\python.exe -m shared.lote `
    --label prueba --pocs 1,2 --scenarios 1,2 --repetitions 1 --cap-usd 0.60

# 2. The same with --accept-cost.
poc1_pipecat\.venv\Scripts\python.exe -m shared.lote `
    --label prueba --pocs 1,2 --scenarios 1,2 --repetitions 1 --cap-usd 0.60 `
    --accept-cost
```

The batch starts the PoC2 worker and keeps it up between calls; if it dies
mid-campaign, the batch brings it back up. With `--no-worker` it leaves it alone
and assumes one is already running in another terminal.

## Following it

```powershell
Get-Content -Wait diagnostics\lote\prueba\progress.jsonl
```

One line per finished call, with the index, the PoC, the scenario, the lines,
the fragmentation, the end reason, that call's p50, the WER and the cumulative
spend.

## Resuming

Relaunch with the **same `--label`**. It skips whatever already has a line in
the progress file and carries on. That covers all three cases:

- **Ctrl+C.** The in-flight call is killed and leaves no line: it is run again.
- **The process dies.** Same, and the previous calls are complete on disk.
- **The cap was reached.** Relaunching with a higher `--cap-usd` continues.

Changing the plan under a label that has already run is refused: the indices
would shift and the batch would skip calls that were never made.

**Killing the batch takes the in-flight call with it, and the worker too.** Not
through a `finally` (`Stop-Process -Force` runs nothing) but through a Windows
Job Object: see `children.py`. The first version did not have it, and the test
on 20 September 2026 left three orphan processes, one of them a call that kept
spending credit with nobody to collect the result.

To rerun a specific call, delete its line from `progress.jsonl`.

## What it leaves in `diagnostics/lote/<label>/`

| | |
|---|---|
| `plan.json` | the calls and their fingerprint |
| `progress.jsonl` | one line per finished call |
| `summaries/<clave>.json` | the summary that call's harness wrote |
| `calls/<clave>.log` | its full output; this is where you read one that failed |
| `quotes/` | the estimate per (PoC, scenario) |
| `worker.log` | the PoC2 worker, with one marker per startup |

Each call's measurement artifacts (log, zero crossing, pipeline JSON, audit WAV)
stay where they always were, in `pocN_*/diagnostics/harness/` and
`pocN_*/recordings/`. Each call's summary carries their paths.
