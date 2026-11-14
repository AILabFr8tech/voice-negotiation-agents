"""PoC3 — negotiating voice agent built on OpenAI's GPT Live API.

Third of the comparison's 3 PoCs. **It replaces the original OpenAI Realtime API
slot**: the model is `gpt-live-1` (in August and September, its alpha
`gpt-live-1-boulder-alpha`), a full-duplex model that listens and speaks at the
same time and delegates the heavy lifting to a Responses API model while the
conversation goes on.

Same import pattern as PoC1 and PoC2: the package is imported from the repo
**root**, not from `poc3_gpt_live/`. It is a uv virtual project (no
`[build-system]`, it is not installed), so the only import root is the repo root —
the same one `pythonpath = .` in `pytest.ini` already uses:

    uv run --project poc3_gpt_live python -m poc3_gpt_live.app.config

Each PoC has its own `pyproject.toml` + `uv.lock`, on purpose: the comparison
needs to be able to document the exact environment of each architecture.

**PoC3 runs on an alpha API, not GA.** The terms forbid production use and
forbid exposing the audio outside the organization, and OpenAI states that the
model and the shape of the API will change before GA. It is valid as an
exploratory probe, not as a candidate eligible on equal terms with PoC1 and
PoC2. See `docs_alpha/README.md` and `docs/METRICS_CONTRACT.md` §2.5.
"""
