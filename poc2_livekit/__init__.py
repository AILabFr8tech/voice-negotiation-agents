"""PoC2 — voice negotiation agent built on LiveKit Agents.

Same import pattern as PoC1: the package is imported from the repo **root**,
not from `poc2_livekit/`. It is a uv virtual project (no `[build-system]`, it
is not installed), so the only import root is the repo root — the same one
that `pythonpath = .` in `pytest.ini` already uses:

    uv run --project poc2_livekit python -m poc2_livekit.app.config

Each PoC has its own `pyproject.toml` + `uv.lock`, on purpose: the comparison
needs to be able to document the exact environment of each architecture, and
pipecat and livekit-agents have no reason to resolve the same versions of
their shared dependencies.
"""
