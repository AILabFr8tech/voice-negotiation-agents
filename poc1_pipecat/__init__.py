"""PoC1 — negotiating voice agent built with Pipecat.

The package is imported from the repo **root**, not from `poc1_pipecat/`:
`poc1_pipecat` is a uv virtual project (it has no `[build-system]` and is not
installed), so the only import root is the repo root — the same one that
`pythonpath = .` in `pytest.ini` already uses. That makes both `shared` and
`poc1_pipecat` importable under the same rule:

    uv run --project poc1_pipecat python -m poc1_pipecat.app.smoke
"""
