"""PoC3's event census with the alpha's names, wherever it comes from.

The analysis was written against the alpha census (campaign of 22-Sep-2026).
The public API sends the same thing in a different shape (see the table in
`poc3_gpt_live/app/bridge.py`): events prefixed with `session.`, the
`response.*` ones wrapped in `response.event`, and without the acknowledgement
of the function output. `events` reads either one and always returns the
alpha's shape:

- `session.delegation.created` -> `delegation.created`
- `session.output_audio.delta` -> `output_audio.delta`, with `audio_ms` (the
  chunk's duration, derived from its bytes; the alpha carried
  `start_ms`/`end_ms`)
- `session.input_transcript.delta` / `output_transcript.delta` ->
  `input_transcript.added` / `output_transcript.added`
- `response.event` -> the inner `response.*`, with its `delegation_id`
- The function output: the alpha acknowledged
  `delegation.function_call_output.created` when it received it. The public
  API does not acknowledge `response.item.create`; the equivalent is the
  `response.created` with which the delegate starts after receiving it (the
  bridge sends the output and right after `response.create`). A
  `delegation.function_call_output.created` is synthesized at that instant,
  with the function's `call_id`.

In the public API `delegation.created` no longer carries the text of the
delegated request.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

RENOMBRES: dict[str, str] = {
    "session.delegation.created": "delegation.created",
    "session.output_audio.delta": "output_audio.delta",
    "session.input_transcript.delta": "input_transcript.added",
    "session.output_transcript.delta": "output_transcript.added",
}
RE_SIZE: re.Pattern[str] = re.compile(r"<audio (\d+) bytes>")


def bytes_per_ms(session: dict[str, Any]) -> float:
    """Audio bytes per millisecond according to the session's confirmed format."""
    format_kind = ((session.get("audio") or {}).get("format") or {})
    sample_rate = int(format_kind.get("rate") or 24000)
    per_sample = 1 if format_kind.get("type") in ("audio/pcmu", "audio/pcma") else 2
    return sample_rate * per_sample / 1000.0


def events(census: Path) -> Iterator[dict[str, Any]]:
    """Each census row as `{n, t_local_seg, type, evento}`, in the alpha's shape."""
    raw_values = [json.loads(l) for l in census.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    published_any = any(str(e.get("type", "")).startswith("session.") and e.get("type") != "session.started"
                  or e.get("type") == "response.event" for e in raw_values)
    if not published_any:
        yield from raw_values
        return
    by_ms = 48.0
    pending_items: list[str] = []  # call_id of functions waiting for their output
    for e in raw_values:
        tipo = str(e.get("type", ""))
        ev = e.get("event") if isinstance(e.get("event"), dict) else {}
        if tipo == "session.started":
            by_ms = bytes_per_ms((ev or {}).get("session") or {})
            yield e
            continue
        if tipo == "response.event":
            internal = (ev or {}).get("event") or {}
            internal_kind = str(internal.get("type", ""))
            if internal_kind == "response.created" and pending_items:
                for call_id in pending_items:
                    yield {**e, "type": "delegation.function_call_output.created",
                           "event": {"type": "delegation.function_call_output.created",
                                      "item": {"call_id": call_id}, "sintetizado": True}}
                pending_items = []
            if internal_kind == "response.output_item.done":
                item = internal.get("item") or {}
                if item.get("type") == "function_call" and item.get("call_id"):
                    pending_items.append(item["call_id"])
            yield {**e, "type": internal_kind,
                   "event": {**internal, "delegation_id": (ev or {}).get("delegation_id")}}
            continue
        new = RENOMBRES.get(tipo)
        if new is None:
            yield e
            continue
        output = {**e, "type": new, "event": {**(ev or {}), "type": new}}
        if new == "output_audio.delta":
            m = RE_SIZE.search(str((ev or {}).get("delta", "")))
            if m:
                # The census stores the approximate size (base64 * 3/4): it is
                # rounded to 20 ms chunks.
                ms = int(m.group(1)) / by_ms
                output["event"]["audio_ms"] = round(ms / 20.0) * 20.0
        yield output
