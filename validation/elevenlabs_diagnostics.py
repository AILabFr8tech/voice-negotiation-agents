"""Why does ElevenLabs return no audio? Key, plan, voice and a minimal synthesis.

When the TTS connects and closes without sending a single frame ("no audio frames
were pushed", or `agente_sin_audio` in the harness), the WebSocket does not say why.
The REST endpoint does: it returns the error with its code. This script:

1. queries the key's account (`/v1/user/subscription`): plan and characters;
2. queries the configured voice (`/v1/voices/<id>`): whether the account can see it
   and which category it is (`premade`, `professional`, `generated`...);
3. synthesizes **one word** ("Hola", 4 characters, ~2 credits) with the voice and
   model from `.env`, and prints the exact error if there is one.

Never prints the key.

    uv run --project poc1_pipecat python -m validation.elevenlabs_diagnostics
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "https://api.elevenlabs.io"


def api_request(ruta: str, key: str, *, cuerpo: dict | None = None) -> tuple[int, bytes]:
    data = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    request = urllib.request.Request(
        BASE + ruta, data=data, method="POST" if data else "GET",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def detail(raw_value: bytes) -> str:
    try:
        d = json.loads(raw_value.decode("utf-8"))
    except Exception:  # noqa: BLE001 - diagnostics
        return raw_value[:200].decode("utf-8", "replace")
    det = d.get("detail", d)
    return json.dumps(det, ensure_ascii=False)[:400]


def main() -> int:
    from poc1_pipecat.app.config import load_settings

    a = load_settings()
    key, voice, model_name = a.eleven_api_key, a.eleven_voice_id, a.eleven_model
    print(f"key: {'present, ends in ...' + key[-4:] if key else 'MISSING ELEVEN_API_KEY'}")
    print(f"voice:   {voice}\nmodel_name: {model_name}\n")
    if not key:
        return 1

    code_value, raw_value = api_request("/v1/user/subscription", key)
    if code_value == 200:
        s = json.loads(raw_value)
        print(f"1. account: plan {s.get('tier')!r} · characters used "
              f"{s.get('character_count')} of {s.get('character_limit')} · status {s.get('status')!r}")
    else:
        print(f"1. account: {code_value} {detail(raw_value)}")

    code_value, raw_value = api_request(f"/v1/voices/{voice}", key)
    if code_value == 200:
        v = json.loads(raw_value)
        print(f"2. voice: visible to the account · name {v.get('name')!r} · category {v.get('category')!r}")
    else:
        print(f"2. voice: {code_value} {detail(raw_value)}")

    code_value, raw_value = api_request(
        f"/v1/text-to-speech/{voice}?output_format=ulaw_8000", key,
        cuerpo={"text": "Hola", "model_id": model_name},
    )
    if code_value == 200:
        print(f"3. test synthesis: OK, {len(raw_value)} bytes of audio")
    else:
        print(f"3. test synthesis: {code_value} {detail(raw_value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
