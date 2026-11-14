"""Does this OpenAI key have access to GPT-Live? Without opening a voice session and at no cost.

Queries `/v1/models/<model>`, which is a free read, and prints only status and
error codes: never the key. It separates "the key cannot see the model" from
"the session fails for some other reason". The bridge uses the public protocol
and does not send the alpha header (whose access was withdrawn:
`quicksilver_v2_access_denied`), so it is not tested here either.

    uv run --project poc3_gpt_live python -m validation.gpt_live_access_diagnostics
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def query(url: str, headers: dict[str, str]) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as r:
            cuerpo = json.loads(r.read().decode("utf-8"))
            return r.status, f"owned_by={cuerpo.get('owned_by')!r}"
    except urllib.error.HTTPError as e:
        try:
            error = json.loads(e.read().decode("utf-8")).get("error", {})
            return e.code, f"{error.get('code')}: {error.get('message')}"
        except Exception:  # noqa: BLE001 - diagnostics
            return e.code, "(no readable body)"


def main() -> int:
    from poc3_gpt_live.app.config import load_settings

    settings = load_settings()
    base = {"Authorization": f"Bearer {settings.openai_api_key}"}
    print(f"configured model (OPENAI_LIVE_MODEL): {settings.live_model}")
    print(f"session endpoint: {settings.url_live()}  (public protocol, no alpha header)")
    print(f"key: ends in ...{settings.openai_api_key[-4:]}  (compare it with the one in the OpenAI dashboard)\n")
    for model_name in dict.fromkeys([settings.live_model, "gpt-live-1"]):
        code_value, detail = query(f"https://api.openai.com/v1/models/{model_name}", base)
        print(f"  {model_name:<28} → {code_value}  {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
