"""PoC3 configuration: reads the repo-root `.env` and validates before starting.

Same `.env` and same criterion as `poc1_pipecat/app/config.py` and
`poc2_livekit/app/config.py`: everything that can change between runs lives
here, never hardcoded. Switching models must be a value change in the
`.env`, never a rewrite of the system prompt (CLAUDE.md).

Validation is split in three so that the smoke test can run without
spending anything:

    load_settings()               -> requires only OPENAI_API_KEY
    verify_for_voice(ajustes)    -> requires the live model and the delegation model
    verify_for_telephony(...)  -> additionally requires Twilio and PUBLIC_URL

No function in this module prints or logs the value of a key.

## Why PoC3 asks for fewer keys than PoC1 and PoC2

PoC1 and PoC2 require all three AI keys (OpenAI, Deepgram, ElevenLabs) because
they build a three-provider pipeline. PoC3 has no discrete STT or TTS: the
model is full-duplex and has them built in, so `DEEPGRAM_API_KEY` and
`ELEVEN_API_KEY` are not read here. **That is not a simplification of the
measurement: it is the architecture being measured**, and the report has to
be able to cite it (less integration, zero visibility into the stages).

## PoC3's own value

`OPENAI_LIVE_MODEL` is a variable of its own and does **not** reuse `OPENAI_MODEL`: that one is
pinned to `gpt-4.1-mini` for the 3 PoCs and describes the *pipeline* LLM, which in
PoC3 does not exist as an isolable stage. Here `OPENAI_MODEL` is read, but for
something else — it is the model GPT-Live **delegates** tool calling to (see
`app/tools.py`), and that is why it has to remain the same one PoC1 and
PoC2 run.

## Public protocol

The August and September campaigns ran on the alpha protocol:
`wss://api.openai.com/v1/live?model=…` with the header `OpenAI-Alpha:
quicksilver=v2`, which was mandatory. That access was withdrawn (the server replies
`quicksilver_v2_access_denied`) and the `gpt-live-1` model moved to the public API
in a different shape: `wss://api.openai.com/v1/live/sessions`, only `Authorization`,
and the model inside the first event (`session.start`) instead of in the URL.
`OPENAI_LIVE_ALPHA_HEADER` is no longer read; if it is still in the `.env` it does no harm.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# poc3_gpt_live/app/config.py -> poc3_gpt_live/app -> poc3_gpt_live -> repo root
ROOT_PATH: Final[Path] = Path(__file__).resolve().parents[2]
RUTA_ENV: Final[Path] = ROOT_PATH / ".env"

RECORDINGS_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "recordings"


def force_utf8_output() -> None:
    """Force UTF-8 on stdout/stderr. Call it at the start of any entry point.

    Same reason as in PoC1 and PoC2: on Windows, as soon as output is redirected to
    a file or goes through a pipe, Python falls back to cp1252 and a single character
    outside that set kills the process with `UnicodeEncodeError`. Where it really
    matters is when dumping the transcript **after** an already-paid call.

    (It is a deliberate copy: `shared/` cannot depend on a PoC, and one PoC
    cannot depend on another. It is ten lines with no business logic.)
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure_stream = getattr(stream, "reconfigure", None)
        if reconfigure_stream is None:
            continue
        try:
            reconfigure_stream(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic terminal
            pass


# --- GPT-Live defaults ---
# The full-duplex model of the public API. It is the same `gpt-live-1` the
# September campaign ran with (back then via the alpha protocol; see
# `docs/hallazgo_gpt_live_ga.md`). August's `-boulder-alpha` no longer exists.
DEFAULT_LIVE_MODEL: Final[str] = "gpt-live-1"

# The public API WebSocket. No query string: the model goes in
# `session.start` (see the module docstring).
LIVE_BASE_URL: Final[str] = "wss://api.openai.com/v1/live/sessions"

# Model GPT-Live delegates tool calling to (`delegation.responses.model`).
# **It is the same gpt-4.1-mini frozen for the 3 PoCs since August 3,
# 2026** (CLAUDE.md): the 4 negotiation functions run against the same
# model in all three architectures, or the tool-calling comparison would measure the
# model and not the framework. It is also the only PoC3 traffic that is billed
# (`docs_alpha/`: traffic to `gpt-live` is not charged, the delegated traffic is).
DEFAULT_DELEGATION_MODEL: Final[str] = "gpt-4.1-mini"
# The format of PoC3's October campaign: μ-law 8 kHz, Twilio's, with no
# resampling. `pcm24` reproduces the earlier campaigns.
DEFAULT_AUDIO_FORMAT: Final[str] = "pcmu"

# The model's output voice. The alpha documents `marin` as the default and exposes no
# catalog; it is left as config so as not to hardcode something that may grow.
#
# **Declared asymmetry**: PoC1 and PoC2 run the same ElevenLabs voice, PoC3
# cannot — its TTS is native. It cannot be corrected, so it is reported
# (`docs/CONTRATO_METRICAS.md` §2.5).
VOICE_DEFAULT: Final[str] = "marin"


class IncompleteConfiguration(RuntimeError):
    """An environment variable needed for what is about to run is missing."""


@dataclass(frozen=True)
class Settings:
    """Effective configuration of a run. Empty fields are optional."""

    # --- OpenAI: the only AI key PoC3 needs ---
    openai_api_key: str
    live_model: str
    voice: str
    # Model of the Responses delegation, where tool calling runs.
    delegation_model: str
    # Session audio format: `pcmu` (μ-law 8 kHz, no resampling) or
    # `pcm24` (PCM16 24 kHz, the one used in the September and October campaigns). See
    # `bridge.formato_de_sesion`.
    audio_format: str = "pcm24"
    # --- Telephony: only needed for a real call ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    twilio_destination_number: str = ""
    public_url: str = ""

    # -- GPT-Live URL and headers --

    def url_live(self) -> str:
        """The GPT-Live WebSocket. The model does NOT go here: it goes in `session.start`."""
        return LIVE_BASE_URL

    def headers_live(self) -> dict[str, str]:
        """Only `Authorization`: the public API takes no selector. Never logged."""
        return {"Authorization": f"Bearer {self.openai_api_key}"}

    # -- Public host, to build webhooks and the Media Stream WebSocket --

    def public_host(self) -> str:
        """`PUBLIC_URL` without scheme or trailing slash."""
        host = self.public_url.strip()
        for prefix in ("https://", "http://", "wss://", "ws://"):
            if host.startswith(prefix):
                host = host[len(prefix) :]
                break
        return host.rstrip("/")

    def url_webhook(self, ruta: str) -> str:
        """https URL of the webhook `ruta` (the one Twilio sees)."""
        return f"https://{self.public_host()}/{ruta.lstrip('/')}"

    def url_websocket(self, ruta: str) -> str:
        """wss URL of the Media Stream `ruta` (the one Twilio opens)."""
        return f"wss://{self.public_host()}/{ruta.lstrip('/')}"

    def safe_summary(self) -> dict[str, str]:
        """Summary for logs. Secrets come out as `<present>`/`<absent>`."""
        secret_values = {
            "openai_api_key",
            "twilio_auth_token",
            "twilio_account_sid",
        }
        summary: dict[str, str] = {}
        for campo in fields(self):
            valor = getattr(self, campo.name)
            if campo.name in secret_values:
                summary[campo.name] = "<present>" if valor else "<absent>"
            else:
                summary[campo.name] = valor or "<absent>"
        return summary


def _leer(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def load_settings(*, ruta_env: Path | None = None) -> Settings:
    """Load `.env` and return the settings with the OpenAI key validated.

    It requires neither the live model nor telephony: `verify_for_voice` and
    `verify_for_telephony` ask for those depending on what is about to run.
    """
    load_dotenv(ruta_env or RUTA_ENV, override=False)

    if not _leer("OPENAI_API_KEY"):
        raise IncompleteConfiguration(
            "OPENAI_API_KEY is missing from the repo-root .env: it is the only AI key "
            "PoC3 needs (the model has built-in STT and TTS)"
        )

    return Settings(
        openai_api_key=_leer("OPENAI_API_KEY"),
        live_model=_leer("OPENAI_LIVE_MODEL", DEFAULT_LIVE_MODEL),
        voice=_leer("OPENAI_LIVE_VOICE", VOICE_DEFAULT),
        # `OPENAI_MODEL` is the frozen variable shared by the 3 PoCs. Here it describes the
        # delegation model, which is where tool calling runs.
        delegation_model=_leer("OPENAI_MODEL", DEFAULT_DELEGATION_MODEL),
        audio_format=_leer("POC3_AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT),
        twilio_account_sid=_leer("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=_leer("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=_leer("TWILIO_PHONE_NUMBER"),
        twilio_destination_number=_leer("TWILIO_DESTINATION_NUMBER"),
        public_url=_leer("PUBLIC_URL"),
    )


def verify_for_voice(settings: Settings) -> None:
    """Require what is needed to open a real session against GPT-Live."""
    if settings.audio_format not in ("pcmu", "pcm24"):
        raise IncompleteConfiguration(
            f"POC3_FORMATO_AUDIO={settings.audio_format!r}: must be pcmu or pcm24"
        )
    if not settings.live_model:
        raise IncompleteConfiguration(
            "OPENAI_LIVE_MODEL is missing from the .env: the GPT-Live model"
        )
    if not settings.delegation_model:
        raise IncompleteConfiguration(
            "OPENAI_MODEL is empty: it is the model GPT-Live delegates "
            "tool calling to, and it has to be the same one PoC1 and PoC2 run"
        )


def verify_for_telephony(settings: Settings) -> None:
    """Require what is needed for a real call through Twilio.

    It does not chain `verify_for_voice`: whoever is going to call must invoke both,
    and keeping them separate makes a half-filled `.env` report everything missing instead
    of only the first thing that broke.

    Reminder from CLAUDE.md: the number in `TWILIO_PHONE_NUMBER` belongs to the
    dispatcher agent. Dialing out with a per-call `url` does **not** touch the
    TwiML configured on that number, and this code never modifies the
    number's configuration. The dedicated number or subaccount
    for SPOTS is still pending.
    """
    absent = [
        name
        for name, valor in (
            ("TWILIO_ACCOUNT_SID", settings.twilio_account_sid),
            ("TWILIO_AUTH_TOKEN", settings.twilio_auth_token),
            ("TWILIO_PHONE_NUMBER", settings.twilio_phone_number),
            ("PUBLIC_URL", settings.public_url),
        )
        if not valor
    ]
    if absent:
        raise IncompleteConfiguration(
            "telephony variables missing from the .env: " + ", ".join(absent)
        )


def main() -> int:
    """Print the state of the PoC3 environment without touching the network or spending anything.

        uv run --project poc3_gpt_live python -m poc3_gpt_live.app.config
    """
    force_utf8_output()

    try:
        settings = load_settings()
    except IncompleteConfiguration as error:
        print(f"[missing] {error}")
        return 1

    print("PoC3 — GPT-Live: environment status")
    for clave, valor in settings.safe_summary().items():
        print(f"  {clave:<24} {valor}")

    print(f"\n  endpoint                 {settings.url_live()}")

    print()
    pending_items: list[str] = []
    for label, check in (
        ("voice", verify_for_voice),
        ("telephony", verify_for_telephony),
    ):
        try:
            check(settings)
        except IncompleteConfiguration as error:
            pending_items.append(f"  [missing] {label}: {error}")
        else:
            print(f"  [ok] {label}")

    for linea in pending_items:
        print(linea)

    print(
        "\nThis only validates configuration; it does not touch the network. To exercise the "
        "contract without spending anything:\n"
        "  python -m poc3_gpt_live.app.smoke"
    )
    return 1 if pending_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
