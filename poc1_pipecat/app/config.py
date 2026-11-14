"""PoC1 configuration: reads the repo-root `.env` and validates before starting.

Everything that may change between runs lives here, never hardcoded in the
pipeline: the LLM model, the STT model and language, the TTS voice. Switching
model provider must be a change of value in `.env`, never a rewrite of the
system prompt (CLAUDE.md).

Validation is split in three so that the offline smoke test can run without
telephony and without voice:

    load_settings()               -> requires only the three AI keys
    verify_for_voice(settings)    -> additionally requires the ElevenLabs voice
    verify_for_telephony(...)  -> additionally requires Twilio and PUBLIC_URL

No function in this module prints or logs the value of a key.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# poc1_pipecat/app/config.py -> poc1_pipecat/app -> poc1_pipecat -> root
ROOT_PATH: Final[Path] = Path(__file__).resolve().parents[2]
RUTA_ENV: Final[Path] = ROOT_PATH / ".env"


def force_utf8_output() -> None:
    """Force UTF-8 on stdout/stderr. Call it at the start of any entry point.

    In a real Windows terminal Python uses UTF-8, but as soon as the output is
    redirected to a file or piped, it falls back to cp1252 — and then a single
    character outside that charset crashes the process with `UnicodeEncodeError`.
    It actually happened: the arrow in "text→TTS" in the smoke test's timing
    table killed the whole run when the output was redirected, after the money
    for the three APIs had already been spent.

    Where it really matters is the telephony server: there the transcript dump
    happens **after** hanging up a call that has already been paid for, and that
    transcript carries arbitrary STT text. Losing the evidence to the console
    encoding would be the worst possible moment to fail.

    `errors="replace"` is the fallback: better an odd character on screen than
    losing the run.

    (`shared/metrics/cli.py` has its own copy of this: `shared/` cannot depend
    on `poc1_pipecat/`.)
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure_stream = getattr(stream, "reconfigure", None)
        if reconfigure_stream is None:
            continue
        try:
            reconfigure_stream(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic terminal
            pass

# --- Model defaults ---
# gpt-4.1-mini, the same LLM in all 3 PoCs (in PoC3 it is the delegate model).
# The LLM dominates turn latency, so the fastest model in the same ecosystem is
# the big lever (measurements in CLAUDE.md). It is not a reasoning model: there
# is no `reasoning_effort` to turn off, and pipecat 1.6 does not expose that
# knob in `OpenAILLMService.Settings` either. Do not change it mid-campaign: it
# would break comparability across PoCs (CLAUDE.md).
DEFAULT_LLM_MODEL: Final[str] = "gpt-4.1-mini"
# nova-3 with language="multi" is what enables the es->en switch requested by
# the last line of the prompt's ESTILO DE VOZ section. If the bake-off shows a
# better WER with monolingual Spanish nova-3, it is a change of value in `.env`.
DEFAULT_STT_MODEL: Final[str] = "nova-3"
DEFAULT_STT_LANGUAGE: Final[str] = "multi"
# eleven_flash_v2_5, NOT eleven_flash_v2: v2 is English-only.
DEFAULT_TTS_MODEL: Final[str] = "eleven_flash_v2_5"

_AI_KEYS: Final[tuple[str, ...]] = (
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "ELEVEN_API_KEY",
)


class IncompleteConfiguration(RuntimeError):
    """An environment variable needed for what is about to run is missing."""


@dataclass(frozen=True)
class Settings:
    """Effective configuration of a run. Empty fields are optional."""

    # --- AI (keys confirmed in CLAUDE.md) ---
    openai_api_key: str
    openai_model: str
    deepgram_api_key: str
    deepgram_model: str
    deepgram_language: str
    eleven_api_key: str
    eleven_model: str
    # ElevenLabs voice: only needed when there is real TTS.
    eleven_voice_id: str = ""
    # --- Telephony: only needed when there is a real call ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    twilio_destination_number: str = ""
    public_url: str = ""

    # -- Public host, to build webhooks and the bot's WebSocket --

    def public_host(self) -> str:
        """`PUBLIC_URL` without scheme or trailing slash."""
        host = self.public_url.strip()
        for prefix in ("https://", "http://", "wss://", "ws://"):
            if host.startswith(prefix):
                host = host[len(prefix) :]
                break
        return host.rstrip("/")

    def url_webhook(self, ruta: str) -> str:
        """https URL of the webhook at `ruta` (the one Twilio sees)."""
        return f"https://{self.public_host()}/{ruta.lstrip('/')}"

    def url_websocket(self, ruta: str) -> str:
        """wss URL of the Media Stream at `ruta` (the one Twilio opens)."""
        return f"wss://{self.public_host()}/{ruta.lstrip('/')}"

    def safe_summary(self) -> dict[str, str]:
        """Summary for logs. Secrets come out as `<presente>`/`<ausente>` (present/absent)."""
        secret_values = {
            "openai_api_key",
            "deepgram_api_key",
            "eleven_api_key",
            "twilio_auth_token",
            "twilio_account_sid",
        }
        summary: dict[str, str] = {}
        for campo in fields(self):
            valor = getattr(self, campo.name)
            if campo.name in secret_values:
                summary[campo.name] = "<presente>" if valor else "<ausente>"
            else:
                summary[campo.name] = valor or "<ausente>"
        return summary


def _leer(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def load_settings(*, ruta_env: Path | None = None) -> Settings:
    """Load `.env` and return the settings with the three AI keys validated.

    It does not require voice or telephony: `verify_for_voice` and
    `verify_for_telephony` ask for those, depending on what is about to run.
    """
    load_dotenv(ruta_env or RUTA_ENV, override=False)

    absent = [key for key in _AI_KEYS if not _leer(key)]
    if absent:
        raise IncompleteConfiguration(
            "missing AI keys in the repo-root .env: " + ", ".join(absent)
        )

    return Settings(
        openai_api_key=_leer("OPENAI_API_KEY"),
        openai_model=_leer("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        deepgram_api_key=_leer("DEEPGRAM_API_KEY"),
        deepgram_model=_leer("DEEPGRAM_MODEL", DEFAULT_STT_MODEL),
        deepgram_language=_leer("DEEPGRAM_LANGUAGE", DEFAULT_STT_LANGUAGE),
        eleven_api_key=_leer("ELEVEN_API_KEY"),
        eleven_model=_leer("ELEVENLABS_MODEL", DEFAULT_TTS_MODEL),
        eleven_voice_id=_leer("ELEVENLABS_VOICE_ID"),
        twilio_account_sid=_leer("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=_leer("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=_leer("TWILIO_PHONE_NUMBER"),
        twilio_destination_number=_leer("TWILIO_DESTINATION_NUMBER"),
        public_url=_leer("PUBLIC_URL"),
    )


def verify_for_voice(settings: Settings) -> None:
    """Require what is needed to synthesize real audio."""
    if not settings.eleven_voice_id:
        raise IncompleteConfiguration(
            "missing ELEVENLABS_VOICE_ID in the .env: pick an ElevenLabs voice "
            "that speaks neutral Spanish and paste its id"
        )


def verify_for_telephony(settings: Settings) -> None:
    """Require what is needed for a real call through Twilio.

    It does not chain `verify_for_voice`: whoever is going to call must invoke
    both, and keeping them separate makes a half-filled `.env` report everything
    that is missing instead of only the first thing that broke.

    Reminder from CLAUDE.md: the number in `TWILIO_PHONE_NUMBER` belongs to the
    dispatcher agent. Dialing out with a per-call `url` does **not** touch the
    TwiML configured on that number, and this code never modifies the number's
    configuration. The dedicated number or subaccount for SPOTS is still
    pending.
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
            "missing telephony variables in the .env: " + ", ".join(absent)
        )
