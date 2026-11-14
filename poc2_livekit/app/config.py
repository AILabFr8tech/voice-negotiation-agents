"""PoC2 configuration: reads the root `.env` and validates before starting.

Same `.env` and same criterion as `poc1_pipecat/app/config.py`: everything that
can change between runs lives here, never hardcoded. The AI models are **the
same frozen values** as PoC1 (CLAUDE.md), because the comparison measures the
voice framework, not the model.

Validation is split into four so that the environment can be checked without
spending anything:

    load_settings()                -> requires only the three AI keys
    verify_for_voice(settings)     -> additionally requires the ElevenLabs voice
    verify_for_livekit(settings) -> requires the LiveKit project and the isolation
    verify_for_telephony(...)   -> additionally requires the outbound SIP trunk

No function in this module prints or logs the value of a key.

## Why there is an isolation check in the code

The LiveKit project (`LIVEKIT_URL`) is the one of the **real dispatcher
agent**, which is in use. We share project, keys and trunk; what we do not
share is the agent name, and all of the isolation hangs on that:

- A worker registered with `agent_name` does **not** receive automatic
  dispatch. It only joins a room when someone requests it by name with
  `CreateAgentDispatch`. As long as PoC2's name differs from the dispatcher's,
  no dispatcher job can land on our agent, nor the other way around.
- That is why `AGENT_NAME` from the `.env` (today `outbound-dispatcher-test`)
  is **never** read from here, and `verify_for_livekit` blows up if someone
  puts that same value in `LIVEKIT_AGENT_NAME`. It is the kind of error that
  only shows up once a real carrier from the other system has already answered.

This module does not create, modify or delete dispatch rules, trunks or inbound
rules. It only reads configuration.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# poc2_livekit/app/config.py -> poc2_livekit/app -> poc2_livekit -> root
ROOT_PATH: Final[Path] = Path(__file__).resolve().parents[2]
RUTA_ENV: Final[Path] = ROOT_PATH / ".env"


def force_utf8_output() -> None:
    """Forces UTF-8 on stdout/stderr. Call it at the start of any entry point.

    Same reason as in PoC1: on Windows, as soon as output is redirected to a
    file or goes through a pipe, Python falls back to cp1252 and a single
    character outside that set kills the process with `UnicodeEncodeError`.
    Where it really matters is when dumping the transcript **after** a call
    that has already been paid for.

    (It is a deliberate copy: `shared/` cannot depend on a PoC, and a PoC
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


# --- Model defaults: identical to PoC1, and they must stay that way ---
# gpt-4.1-mini is frozen for all 3 PoCs. If it changes, it changes in all three
# and the calls already made are repeated (CLAUDE.md).
DEFAULT_LLM_MODEL: Final[str] = "gpt-4.1-mini"
DEFAULT_STT_MODEL: Final[str] = "nova-3"
DEFAULT_STT_LANGUAGE: Final[str] = "multi"
# eleven_flash_v2_5, NOT eleven_flash_v2: v2 is English only.
DEFAULT_TTS_MODEL: Final[str] = "eleven_flash_v2_5"

# Name under which PoC2's worker registers. It must differ from the
# dispatcher's: it is the isolation mechanism, not a cosmetic label.
DEFAULT_AGENT_NAME: Final[str] = "spots-poc2-negociador"

# Audio turn detector version. **It is pinned on purpose instead of letting
# LiveKit choose it.**
#
# `TurnDetector` without `version=` resolves like this (inference/eot/detector.py):
#
#     "v1" if (utils.is_hosted() or utils.is_dev_mode()) else "v1-mini"
#
# In other words: the detector architecture would depend on **how the worker
# was launched** —`dev`/`console` gives v1 (remote), `start` gives v1-mini
# (local)—, which is an indefensible basis for a benchmark. It is pinned so the
# run does not depend on that.
#
# `v1-mini` is chosen: it runs **locally, on CPU**, which is how all of PoC2
# runs — the worker is launched from the terminal and nothing is deployed on
# LiveKit Cloud. Requesting `v1` would add a network hop to LiveKit Inference
# on the turn-taking path that the rest of the architecture does not have, and
# on top of that one that PoC1 does not pay either, because its Smart Turn v3
# runs in-process.
#
# So `v1-mini` is not a concession: it is **local detector against local
# detector**, which removes a confound from the latency comparison instead of
# adding one. If PoC2 is ever deployed on Cloud, `v1` is the correct value and
# it is a one-line change in `.env` — but then the calls have to be
# repeated.
VERSION_DETECTOR_DEFAULT: Final[str] = "v1-mini"
VALID_DETECTOR_VERSIONS: Final[tuple[str, ...]] = ("v1", "v1-mini")

# Prefix of PoC2's rooms. It makes them recognizable at a glance in the shared
# project's dashboard and keeps them from colliding with the dispatcher's.
ROOM_PREFIX: Final[str] = "spots-poc2"

# Identity of the SIP participant inside the room. It lives here because both
# sides use it and they must match: `app/call.py` sets it when creating the
# participant and `app/agent.py` waits for it to know when the call was
# answered. If they drift apart, the agent waits for someone who never arrives.
CARRIER_IDENTITY: Final[str] = "carrier"

# The dispatcher's variable. It is named here so that it is **not** read, and
# so it can be compared against in the isolation check.
DISPATCHER_AGENT_KEY: Final[str] = "AGENT_NAME"

_AI_KEYS: Final[tuple[str, ...]] = (
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "ELEVEN_API_KEY",
)


class IncompleteConfiguration(RuntimeError):
    """An environment variable needed for what is about to run is missing."""


class CollisionWithDispatcher(RuntimeError):
    """PoC2's configuration overlaps with that of the real dispatcher agent."""


@dataclass(frozen=True)
class Settings:
    """Effective configuration of a run. Empty fields are optional."""

    # --- AI: same providers and models as PoC1 ---
    openai_api_key: str
    openai_model: str
    deepgram_api_key: str
    deepgram_model: str
    deepgram_language: str
    eleven_api_key: str
    eleven_model: str
    # ElevenLabs voice: only needed when there is real TTS.
    eleven_voice_id: str = ""
    # --- LiveKit: project and worker ---
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    agent_name: str = DEFAULT_AGENT_NAME
    turn_detector_version: str = VERSION_DETECTOR_DEFAULT
    # --- Telephony: outbound SIP trunk, only for real calls ---
    sip_outbound_trunk_id: str = ""
    destination_number: str = ""

    def room_name(
        self,
        *,
        scenario: int | str | None = None,
        repetition: int | str | None = None,
        suffix: str = "",
    ) -> str:
        """Room name of a run: `spots-poc2-e2-r1-<suffix>`.

        The prefix makes it obvious in the shared project's dashboard which
        rooms belong to SPOTS and which to the dispatcher. Scenario and
        repetition go in the name because the room name is the only thing that
        survives in the LiveKit console even if the local process dies.

        `suffix` is set by the caller (normally a timestamp): two runs of the
        same scenario and repetition must not land in the same room.
        """
        pieces = [ROOM_PREFIX]
        if scenario is not None and str(scenario).strip():
            pieces.append(f"e{str(scenario).strip()}")
        if repetition is not None and str(repetition).strip():
            pieces.append(f"r{str(repetition).strip()}")
        if suffix.strip():
            pieces.append(suffix.strip())
        return "-".join(pieces)

    def safe_summary(self) -> dict[str, str]:
        """Summary for logs. Secrets come out as `<present>`/`<absent>`."""
        secret_values = {
            "openai_api_key",
            "deepgram_api_key",
            "eleven_api_key",
            "livekit_api_key",
            "livekit_api_secret",
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
    """Loads `.env` and returns the settings with the three AI keys validated.

    It does not require voice, LiveKit or telephony: those are requested by the
    `verify_*` functions depending on what is about to run.
    """
    load_dotenv(ruta_env or RUTA_ENV, override=False)

    absent = [key for key in _AI_KEYS if not _leer(key)]
    if absent:
        raise IncompleteConfiguration(
            "AI keys missing from the root .env: " + ", ".join(absent)
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
        livekit_url=_leer("LIVEKIT_URL"),
        livekit_api_key=_leer("LIVEKIT_API_KEY"),
        livekit_api_secret=_leer("LIVEKIT_API_SECRET"),
        # Deliberately does NOT fall back to `AGENT_NAME`: that one is the dispatcher's.
        agent_name=_leer("LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME),
        turn_detector_version=_leer(
            "LIVEKIT_TURN_DETECTOR_VERSION", VERSION_DETECTOR_DEFAULT
        ),
        sip_outbound_trunk_id=_leer("SIP_OUTBOUND_TRUNK_ID"),
        # PoC1's destination is reused: the comparison calls go to our own
        # cell phone, and it must be the same number in all 3 PoCs.
        destination_number=_leer("TWILIO_DESTINATION_NUMBER"),
    )


def verify_for_voice(settings: Settings) -> None:
    """Requires what is needed to synthesize real audio."""
    if not settings.eleven_voice_id:
        raise IncompleteConfiguration(
            "ELEVENLABS_VOICE_ID missing from the .env: it must be the SAME voice "
            "PoC1 uses, or the naturalness comparison does not measure the framework"
        )


def verify_isolation(settings: Settings) -> None:
    """PoC2's agent cannot have the same name as the dispatcher's.

    A worker with its own `agent_name` only joins the rooms where it is
    requested by name. If the two names match, that isolation disappears and a
    dispatcher job can land on this agent — with a real carrier from the other
    system on the phone.
    """
    if not settings.agent_name.strip():
        raise IncompleteConfiguration(
            "LIVEKIT_AGENT_NAME is empty: without an agent name the worker would "
            "receive automatic dispatch and join rooms that are not its own"
        )

    of_dispatcher = (os.getenv(DISPATCHER_AGENT_KEY) or "").strip()
    if of_dispatcher and settings.agent_name.strip() == of_dispatcher:
        raise CollisionWithDispatcher(
            f"LIVEKIT_AGENT_NAME is {settings.agent_name!r}, the same value as "
            f"{DISPATCHER_AGENT_KEY} (the real dispatcher agent). The agent name "
            "is what isolates the two PoCs within the same LiveKit project: give "
            "PoC2 a different one"
        )


def verify_for_livekit(settings: Settings) -> None:
    """Requires the LiveKit project and checks isolation from the dispatcher."""
    absent = [
        name
        for name, valor in (
            ("LIVEKIT_URL", settings.livekit_url),
            ("LIVEKIT_API_KEY", settings.livekit_api_key),
            ("LIVEKIT_API_SECRET", settings.livekit_api_secret),
        )
        if not valor
    ]
    if absent:
        raise IncompleteConfiguration(
            "LiveKit variables missing from the .env: " + ", ".join(absent)
        )
    if settings.turn_detector_version not in VALID_DETECTOR_VERSIONS:
        raise IncompleteConfiguration(
            f"LIVEKIT_TURN_DETECTOR_VERSION={settings.turn_detector_version!r} is not "
            f"valid: use one of {list(VALID_DETECTOR_VERSIONS)}. Leaving it on "
            "automatic would make the detector architecture depend on how the "
            "worker was launched"
        )
    verify_isolation(settings)


def verify_for_telephony(settings: Settings) -> None:
    """Requires what is needed to dial over the outbound SIP trunk.

    It does not chain the other `verify_*` functions: whoever is about to call
    must invoke all of them, and keeping them separate makes a half-filled
    `.env` report everything that is missing instead of only the first thing
    that broke.

    The trunk (`SIP_OUTBOUND_TRUNK_ID`) is referenced by id and **never**
    modified: PoC2 only names it when creating the SIP participant. Creating,
    editing or deleting trunks and dispatch rules is entirely outside this code.
    """
    absent = [
        name
        for name, valor in (
            ("SIP_OUTBOUND_TRUNK_ID", settings.sip_outbound_trunk_id),
            ("TWILIO_DESTINATION_NUMBER", settings.destination_number),
        )
        if not valor
    ]
    if absent:
        raise IncompleteConfiguration(
            "telephony variables missing from the .env: " + ", ".join(absent)
        )


def main() -> int:
    """Prints the state of PoC2's environment without touching the network or spending anything.

        uv run --project poc2_livekit python -m poc2_livekit.app.config
    """
    force_utf8_output()

    try:
        settings = load_settings()
    except IncompleteConfiguration as error:
        print(f"[missing] {error}")
        return 1

    print("PoC2 — LiveKit Agents: environment status")
    for clave, valor in settings.safe_summary().items():
        print(f"  {clave:<24} {valor}")

    print()
    pending_items: list[str] = []
    for label, check in (
        ("voice", verify_for_voice),
        ("LiveKit", verify_for_livekit),
        ("telephony", verify_for_telephony),
    ):
        try:
            check(settings)
        except (IncompleteConfiguration, CollisionWithDispatcher) as error:
            pending_items.append(f"  [{label}] {error}")
        else:
            print(f"  [ok] {label}")

    for linea in pending_items:
        print(linea)

    print(
        "\nThis only validates configuration; it does not touch the network. To "
        "exercise the pipeline without spending a call:\n"
        "  python -m poc2_livekit.app.smoke"
    )
    return 1 if pending_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
