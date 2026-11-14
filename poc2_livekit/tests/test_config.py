"""PoC2 configuration: tiered validation and isolation from the dispatcher.

No test touches the network, the real `.env` or a key: each one builds its own
environment with `monkeypatch` and points `ruta_env` at a file that does not
exist (which is what makes `load_dotenv` harmless here).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from poc2_livekit.app.config import (
    DEFAULT_STT_LANGUAGE,
    DISPATCHER_AGENT_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_AGENT_NAME,
    ROOM_PREFIX,
    Settings,
    CollisionWithDispatcher,
    IncompleteConfiguration,
    load_settings,
    verify_isolation,
    verify_for_livekit,
    verify_for_telephony,
    verify_for_voice,
)

# Every variable this module can read. They are cleared at the start of each
# test so that the real `.env` —or whatever another PoC loaded into the
# process— does not leak into the result.
KEYS = (
    "LIVEKIT_TURN_DETECTOR_VERSION",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "DEEPGRAM_API_KEY",
    "DEEPGRAM_MODEL",
    "DEEPGRAM_LANGUAGE",
    "ELEVEN_API_KEY",
    "ELEVENLABS_MODEL",
    "ELEVENLABS_VOICE_ID",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_AGENT_NAME",
    "SIP_OUTBOUND_TRUNK_ID",
    "TWILIO_DESTINATION_NUMBER",
    DISPATCHER_AGENT_KEY,
)

NONEXISTENT_ENV = Path(__file__).with_name("_no_existe.env")


@pytest.fixture()
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _with_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-prueba")
    monkeypatch.setenv("ELEVEN_API_KEY", "el-prueba")


def _load() -> Settings:
    return load_settings(ruta_env=NONEXISTENT_ENV)


# --------------------------------------------------------------------------
# Validation tiers
# --------------------------------------------------------------------------


def test_without_ai_keys_it_does_not_load(clean_environment):
    with pytest.raises(IncompleteConfiguration) as error:
        _load()
    for key in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVEN_API_KEY"):
        assert key in str(error.value)


def test_with_the_three_ai_keys_it_loads_without_requiring_livekit(clean_environment):
    """Level 0 must not require LiveKit: the environment is validated without a project."""
    _with_ai(clean_environment)
    settings = _load()
    assert settings.openai_api_key == "sk-prueba"
    assert settings.livekit_url == ""


def test_elevenlabs_voice_is_missing(clean_environment):
    _with_ai(clean_environment)
    with pytest.raises(IncompleteConfiguration, match="ELEVENLABS_VOICE_ID"):
        verify_for_voice(_load())


def test_livekit_reports_all_missing_variables_at_once(clean_environment):
    """A half-filled `.env` must report everything that is missing, not just the first item."""
    _with_ai(clean_environment)
    clean_environment.setenv("LIVEKIT_URL", "wss://ejemplo.livekit.cloud")
    with pytest.raises(IncompleteConfiguration) as error:
        verify_for_livekit(_load())
    assert "LIVEKIT_API_KEY" in str(error.value)
    assert "LIVEKIT_API_SECRET" in str(error.value)
    assert "LIVEKIT_URL" not in str(error.value)


def test_telephony_requires_trunk_and_destination(clean_environment):
    _with_ai(clean_environment)
    with pytest.raises(IncompleteConfiguration) as error:
        verify_for_telephony(_load())
    assert "SIP_OUTBOUND_TRUNK_ID" in str(error.value)
    assert "TWILIO_DESTINATION_NUMBER" in str(error.value)


# --------------------------------------------------------------------------
# Isolation from the real dispatcher agent
# --------------------------------------------------------------------------


def test_the_agent_name_does_not_fall_back_to_the_dispatcher_variable(clean_environment):
    """`AGENT_NAME` belongs to the dispatcher: PoC2 must not inherit it by accident."""
    _with_ai(clean_environment)
    clean_environment.setenv(DISPATCHER_AGENT_KEY, "outbound-dispatcher-test")
    settings = _load()
    assert settings.agent_name == DEFAULT_AGENT_NAME
    assert settings.agent_name != "outbound-dispatcher-test"


def test_name_collision_with_the_dispatcher_blows_up(clean_environment):
    _with_ai(clean_environment)
    clean_environment.setenv(DISPATCHER_AGENT_KEY, "outbound-dispatcher-test")
    clean_environment.setenv("LIVEKIT_AGENT_NAME", "outbound-dispatcher-test")
    with pytest.raises(CollisionWithDispatcher):
        verify_isolation(_load())


def test_a_blank_name_in_the_env_falls_back_to_the_safe_default(clean_environment):
    """A blank value does not leave the worker without a name: it falls back to PoC2's default.

    This matters because an empty `agent_name` would be dangerous —the worker
    would receive automatic dispatch and join other people's rooms—, so the
    `.env` path must never be able to produce one.
    """
    _with_ai(clean_environment)
    clean_environment.setenv("LIVEKIT_AGENT_NAME", "   ")
    settings = _load()
    assert settings.agent_name == DEFAULT_AGENT_NAME
    verify_isolation(settings)  # does not blow up


def test_empty_agent_name_blows_up(clean_environment):
    """The other path: `Settings` built by hand without a name, from code."""
    _with_ai(clean_environment)
    settings = replace(_load(), agent_name="")
    with pytest.raises(IncompleteConfiguration, match="LIVEKIT_AGENT_NAME"):
        verify_isolation(settings)


def test_livekit_complete_but_with_collision_still_blows_up(clean_environment):
    """The isolation check is part of the LiveKit gate, not an extra."""
    _with_ai(clean_environment)
    clean_environment.setenv("LIVEKIT_URL", "wss://ejemplo.livekit.cloud")
    clean_environment.setenv("LIVEKIT_API_KEY", "API-prueba")
    clean_environment.setenv("LIVEKIT_API_SECRET", "secreto-prueba")
    clean_environment.setenv(DISPATCHER_AGENT_KEY, "outbound-dispatcher-test")
    clean_environment.setenv("LIVEKIT_AGENT_NAME", "outbound-dispatcher-test")
    with pytest.raises(CollisionWithDispatcher):
        verify_for_livekit(_load())


# --------------------------------------------------------------------------
# Frozen config of the comparison
# --------------------------------------------------------------------------


def test_the_default_models_are_the_same_as_poc1(clean_environment):
    """If these values drift from PoC1, the comparison stops measuring the
    framework and starts measuring the model (CLAUDE.md)."""
    _with_ai(clean_environment)
    settings = _load()
    assert settings.openai_model == DEFAULT_LLM_MODEL == "gpt-4.1-mini"
    assert settings.deepgram_model == DEFAULT_STT_MODEL == "nova-3"
    assert settings.deepgram_language == DEFAULT_STT_LANGUAGE == "multi"
    assert settings.eleven_model == DEFAULT_TTS_MODEL == "eleven_flash_v2_5"


def test_the_model_can_be_changed_from_the_env(clean_environment):
    """Changing the model is config, never a prompt rewrite."""
    _with_ai(clean_environment)
    clean_environment.setenv("OPENAI_MODEL", "gpt-4.1")
    assert _load().openai_model == "gpt-4.1"


# --------------------------------------------------------------------------
# Turn detector
# --------------------------------------------------------------------------


def test_an_invalid_detector_version_blows_up(clean_environment):
    """Leaving it on automatic would make the architecture depend on the launcher."""
    _with_ai(clean_environment)
    for key, valor in (
        ("LIVEKIT_URL", "wss://ejemplo.livekit.cloud"),
        ("LIVEKIT_API_KEY", "API-prueba"),
        ("LIVEKIT_API_SECRET", "secreto-prueba"),
        ("LIVEKIT_TURN_DETECTOR_VERSION", "auto"),
    ):
        clean_environment.setenv(key, valor)
    with pytest.raises(IncompleteConfiguration, match="LIVEKIT_TURN_DETECTOR_VERSION"):
        verify_for_livekit(_load())


def test_the_default_detector_is_v1_mini(clean_environment):
    """Local on CPU, the way all of PoC2 runs.

    The default matters: if the `.env` does not carry the variable, the
    campaign must not fall onto the remote model without anyone asking for it.
    And the SDK does not help —with an explicit `version` that is not exactly
    "v1", `TurnDetector` silently resolves to v1-mini—, so the correct value
    has to be here.
    """
    _with_ai(clean_environment)
    assert _load().turn_detector_version == "v1-mini"


# --------------------------------------------------------------------------
# Rooms and secrets
# --------------------------------------------------------------------------


def test_the_room_name_carries_prefix_scenario_and_repetition(clean_environment):
    _with_ai(clean_environment)
    room = _load().room_name(scenario=2, repetition=1, suffix="20260805")
    assert room == f"{ROOM_PREFIX}-e2-r1-20260805"


def test_the_room_name_omits_the_empty_pieces(clean_environment):
    _with_ai(clean_environment)
    assert _load().room_name() == ROOM_PREFIX


def test_the_summary_does_not_leak_secrets(clean_environment):
    _with_ai(clean_environment)
    clean_environment.setenv("LIVEKIT_API_SECRET", "secreto-que-no-debe-salir")
    summary = _load().safe_summary()
    assert summary["livekit_api_secret"] == "<present>"
    assert "secreto-que-no-debe-salir" not in " ".join(summary.values())
    assert "sk-prueba" not in " ".join(summary.values())
