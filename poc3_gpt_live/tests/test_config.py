"""PoC3 configuration: the GPT-Live endpoint and what is frozen."""

from __future__ import annotations

import pytest

from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    DEFAULT_LIVE_MODEL,
    LIVE_BASE_URL,
    Settings,
    IncompleteConfiguration,
    load_settings,
    verify_for_telephony,
    verify_for_voice,
)


def _settings(**changes) -> Settings:
    base = {
        "openai_api_key": "sk-prueba",
        "live_model": DEFAULT_LIVE_MODEL,
        "voice": "marin",
        "delegation_model": DEFAULT_DELEGATION_MODEL,
    }
    base.update(changes)
    return Settings(**base)


class TestPublicEndpoint:
    def test_the_url_is_the_sessions_one_without_query_string(self):
        """The model goes in `session.start`, not in the URL (with the alpha it was the other way round)."""
        url = _settings().url_live()
        assert url == LIVE_BASE_URL == "wss://api.openai.com/v1/live/sessions"

    def test_only_sends_authorization(self):
        """No `OpenAI-Alpha`: that access was withdrawn (`quicksilver_v2_access_denied`)."""
        assert _settings().headers_live() == {"Authorization": "Bearer sk-prueba"}

    def test_the_default_model_is_the_public_api_one(self):
        assert DEFAULT_LIVE_MODEL == "gpt-live-1"


class TestFrozenModel:
    def test_the_delegation_runs_the_model_of_the_3_poc(self):
        """If PoC3 delegated to another model, the comparison would be measuring the model."""
        assert DEFAULT_DELEGATION_MODEL == "gpt-4.1-mini"

    def test_the_live_model_is_not_the_pipeline_one(self):
        """They are two different variables on purpose: they describe two different things."""
        assert DEFAULT_LIVE_MODEL != DEFAULT_DELEGATION_MODEL


class TestSecrets:
    def test_the_summary_does_not_leak_keys(self):
        summary = _settings(twilio_auth_token="token-real").safe_summary()
        assert summary["openai_api_key"] == "<present>"
        assert summary["twilio_auth_token"] == "<present>"
        assert "sk-prueba" not in str(summary)
        assert "token-real" not in str(summary)

    def test_what_is_not_secret_is_visible(self):
        summary = _settings().safe_summary()
        assert summary["live_model"] == DEFAULT_LIVE_MODEL
        assert summary["voice"] == "marin"


class TestValidations:
    def test_voice_requires_the_live_model(self):
        verify_for_voice(_settings())
        with pytest.raises(IncompleteConfiguration, match="OPENAI_LIVE_MODEL"):
            verify_for_voice(_settings(live_model=""))

    def test_voice_requires_the_delegation_model(self):
        with pytest.raises(IncompleteConfiguration, match="OPENAI_MODEL"):
            verify_for_voice(_settings(delegation_model=""))

    def test_telephony_reports_everything_missing_at_once(self):
        """A half-filled .env must report everything, not just the first thing that blew up."""
        with pytest.raises(IncompleteConfiguration) as error:
            verify_for_telephony(_settings())
        message = str(error.value)
        for key in (
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
            "PUBLIC_URL",
        ):
            assert key in message

    def test_telephony_does_not_chain_the_voice_verification(self):
        """They are separate so that each one reports its own gap."""
        verify_for_telephony(
            _settings(
                live_model="",
                twilio_account_sid="AC",
                twilio_auth_token="t",
                twilio_phone_number="+1",
                public_url="ejemplo.ngrok.dev",
            )
        )


class TestUrls:
    @pytest.mark.parametrize(
        "raw_value",
        [
            "ejemplo.ngrok.dev",
            "https://ejemplo.ngrok.dev",
            "wss://ejemplo.ngrok.dev/",
            "http://ejemplo.ngrok.dev//",
        ],
    )
    def test_the_host_is_normalized_however_it_comes(self, raw_value):
        assert _settings(public_url=raw_value).public_host() == "ejemplo.ngrok.dev"

    def test_builds_the_webhook_and_the_websocket(self):
        settings = _settings(public_url="https://ejemplo.ngrok.dev")
        assert settings.url_webhook("twiml") == "https://ejemplo.ngrok.dev/twiml"
        assert settings.url_websocket("/ws") == "wss://ejemplo.ngrok.dev/ws"


class TestCarga:
    def test_without_openai_key_it_does_not_start(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        empty = tmp_path / ".env"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(IncompleteConfiguration, match="OPENAI_API_KEY"):
            load_settings(ruta_env=empty)

    def test_the_alpha_defaults_are_applied(self, tmp_path, monkeypatch):
        for key in (
            "OPENAI_LIVE_MODEL",
            "OPENAI_LIVE_ALPHA_HEADER",
            "OPENAI_LIVE_VOICE",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
        empty = tmp_path / ".env"
        empty.write_text("", encoding="utf-8")

        settings = load_settings(ruta_env=empty)
        assert settings.live_model == DEFAULT_LIVE_MODEL
        assert settings.voice == "marin"
