"""El humo nivel 1 tampoco puede subcontar la sesión.

Copiar el uso antes de cerrar pierde los últimos segundos de sesión. El humo
puede tener ese defecto en su propio camino aunque `telefonia.create_app` no lo
tenga, y la prueba de aquél no lo cubre porque `run_level_1` no pasa por
`create_app`: arma su cliente, su censo y su volcado por separado. Con la tarifa
por sesión el humo se cotiza con ese contador, así que no es un dato de
diagnóstico.

Aquí el alpha es falso y todo lo demás es el camino real del humo: el
`PuenteDeAudio`, el colector, el censo en disco y `save_run`.
"""

from __future__ import annotations

import asyncio
import json
import wave
from typing import Any, AsyncIterator

import numpy as np
import pytest

from poc3_gpt_live.app import bridge, smoke
from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    DEFAULT_LIVE_MODEL,
    Settings,
)

# El penúltimo aviso y el total final, con los valores que de verdad salieron.
USAGE_MID_RUN = {"audio_duration_ms": 47_000, "backend_model_usage": []}
USAGE_AT_CLOSE = {"audio_duration_ms": 55_000, "backend_model_usage": []}


class _FakeAlpha:
    """Ciclo de vida del cliente real, con el total solo al cerrar."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        initial_session: dict[str, Any],
        *,
        census: Any = None,
    ) -> None:
        self.census = census
        self.usage: dict[str, Any] = {}
        self.closed = False

    async def connect(self) -> dict[str, Any]:
        return {"id": "fake_sess", "status": "started"}

    async def add_context(self, text: str) -> None:
        return None

    async def send_audio(self, pcm24: bytes) -> None:
        return None

    async def answer_function(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        event = {
            "type": bridge.EV_SESSION_USAGE,
            "usage": dict(USAGE_MID_RUN),
        }
        self.usage = dict(USAGE_MID_RUN)
        self._annotate(event)
        yield event
        # Y se queda escuchando hasta que la alimentación termine y la cancelen,
        # igual que el de verdad mientras la sesión sigue abierta.
        await asyncio.Event().wait()

    async def shut(self) -> None:
        if self.closed:
            return
        event = {"type": bridge.EV_SESSION_CLOSED, "usage": dict(USAGE_AT_CLOSE)}
        self._annotate(event)
        self.usage = dict(USAGE_AT_CLOSE)
        self.closed = True

    def _annotate(self, event: dict[str, Any]) -> None:
        if self.census is not None:
            self.census.registrar(event)


def _short_wav(destination, seconds: float = 0.4) -> Any:
    """Un WAV de voz sintética. Corto: cada segundo son 50 tramas de espera."""
    n = int(8_000 * seconds)
    t = np.arange(n) / 8_000.0
    samples = (np.sin(2 * np.pi * 300 * t) * 6_000).astype("<i2")
    with wave.open(str(destination), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8_000)
        f.writeframes(samples.tobytes())
    return destination


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Corre `run_level_1` entero contra el alpha falso."""
    monkeypatch.setattr(smoke, "RECORDINGS_PATH", tmp_path)
    monkeypatch.setattr(bridge, "GptLiveClient", _FakeAlpha)
    monkeypatch.setattr(
        smoke,
        "load_settings",
        lambda: Settings(
            openai_api_key="sk-prueba",
            live_model=DEFAULT_LIVE_MODEL,
            voice="marin",
            delegation_model=DEFAULT_DELEGATION_MODEL,
        ),
    )
    monkeypatch.setattr(smoke, "verify_for_voice", lambda settings: None)
    # El respiro final del alimentador son 5 s de reloj real que aquí no
    # compran nada: lo único que tiene que pasar después es el `finally`.
    monkeypatch.setattr(smoke, "MAX_AUDIO_SECONDS", 2.0)

    asyncio.run(smoke.run_level_1(ruta_wav=_short_wav(tmp_path / "voice.wav")))

    census = [
        json.loads(linea)
        for linea in (tmp_path / "humo" / "census_humo-nivel1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if linea.strip()
    ]
    content = json.loads(
        (tmp_path / "humo" / "call_humo-nivel1.json").read_text(encoding="utf-8")
    )
    return content, census


class TestTheSmokeJsonSaysTheSameAsItsCensus:
    def test_the_duration_is_the_same_in_both(self, run) -> None:
        content, census = run
        of_census = [
            f for f in census if f["type"] == bridge.EV_SESSION_CLOSED
        ][-1]["event"]["usage"]["audio_duration_ms"]

        assert content["gpt_live_extra"]["usage"]["audio_duration_ms"] == of_census

    def test_and_is_the_closing_total_and_not_the_previous_warning(self, run) -> None:
        """Contra el literal: si los dos se quedaran en el penúltimo valor, la
        prueba de arriba seguiría en verde y el humo seguiría subcontando."""
        content, _ = run
        usage = content["gpt_live_extra"]["usage"]

        assert usage["audio_duration_ms"] == USAGE_AT_CLOSE["audio_duration_ms"]
        assert usage["audio_duration_ms"] != USAGE_MID_RUN["audio_duration_ms"]

    def test_the_census_is_closed_with_the_close_event_inside(
        self, run
    ) -> None:
        """El censo se cierra DESPUÉS del drenado; si se cerrara antes, el
        `session.closed` no estaría en el archivo y no habría con qué cotizar."""
        _, census = run
        kinds = [f["type"] for f in census]

        assert kinds.index(bridge.EV_SESSION_USAGE) < kinds.index(
            bridge.EV_SESSION_CLOSED
        )


class TestAClosingThatBlowsUpDoesNotTakeTheRest:
    def test_usage_and_census_survive(self, tmp_path, monkeypatch) -> None:
        """Antes el `await cliente.cerrar()` iba desnudo: una excepción ahí
        dejaba el censo sin cerrar y la corrida sin volcado."""

        class _Explodes(_FakeAlpha):
            async def shut(self) -> None:
                self.usage = dict(USAGE_AT_CLOSE)
                raise RuntimeError("el socket ya no está")

        monkeypatch.setattr(smoke, "RECORDINGS_PATH", tmp_path)
        monkeypatch.setattr(bridge, "GptLiveClient", _Explodes)
        monkeypatch.setattr(
            smoke,
            "load_settings",
            lambda: Settings(
                openai_api_key="sk-prueba",
                live_model=DEFAULT_LIVE_MODEL,
                voice="marin",
                delegation_model=DEFAULT_DELEGATION_MODEL,
            ),
        )
        monkeypatch.setattr(smoke, "verify_for_voice", lambda settings: None)
        monkeypatch.setattr(smoke, "MAX_AUDIO_SECONDS", 2.0)

        asyncio.run(smoke.run_level_1(ruta_wav=_short_wav(tmp_path / "voice.wav")))

        content = json.loads(
            (tmp_path / "humo" / "call_humo-nivel1.json").read_text(
                encoding="utf-8"
            )
        )
        assert content["gpt_live_extra"]["usage"]["audio_duration_ms"] == 55_000
        assert (tmp_path / "humo" / "census_humo-nivel1.jsonl").is_file()
