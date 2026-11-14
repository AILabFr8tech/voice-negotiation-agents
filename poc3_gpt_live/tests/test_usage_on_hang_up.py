"""El JSON de la llamada tiene que decir la misma duración que el censo.

La tarifa de `gpt-live-1` se cobra por tiempo de **sesión**, así que
`usage.audio_duration_ms` no es un dato de diagnóstico: es el contador con el
que se cotiza. El total final llega en `session.closed`, y `session.closed`
llega mientras `ClienteGptLive.cerrar` drena — o sea **después** de que el
puente terminó. Copiar el uso antes de cerrar dejaba en el JSON el penúltimo
`session.usage.updated`.

Lo que se prueba aquí es el orden del `finally` de `create_app`, no una función
suelta: el defecto vivía justo ahí, y una prueba sobre una pieza aparte no lo
habría visto. El cliente es falso porque el alpha cuesta dinero; el censo, la
ruta del websocket, el puente y la escritura a disco son los de producción.

Las cifras de cuánto costaba el sesgo están en `docs/costos_poc3.md`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from poc3_gpt_live.app import bridge, telephony
from poc3_gpt_live.app.config import (
    DEFAULT_DELEGATION_MODEL,
    DEFAULT_LIVE_MODEL,
    Settings,
)
from shared.negotiation_engine import TEST_TRIP

CALL_SID = "CAusagetest"
STREAM_SID = "MZprueba"

# El penúltimo aviso y el total final. La distancia entre los dos es el sesgo
# que esta prueba existe para impedir.
USAGE_MID_CALL = {"audio_duration_ms": 75_000}
USAGE_AT_CLOSE = {"audio_duration_ms": 89_000}


class _FakeClient:
    """Un `ClienteGptLive` con el ciclo de vida del de verdad y sin WebSocket.

    Reproduce lo único que importa aquí y que el de verdad hace: `escuchar`
    deja `uso` en el último `session.usage.updated`, y el total no aparece
    hasta `cerrar`, que es quien drena el `session.closed`.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        initial_session: dict[str, Any],
        *,
        census: bridge.EventCensus | None = None,
    ) -> None:
        self.census = census
        self.usage: dict[str, Any] = {}
        self.closed = False
        self.times_closed = 0

    async def connect(self) -> dict[str, Any]:
        session = {"id": "test_sess", "status": "started"}
        self._annotate({"type": bridge.EV_SESSION_STARTED, "session": session})
        return session

    async def add_context(self, text: str) -> None:
        return None

    async def send_audio(self, pcm: bytes) -> None:
        return None

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        event = {
            "type": bridge.EV_SESSION_USAGE,
            "usage": dict(USAGE_MID_CALL),
        }
        self.usage = dict(USAGE_MID_CALL)
        self._annotate(event)
        yield event
        # Y se acaba: es lo que hace que `correr` vuelva y se entre al `finally`.

    async def shut(self) -> None:
        self.times_closed += 1
        if self.closed:
            return
        event = {"type": bridge.EV_SESSION_CLOSED, "usage": dict(USAGE_AT_CLOSE)}
        self._annotate(event)
        self.usage = dict(USAGE_AT_CLOSE)
        self.closed = True

    def _annotate(self, event: dict[str, Any]) -> None:
        if self.census is not None:
            self.census.registrar(event)


class _WebSocketThatSaysNothing:
    """Twilio del lado de acá: saluda, arranca y se queda callado."""

    def __init__(self) -> None:
        self.sent_items: list[str] = []
        self.closed_one = False
        self._pending_items = [
            json.dumps({"event": "connected"}),
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": STREAM_SID,
                        "callSid": CALL_SID,
                        "customParameters": {
                            "poc": "3",
                            "scenario": "1",
                            "repetition": "1",
                        },
                    },
                }
            ),
        ]

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        if self._pending_items:
            return self._pending_items.pop(0)
        # Nunca vuelve: la tarea que lee de Twilio se queda aquí hasta que
        # `correr` la cancela, igual que en una llamada en la que el
        # transportista deja de hablar.
        await asyncio.Event().wait()
        raise AssertionError("inalcanzable")

    async def send_text(self, raw_value: str) -> None:
        self.sent_items.append(raw_value)

    async def close(self) -> None:
        self.closed_one = True


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-prueba",
        live_model=DEFAULT_LIVE_MODEL,
        voice="marin",
        delegation_model=DEFAULT_DELEGATION_MODEL,
    )


def _media_stream_path(app: Any) -> Any:
    for ruta in app.routes:
        if getattr(ruta, "path", "") == f"/{telephony.RUTA_WS}":
            return ruta.endpoint
    raise AssertionError(f"la app no expone /{telephony.RUTA_WS}")


def _launch(tmp_path, monkeypatch, client=_FakeClient):
    monkeypatch.setattr(telephony, "RECORDINGS_PATH", tmp_path)
    monkeypatch.setattr(bridge, "GptLiveClient", client)

    app = telephony.create_app(_settings(), trip=TEST_TRIP, text_echo=False)
    ws = _WebSocketThatSaysNothing()
    asyncio.run(_media_stream_path(app)(ws))
    return ws


@pytest.fixture
def run(tmp_path, monkeypatch):
    """La ruta de producción entera, contra el cliente falso."""
    _launch(tmp_path, monkeypatch)

    (destination,) = list(tmp_path.glob("*.json"))
    content = json.loads(destination.read_text(encoding="utf-8"))
    census = [
        json.loads(linea)
        for linea in (tmp_path / f"census_{CALL_SID}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if linea.strip()
    ]
    return content, census


def _census_duration(census: list[dict[str, Any]]) -> int:
    closed_ones = [row for row in census if row["type"] == bridge.EV_SESSION_CLOSED]
    assert closed_ones, "el censo no registró el cierre de la sesión"
    return int(closed_ones[-1]["event"]["usage"]["audio_duration_ms"])


class TestTheJsonAndTheCensusSayTheSame:
    def test_the_duration_is_the_same_in_both_artifacts(self, run) -> None:
        content, census = run
        of_json = content["gpt_live_extra"]["usage"]["audio_duration_ms"]

        assert of_json == _census_duration(census)

    def test_and_is_the_closing_total_and_not_the_previous_warning(self, run) -> None:
        """Igualdad contra un literal: si los dos artefactos se quedaran en el
        penúltimo valor, la prueba de arriba seguiría en verde y el JSON
        seguiría subcontando."""
        content, _ = run
        usage = content["gpt_live_extra"]["usage"]

        assert usage["audio_duration_ms"] == USAGE_AT_CLOSE["audio_duration_ms"]
        assert usage["audio_duration_ms"] != USAGE_MID_CALL["audio_duration_ms"]

    def test_the_census_records_the_close_after_the_warning(self, run) -> None:
        """El orden que hace necesario el arreglo: el total llega al final."""
        _, census = run
        kinds = [row["type"] for row in census]

        assert kinds.index(bridge.EV_SESSION_USAGE) < kinds.index(
            bridge.EV_SESSION_CLOSED
        )


class TestTheCloseStillHappensOnlyOnce:
    def test_the_session_closes_and_the_websocket_too(
        self, tmp_path, monkeypatch
    ) -> None:
        """Mover la copia del uso no puede haber movido el cierre de sitio."""
        clients: list[_FakeClient] = []

        class _Recorded(_FakeClient):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                clients.append(self)

        ws = _launch(tmp_path, monkeypatch, client=_Recorded)

        (client,) = clients
        assert client.closed is True
        assert client.times_closed == 1
        assert ws.closed_one is True
