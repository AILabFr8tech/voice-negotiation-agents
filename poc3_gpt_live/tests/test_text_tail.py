"""La cola izquierda de la ventana de texto de PoC3, en tres capas.

El texto del alpha llega **antes** que su audio (−531 ms de p50 sobre
`CA221584…`), así que la ventana de `shared/` —que abre en `racha.inicio`— deja
fuera el texto de las rachas cortas. `ObservadoDePoC3` la abre
`TEXT_LEFT_TAIL_S` antes. Lo que hay que demostrar es que eso
arregla PoC3 **sin mover PoC1 ni PoC2**.

## Las tres capas, y qué prueba cada una

1. **Identidad del camino** (`TestPoC1YPoC2NoEjecutanCodigoNuevo`). No
   "equivalente": se comprueba que en todo el repo existan exactamente dos
   definiciones de `agent_text_during`, la de `shared/` y la de PoC3.
   Si nadie más la sobreescribe, PoC1 y PoC2 **no pueden** verse afectados.
2. **Replay sobre los artefactos de campaña** (`TestElClasificadorNoCambia`).
   Los 116 fragmentos descartados que las 216 bitácoras guardaron, con su
   duración, su texto y su motivo. Se vuelve a correr `_streak_verdict`
   y se exige el mismo motivo, cadena por cadena.
3. **Detector de deriva de la copia** (`TestLaColaNoSeSepara`). La
   sobreescritura con la cola forzada a 0 tiene que devolver salida byte a byte
   idéntica a la de `Observado`, sobre fragmentos reales.

## Lo que estas pruebas NO hacen, y no se puede hacer con lo que hay

**No se replaya la VENTANA sobre datos de PoC1 ni de PoC2.** Las bitácoras de
campaña persisten el texto **ya unido** —por turno y por fragmento
descartado— y nunca escribieron a disco la marca de llegada de cada
`FragmentoDeTexto`. Sin esos instantes no se puede reconstruir qué caía dentro
de `[racha.inicio, racha.fin + 3.0]` en aquellas corridas, así que sobre
artefactos reales queda verificado el **clasificador** (capa 2) y no la ventana.
La ventana queda cubierta por construcción (capa 1) y contra deriva (capa 3).
Esto no es una omisión que se pueda cerrar más adelante con los mismos
files: el dato no existe.

**Y la capa 1 no importa `poc1_pipecat` ni `poc2_livekit`.** Hacerlo ataría el
entorno de PoC3 al de las otras dos —justo lo que el encabezado de
`app/media_streams.py` evita al no importar el transporte de PoC1— y además
cargaría pipecat para comprobar una propiedad del código fuente. Se lee el
fuente.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from poc3_gpt_live.app.media_streams import (
    TEXT_LEFT_TAIL_S,
    FROM_AGENT,
    ObservedFromPoC3,
)
from shared.carrier_simulator import (
    MIN_SUBSTANTIVE_SPEECH_S,
    BACKCHANNEL_CAP_S,
    Observed,
    SpeechStreak,
)
from shared.carrier_simulator.player import _streak_verdict

RAIZ = Path(__file__).resolve().parents[2]

# Fragmentos reales del censo `CA221584…`, con el adelanto que motiva la cola:
# el texto se fecha ~0.5 s antes del audio que describe.
REAL_FRAGMENTS: list[str] = [
    " ¿H", "ablo con", " Carlos?", " una tarifa", " de", " diecis", "iete",
    " mil",
]


def _observed_with(texts: list[tuple[str, float]]) -> ObservedFromPoC3:
    """Un observado con fragmentos del agente colocados en instantes exactos."""
    observed = ObservedFromPoC3(silent=True)
    for text, t in texts:
        fragment = observed.annotate_bridge_text(
            {"de": FROM_AGENT, "text": text, "has_spaces": True}
        )
        # `anotar_texto` fecha con el reloj real; para la prueba manda el
        # instante que se le pide, que es lo que la ventana filtra.
        observed.fragments[-1].t = t
    return observed


# --------------------------------------------------------------------------
# Capa 1
# --------------------------------------------------------------------------


class TestPoC1AndPoC2DoNotRunNewCode:
    def test_only_shared_and_poc3_define_the_window(self) -> None:
        """Si nadie más la sobreescribe, nadie más puede verse afectado."""
        patron = re.compile(r"^\s*def agent_text_during\(", re.M)
        found_ones = sorted(
            str(p.relative_to(RAIZ)).replace("\\", "/")
            for p in RAIZ.rglob("*.py")
            if ".venv" not in p.parts
            and "__pycache__" not in p.parts
            and patron.search(p.read_text(encoding="utf-8", errors="replace"))
        )
        assert found_ones == [
            "poc3_gpt_live/app/media_streams.py",
            "shared/carrier_simulator/observed.py",
        ]

    def test_the_base_is_still_the_base(self) -> None:
        """PoC3 la sobreescribe; `Observado` conserva la suya intacta."""
        assert (
            ObservedFromPoC3.agent_text_during
            is not Observed.agent_text_during
        )
        assert Observed.agent_text_during.__qualname__ == (
            "Observed.agent_text_during"
        )

    def test_the_only_caller_dispatches_per_instance(self) -> None:
        """De ahí que baste con sobreescribir, sin tocar `shared/`."""
        source = (RAIZ / "shared/carrier_simulator/player.py").read_text(
            encoding="utf-8"
        )
        assert source.count("agent_text_during") == 1
        assert "observed.agent_text_during(streak)" in source


# --------------------------------------------------------------------------
# Capa 2
# --------------------------------------------------------------------------


def _campaign_discards() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for poc, patron in (
        ("poc1", "poc1_pipecat/diagnostics/harness/event_log_*.json"),
        ("poc2", "poc2_livekit/diagnostics/harness/event_log_*.json"),
    ):
        for ruta in sorted(RAIZ.glob(patron)):
            event_log = json.loads(ruta.read_text(encoding="utf-8"))
            for turn in event_log.get("turns") or []:
                for fragment in turn.get("discarded_fragments") or []:
                    rows.append((f"{poc}/{ruta.stem}/{turn.get('turn')}", fragment))
    return rows


class TestTheClassifierDoesNotChange:
    """Replay exacto sobre lo que la campaña dejó escrito.

    Se salta si los artefactos no están: son gitignoreados por LFPDPPP y viven
    solo en la máquina que corrió la campaña. Saltar es correcto — lo que no
    sería correcto es que la prueba pase sin haber mirado nada, y por eso
    `test_there_are_artifacts_to_inspect` falla en vez de saltarse cuando hay
    archivos pero ningún descarte dentro.
    """

    def test_there_are_artifacts_to_inspect(self) -> None:
        logs = list(RAIZ.glob("poc1_pipecat/diagnostics/harness/event_log_*.json"))
        if not logs:
            pytest.skip("sin bitácoras de campaña en esta máquina")
        assert _campaign_discards(), (
            "hay bitácoras pero ningún fragmento descartado: el replay no "
            "estaría comprobando nada"
        )

    def test_each_recorded_discard_gives_the_same_reason_again(self) -> None:
        descartes = _campaign_discards()
        if not descartes:
            pytest.skip("sin bitácoras de campaña en esta máquina")

        for where, fragment in descartes:
            streak = SpeechStreak(
                start=float(fragment["start"]),
                end=float(fragment["start"])
                + float(fragment["duration_ms"]) / 1000.0,
            )
            verdict = _streak_verdict(
                streak, str(fragment.get("text") or ""), opened=False
            )
            assert verdict == fragment["motivo"], where


# --------------------------------------------------------------------------
# Capa 3
# --------------------------------------------------------------------------


class TestTheTailIsNotSplit:
    """La copia de PoC3 con la cola en 0 tiene que ser la base, byte a byte."""

    def _fragments(self) -> list[tuple[str, float]]:
        # El audio de la racha va de 10.0 a 11.0 s; el texto llega 0.5 s antes.
        return [(t, 9.5 + i * 0.1) for i, t in enumerate(REAL_FRAGMENTS)]

    def test_with_the_tail_at_zero_returns_the_same_as_shared(
        self, monkeypatch
    ) -> None:
        streak = SpeechStreak(start=10.0, end=11.0)
        texts = self._fragments()

        base = Observed.agent_text_during(
            _observed_with(texts), streak
        )

        monkeypatch.setattr(
            "poc3_gpt_live.app.media_streams.TEXT_LEFT_TAIL_S", 0.0
        )
        self_reported = _observed_with(texts).agent_text_during(streak)

        assert self_reported == base

    def test_with_the_tail_set_it_does_change(self) -> None:
        """Si no cambiara, la cola no estaría haciendo nada.

        Con el adelanto real, la ventana de la base se come el **principio** de
        la frase: no devuelve vacío, devuelve un tronco. Para el WER eso ya es
        daño, y para el clasificador lo es cuando el tronco cae del lado
        equivocado de `is_backchannel`. El caso de pérdida total —un apoyo
        verbal que viaja en un solo fragmento— lo cubre
        `TestElRastroDeLaCola.test_marks_when_the_tail_changes_the_verdict`.
        """
        streak = SpeechStreak(start=10.0, end=11.0)
        texts = self._fragments()

        base = Observed.agent_text_during(_observed_with(texts), streak)
        self_reported = _observed_with(texts).agent_text_during(streak)

        full = "¿Hablo con Carlos? una tarifa de diecisiete mil"
        assert self_reported == full
        assert base == "diecisiete mil"
        assert full.endswith(base) and base != full


# --------------------------------------------------------------------------
# El rastro
# --------------------------------------------------------------------------


class TestTheTailTrace:
    """La cola tiene que poder auditarse, o es una perilla invisible."""

    def test_counts_the_fragments_that_entered_only_through_the_tail(self) -> None:
        streak = SpeechStreak(start=10.0, end=11.0)
        # Tres llegan antes del inicio de la racha, dos dentro.
        texts = [(" uno", 9.4), (" dos", 9.6), (" tres", 9.8),
                  (" cuatro", 10.2), (" cinco", 10.6)]
        observed = _observed_with(texts)

        observed.agent_text_during(streak)
        block = observed.left_tail_block()

        assert block["left_tail_s"] == TEXT_LEFT_TAIL_S
        assert block["evaluated_streaks"] == 1
        assert block["fragments_only_via_queue"] == 3
        assert block["per_streak"][0]["fragments_in_window"] == 5

    def test_marks_when_the_tail_changes_the_verdict(self) -> None:
        """El caso que motiva todo: un apoyo verbal de 1.0 s.

        Sin cola el texto sale vacío y `is_backchannel("")` es `False`, o sea
        que la racha se toma como la respuesta y el guion avanza encima. Con
        cola el texto es "Ajá" y se descarta, que es lo correcto.
        """
        streak = SpeechStreak(start=10.0, end=11.0)
        assert MIN_SUBSTANTIVE_SPEECH_S <= streak.duration < BACKCHANNEL_CAP_S
        observed = _observed_with([(" Ajá", 9.5)])

        observed.agent_text_during(streak)
        trail = observed.left_tail_block()["per_streak"][0]

        assert trail["text_without_tail"] == ""
        assert trail["text_with_tail"] == "Ajá"
        assert trail["in_the_band_where_text_decides"] is True
        assert trail["the_tail_decided"] is True
        assert observed.left_tail_block()["times_decided"] == 1

    def test_outside_the_band_the_tail_does_not_decide_even_if_it_adds_text(self) -> None:
        """Por encima de 2 s el clasificador acepta sin preguntarle al texto."""
        streak = SpeechStreak(start=10.0, end=13.0)
        observed = _observed_with([(" Ajá", 9.5)])

        observed.agent_text_during(streak)
        trail = observed.left_tail_block()["per_streak"][0]

        assert trail["fragments_only_via_queue"] == 1
        assert trail["in_the_band_where_text_decides"] is False
        assert trail["the_tail_decided"] is False

    def test_a_streak_evaluated_several_times_counts_once(self) -> None:
        """El bucle de espera reevalúa cada 50 ms; el rastro no se infla."""
        streak = SpeechStreak(start=10.0, end=11.0)
        observed = _observed_with([(" Ajá", 9.5)])

        for _ in range(20):
            observed.agent_text_during(streak)

        block = observed.left_tail_block()
        assert block["evaluated_streaks"] == 1
        assert block["times_decided"] == 1

    def test_the_block_comes_out_also_when_the_tail_did_nothing(self) -> None:
        """Si solo apareciera al actuar, "no decidió" y "nadie midió" se
        leerían igual."""
        block = ObservedFromPoC3(silent=True).left_tail_block()

        assert block["evaluated_streaks"] == 0
        assert block["times_decided"] == 0
        assert block["per_streak"] == []
