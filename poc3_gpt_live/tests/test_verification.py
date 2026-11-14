"""Las siete revisiones previas a la campaña, sobre artefactos en disco.

Lo que se prueba aquí no es que la corrida salga verde —eso depende de la
corrida— sino que **cada revisión detecte lo suyo**. Una revisión que devuelve
`ok` pase lo que pase es peor que no tenerla: da permiso para gastar.

Cada caso construye el artefacto mínimo que hace fallar una sola revisión y
comprueba que las otras seis no se contagian. Los números del cuadre salen de
las llamadas de agosto de 2026, donde la fórmula se pasa por uno.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from poc3_gpt_live.diagnostics.verification import (
    TOLERATED_LONG_GAPS,
    SUSPICIOUS_SAMPLE_MS,
    ROJO,
    GREEN,
    review,
)

CALL_SID = "ARNES-call-prueba"


def _turn_taking(**changes: Any) -> dict[str, Any]:
    base = {
        "carrier_streaks": 10,
        "discarded_streaks": 2,
        "streaks_without_response": 1,
        "zeros_vencidos_por_apoyo_verbal": 0,
        "zeros_absorbidos_por_la_apertura": 0,
        "onsets_over_carrier": 0,
        "incoming_hangover_s": 0.8,
        "incoming_gaps_ms": [109, 250, 312],
    }
    base.update(changes)
    return base


def _call(**changes: Any) -> dict[str, Any]:
    """Una llamada sana: 1 apertura + 7 contestadas = 8 muestras."""
    samples = [{"indice": 0, "full_turn_ms": 1_469.0, "is_opening": True}]
    samples += [
        {"indice": i, "full_turn_ms": 1_200.0, "is_opening": False}
        for i in range(1, 8)
    ]
    base = {
        "latencies": samples,
        "gpt_live_extra": {"turn_taking": _turn_taking()},
        "text_echo": {
            "enabled": True,
            "emitted_fragments": 42,
            "eligible": True,
            "motivo": "",
        },
        "clock": {
            "interpretable": True,
            "umbral_ms": 2.0,
            "the_choice_of_R_changes_the_verdict": False,
            "effective_resolution": {"used_ms": 0.5, "inestable": False},
        },
    }
    base.update(changes)
    return base


def _log(**changes: Any) -> dict[str, Any]:
    base = {
        "text": {"received_fragments": 42, "eligible": True},
        "text_left_tail": {
            "left_tail_s": 1.0,
            "evaluated_streaks": 9,
            "fragments_only_via_queue": 4,
            "times_decided": 1,
        },
    }
    base.update(changes)
    return base


def _write(tmp_path, *, call=None, event_log=None, zeros=True):
    """Deja en disco los tres artefactos y devuelve las dos carpetas."""
    folder = tmp_path / "harness"
    recordings = tmp_path / "recordings"
    folder.mkdir(parents=True, exist_ok=True)
    recordings.mkdir(parents=True, exist_ok=True)

    if call is not None:
        (recordings / f"call_{CALL_SID}.json").write_text(
            json.dumps(call), encoding="utf-8"
        )
    if event_log is not None:
        (folder / f"event_log_{CALL_SID}.json").write_text(
            json.dumps(event_log), encoding="utf-8"
        )
    if zeros:
        # El emisor deja un `.estado.json` al cerrar; sin él, `bloque_de_ceros`
        # no puede afirmar que la corrida estuvo bien, y eso es lo correcto.
        from poc3_gpt_live.app.marks import state_path

        state_path(CALL_SID, folder=folder).write_text(
            json.dumps({"eligible_for_analysis": True, "why_not": []}),
            encoding="utf-8",
        )
    return folder, recordings


def _review(tmp_path, **kwargs) -> dict[str, Any]:
    folder, recordings = _write(tmp_path, **kwargs)
    return review(CALL_SID, folder=folder, recordings=recordings)


def _of(block: dict[str, Any], what: str) -> dict[str, Any]:
    return next(r for r in block["reviews"] if r["que"] == what)


class TestTheHealthyRun:
    def test_the_seven_pass(self, tmp_path) -> None:
        block = _review(tmp_path, call=_call(), event_log=_log())

        failing = [r["que"] for r in block["reviews"] if not r["ok"]]
        assert failing == []
        assert block["verdict"] == GREEN
        assert len(block["reviews"]) == 7


class TestWithoutArtifactsThereIsNoReview:
    def test_without_bridge_json_it_is_red_with_the_reason(self, tmp_path) -> None:
        """Y no un bloque a medias que parezca una revisión hecha."""
        block = _review(tmp_path, call=None, event_log=_log())

        assert block["verdict"] == ROJO
        assert block["reviews"] == []
        assert "no hay JSON del puente" in block["could_not_be_reviewed"][0]

    def test_without_a_log_it_says_so_and_does_not_take_it_as_good(self, tmp_path) -> None:
        block = _review(tmp_path, call=_call(), event_log=None)

        assert block["could_not_be_reviewed"] == ["no hay bitácora del arnés"]
        assert _of(block, "left_tail")["ok"] is False
        assert block["verdict"] == ROJO


class TestTheSampleReconciliation:
    def test_the_august_mismatch_is_detected(self, tmp_path) -> None:
        """La racha que absorbe la apertura y que nadie restaba: sobra una.

        Es el caso real de 10 de las 16 llamadas de agosto. Con el arnés no
        puede pasar —el agente abre y el arnés espera a que se calle—, así que
        aquí el cuadre tiene que dar exacto.
        """
        call = _call()
        call["latencies"] = call["latencies"][:-1]  # una muestra menos

        block = _review(tmp_path, call=call, event_log=_log())
        reconciliation = _of(block, "sample_reconciliation")

        assert reconciliation["ok"] is False
        assert reconciliation["valor"]["residue"] == -1
        assert block["verdict"] == ROJO

    def test_onsets_over_the_carrier_count_as_a_sample(
        self, tmp_path
    ) -> None:
        """Salen sin número, pero salen: si se restaran, el cuadre no cerraría."""
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            onsets_over_carrier=3
        )
        call["latencies"] += [
            {"indice": i, "full_turn_ms": None, "is_opening": False}
            for i in (8, 9, 10)
        ]

        assert _of(_review(tmp_path, call=call, event_log=_log()),
                   "sample_reconciliation")["ok"] is True

    def test_publishes_the_terms_to_know_which_is_surplus(self, tmp_path) -> None:
        block = _review(tmp_path, call=_call(), event_log=_log())
        detail = _of(block, "sample_reconciliation")["detail"]

        assert detail["carrier_streaks"] == 10
        assert "zeros_absorbidos_por_la_apertura" in detail
        assert "zeros_colgados_al_final" in detail

    def test_the_zero_hung_up_at_the_end_does_not_count_as_answered(
        self, tmp_path
    ) -> None:
        """El caso real del escenario 3: el transportista habla y ya colgaron.

        `close_pending_ones` cuenta esa racha en `zeros_colgados_al_final`. Existió
        como turno y no produjo muestra, así que hay que restarla — antes no se
        restaba y la llamada de verificación salió en rojo por uno.
        """
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            zeros_colgados_al_final=1
        )
        call["latencies"] = call["latencies"][:-1]  # 1 apertura + 6

        reconciliation = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "sample_reconciliation",
        )

        assert reconciliation["ok"] is True
        assert reconciliation["valor"]["residue"] == 0

    def test_against_the_literal_the_term_carries_weight(self, tmp_path) -> None:
        """Sin el término la prueba de arriba pasaría igual con la fórmula vieja.

        Con las MISMAS muestras y el contador en cero el residuo tiene que salir
        +1: es exactamente el rojo que hubo que diagnosticar a mano.
        """
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            zeros_colgados_al_final=0
        )
        call["latencies"] = call["latencies"][:-1]

        reconciliation = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "sample_reconciliation",
        )

        assert reconciliation["ok"] is False
        assert reconciliation["valor"]["residue"] == -1

    @pytest.mark.parametrize(
        "label, tt, samples",
        [
            # escenario 1: el agente contestó la última racha; 1 arranque encima.
            ("escenario 1", {"carrier_streaks": 8, "discarded_streaks": 0,
                             "streaks_without_response": 1,
                             "zeros_vencidos_por_apoyo_verbal": 0,
                             "zeros_absorbidos_por_la_apertura": 0,
                             "zeros_colgados_al_final": 0,
                             "onsets_over_carrier": 1}, 9),
            # escenario 3: T6 partido en dos (sin_respuesta) y T8 colgado.
            ("escenario 3", {"carrier_streaks": 9, "discarded_streaks": 0,
                             "streaks_without_response": 1,
                             "zeros_vencidos_por_apoyo_verbal": 0,
                             "zeros_absorbidos_por_la_apertura": 0,
                             "zeros_colgados_al_final": 1,
                             "onsets_over_carrier": 0}, 8),
        ],
    )
    def test_both_already_spent_runs_close(
        self, tmp_path, label, tt, samples
    ) -> None:
        """Los contadores tal como los publicaron las dos llamadas del 22/09/2026.

        Van como caso porque son las únicas dos corridas de arnés de PoC3 que
        existen y la campaña se decide con ellas: si la fórmula vuelve a
        desajustarse, falla aquí y no después de gastar.
        """
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(**tt)
        call["latencies"] = [
            {"indice": 0, "full_turn_ms": 1_469.0, "is_opening": True}
        ] + [
            {"indice": i, "full_turn_ms": 2_100.0, "is_opening": False}
            for i in range(1, samples)
        ]

        reconciliation = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "sample_reconciliation",
        )

        assert reconciliation["ok"] is True, f"{label}: {reconciliation['valor']}"


class TestTheShortSamples:
    def test_a_300_ms_sample_is_not_a_response(self, tmp_path) -> None:
        call = _call()
        call["latencies"][3]["full_turn_ms"] = 313.0

        revision = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "too_short_samples",
        )

        assert revision["ok"] is False
        assert revision["valor"] == 1
        assert revision["detail"][0]["full_turn_ms"] == 313.0

    def test_the_opening_does_not_count(self, tmp_path) -> None:
        """La apertura mide otra cosa: del `connected` a la primera voz."""
        call = _call()
        call["latencies"][0]["full_turn_ms"] = 200.0

        assert _of(
            _review(tmp_path, call=call, event_log=_log()),
            "too_short_samples",
        )["ok"] is True

    def test_just_above_the_floor_passes(self, tmp_path) -> None:
        call = _call()
        call["latencies"][2]["full_turn_ms"] = SUSPICIOUS_SAMPLE_MS + 1

        assert _of(
            _review(tmp_path, call=call, event_log=_log()),
            "too_short_samples",
        )["ok"] is True


class TestTheTextEcho:
    def test_a_deaf_run_does_not_pass(self, tmp_path) -> None:
        call = _call()
        call["text_echo"] = {
            "enabled": True,
            "emitted_fragments": 0,
            "eligible": False,
            "motivo": "el agente habló y el arnés no se enteró",
        }
        event_log = _log(text={"received_fragments": 0, "eligible": False})

        revision = _of(
            _review(tmp_path, call=call, event_log=event_log), "text_echo"
        )

        assert revision["ok"] is False
        assert revision["detail"]["bridge_reason"]

    def test_if_the_two_ends_do_not_count_the_same_it_fails(self, tmp_path) -> None:
        """El puente emitió 42 y al arnés llegaron 12: falló el transporte, no
        el modelo. Con cada lado mirándose solo, esto pasa desapercibido."""
        event_log = _log(text={"received_fragments": 12, "eligible": True})

        revision = _of(
            _review(tmp_path, call=_call(), event_log=event_log), "text_echo"
        )

        assert revision["ok"] is False
        assert revision["valor"] == {
            "enabled": True,
            "emitted_by_the_bridge": 42,
            "received_by_the_harness": 12,
        }

    def test_the_manually_disabled_echo_does_not_pass_either(self, tmp_path) -> None:
        call = _call()
        call["text_echo"] = {
            "enabled": False,
            "emitted_fragments": 0,
            "eligible": False,
            "motivo": "apagado por POC3_SIN_ECO_DE_TEXTO",
        }

        assert _of(
            _review(tmp_path, call=call, event_log=_log()), "text_echo"
        )["ok"] is False


class TestTheClockAndTheCrossing:
    def test_an_uninterpretable_clock_removes_the_call(self, tmp_path) -> None:
        call = _call()
        call["clock"]["interpretable"] = False

        block = _review(tmp_path, call=call, event_log=_log())

        assert _of(block, "interpretable_clock")["ok"] is False
        assert block["verdict"] == ROJO

    def test_publishes_the_two_flags_of_step_3(self, tmp_path) -> None:
        valor = _of(
            _review(tmp_path, call=_call(), event_log=_log()),
            "interpretable_clock",
        )["valor"]

        assert valor["R_unstable"] is False
        assert valor["the_choice_of_R_changes_the_verdict"] is False

    def test_without_sender_state_the_crossing_is_not_taken_as_good(
        self, tmp_path
    ) -> None:
        """Que no haya archivo de estado ES el motivo, y se dice con esas
        palabras en vez de desaparecer de la cuenta."""
        block = _review(
            tmp_path, call=_call(), event_log=_log(), zeros=False
        )
        revision = _of(block, "zero_crossing")

        assert revision["ok"] is False
        assert revision["detail"]["why_not"]


class TestTheInboundGaps:
    def test_a_few_long_ones_are_tolerated(self, tmp_path) -> None:
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            incoming_gaps_ms=[250, 900, 1_100]
        )

        revision = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "incoming_gaps",
        )

        assert revision["valor"]["above_hangover"] == 2
        assert revision["ok"] is True

    def test_many_long_ones_say_the_hangover_was_too_short(
        self, tmp_path
    ) -> None:
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            incoming_gaps_ms=[900] * (TOLERATED_LONG_GAPS + 1)
        )

        revision = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "incoming_gaps",
        )

        assert revision["ok"] is False
        assert "se cambia a media campaña" in revision["why_it_matters"]

    def test_threshold_comes_from_the_call_hangover(self, tmp_path) -> None:
        """Y no de una constante de aquí: si el hangover cambiara, esto lo
        seguiría, y un cambio a media campaña se vería en el propio archivo."""
        call = _call()
        call["gpt_live_extra"]["turn_taking"] = _turn_taking(
            incoming_hangover_s=0.25, incoming_gaps_ms=[300, 400, 500]
        )

        revision = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "incoming_gaps",
        )

        assert revision["valor"]["hangover_ms"] == 250.0
        assert revision["valor"]["above_hangover"] == 3


class TestTheHarnessRunPassesItsOwnChecks:
    """Sobre la corrida de verdad de `test_harness.py`, no sobre artefactos a mano.

    Es la que dice si las siete revisiones se pueden contestar con lo que una
    corrida deja de verdad en disco. No se exige verde entero —el alpha falso
    no negocia, así que el guion no llega a los turnos de una llamada real—
    sino que ninguna revisión se caiga por falta de campos.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def reviewed(run):
        _, call_sid, folder, recordings = run
        # La bitácora la escribe `main`, no `actuar_el_guion`: se arma aquí con
        # lo mismo que pondría él.
        observed = run[0]
        (folder / f"event_log_{call_sid}.json").write_text(
            json.dumps(
                {
                    "text": observed.text_block(),
                    "branches": observed.branches_block(),
                    "text_left_tail": (
                        observed.left_tail_block()
                    ),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return review(call_sid, folder=folder, recordings=recordings)

    def test_the_seven_can_be_answered(self, reviewed) -> None:
        assert len(reviewed["reviews"]) == 7
        assert reviewed["could_not_be_reviewed"] == []

    def test_the_text_echo_adds_up_between_the_two_ends(self, reviewed) -> None:
        valor = _of(reviewed, "text_echo")["valor"]

        assert valor["emitted_by_the_bridge"] == valor["received_by_the_harness"]
        assert _of(reviewed, "text_echo")["ok"] is True

    def test_the_anchored_clock_comes_out_interpretable(self, reviewed) -> None:
        assert _of(reviewed, "interpretable_clock")["ok"] is True

    def test_the_zero_crossing_ended_up_fit(self, reviewed) -> None:
        assert _of(reviewed, "zero_crossing")["ok"] is True

    def test_the_left_queue_left_a_trace(self, reviewed) -> None:
        assert _of(reviewed, "left_tail")["ok"] is True


class TestTheCrossingRequiresBothHalves:
    """Marcas en disco **y** sesgo interpretable.

    Antes del paso 3 el cruce solo podía preguntar por lo primero, porque el
    acuse del puente vivía en otro eje y la resta no era una latencia pasara lo
    que pasara. Desde que el eje es común, una llamada puede tener sus ocho
    marcas impecables y aun así no servir: si su reloj derivó por encima del
    umbral, los ceros están ahí y no se pueden restar.
    """

    def test_with_both_it_comes_out_fit(self, tmp_path) -> None:
        revision = _of(
            _review(tmp_path, call=_call(), event_log=_log()),
            "zero_crossing",
        )

        assert revision["ok"] is True
        assert revision["valor"]["same_axis"] is True
        assert revision["valor"]["interpretable_bias"] is True

    def test_with_the_marks_but_no_interpretable_clock_it_is_not_fit(
        self, tmp_path
    ) -> None:
        """La cobertura sigue completa: lo que falla es la otra mitad."""
        call = _call()
        call["clock"]["interpretable"] = False

        block = _review(tmp_path, call=call, event_log=_log())
        revision = _of(block, "zero_crossing")

        assert revision["valor"]["eligible_for_analysis"] is True
        assert revision["valor"]["interpretable_bias"] is False
        assert revision["ok"] is False

    def test_without_a_bridge_block_the_bias_is_null_and_it_does_not_pass_either(
        self, tmp_path
    ) -> None:
        """Nulo no es `false`, pero tampoco es permiso: nadie lo miró."""
        call = _call()
        del call["clock"]

        revision = _of(
            _review(tmp_path, call=call, event_log=_log()),
            "zero_crossing",
        )

        assert revision["valor"]["interpretable_bias"] is None
        assert revision["ok"] is False
