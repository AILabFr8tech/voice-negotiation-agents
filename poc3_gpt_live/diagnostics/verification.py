"""Lo que la llamada de verificación tiene que confirmar, leído de los archivos.

Antes de las 84 de campaña falta **una** llamada que confirme que el
instrumento quedó bien. Lo que hay que mirar en ella está repartido en tres
archivos y en dos procesos, así que revisarlo a mano es exactamente el tipo de
cosa que se hace mal una vez y nadie vuelve a notar. Esto lo hace el programa,
sobre los artefactos que la llamada ya dejó, y se puede volver a correr sobre
una llamada vieja sin gastar nada.

Las seis revisiones, y por qué cada una:

1. **Cuadre de muestras.** `muestras = 1 apertura + rachas contestadas`. Si no
   cierra, el número de turnos medidos no es el número de turnos que hubo, y
   ningún percentil que salga de ahí significa lo que dice. Sobre las 15 de
   agosto la fórmula se pasa por uno en 10 de 16 llamadas, porque el
   transportista contestaba el teléfono hablando y esa racha la absorbe la
   apertura sin que nadie la restara. Con el arnés eso no puede pasar —el
   agente abre y el arnés espera a que se calle—, así que aquí el cuadre
   **tiene que dar exacto**.

2. **Eco de texto.** Sin texto no hay referencia del WER del agente, las ramas
   del guion caen todas a su línea por defecto y `turno.respuesta` sale en
   blanco. Los dos extremos del cable lo publican por separado y esto los
   compara: si el puente dice que emitió N y el arnés recibió 0, lo que falló
   es el transporte.

3. **Rastro de la cola izquierda.** La cola de 1.0 s se fijó antes de la
   campaña y su tamaño no se elige después de verla correr. Lo que sí hay que
   saber es cuántas veces **decidió**, o sea cuántas veces cambió el veredicto
   de una racha. Que el bloque exista es lo que hace auditable la perilla.

4. **Reloj interpretable.** Es la condición de entrada al cruce del cero: si la
   deriva se pasó de su umbral, la llamada no se corrige, se declara y se
   queda fuera.

5. **Cruce del cero.** Que las marcas estén en disco, que el emisor haya
   cerrado con su resumen y que el sesgo salga interpretable. Sin lo primero no
   hay etapa 1 del discriminante; sin lo segundo la hay pero no se puede
   restar, porque el eje común es de construcción y la deriva es por llamada.

6. **Huecos entrantes.** El hangover está congelado en 800 ms. Si una corrida
   trae muchos huecos por encima, el hangover quedó corto y parte frases en
   dos — pero **no se cambia a media campaña**: se anota y se repiten las
   llamadas ya hechas.

Y una séptima, que sale del mismo pendiente: **ninguna muestra por debajo de
~600 ms**. Una muestra de 300 ms no es una respuesta rápida, es el hueco entre
dos palabras del transportista medido como si fuera un turno.

Este módulo **no decide nada de la campaña**: lee, compara contra criterios
escritos y dice verde o rojo con el motivo. Cambiar un criterio es cambiar este
archivo, que es lo que se quiere — que quede en el diff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

from poc3_gpt_live.app.config import RECORDINGS_PATH
from poc3_gpt_live.app.marks import DIAGNOSTICS_PATH, zero_block

# Por debajo de esto una "latencia" es un hueco interno del habla del
# transportista y no una respuesta. Sale de la calibración del hangover
# (docs/CONTRATO_METRICAS.md §2.7).
SUSPICIOUS_SAMPLE_MS: Final[float] = 600.0

# Cuántos huecos entrantes por encima del hangover se toleran antes de decir
# que el hangover quedó corto. Uno suelto es una pausa larga de verdad; varios
# son el hangover partiendo frases.
TOLERATED_LONG_GAPS: Final[int] = 2

GREEN: Final[str] = "verde"
ROJO: Final[str] = "rojo"


def _revision(
    what: str, ok: bool, *, valor: Any, expected: str, why: str, detail: Any = None
) -> dict[str, Any]:
    row = {
        "que": what,
        "ok": bool(ok),
        "valor": valor,
        "expected": expected,
        "why_it_matters": why,
    }
    if detail is not None:
        row["detail"] = detail
    return row


# --------------------------------------------------------------------------
# Las revisiones, una por una
# --------------------------------------------------------------------------


def sample_reconciliation(content: dict[str, Any]) -> dict[str, Any]:
    """`muestras = 1 apertura + rachas contestadas`, con los términos a la vista.

    La fórmula está escrita en el propio JSON de la llamada
    (`turn_taking.nota_de_las_rachas`). Se reproduce aquí término a término para
    que el residuo diga **cuál** sobra o falta, no solo que no cierra.

    **`zeros_colgados_al_final` entra en la fórmula.** Si el transportista suelta
    su última línea y el agente ya colgó, esa racha existió y no produjo muestra.
    `close_pending_ones` la cuenta en `zeros_colgados_al_final`
    (`app/latency.py`), disjunta de `rachas_sin_respuesta` —que se incrementa
    cuando llega una racha nueva con un cero pendiente, no al cerrar—; sin
    restarla, el residuo sale −1.
    """
    tt = content.get("gpt_live_extra", {}).get("turn_taking", {})
    samples = len(content.get("latencies", []))

    terms = {
        "carrier_streaks": int(tt.get("carrier_streaks", 0)),
        "discarded_streaks": int(tt.get("discarded_streaks", 0)),
        "streaks_without_response": int(tt.get("streaks_without_response", 0)),
        "zeros_vencidos_por_apoyo_verbal": int(
            tt.get("zeros_vencidos_por_apoyo_verbal", 0)
        ),
        "zeros_absorbidos_por_la_apertura": int(
            tt.get("zeros_absorbidos_por_la_apertura", 0)
        ),
        "zeros_colgados_al_final": int(tt.get("zeros_colgados_al_final", 0)),
        "onsets_over_carrier": int(
            tt.get("onsets_over_carrier", 0)
        ),
    }
    answered_calls = (
        terms["carrier_streaks"]
        - terms["discarded_streaks"]
        - terms["streaks_without_response"]
        - terms["zeros_vencidos_por_apoyo_verbal"]
        - terms["zeros_absorbidos_por_la_apertura"]
        - terms["zeros_colgados_al_final"]
    )
    # Los arranques sobre el transportista SÍ producen muestra: sale sin número,
    # con el motivo anotado. Por eso entran al esperado y no se restan.
    expected_variables = 1 + answered_calls + terms["onsets_over_carrier"]

    return _revision(
        "sample_reconciliation",
        samples == expected_variables,
        valor={"samples": samples, "expected_variables": expected_variables, "residue": samples - expected_variables},
        expected="residuo 0: muestras = 1 apertura + rachas contestadas + arranques sobre el transportista",
        why=(
            "si no cierra, el número de turnos medidos no es el de turnos que "
            "hubo y ningún percentil de esa llamada significa lo que dice"
        ),
        detail=terms,
    )


def too_short_samples(content: dict[str, Any]) -> dict[str, Any]:
    """Ninguna por debajo de ~600 ms: eso sería un hueco interno, no una respuesta."""
    short_clips = [
        {"indice": m.get("indice"), "full_turn_ms": m.get("full_turn_ms")}
        for m in content.get("latencies", [])
        if m.get("full_turn_ms") is not None
        and not m.get("is_opening")
        and float(m["full_turn_ms"]) < SUSPICIOUS_SAMPLE_MS
    ]
    return _revision(
        "too_short_samples",
        not short_clips,
        valor=len(short_clips),
        expected=f"0 muestras por debajo de {SUSPICIOUS_SAMPLE_MS:.0f} ms",
        why=(
            "una muestra de 300 ms no es una respuesta rápida: es el silencio "
            "entre dos palabras del transportista medido como si fuera un turno"
        ),
        detail=short_clips or None,
    )


def text_echo(
    content: dict[str, Any], event_log: dict[str, Any] | None
) -> dict[str, Any]:
    """Lo que el puente dice que emitió, contra lo que el arnés dice que recibió."""
    of_bridge = content.get("text_echo", {})
    emitted = int(of_bridge.get("emitted_fragments", 0))
    bridge_eligible = bool(of_bridge.get("eligible"))

    received_items: int | None = None
    harness_eligible: bool | None = None
    if event_log is not None:
        of_harness = event_log.get("text", {})
        received_items = int(of_harness.get("received_fragments", 0))
        harness_eligible = bool(of_harness.get("eligible"))

    ok = (
        bool(of_bridge.get("enabled"))
        and emitted > 0
        and bridge_eligible
        and (received_items is None or (received_items == emitted and harness_eligible))
    )
    return _revision(
        "text_echo",
        ok,
        valor={
            "enabled": of_bridge.get("enabled"),
            "emitted_by_the_bridge": emitted,
            "received_by_the_harness": received_items,
        },
        expected="encendido, con fragmentos, y los dos extremos contando lo mismo",
        why=(
            "sin texto no hay referencia del WER del agente, `turno.respuesta` "
            "sale en blanco y las ramas caen todas a su línea por defecto"
        ),
        detail={"bridge_reason": of_bridge.get("motivo") or None},
    )


def left_tail(event_log: dict[str, Any] | None) -> dict[str, Any]:
    """Que la cola haya dejado rastro, y cuántas veces decidió."""
    if event_log is None:
        return _revision(
            "left_tail",
            False,
            valor=None,
            expected="el bloque publicado por el arnés",
            why="sin bitácora del arnés no se puede auditar la perilla",
        )
    block = event_log.get("text_left_tail", {})
    evaluated = int(block.get("evaluated_streaks", 0))
    return _revision(
        "left_tail",
        evaluated > 0,
        valor={
            "cola_seg": block.get("left_tail_s"),
            "evaluated_streaks": evaluated,
            "fragments_only_via_queue": block.get("fragments_only_via_queue"),
            "times_decided": block.get("times_decided"),
        },
        expected="al menos una racha evaluada, con el conteo de cuántas decidió",
        why=(
            "el tamaño de la cola se fijó antes de la campaña y no se elige "
            "después de verla correr; lo que sí hay que poder contar es cuántas "
            "veces cambió un veredicto"
        ),
    )


def interpretable_clock(content: dict[str, Any]) -> dict[str, Any]:
    """La condición de entrada al cruce del cero."""
    clock = content.get("clock", {})
    resolution = clock.get("effective_resolution", {})
    return _revision(
        "interpretable_clock",
        bool(clock.get("interpretable")),
        valor={
            "interpretable": clock.get("interpretable"),
            "umbral_ms": clock.get("umbral_ms"),
            "R_used_ms": resolution.get("used_ms"),
            "R_unstable": resolution.get("inestable"),
            "the_choice_of_R_changes_the_verdict": clock.get(
                "the_choice_of_R_changes_the_verdict"
            ),
        },
        expected="interpretable, con la deriva por debajo de su umbral",
        why=(
            "una deriva por encima del umbral no se corrige: se declara, y la "
            "llamada queda fuera del cruce contra los ceros del arnés"
        ),
    )


def zero_crossing(
    call_sid: str, *, folder: Path, bridge_clock: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Que las marcas estén en disco, el emisor haya cerrado y el eje se pueda restar.

    `bridge_clock` es el bloque `reloj` del JSON de la llamada. Va aquí
    porque `zero_block` se arma leyendo las marcas del arnés y por sí solo
    no puede decir si la deriva de **esta** llamada dejó el sesgo interpretable:
    sin él, `sesgo_interpretable` sale `null` y el cruce no se declara apto por
    la mitad.
    """
    block = zero_block(
        call_sid, folder=folder, bridge_clock=bridge_clock
    )
    clock = block.get("clock", {})
    ok = bool(block.get("eligible_for_analysis")) and bool(
        clock.get("interpretable_bias")
    )
    return _revision(
        "zero_crossing",
        ok,
        valor={
            "eligible_for_analysis": block.get("eligible_for_analysis"),
            "same_axis": clock.get("same_axis"),
            "interpretable_bias": clock.get("interpretable_bias"),
            "cobertura": block.get("cobertura"),
        },
        expected="apto, con una marca por turno actuado y el sesgo interpretable",
        why=(
            "sin los ceros reales no hay etapa 1 del discriminante, y sin el "
            "sesgo interpretable los hay pero no se pueden restar"
        ),
        detail={"why_not": block.get("why_not") or None},
    )


def incoming_gaps(content: dict[str, Any]) -> dict[str, Any]:
    """Cuántos silencios del transportista pasan del hangover congelado."""
    tt = content.get("gpt_live_extra", {}).get("turn_taking", {})
    hangover_ms = float(tt.get("incoming_hangover_s", 0.8)) * 1000.0
    gaps = [float(h) for h in tt.get("incoming_gaps_ms", [])]
    longs = [h for h in gaps if h > hangover_ms]

    return _revision(
        "incoming_gaps",
        len(longs) <= TOLERATED_LONG_GAPS,
        valor={
            "hangover_ms": hangover_ms,
            "gaps": len(gaps),
            "above_hangover": len(longs),
            "max_ms": max(gaps) if gaps else None,
        },
        expected=f"a lo más {TOLERATED_LONG_GAPS} por encima del hangover",
        why=(
            "cada hueco por encima del hangover parte una frase en dos y arma "
            "un cero falso. Si salen muchos, el hangover quedó corto — pero NO "
            "se cambia a media campaña: se anota y se repiten las hechas"
        ),
        detail={"the_long_ones_ms": longs or None},
    )


# --------------------------------------------------------------------------
# El veredicto
# --------------------------------------------------------------------------


def _leer(ruta: Path) -> dict[str, Any] | None:
    try:
        content = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None


def review(
    call_sid: str,
    *,
    folder: Path | None = None,
    recordings: Path | None = None,
) -> dict[str, Any]:
    """Las siete revisiones sobre los artefactos de `call_sid`.

    Sin el JSON del puente no hay nada que revisar y se devuelve rojo con el
    motivo, en vez de un bloque a medias que parezca una revisión hecha.
    """
    folder = folder or DIAGNOSTICS_PATH
    recordings = recordings or RECORDINGS_PATH

    call_path = recordings / f"call_{call_sid}.json"
    content = _leer(call_path)
    if content is None:
        return {
            "call_sid": call_sid,
            "verdict": ROJO,
            "reviews": [],
            "could_not_be_reviewed": [f"no hay JSON del puente en {call_path}"],
        }

    event_log = _leer(folder / f"event_log_{call_sid}.json")

    reviews = [
        sample_reconciliation(content),
        too_short_samples(content),
        text_echo(content, event_log),
        left_tail(event_log),
        interpretable_clock(content),
        zero_crossing(
            call_sid, folder=folder, bridge_clock=content.get("clock")
        ),
        incoming_gaps(content),
    ]
    absent = [] if event_log is not None else ["no hay bitácora del arnés"]

    return {
        "call_sid": call_sid,
        "verdict": GREEN if all(r["ok"] for r in reviews) else ROJO,
        "reviews": reviews,
        "could_not_be_reviewed": absent,
        "artifacts": {
            "bridge_json": str(call_path),
            "census": str(recordings / f"census_{call_sid}.jsonl"),
            "event_log": str(folder / f"event_log_{call_sid}.json"),
        },
    }


def display(block: dict[str, Any]) -> None:
    """La tabla, para leerla sin abrir el JSON."""
    print("\n" + "=" * 70)
    print(f"Verificación de la corrida — {block['call_sid']}")
    print("=" * 70 + "\n")

    for revision in block["reviews"]:
        mark = "ok  " if revision["ok"] else "FALLA"
        print(f"  [{mark}] {revision['que']}")
        print(f"          {revision['valor']}")
        if not revision["ok"]:
            print(f"          expected: {revision['expected']}")
            if revision.get("detail"):
                print(f"          detail:  {revision['detail']}")

    for missing in block.get("could_not_be_reviewed", []):
        print(f"\n  [!] {missing}")

    print(f"\n  verdict: {block['verdict'].upper()}")
    if block["verdict"] == ROJO:
        print(
            "  Con una revisión en rojo NO se arranca la campaña: lo que falla "
            "aquí\n  falla igual en las 84, y ahí ya cuesta dinero."
        )


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.diagnostics.verification",
        description=(
            "Revisa los artefactos de una corrida de PoC3 contra lo que la "
            "llamada de verificación tiene que confirmar. No gasta nada."
        ),
    )
    arg_parser.add_argument("call_sid", help="la llave que comparten los tres archivos")
    arg_parser.add_argument(
        "--folder",
        default=str(DIAGNOSTICS_PATH),
        help="dónde dejó el arnés sus marcas y su bitácora",
    )
    arg_parser.add_argument(
        "--recordings",
        default=str(RECORDINGS_PATH),
        help="dónde dejó el puente el JSON de la llamada y el censo",
    )
    arg_parser.add_argument(
        "--json", action="store_true", help="el bloque crudo en vez de la tabla"
    )
    args = arg_parser.parse_args(argv)

    block = review(
        args.call_sid,
        folder=Path(args.folder),
        recordings=Path(args.recordings),
    )
    if args.json:
        print(json.dumps(block, ensure_ascii=False, indent=2))
    else:
        display(block)
    return 0 if block["verdict"] == GREEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
