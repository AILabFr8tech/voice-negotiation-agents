"""Cuánto de la latencia es transporte de vuelta, y no framework.

## Por qué existe

La latencia por intervención (§2.2.5) se mide **desde fuera**: del fin de voz
del arnés al primer audio del agente que el arnés oye. Eso lleva dentro tres
cosas, y solo una es del framework:

    latencia = camino de ida  +  lo que tarda el pipeline  +  camino de vuelta

El camino de ida ya está acotado: es el `speech_bias` —141 ms en PoC1, 408 en
PoC2—. Lo que tarda el pipeline lo publica cada framework (`full_turn_ms`).
El de vuelta era el que faltaba, y sin él la sección de latencia del reporte
compara **despliegues** —PoC1 entera en la máquina contra el SFU de LiveKit en la
nube— y no frameworks.

## Cómo se mide, y por qué no se infiere

Se podría despejar restando medianas (`latencia − full_turn − sesgo`), pero
eso es una cota con tres incertidumbres dentro y no sobrevive a que alguien
pregunte por un turno concreto. Así que se **mide**, con las dos puntas en reloj
de pared y la resta directa:

- **La punta izquierda** es el instante en que el audio dejó el pipeline. En
  PoC1 lo publica `SondaDeSalidaDeAudio`, un procesador entre el TTS y el
  transporte (`audio_outputs` en el cruce). En PoC2 lo publica el propio
  LiveKit como `started_speaking_at`, que ya venía en el JSON del worker y no se
  estaba usando.
- **La punta derecha** es cuándo el arnés lo oyó: `agent_first_speech`, que
  ya se mide por intervención.

Las dos son `time.time()` de procesos en la misma máquina, así que la resta vale
por la misma razón por la que vale el `speech_bias` — y si los hosts no
coincidieran, el cruce ya lo declara no interpretable.

## Lo que queda dentro de cada número

No es el mismo camino en las dos, y ese **es** el hallazgo:

- **PoC1**: serializar a μ-law base64, el WebSocket local, y el búfer del cliente
  del arnés. Todo en la misma máquina.
- **PoC2**: el encoder Opus, la subida al SFU de LiveKit Cloud, la bajada al
  suscriptor y su jitter buffer. Un viaje redondo a internet.

`playback_latency` de LiveKit (3–7 ms medidos) **no** es esto: solo cubre el
búfer local del worker antes de publicar, y por eso no sirve para descontar el
despliegue.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final

# Cuánto antes de oír al agente puede haber salido su audio para que el
# emparejamiento valga. Generoso por arriba —un SFU con mala red puede tardar—
# y cerrado por abajo: el audio no se puede oír antes de haber salido.
PAIRING_WINDOW_S: Final[float] = 3.0


@dataclass(frozen=True)
class Outbound:
    """El camino de vuelta de una intervención, con sus dos puntas."""

    turn: str
    left: float
    was_heard: float

    @property
    def ms(self) -> float:
        return (self.was_heard - self.left) * 1000.0

    def a_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "left_the_pipeline": self.left,
            "harness_heard_it": self.was_heard,
            "return_path_ms": round(self.ms, 1),
        }


def pair_outputs(
    lines: list[dict[str, Any]],
    outputs: list[float],
    *,
    window_s: float = PAIRING_WINDOW_S,
) -> tuple[list[Outbound], list[str]]:
    """Cada intervención con la salida de audio que la produjo.

    Se toma la salida **más tardía que no sea posterior** a cuando el arnés oyó:
    el audio no puede oírse antes de haber salido, así que una salida posterior
    es de la respuesta siguiente. Y se exige que esté dentro de la ventana, para
    que una intervención sin respuesta no se quede con la salida de otra.
    """
    ordered = sorted(outputs)
    outbounds: list[Outbound] = []
    unpaired: list[str] = []
    for line in lines:
        was_heard = line.get("agent_first_speech")
        turn = str(line.get("turn", ""))
        if was_heard is None:
            continue
        candidate_ones = [s for s in ordered if s <= was_heard and was_heard - s <= window_s]
        if not candidate_ones:
            unpaired.append(turn)
            continue
        outbounds.append(Outbound(turn=turn, left=candidate_ones[-1], was_heard=float(was_heard)))
    return outbounds, unpaired


def summarize(outbounds: list[Outbound]) -> dict[str, Any]:
    values = sorted(v.ms for v in outbounds)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(values[0], 1),
        "p50": round(median(values), 1),
        "max": round(values[-1], 1),
    }


# --------------------------------------------------------------------------
# Lectura de cada PoC
# --------------------------------------------------------------------------


def _lines_of(block: Any) -> list[dict[str, Any]]:
    if not isinstance(block, dict):
        return []
    detail = block.get("per_line")
    return detail if isinstance(detail, list) else []


def _numero(row: dict[str, Any], *keys: str) -> float | None:
    """El primer valor presente de entre varias llaves.

    Las dos PoC escriben la latencia por intervención con el nombre largo
    (`line_latency_ms`, que pone cada `marcas.py`), pero el resumen
    genérico de `carrier_simulator.cruce` la escribe corta. Leer las dos evita
    que un cambio de productor deje la tabla en blanco sin avisar.
    """
    for clave in keys:
        valor = row.get(clave)
        if valor is not None:
            return float(valor)
    return None


def _observed_values_of(block: Any) -> list[dict[str, Any]]:
    """Por intervención: lo observado y su camino de ida, del mismo archivo.

    Van juntas a propósito. La cifra observada y su corrección tienen que venir
    de la misma llamada **y de la misma intervención**, o la resta estaría
    mezclando turnos con condiciones distintas.
    """
    rows = []
    for i in _lines_of(block):
        observed_value = _numero(i, "line_latency_ms", "latency_ms")
        if observed_value is None:
            continue
        rows.append(
            {
                "turn": str(i.get("turn", "")),
                "observed_ms": observed_value,
                "ida_ms": _numero(i, "speech_bias_ms"),
            }
        )
    return rows


def of_poc1(crossing_path: Path) -> dict[str, Any] | None:
    """Lee `cero_*.json`: `audio_outputs` contra `por_intervencion`."""
    if not crossing_path.is_file():
        return None
    cruce = json.loads(crossing_path.read_text(encoding="utf-8"))
    lines = _lines_of(cruce.get("lines"))
    # La punta audible es la buena: el arnés fecha cuándo **oyó** al agente, así
    # que la izquierda tiene que ser cuándo el pipeline soltó audio que suena.
    # Con la primera trama a secas, el arranque del TTS entraría dentro. Las
    # corridas viejas no traen la audible: se usa la otra y el resultado se marca.
    audible = [float(s) for s in cruce.get("audible_outputs") or []]
    any_ones = [float(s) for s in cruce.get("audio_outputs") or []]
    outputs = audible or any_ones
    if not outputs or not lines:
        return None
    outbounds, unpaired = pair_outputs(lines, outputs)

    data = {
        "poc": 1,
        "run": crossing_path.stem,
        "left_tip": (
            "`SondaDeSalidaDeAudio`, entre el TTS y el transporte del pipeline"
            + (", desde la primera trama CON VOZ" if audible else "")
        ),
        "same_definition_at_both_ends": bool(audible),
        "what_it_carries": (
            "serializar a μ-law base64, el WebSocket local y el búfer del "
            "cliente del arnés. Todo en la misma máquina"
        ),
        "return_path_ms": summarize(outbounds),
        "per_line": _observed_values_of(cruce.get("lines")),
        "lines_without_output": unpaired,
        "detail": [v.a_dict() for v in outbounds],
    }
    if audible and any_ones:
        # Cuánto tarda el TTS en soltar algo que suene, desde que suelta la
        # primera trama. No es transporte, y PoC2 no lo puede separar —su punta
        # izquierda es `started_speaking_at`, que no distingue— así que este
        # número es la corrección que hay que tener en cuenta al comparar.
        no_speech, _ = pair_outputs(lines, any_ones)
        per_turn = {v.turn: v.ms for v in outbounds}
        onsets = [
            no_speech_v.ms - per_turn[no_speech_v.turn]
            for no_speech_v in no_speech
            if no_speech_v.turn in per_turn
        ]
        data["tts_onset_ms"] = summarize(
            [Outbound(turn="", left=0.0, was_heard=a / 1000.0) for a in onsets]
        )
    return data


def of_poc2(worker_path: Path) -> dict[str, Any] | None:
    """Lee el JSON del worker: `started_speaking_at` contra `por_intervencion`."""
    if not worker_path.is_file():
        return None
    data = json.loads(worker_path.read_text(encoding="utf-8"))
    extra = data.get("livekit_extra") or {}
    outputs = [
        float(t["assistant"]["started_speaking_at"])
        for t in extra.get("per_turn") or []
        if isinstance(t.get("assistant"), dict)
        and t["assistant"].get("started_speaking_at") is not None
    ]
    harness = extra.get("harness") or {}
    lines = _lines_of(harness.get("lines"))
    if not outputs or not lines:
        return None
    outbounds, unpaired = pair_outputs(lines, outputs)
    return {
        "poc": 2,
        "run": worker_path.stem,
        "left_tip": (
            "`started_speaking_at` del propio LiveKit: cuándo empezó a reproducir "
            "hacia la sala"
        ),
        "what_it_carries": (
            "el encoder Opus, la subida al SFU de LiveKit Cloud, la bajada al "
            "suscriptor y su jitter buffer. Un viaje redondo a internet"
        ),
        "return_path_ms": summarize(outbounds),
        "per_line": _observed_values_of(harness.get("lines")),
        "lines_without_output": unpaired,
        "detail": [v.a_dict() for v in outbounds],
    }


def leer(ruta: Path) -> dict[str, Any] | None:
    """Decide por el contenido, no por el nombre: sirve para las dos PoC."""
    return of_poc1(ruta) or of_poc2(ruta)


def latency_table(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Las TRES cifras por PoC: observada, transporte descontado, y corregida.

    El contrato (§2.2.8) prohíbe publicar solo la corregida. La razón es que el
    ajuste es del mismo orden que la diferencia que pretende explicar —descuenta
    ~146 ms de PoC1 contra ~610 de PoC2—, así que sin las otras dos nadie puede
    verificar de dónde salió ni rehacerlo si mañana se remide una punta.

    La corrección se hace **por intervención** y no entre medianas: las dos PoC
    publican el `speech_bias` de cada intervención, así que cada turno se
    corrige con su propio transporte y el p50 de lo corregido es un percentil de
    verdad. Como consecuencia, la columna corregida **no** es la resta de las
    otras columnas; la diferencia entre las dos formas se publica en
    `deriva_por_restar_percentiles_ms` para que el que quiera cuadrar a mano vea
    de cuánto es el desacuerdo en vez de creer que hay un error.
    """
    per_poc: dict[int, list[dict[str, float | None]]] = {}
    for run in runs:
        poc = int(run["poc"])
        outbound_of = {
            str(f.get("turn", "")): float(f["return_path_ms"])
            for f in run["detail"]
        }
        rows = per_poc.setdefault(poc, [])
        for i in run.get("per_line") or []:
            rows.append(
                {
                    "observed_value": float(i["observed_ms"]),
                    "inbound": None if i.get("ida_ms") is None else float(i["ida_ms"]),
                    "outbound": outbound_of.get(str(i.get("turn", ""))),
                }
            )
        # Las vueltas de turnos que no dejaron latencia observada igual cuentan
        # para el p50 de la vuelta: es una medición propia, no un derivado.
        seen_items = {str(i.get("turn", "")) for i in run.get("per_line") or []}
        for turn, ms in outbound_of.items():
            if turn not in seen_items:
                rows.append({"observed_value": None, "inbound": None, "outbound": ms})

    def _p50(values: list[float]) -> float | None:
        return round(median(values), 1) if values else None

    resultado: dict[str, dict[str, Any]] = {}
    for poc, rows in sorted(per_poc.items()):
        obs = [f["observed_value"] for f in rows if f["observed_value"] is not None]
        inbound_legs = [f["inbound"] for f in rows if f["inbound"] is not None]
        outbs = [f["outbound"] for f in rows if f["outbound"] is not None]
        completed = [
            f
            for f in rows
            if f["observed_value"] is not None
            and f["inbound"] is not None
            and f["outbound"] is not None
        ]
        transport_modes = [f["inbound"] + f["outbound"] for f in completed]
        corrected_ones = [f["observed_value"] - f["inbound"] - f["outbound"] for f in completed]
        row: dict[str, Any] = {
            "n_lines": len(obs),
            "observed_p50_ms": _p50(obs),
            "ida_p50_ms": _p50(inbound_legs),
            "outbound_p50_ms": _p50(outbs),
            "n_round_trips": len(outbs),
            "n_correctable": len(completed),
            "transport_p50_ms": _p50(transport_modes),
            "corrected_p50_ms": _p50(corrected_ones),
        }
        if row["corrected_p50_ms"] is not None and row["observed_p50_ms"] is not None:
            a_mano = (
                row["observed_p50_ms"]
                - (row["ida_p50_ms"] or 0.0)
                - (row["outbound_p50_ms"] or 0.0)
            )
            row["drift_by_subtracting_percentiles_ms"] = round(
                a_mano - row["corrected_p50_ms"], 1
            )
        resultado[str(poc)] = row

    return {
        "per_poc": resultado,
        "how_it_is_published": (
            "las tres cifras van juntas en la misma tabla del reporte y en este "
            "order: observada, transporte (ida + vuelta), corregida. Publicar "
            "solo la corregida deja el ajuste sin auditar. Ver "
            "docs/METRICS_CONTRACT.md §2.2.8"
        ),
        "what_each_one_answers": {
            "observed_value": (
                "cuánto va a tardar quien despliegue ese framework como su "
                "fabricante lo pide. Es la verdad de uso e incluye el SFU que "
                "LiveKit requiere por diseño"
            ),
            "transport": (
                "cuánto de eso es dónde vive el servidor y no qué hace el "
                "framework. El de ida es el `speech_bias` de la corrida; el de "
                "vuelta se mide punta a punta (§2.2.7)"
            ),
            "corrected": (
                "cuánto trabajo hace el framework por turno, aislado del "
                "despliegue. No es lo que va a medir quien lo despliegue"
            ),
        },
        "caution": (
            "la corregida es el p50 de la resta por intervención, NO la resta "
            "de los p50: los percentiles no se restan entre sí. "
            "`deriva_por_restar_percentiles_ms` dice de cuánto es el desacuerdo "
            "entre las dos formas"
        ),
    }


def compare(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Junta las corridas por PoC y dice qué queda cuando se descuenta."""
    per_poc: dict[int, list[float]] = {}
    for run in runs:
        for row in run["detail"]:
            per_poc.setdefault(int(run["poc"]), []).append(
                float(row["return_path_ms"])
            )
    summaries = {
        str(poc): {
            "n": len(values),
            "min": round(min(values), 1),
            "p50": round(median(values), 1),
            "max": round(max(values), 1),
        }
        for poc, values in sorted(per_poc.items())
        if values
    }
    difference = None
    if "1" in summaries and "2" in summaries:
        difference = round(summaries["2"]["p50"] - summaries["1"]["p50"], 1)
    return {
        "per_poc": summaries,
        "p50_difference_ms": difference,
        "how_to_read": (
            "la diferencia es lo que el despliegue le cobra a PoC2 y no a PoC1 "
            "SOLO en el camino de vuelta. El de ida se descuenta aparte con el "
            "`speech_bias`. La suma de los dos es lo que hay que restarle a la "
            "latencia por intervención antes de atribuirle nada al framework"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="shared.metrics.return_path",
        description=(
            "Mide el transporte de vuelta de cada PoC: del audio que sale del "
            "pipeline al audio que el arnés oye. Offline y gratis."
        ),
    )
    arg_parser.add_argument(
        "files", nargs="+", help="`cero_*.json` de PoC1 o `llamada_*.json` de PoC2"
    )
    arg_parser.add_argument("--json", action="store_true")
    args = arg_parser.parse_args(argv)

    runs = [c for c in (leer(Path(a)) for a in args.files) if c is not None]
    if not runs:
        print(
            "[FALLA] ninguna corrida trae las dos puntas. PoC1 necesita "
            "`audio_outputs` (corridas del 20/09/2026 en adelante); las dos "
            "necesitan `por_intervencion` en su bloque del arnés."
        )
        return 1

    final = compare(runs)
    if args.json:
        print(json.dumps({"runs": runs, "comparacion": final}, ensure_ascii=False, indent=2))
        return 0

    for run in runs:
        summary = run["return_path_ms"]
        print(f"\n  PoC{run['poc']} · {run['run'][-28:]}")
        print(f"    punta izquierda   {run['left_tip']}")
        if summary["n"]:
            print(
                f"    camino de vuelta  n={summary['n']} · "
                f"min {summary['min']:,.0f} · p50 {summary['p50']:,.0f} · "
                f"max {summary['max']:,.0f} ms"
            )
        onset = run.get("tts_onset_ms")
        if onset and onset.get("n"):
            print(
                f"    arranque del TTS  p50 {onset['p50']:,.0f} · "
                f"max {onset['max']:,.0f} ms (NO es transporte; PoC2 no lo "
                "puede separar)"
            )
        if run.get("same_definition_at_both_ends") is False:
            print(
                "    [!] corrida vieja: la punta izquierda es la primera trama, "
                "no la primera con voz.\n        Lleva dentro el arranque del TTS."
            )
        if run["lines_without_output"]:
            print(
                f"    sin pareja        {', '.join(run['lines_without_output'])}"
            )

    print("\n  camino de vuelta, por PoC\n")
    for poc, summary in final["per_poc"].items():
        print(
            f"    PoC{poc}  n={summary['n']:>3} · min {summary['min']:>7,.0f} · "
            f"p50 {summary['p50']:>7,.0f} · max {summary['max']:>7,.0f} ms"
        )
    if final["p50_difference_ms"] is not None:
        print(
            f"\n  PoC2 paga {final['p50_difference_ms']:,.0f} ms más de vuelta que "
            "PoC1. Eso es despliegue, no framework."
        )

    tabla = latency_table(runs)
    print("\n  latencia, con las TRES cifras (contrato §2.2.8)\n")
    print(
        f"    {'':6} {'observed_value':>11} {'inbound':>9} {'outbound':>9} "
        f"{'transport':>11} {'corrected':>11}"
    )
    for poc, row in tabla["per_poc"].items():

        def _c(clave: str, row: dict[str, Any] = row) -> str:
            valor = row.get(clave)
            return "—" if valor is None else f"{valor:,.0f}"

        print(
            f"    PoC{poc:<3} {_c('observed_p50_ms'):>11} {_c('ida_p50_ms'):>9} "
            f"{_c('outbound_p50_ms'):>9} {_c('transport_p50_ms'):>11} "
            f"{_c('corrected_p50_ms'):>11}"
        )
        drift = row.get("drift_by_subtracting_percentiles_ms")
        if drift:
            print(
                f"           (restar los p50 a mano daría {drift:+,.0f} ms de "
                f"diferencia; n corregibles = {row['n_correctable']})"
            )
    print(f"\n    [!] {tabla['caution']}.")
    print("        Las tres van juntas al reporte; la corregida sola no se audita.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
