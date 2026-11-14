"""Consolidado por PoC: junta las 15 llamadas en un Markdown y un CSV.

Es la salida que alimenta `docs/comparativo_results.md`. Dos formatos porque
sirven a dos cosas distintas: el Markdown se lee y se pega en el reporte de
residencia, el CSV se abre en una hoja de cálculo para cruzar las 3 PoC.

## Cómo se agrega cada familia de métricas

- **Latencia:** se juntan las muestras crudas de las 15 llamadas en una sola
  poza por cubeta y de ahí sale el percentil. **No** se promedian los p95 de
  cada llamada: el promedio de percentiles no es un percentil de nada, y con ~8
  turnos por llamada el p95 individual es casi el máximo.
- **WER/CER/amounts:** se suman los conteos de errores y de palabras de
  referencia, y la tasa se calcula al final. Promediar los WER por llamada le
  daría el mismo peso a un turno de 3 palabras que a uno de 40.
- **Naturalidad y turn-taking:** son ordinales 1-5 puestos a mano. Se reporta
  media, mediana y mínimo — la media sola esconde una llamada con 1.
- **Confiabilidad:** duración y cobertura de la captura.

Aquí solo entra lo que distingue una tecnología de voz de otra. Lo que el agente
negocie es el vehículo de la conversación —da vocabulario de dominio al WER y
ejercita el tool calling— pero no es objeto de medición, así que no aparece en
este reporte ni en el CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Final, Sequence

from shared.metrics import latency as lat
from shared.metrics import wer as wer_mod
from shared.metrics.calls import CallMetrics

# Columnas del CSV por llamada. Fijas y en este orden: el CSV se cruza entre las
# 3 PoC y las columnas tienen que coincidir.
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "poc",
    "scenario",
    "repetition",
    "call_sid",
    "duration_s",
    "measured_turns",
    "turns_with_number",
    "measurement_coverage",
    "full_turn_p50_regime_ms",
    "full_turn_p95_regime_ms",
    "full_turn_opening_ms",
    # ⚠ `llm_ttft_p50_regimen_ms` **no es comparable entre PoC1 y PoC2**: PoC1
    # para en el primer token de TEXTO (con el round trip de la herramienta
    # adentro) y PoC2 en el primer token del nodo (sin él). Además la de PoC1 es
    # bimodal, así que su p50 cae en el valle. Ver `CONTRATO_METRICAS.md`
    # §2.7.1 antes de ponerlas lado a lado.
    "llm_ttft_p50_regime_ms",
    "tts_ttfb_p50_regime_ms",
    "stt_p50_regimen_ms",
    "wer",
    "cer",
    "amounts_error",
    "words_reference",
    "naturalness",
    "turn_taking",
    "unwanted_interruptions",
    "false_end_of_turn",
    # Round trip al LLM auxiliar que la PoC pudo aislar. Hoy solo PoC3 los llena
    # (su delegación a `gpt-4.1-mini`); en PoC1 y PoC2 salen vacías. Ver
    # `llamadas.LLAVE_ROUND_TRIPS_LLM`: **no** es `llm_ttft_ms` y no se compara
    # contra esa columna.
    "llm_delegate_n",
    "llm_delegate_p50_ms",
    "llm_delegate_max_ms",
    "llm_delegate_with_function_n",
    "llm_delegate_covered_by_speech_n",
    "warning_list",
)


@dataclass
class Consolidated:
    """Agregado de una PoC completa."""

    poc: str = ""
    calls: list[CallMetrics] = field(default_factory=list)
    latency: dict[str, Any] = field(default_factory=dict)
    stt: dict[str, Any] = field(default_factory=dict)
    subjective: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    warning_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "poc": self.poc,
            "n_calls": len(self.calls),
            "latency": self.latency,
            "stt": self.stt,
            "subjective": self.subjective,
            "reliability": self.reliability,
            "warning_list": list(self.warning_list),
        }


# --------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------


def _add_latency(calls: Sequence[CallMetrics]) -> dict[str, Any]:
    """Junta las muestras crudas de todas las llamadas y saca los percentiles.

    Las llamadas marcadas con `exclude_from_pooled` **no entran a la poza**. Es
    para corridas cuya arquitectura no fue la que se quería medir —el caso real
    es el turn detector de PoC2 cayendo de remoto a local a media llamada—, donde
    juntar sus turnos con los demás mezclaría dos cosas distintas en el mismo
    percentil.

    No es lo mismo que descartarlas: siguen apareciendo en el detalle por llamada
    y su WER sigue contando. Lo que se les quita es el voto en los agregados, y
    el reporte dice cuántas fueron y por qué.
    """
    all_items: list[lat.TurnSample] = []
    with_samples = 0
    excluded_calls: list[str] = []

    for call in calls:
        if call.exclude_from_pooled is not None:
            motivo = str(
                call.exclude_from_pooled.get("motivo")
                or call.exclude_from_pooled.get("detail")
                or "sin motivo"
            )
            excluded_calls.append(f"{call.label}: {motivo}")
            continue
        if call._samples:
            with_samples += 1
        all_items.extend(call._samples)

    summary = lat.summarize_samples(all_items, calls=max(with_samples, 1))
    if excluded_calls:
        summary["excluded_from_pool"] = excluded_calls
        summary.setdefault("advertencias", []).append(
            f"{len(excluded_calls)} llamada(s) fuera de la poza de percentiles "
            "(arquitectura distinta de la que se quería medir)"
        )
    return summary


def _add_stt(calls: Sequence[CallMetrics]) -> dict[str, Any]:
    accumulator = wer_mod.AccumulatorSTT()
    no_reference = 0
    for call in calls:
        if call._text is None or call._amounts is None:
            no_reference += 1
            continue
        accumulator.add(call._text, call._amounts)
    output = accumulator.to_dict()
    output["llamadas_sin_reference"] = no_reference
    return output


def _add_reliability(calls: Sequence[CallMetrics]) -> dict[str, Any]:
    """Cuánto duraron las llamadas y qué tan completa quedó la captura."""
    total = len(calls)
    durations: list[float] = []
    no_latency = 0
    turns = 0

    # Cobertura de la medición: de los turnos capturados, cuántos admitieron un
    # número. **Es resultado, no hueco de instrumentación.** En PoC3 el modelo es
    # full-duplex y arranca encima del interlocutor, así que una fracción alta de
    # turnos no tiene un fin de habla contra el cual medir; en PoC1 y PoC2 eso es
    # raro. La diferencia entre las tres es justo lo que la comparativa quiere ver,
    # y antes quedaba enterrada en una advertencia de exclusión.
    with_number = 0
    no_number = 0

    for call in calls:
        if isinstance(call.duration_s, (int, float)):
            durations.append(float(call.duration_s))
        n = call.latency.get("n_turns")
        if isinstance(n, int):
            turns += n
        if not call._samples:
            no_latency += 1
        for sample in call._samples:
            if getattr(sample, lat.CAMPO_PRINCIPAL) is None:
                no_number += 1
            else:
                with_number += 1

    captured_ones = with_number + no_number
    return {
        "n_calls": total,
        "mean_duration_s": (
            round(sum(durations) / len(durations), 3) if durations else None
        ),
        "total_duration_s": round(sum(durations), 3) if durations else None,
        "measured_turns": turns,
        "calls_without_latency": no_latency,
        "turns_with_number": with_number,
        "turns_without_number": no_number,
        "measurement_coverage": (
            round(with_number / captured_ones, 4) if captured_ones else None
        ),
    }


def _add_subjective(calls: Sequence[CallMetrics]) -> dict[str, Any]:
    def collect(campo: str) -> list[float]:
        values: list[float] = []
        for call in calls:
            notes = call.manual_notes or {}
            valor = notes.get(campo)
            if isinstance(valor, (int, float)):
                values.append(float(valor))
        return values

    output: dict[str, Any] = {}
    for campo in ("naturalness", "turn_taking"):
        values = collect(campo)
        output[campo] = {
            "n": len(values),
            "mean": round(sum(values) / len(values), 2) if values else None,
            "median_value": round(median(values), 2) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    for campo in ("unwanted_interruptions", "false_end_of_turn"):
        values = collect(campo)
        output[campo] = {
            "n": len(values),
            "total": int(sum(values)) if values else None,
        }
    return output


def _global_warnings(
    calls: Sequence[CallMetrics], consolidated: Consolidated
) -> list[str]:
    warning_list: list[str] = []

    if not calls:
        return ["no se encontró ninguna llamada"]

    expected_variables = 15
    if len(calls) < expected_variables:
        warning_list.append(
            f"{len(calls)} llamadas de las {expected_variables} del guion "
            "(5 escenarios × 3 repeticiones)"
        )

    no_ref = consolidated.stt.get("llamadas_sin_reference") or 0
    if no_ref:
        warning_list.append(
            f"{no_ref} llamada(s) sin archivo de referencia corregido: no entran "
            "al WER"
        )

    missing_notes = sum(
        1
        for call in calls
        if not (call.manual_notes or {}).get("completed")
    )
    if missing_notes:
        warning_list.append(
            f"{missing_notes} llamada(s) sin las puntuaciones 1-5 de naturalidad / "
            "turn-taking"
        )

    no_latency = consolidated.reliability.get("calls_without_latency") or 0
    if no_latency:
        warning_list.append(
            f"{no_latency} llamada(s) sin muestras de latencia: revisa la captura "
            "con `python -m shared.metrics.cli verificar`"
        )

    warning_list.extend(consolidated.latency.get("advertencias") or [])

    # Cobertura por escenario: falta detectar el escenario que nunca se corrió.
    per_scenario: dict[Any, int] = {}
    for call in calls:
        per_scenario[call.scenario] = per_scenario.get(call.scenario, 0) + 1
    absent = [str(e) for e in range(1, 6) if e not in per_scenario]
    if absent:
        warning_list.append(f"escenarios sin llamadas: {', '.join(absent)}")
    if None in per_scenario:
        warning_list.append(
            f"{per_scenario[None]} llamada(s) sin escenario identificado "
            "(falta el campo en el JSON o en el archivo de referencia)"
        )

    return warning_list


def consolidate(
    calls: Sequence[CallMetrics], *, poc: str = ""
) -> Consolidated:
    """Agrega las métricas de todas las llamadas de una PoC."""
    ordered = sorted(
        calls,
        key=lambda m: (
            m.scenario if m.scenario is not None else 99,
            m.repetition if m.repetition is not None else 99,
            m.call_sid,
        ),
    )
    consolidated = Consolidated(poc=poc, calls=list(ordered))
    consolidated.latency = _add_latency(ordered)
    consolidated.stt = _add_stt(ordered)
    consolidated.subjective = _add_subjective(ordered)
    consolidated.reliability = _add_reliability(ordered)
    consolidated.warning_list = _global_warnings(ordered, consolidated)
    return consolidated


# --------------------------------------------------------------------------
# Salida: CSV
# --------------------------------------------------------------------------


def bucket_value(
    block: dict[str, Any], bucket: str, campo: str, label: str
) -> Any:
    """Saca un percentil del resumen de latencia sin reventar si no está."""
    buckets = block.get("buckets") or {}
    data = buckets.get(bucket) or {}
    if campo == "full_turn":
        return (data.get("full_turn") or {}).get(label)
    stages = data.get("stages") or {}
    return (stages.get(campo) or {}).get(label)


def _coverage_of(call: CallMetrics) -> dict[str, Any]:
    """Turnos con número y su fracción. Ver `_add_reliability`."""
    having = sum(
        1
        for m in call._samples
        if getattr(m, lat.CAMPO_PRINCIPAL) is not None
    )
    total = len(call._samples)
    return {
        "turns_with_number": having,
        "measurement_coverage": round(having / total, 4) if total else None,
    }


def _delegate_llm_columns(call: CallMetrics) -> dict[str, Any]:
    """Las columnas del round trip al LLM auxiliar. Vacías si la PoC no lo publica.

    `llm_delegado_tapado_por_voz_n` es el dato que hace comparable a PoC3 con las
    otras dos: cuántos round trips duraron **más** que el peor turno medido de la
    llamada, o sea que el modelo de voz ya estaba hablando cuando el delegado
    contestó. Esos round trips NO están dentro de `full_turn_ms` — el relleno
    los tapa— mientras que en PoC1 el mismo ida y vuelta sí se paga como aire
    muerto. Sin esta columna, comparar los p50 crudos entre PoC engaña.
    """
    empty_ones = dict.fromkeys(
        (
            "llm_delegate_n",
            "llm_delegate_p50_ms",
            "llm_delegate_max_ms",
            "llm_delegate_with_function_n",
            "llm_delegate_covered_by_speech_n",
        ),
        None,
    )
    round_trips = call.round_trips_llm
    if not round_trips:
        return empty_ones

    values = [float(r["ms"]) for r in round_trips]
    worst_turn = bucket_value(
        call.latency, lat.BUCKET_REGIMEN, "full_turn", "max"
    )
    masked = (
        sum(1 for v in values if v > float(worst_turn))
        if isinstance(worst_turn, (int, float))
        else None
    )
    return {
        "llm_delegate_n": len(values),
        "llm_delegate_p50_ms": lat.percentile(sorted(values), 50),
        "llm_delegate_max_ms": max(values),
        "llm_delegate_with_function_n": sum(1 for r in round_trips if r["with_function"]),
        "llm_delegate_covered_by_speech_n": masked,
    }


def csv_row(call: CallMetrics) -> dict[str, Any]:
    """Una llamada como fila plana del CSV."""
    latency = call.latency
    stt = call.stt or {}
    canal = stt.get("full_channel") or {}
    amounts = stt.get("amounts") or {}
    notes = call.manual_notes or {}

    return {
        "poc": call.poc,
        "scenario": call.scenario,
        "repetition": call.repetition,
        "call_sid": call.call_sid,
        "duration_s": call.duration_s,
        "measured_turns": latency.get("n_turns"),
        **_coverage_of(call),
        "full_turn_p50_regime_ms": bucket_value(
            latency, lat.BUCKET_REGIMEN, "full_turn", "p50"
        ),
        "full_turn_p95_regime_ms": bucket_value(
            latency, lat.BUCKET_REGIMEN, "full_turn", "p95"
        ),
        "full_turn_opening_ms": bucket_value(
            latency, lat.BUCKET_OPENING, "full_turn", "p50"
        ),
        "llm_ttft_p50_regime_ms": bucket_value(
            latency, lat.BUCKET_REGIMEN, "llm_ttft", "p50"
        ),
        "tts_ttfb_p50_regime_ms": bucket_value(
            latency, lat.BUCKET_REGIMEN, "tts_ttfb", "p50"
        ),
        "stt_p50_regimen_ms": bucket_value(latency, lat.BUCKET_REGIMEN, "stt", "p50"),
        "wer": (canal.get("wer") or {}).get("sample_rate"),
        "cer": (canal.get("cer") or {}).get("sample_rate"),
        "amounts_error": amounts.get("error_rate"),
        "words_reference": canal.get("words_reference"),
        "naturalness": notes.get("naturalness"),
        "turn_taking": notes.get("turn_taking"),
        "unwanted_interruptions": notes.get("unwanted_interruptions"),
        "false_end_of_turn": notes.get("false_end_of_turn"),
        **_delegate_llm_columns(call),
        "warning_list": " / ".join(call.warning_list),
    }


def write_csv(consolidated: Consolidated, ruta: Path | str) -> Path:
    """Un CSV con una fila por llamada. Es lo que se abre en la hoja de cálculo."""
    destination = Path(ruta)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # newline="" es lo que pide el módulo csv para no duplicar saltos de línea
    # en Windows.
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for call in consolidated.calls:
            writer.writerow(csv_row(call))
    return destination


# --------------------------------------------------------------------------
# Salida: Markdown
# --------------------------------------------------------------------------


def _fmt(valor: Any, *, decimals: int = 1, suffix: str = "") -> str:
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:,.{decimals}f}{suffix}"
    return str(valor)


def _pct(valor: Any, *, decimals: int = 1) -> str:
    if not isinstance(valor, (int, float)):
        return "—"
    return f"{valor * 100:.{decimals}f}%"


def _tabla(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    text_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        text_lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return text_lines


def _excluded_block(excluded_calls: Sequence[str]) -> list[str]:
    """Qué llamadas quedaron fuera de la poza, y por qué."""
    if not excluded_calls:
        return []
    return [
        f"**{len(excluded_calls)} llamada(s) fuera de estos percentiles**, porque su "
        "arquitectura no fue la que se quería medir. Siguen en el detalle por "
        "llamada y su WER sigue contando; lo único que se les quita es el voto "
        "en los agregados:",
        "",
        *(f"- {motivo}" for motivo in excluded_calls),
        "",
    ]


def _section_latency(block: dict[str, Any]) -> list[str]:
    text_lines = ["## Latencia", ""]
    excluded_calls = block.get("excluded_from_pool") or []

    if not block.get("n_turns"):
        # Sin la distinción de abajo, una corrida donde TODAS las llamadas
        # quedaron excluidas se leería como "falta instrumentación", que es un
        # diagnóstico completamente distinto.
        if excluded_calls:
            text_lines += _excluded_block(excluded_calls)
            text_lines += [
                "No queda ninguna llamada en la poza, así que no hay percentiles "
                "agregados que reportar. **No es falta de instrumentación**: es que "
                "todas las corridas midieron algo distinto de lo que se quería "
                "medir, y hay que repetirlas.",
                "",
            ]
        else:
            text_lines += [
                "Sin muestras de latencia: ninguna llamada trae datos en `latencias`",
                "(falta instrumentación), así que esta sección queda vacía aunque las",
                "llamadas se hayan corrido.",
                "",
            ]
        return text_lines

    text_lines += [
        f"Turnos medidos: **{block['n_turns']}**. Todo en milisegundos.",
        "",
    ]
    text_lines += _excluded_block(excluded_calls)

    text_lines += [
        "### Turno completo (fin de habla del transportista → primer audio del agente)",
        "",
    ]

    rows = []
    for bucket in lat.BUCKETS:
        data = (block.get("buckets") or {}).get(bucket) or {}
        ft = data.get("full_turn") or {}
        if not ft.get("n"):
            continue
        reliable = ft.get("reliable") or {}
        mark = "" if reliable.get("p95") else " ⚠"
        rows.append(
            [
                bucket,
                ft.get("n", 0),
                _fmt(ft.get("p25")),
                _fmt(ft.get("p50")),
                _fmt(ft.get("p75")),
                _fmt(ft.get("p95")) + mark,
                _fmt(ft.get("p99")) + mark,
                _fmt(ft.get("max")),
            ]
        )

    if rows:
        text_lines += _tabla(
            ["bucket", "n", "p25", "p50", "p75", "p95", "p99", "máx"], rows
        )
        text_lines += [
            "",
            "⚠ = muestra insuficiente para citar ese percentil (ver advertencias).",
            "",
        ]

    # Desglose por etapa, solo de régimen: es el que describe la conversación.
    regimen = (block.get("buckets") or {}).get(lat.BUCKET_REGIMEN) or {}
    stages = regimen.get("stages") or {}
    if stages:
        text_lines += ["### Desglose por etapa (solo régimen)", ""]
        stage_rows = [
            [
                name,
                data.get("n", 0),
                _fmt(data.get("p50")),
                _fmt(data.get("p95")),
                _fmt(data.get("max")),
            ]
            for name, data in stages.items()
        ]
        text_lines += _tabla(["stage", "n", "p50", "p95", "máx"], stage_rows)
        text_lines += [
            "",
            "En PoC3 (GPT-Live) esta tabla no aplica: no hay etapas que aislar.",
            "",
        ]
        # El aviso se imprime siempre que haya `llm_ttft`, no solo cuando el
        # lector sepa buscarlo: es el renglón que ya se citó mal una vez.
        if "llm_ttft" in stages:
            text_lines += [
                "> ⚠ **`llm_ttft` no es comparable entre PoC1 y PoC2, y su p50 de",
                "> PoC1 no describe ningún turno.** PoC1 para el cronómetro en el",
                "> primer token de *texto*, así que un turno con tool call lleva",
                "> adentro el round trip completo de la función; PoC2 lee",
                "> `llm_node_ttft`, que para en la llamada a función. Eso vuelve la",
                "> distribución de PoC1 **bimodal** (672 ms sin round trip contra",
                "> 1,687 con él) y su p50 cae en el valle. Partir por modo antes de",
                "> percentilar: `docs/METRICS_CONTRACT.md` §2.7.1.",
                "",
            ]

    return text_lines


def _section_stt(block: dict[str, Any]) -> list[str]:
    text_lines = ["## Calidad de STT", ""]

    if not block.get("calls"):
        text_lines += [
            "Sin archivos de referencia corregidos: no hay WER. Genera las",
            "referencias con `python -m shared.metrics.cli referencia` y corrige",
            "las líneas `yo:` (2-3 min por llamada).",
            "",
        ]
        return text_lines

    wer = block.get("wer") or {}
    cer = block.get("cer") or {}
    amounts = block.get("amounts") or {}

    text_lines += [
        f"Sobre **{block['calls']}** llamada(s) con referencia corregida. "
        "Las tasas se calculan sumando errores y palabras de todas las llamadas, "
        "no promediando WER por llamada.",
        "",
    ]
    text_lines += _tabla(
        ["métrica", "sample_rate", "sust.", "borr.", "inser.", "reference"],
        [
            [
                "WER",
                _pct(wer.get("sample_rate"), decimals=2),
                wer.get("substitutions", 0),
                wer.get("deleted", 0),
                wer.get("insertions", 0),
                f"{wer.get('longitud_reference', 0)} palabras",
            ],
            [
                "CER",
                _pct(cer.get("sample_rate"), decimals=2),
                cer.get("substitutions", 0),
                cer.get("deleted", 0),
                cer.get("insertions", 0),
                f"{cer.get('longitud_reference', 0)} caracteres",
            ],
            [
                "Montos",
                _pct(amounts.get("sample_rate"), decimals=2),
                amounts.get("substitutions", 0),
                amounts.get("deleted", 0),
                amounts.get("insertions", 0),
                f"{amounts.get('longitud_reference', 0)} montos",
            ],
        ],
    )
    text_lines += [
        "",
        "En **Montos**, `sust.` = oyó otro número, `borr.` = se lo comió, "
        "`inser.` = inventó uno. Un WER bajo con montos mal oídos no sirve: "
        "equivocar un monto es peor que equivocar una palabra cualquiera.",
        "",
        "> ⚠ **Este WER sale de los archivos de referencia, y NO es el que el",
        "> arnés publica por llamada.** El `wer_indicativo` de `progreso.jsonl`",
        "> y el bloque `stt` de los resúmenes del lote son para vigilar la",
        "> corrida en vivo: su hipótesis es el STT que el arnés acumuló, que en",
        "> PoC2 concatena resultados interinos y repite texto. **No se citan y",
        "> no se mezclan con esta tabla.** Y si estas cifras salen de corridas",
        "> del arnés, comparan PoC1 contra PoC2 sólo con la poza **pareada**:",
        "> `docs/METRICS_CONTRACT.md` §2.10.",
        "",
    ]
    return text_lines


def _section_reliability(block: dict[str, Any]) -> list[str]:
    text_lines = ["## Confiabilidad", ""]
    rows = [
        ["Llamadas", block.get("n_calls", 0)],
        ["Duración promedio", _fmt(block.get("mean_duration_s"), suffix=" s")],
        ["Duración total", _fmt(block.get("total_duration_s"), suffix=" s")],
        ["Turnos medidos", block.get("measured_turns", 0)],
        ["Llamadas sin latencia capturada", block.get("calls_without_latency", 0)],
        [
            "Cobertura de la medición",
            _pct(block.get("measurement_coverage"))
            + f" ({block.get('turns_with_number', 0)} de "
            + f"{block.get('turns_with_number', 0) + block.get('turns_without_number', 0)}"
            " turnos con número)",
        ],
    ]
    text_lines += _tabla(["métrica", "valor"], rows)
    if block.get("turns_without_number"):
        text_lines += [
            "",
            "**La cobertura es un resultado, no un hueco de instrumentación.** Un "
            "turno queda sin número cuando no hubo un fin de habla contra el cual "
            "medir — en una arquitectura full-duplex el modelo arranca encima del "
            "interlocutor y eso es su modo normal de operación, no un fallo. La "
            "razón de cada turno sin número está en sus `notas`.",
            "",
            "⚠ **Sesgo que hay que declarar al citar el p50:** si el agente se "
            "encima justo cuando responde rápido, los turnos que quedan medidos son "
            "los lentos y el percentil sale hacia arriba. El sesgo es del mecanismo, "
            "no de la muestra, así que no se corrige descartando llamadas: se lee el "
            "p50 junto con esta cobertura.",
        ]
    text_lines.append("")
    return text_lines


def _section_subjective(block: dict[str, Any]) -> list[str]:
    text_lines = ["## Naturalidad y turn-taking (manual, 1-5)", ""]
    rows = []
    for campo in ("naturalness", "turn_taking"):
        data = block.get(campo) or {}
        rows.append(
            [
                campo,
                data.get("n", 0),
                _fmt(data.get("mean"), decimals=2),
                _fmt(data.get("median_value"), decimals=2),
                _fmt(data.get("min"), decimals=0),
                _fmt(data.get("max"), decimals=0),
            ]
        )
    text_lines += _tabla(["métrica", "n", "mean", "median_value", "mín", "máx"], rows)

    counts = []
    for campo in ("unwanted_interruptions", "false_end_of_turn"):
        data = block.get(campo) or {}
        counts.append([campo, data.get("n", 0), _fmt(data.get("total"), decimals=0)])
    text_lines += ["", "### Conteos de la rúbrica", ""]
    text_lines += _tabla(["count", "llamadas anotadas", "total"], counts)
    text_lines.append("")
    return text_lines


def _section_per_call(consolidated: Consolidated) -> list[str]:
    text_lines = ["## Detalle por llamada", ""]
    rows = []
    for call in consolidated.calls:
        stt = call.stt or {}
        canal = stt.get("full_channel") or {}
        wer = (canal.get("wer") or {}).get("sample_rate")
        amounts = (stt.get("amounts") or {}).get("error_rate")
        notes = call.manual_notes or {}
        rows.append(
            [
                call.label,
                _fmt(
                    bucket_value(
                        call.latency, lat.BUCKET_REGIMEN, "full_turn", "p50"
                    ),
                    decimals=0,
                ),
                _pct(wer, decimals=1),
                _pct(amounts, decimals=1),
                _fmt(call.duration_s, decimals=0),
                _fmt(notes.get("naturalness"), decimals=0),
                _fmt(notes.get("turn_taking"), decimals=0),
            ]
        )
    text_lines += _tabla(
        [
            "scen.rep",
            "p50 régimen (ms)",
            "WER",
            "err. montos",
            "duración (s)",
            "nat.",
            "t-t",
        ],
        rows,
    )
    text_lines.append("")
    return text_lines


def generate_markdown(consolidated: Consolidated) -> str:
    """El reporte consolidado en Markdown, listo para pegar en el comparativo."""
    title = f"# Métricas consolidadas — {consolidated.poc or 'PoC sin nombre'}"
    text_lines: list[str] = [
        title,
        "",
        f"{len(consolidated.calls)} llamada(s). Generado por "
        "`shared/metrics/` — todo lo numérico es automático; lo único manual son "
        "las líneas `yo:` de las referencias y las dos notas 1-5.",
        "",
    ]

    if consolidated.warning_list:
        text_lines += ["## Advertencias", ""]
        text_lines += [f"- {aviso}" for aviso in consolidated.warning_list]
        text_lines.append("")

    text_lines += _section_latency(consolidated.latency)
    text_lines += _section_stt(consolidated.stt)
    text_lines += _section_subjective(consolidated.subjective)
    text_lines += _section_reliability(consolidated.reliability)
    text_lines += _section_per_call(consolidated)

    return "\n".join(text_lines).rstrip() + "\n"


def write_markdown(consolidated: Consolidated, ruta: Path | str) -> Path:
    destination = Path(ruta)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(generate_markdown(consolidated), encoding="utf-8")
    return destination
