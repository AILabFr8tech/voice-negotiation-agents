"""El resumen de una llamada del arnés, en una sola estructura legible a máquina.

Existe por el modo lote. Una campaña de cien llamadas no se puede seguir leyendo
reportes en pantalla, y raspar la salida del arnés sería atar el lote al formato
de un `print`. Así que cada corrida deja además un JSON chico con lo que hace
falta para decidir: cómo terminó, cuántas intervenciones hubo, qué tan partidas,
cuánto tardó el agente y qué tan bien la oyó el STT.

Es **derivado**: todo lo que trae ya está en la bitácora, en el cruce del cero y
en el JSON del pipeline. No agrega medición nueva ni cambia ninguna. Si el
resumen y esos archivos se contradicen, mandan ellos.

El WER sí se calcula aquí, y se puede porque el arnés actúa clips pregrabados:
la referencia es exactamente la línea que se reprodujo, no una transcripción que
alguien tenga que corregir a mano. En una llamada actuada por una persona eso no
se podría; por eso este WER es el del arnés y no sustituye al de
`shared/metrics/`, que sigue siendo el de la campaña con referencia corregida.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from shared.carrier_simulator.run import RunResult
from shared.carrier_simulator.script import ScriptStep
from shared.carrier_simulator.observed import Observed
from shared.metrics.latency import percentile
from shared.metrics.clips_reference import is_dead_tail
from shared.metrics.wer import Alignment, compute_amounts, compute_text

# Versión del formato. El lote la verifica: un resumen viejo mezclado con uno
# nuevo en el mismo `progreso.jsonl` daría columnas que no significan lo mismo.
VERSION: int = 1


def _comparable_turns(
    observed: Observed, reached: int
) -> list[tuple[ScriptStep, str]]:
    """Los turnos cuyo STT se puede comparar contra lo que se reprodujo.

    Tres exclusiones, y las tres importan. Los turnos de **cola muerta** —el
    clip sonó cuando la conversación ya había terminado— no se reprodujeron
    contra nadie, y su STT vacío sería un borrado completo que hundiría el WER.
    Los de silencio no tienen línea. Y los que no vienen del guion (`--repeat`
    sobre un clip suelto) no tienen referencia contra la cual medir.

    **El corte no es `alcanzados`**, aunque se le parezca. `alcanzados` cuenta
    los turnos que el agente *contestó*, y hay un caso que cae fuera de esa
    cuenta sin ser cola muerta: el clip sonó, el STT lo transcribió, y el agente
    ya no contestó. Esa transcripción existe y trae errores reales, y
    descartarla puede invertir el signo de una brecha de WER chica. Ver
    `CONTRATO_METRICAS.md` §2.10.
    """
    del reached  # se conserva en la firma por compatibilidad; ya no corta aquí

    end = len(observed.turns)
    while end > 0:
        candidate = observed.turns[end - 1]
        if not is_dead_tail(
            had_transcript=bool((candidate.stt or "").strip()),
            had_response=bool(candidate.responded),
        ):
            break
        end -= 1

    return [
        (t.step, t.stt)
        for t in observed.turns[:end]
        if t.step.of_script and not t.step.is_silence
    ]


def run_wer(observed: Observed, reached: int) -> dict[str, Any]:
    """WER, CER y error de montos de la llamada, sumando conteos por turno.

    Se suman conteos y no se promedian tasas, por la misma razón que en
    `shared/metrics/wer.py`: promediar tasas le da el mismo peso a un turno de
    tres palabras que a uno de treinta.
    """
    pairs = _comparable_turns(observed, reached)
    wer = Alignment()
    cer = Alignment()
    amounts = Alignment()
    for step, stt in pairs:
        text = compute_text(step.text, stt)
        wer = wer + text.wer
        cer = cer + text.cer
        amounts = amounts + compute_amounts(step.text, stt).alignment

    return {
        "what_it_is": (
            "INDICATIVO, NO PUBLICABLE. Sirve para vigilar la corrida en vivo; "
            "la cifra del reporte se calcula aparte, con la referencia derivada "
            "de los clips y el pareo entre PoC (CONTRATO_METRICAS.md §2.10). "
            "Aquí la hipótesis es el STT que el arnés acumuló, y en PoC2 eso "
            "concatena resultados interinos y repite texto. Referencia = la "
            "línea que el arnés reprodujo; excluye los turnos de cola muerta "
            "—el clip sonó cuando la conversación ya había terminado, sin STT "
            "y sin respuesta— y los de silencio. Un turno con STT y sin "
            "respuesta del agente SÍ cuenta"
        ),
        "turns": len(pairs),
        "wer": round(wer.sample_rate, 6) if wer.longitud_reference else None,
        "cer": round(cer.sample_rate, 6) if cer.longitud_reference else None,
        "amount_error": round(amounts.sample_rate, 6) if amounts.longitud_reference else None,
        "counts": {
            "wer": wer.to_dict(),
            "cer": cer.to_dict(),
            "amounts": amounts.to_dict(),
        },
    }


def _latency_of(lines: dict[str, Any] | None) -> dict[str, Any] | None:
    if not lines:
        return None
    block = lines.get("line_latency_ms")
    return block if isinstance(block, dict) else None


def summarize_round_trips(raw_values: Any) -> dict[str, Any] | None:
    """Los round trips al LLM de herramienta, agregados para el lote.

    `None` cuando la PoC no los llena, que **no** es lo mismo que cero: PoC1 y
    PoC2 dejan la llave vacía (pendiente transversal de `CLAUDE.md`) y PoC3 la
    llena con su delegación. Un cero aquí se leería como
    "el agente no llamó a ninguna función", que es un dato de la corrida.

    El detalle por round trip sigue viviendo en el JSON de la llamada y lo agrega
    `shared/metrics/report.py` para el CSV; esto es el conteo que se sigue en
    vivo mientras la campaña corre.
    """
    if not isinstance(raw_values, list) or not raw_values:
        return None
    values = sorted(
        float(r["ms"]) for r in raw_values
        if isinstance(r, dict) and isinstance(r.get("ms"), (int, float))
    )
    if not values:
        return None
    return {
        "n": len(values),
        "with_function_n": sum(
            1 for r in raw_values if isinstance(r, dict) and r.get("with_function")
        ),
        "p50_ms": percentile(values, 50),
        "max_ms": values[-1],
    }


def call_summary(
    observed: Observed,
    resultado: RunResult,
    *,
    poc: int,
    scenario: int | None,
    repetition: int | None,
    label: str,
    duration_s: float,
    cost_usd: float,
    failed_criteria: int,
    lines: dict[str, Any] | None = None,
    overlap: dict[str, Any] | None = None,
    return_path: dict[str, Any] | None = None,
    llm_round_trips: Any = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Todo lo que el lote necesita de una llamada, en un diccionario plano."""
    latency = _latency_of(lines)
    return {
        "version": VERSION,
        "poc": poc,
        "scenario": scenario,
        "repetition": repetition,
        "label": label,
        "duration_s": round(duration_s, 1),
        "cost_usd": round(cost_usd, 4),
        "end": resultado.as_json(),
        "failed_criteria": failed_criteria,
        # La unidad de análisis es la intervención (§2.2.5 del contrato). El
        # bloque entero va aquí, no solo el p50, para que el lote no tenga que
        # volver a abrir el cruce para explicar un número raro.
        "lines": lines,
        "line_latency_ms": latency,
        "fragmentation": (lines or {}).get("fragmentation"),
        # Turn-taking medido sobre el WAV de auditoría, cuando se grabó. Va
        # aquí y no en `latencias` a propósito: son conteos de quién habló
        # encima de quién, no tiempos. Ver §2.3.1 del contrato.
        "overlap": overlap,
        "agent_interruptions": (overlap or {}).get("the_agent_talked_over"),
        # Cuánto de la latencia es transporte de vuelta y no framework.
        # Ver §2.2.7 del contrato.
        "return_path": return_path,
        "return_path_p50_ms": (
            (return_path or {}).get("return_path_ms") or {}
        ).get("p50"),
        # Cuántas veces el agente fue y volvió al LLM de herramienta en esta
        # llamada. No es métrica de negociación —esa familia no está en el
        # reporte— pero el número se reporta: es el término que
        # hace comparables los p50 de las tres (§2.7.1 del contrato).
        "llm_round_trips": summarize_round_trips(llm_round_trips),
        "stt": run_wer(observed, resultado.reached),
        "ramificada": bool(observed.decisions),
        "camino": [f"{d.turn}:{d.rama}" for d in observed.decisions],
        "turns_with_discard": [
            t.step.turn for t in observed.turns if t.out_of_sync
        ],
        "incidents": list(observed.incidents),
        "artifacts": dict(artifacts or {}),
        # Escenario de interrupciones: un registro por evento, con su turno.
        # Solo aparece cuando la corrida los tuvo.
        **(
            {"events": [
                {"turn": t.step.turn, **t.event}
                for t in observed.turns if t.event is not None
            ]}
            if any(t.event is not None for t in observed.turns) else {}
        ),
    }


def quote_summary(
    *,
    poc: int,
    scenario: int | None,
    script: list[ScriptStep],
    seconds: float,
    cost_usd: float,
) -> dict[str, Any]:
    """Lo que costaría la llamada, sin hacerla.

    El arnés ya imprime este aviso en pantalla; esto es el mismo número en JSON,
    para que el lote pueda armar su tabla de costos **antes** de gastar y sin
    raspar texto. Lo calcula el propio arnés a propósito: que el lote llevara su
    copia de las tarifas sería una segunda fuente de verdad del dinero.
    """
    return {
        "version": VERSION,
        "estimate_only": True,
        "poc": poc,
        "scenario": scenario,
        "turns": len(script),
        "seconds_cap": round(seconds, 1),
        "estimated_usd": round(cost_usd, 4),
    }


def measure_overlap(ruta_wav: Path | None) -> dict[str, Any] | None:
    """Cuenta la voz simultánea del WAV de auditoría, si se grabó uno.

    Envuelto en un `except` amplio y a propósito: una corrida buena no se tira
    porque el análisis de un archivo de diagnóstico falle. Sin WAV devuelve
    `None`, que es distinto de cero — cero significa que se midió y no hubo.
    """
    if ruta_wav is None or not Path(ruta_wav).is_file():
        return None
    try:
        from shared.metrics.overlap import measure

        data = measure(ruta_wav)
    except Exception as error:  # noqa: BLE001 - diagnóstico: reporta y sigue
        print(f"  [!] no se pudo medir el solapamiento: {error!r}")
        return None
    # El detalle entero no: el resumen es para leerlo de un vistazo y el WAV
    # sigue en disco para quien quiera volver a medirlo.
    data.pop("detail", None)
    return data


def measure_return_path(ruta: Path | None) -> dict[str, Any] | None:
    """El transporte de vuelta de esta llamada, si el archivo trae las dos puntas.

    Igual de defensivo que el solapamiento y por la misma razón: es un análisis
    derivado, y una corrida buena no se tira porque falle. `None` significa que
    no se pudo medir —una corrida vieja sin la sonda de salida, por ejemplo—, no
    que el camino de vuelta sea cero.
    """
    if ruta is None or not Path(ruta).is_file():
        return None
    try:
        from shared.metrics.return_path import leer

        data = leer(Path(ruta))
    except Exception as error:  # noqa: BLE001 - diagnóstico: reporta y sigue
        print(f"  [!] no se pudo medir el camino de vuelta: {error!r}")
        return None
    if data is None:
        return None
    data.pop("detail", None)
    return data


def write_summary(ruta: Path, data: dict[str, Any]) -> None:
    """Escribe el JSON de forma atómica: o está entero o no está.

    El lote lo lee justo después de que el proceso hijo terminó, y un archivo a
    medias sería indistinguible de una llamada que falló de otra forma. Se
    escribe a un temporal en la misma carpeta y se renombra, que en Windows y en
    POSIX es la operación que no deja estados intermedios visibles.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        dir=str(ruta.parent), prefix=f".{ruta.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporal, ruta)
    except BaseException:
        with_suppress = Path(temporal)
        if with_suppress.exists():
            with_suppress.unlink(missing_ok=True)
        raise
