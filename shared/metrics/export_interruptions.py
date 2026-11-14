"""Exporta los eventos de una campaña de interrupciones a CSV, con su resumen por PoC.

Lee los resúmenes que deja el lote (`summaries/<clave>.json`, llave `eventos`,
que escribe `shared.carrier_simulator.interrupciones`) de las llamadas `ok` de
`progreso.jsonl`. No mide nada nuevo: las métricas ya las calculó el arnés en
vivo, con los umbrales fijados antes de correr.

    uv run --project poc3_gpt_live python -m shared.metrics.export_interruptions --campana diagnostics/lote/<etiqueta>

Escribe en `--output` `interrupciones_eventos.csv` (una fila por evento) y
`interrupciones_resumen.txt`.

**`--include-failed`** suma los eventos de las llamadas en estado `roja`, con la
columna `estado` para poder separarlas. Una llamada sale roja por criterios de
la **latencia** (menos de 5 turnos medibles, un turno con texto y sin audio,
cierre anticipado del agente), no por la medición de interrupciones: un evento
que se disparó antes de que la llamada se cortara se midió igual que en una
`ok`, y los que no alcanzaron a dispararse salen con `disparado = 0` y no entran
a las tasas. Por default solo entran las `ok`, como en el resto del reporte;
incluirlas es una decisión de población y el resumen dice cuántas son.

**Lo que el STT oyó de cada evento**, en su propio apartado y fuera del WER
global: el arnés guarda por separado lo que se transcribió desde que arrancó el
clip del evento (`event_stt`), y aquí se compara contra el texto del clip
con la misma normalización que el WER del reporte (`shared.metrics.wer`). Es el
STT oyendo al transportista **mientras habla el agente**. Se da el WER y el CER
agregados por PoC y clase (errores sumados sobre palabras de referencia
sumadas, no promedio de tasas: un "Ajá." pesaría lo mismo que una frase), y la
tasa de omisión: eventos en que el STT no devolvió nada.

**La línea anterior pegada al evento** (igual en las tres PoC). El arnés reparte lo transcrito por la hora de
arrival: lo que llega después del arranque del clip cuenta como del evento.
Si la transcripción de la línea del transportista llega tarde —pasó en PoC3,
cuyo modelo transcribe a su ritmo— se pega al principio del texto del evento
("No le pedí más dinero… Ándele" por "Ándale") y el WER del evento sale
inflado sin que el reconocimiento se haya equivocado. `remove_glued_line`
quita ese prefijo **solo si es la línea**: el prefijo más largo del texto del
evento que, comparado con el final de la línea del turno, tiene un WER de a lo
más 1/3 (y deja al menos una palabra). El CSV guarda el texto original, el
corregido y `linea_pegada` (1 si se quitó algo), y el resumen dice cuántos.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Final

from shared.metrics.wer import compute_text, combine

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"

COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "turn", "tipo", "clip_id", "fired",
    "hablaba_en_la_reference", "stopped", "stop_time_ms",
    "reply_latency_ms", "line_duration_ms", "pause_duration_ms",
    "endless_pause", "speaks_again_ms", "advance_reason",
    "text", "event_stt", "agent_before_event", "agent_after_event",
    "turn_line", "corrected_event_stt", "glued_line", "final_turn",
    "event_wer", "cer_of_event",
]
# Las que se copian tal cual del registro del evento.
OF_LOG: Final[list[str]] = COLUMNS[4:-6]
# Prefijo que se acepta como la línea anterior pegada al texto del evento.
MAX_WER_OF_THE_PASTED_LINE: Final[float] = 1 / 3
MIN_WORDS_OF_GLUED_LINE: Final[int] = 3
# Lo que sale al CSV: lo anterior más el estado de la llamada en el lote.
OUTPUT_COLUMNS: Final[list[str]] = [*COLUMNS, "estado"]


def _valor(x: Any) -> Any:
    if isinstance(x, bool):
        return int(x)
    return "" if x is None else x


def remove_glued_line(linea: str, event_stt: str) -> tuple[str, bool]:
    """El texto del evento sin la línea anterior pegada al principio. Ver el docstring.

    Se prueba cada prefijo del texto del evento contra cada final de la línea y
    se quita el que **mejor** coincide (menor WER; en empate, el más corto), si
    no pasa de `MAX_WER_OF_THE_PASTED_LINE`. Quitar el más largo que pase el
    umbral se comería también las primeras palabras del clip.
    """
    words = event_stt.split()
    of_the_line = linea.split()
    if len(words) < 2 or len(of_the_line) < MIN_WORDS_OF_GLUED_LINE:
        return event_stt, False
    best: tuple[float, int] | None = None
    for k in range(MIN_WORDS_OF_GLUED_LINE, len(words)):
        prefix = " ".join(words[:k])
        wer = min(
            compute_text(" ".join(of_the_line[since:]), prefix).wer.sample_rate
            for since in range(0, len(of_the_line) - MIN_WORDS_OF_GLUED_LINE + 1)
        )
        if wer <= MAX_WER_OF_THE_PASTED_LINE and (best is None or wer < best[0]):
            best = (wer, k)
    if best is None:
        return event_stt, False
    return " ".join(words[best[1]:]), True


def _of_the_log(summary: dict[str, Any]) -> tuple[dict[str, str], str | None]:
    """Turno → línea que el arnés reprodujo, y el último turno que se actuó.

    El último turno actuado es el de la despedida: ahí la llamada se cierra
    con la regla de colgar de cada PoC, que no es del framework. El CSV lo
    marca en `turno_final` para poder separar esos eventos.
    """
    ruta = Path((summary.get("artifacts") or {}).get("event_log") or "")
    if not ruta.is_file():
        return {}, None
    turns = json.loads(ruta.read_text(encoding="utf-8")).get("turns") or []
    text_lines = {t.get("turn"): t.get("played_line") or "" for t in turns}
    return text_lines, (turns[-1].get("turn") if turns else None)


def campaign_rows(folder: Path, *, include_failed: bool = False) -> list[dict[str, Any]]:
    progress = folder / "progress.jsonl"
    if not progress.is_file():
        raise SystemExit(f"no encuentro {progress}")
    rows: list[dict[str, Any]] = []
    for linea in progress.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        p = json.loads(linea)
        if p.get("estado") not in (("ok", "roja") if include_failed else ("ok",)):
            continue
        ruta = folder / "summaries" / f"{p['clave']}.json"
        if not ruta.is_file():
            continue
        summary = json.loads(ruta.read_text(encoding="utf-8"))
        text_lines, last_turn = _of_the_log(summary)
        for ev in summary.get("events") or []:
            row = {
                "poc": p.get("poc"), "clave": p["clave"],
                "scenario": p.get("scenario"), "repetition": p.get("repetition"),
                **{c: _valor(ev.get(c)) for c in OF_LOG},
                "advance_reason": ev.get("advance_reason", ev.get("motivo", "")),
                "turn_line": text_lines.get(ev.get("turn"), ""),
                "corrected_event_stt": "", "glued_line": "",
                "final_turn": "" if last_turn is None else int(ev.get("turn") == last_turn),
                "event_wer": "", "cer_of_event": "",
                "estado": p.get("estado"),
            }
            if ev.get("text") and ev.get("event_stt") is not None and ev.get("fired"):
                corrected_value, glued_line = remove_glued_line(row["turn_line"], ev["event_stt"])
                row["corrected_event_stt"] = corrected_value
                row["glued_line"] = int(glued_line)
                r = compute_text(ev["text"], corrected_value)
                row["event_wer"] = round(r.wer.sample_rate, 4)
                row["cer_of_event"] = round(r.cer.sample_rate, 4)
            rows.append(row)
    return rows


def what_the_stt_heard(rows: list[dict[str, Any]]) -> str:
    """WER y CER agregados de los clips de evento, por PoC y clase."""
    text_lines = [
        "Lo que el STT oyó del evento (mientras hablaba el agente; fuera del WER global)",
        f"{'':6}{'kind':<14}{'n':>4}{'WER':>9}{'CER':>9}{'sin texto':>12}",
    ]
    for poc in sorted({f["poc"] for f in rows}):
        for tipo in ("interrupcion", "apoyo"):
            fs = [
                f for f in rows
                if f["poc"] == poc and f["tipo"] == tipo and f["fired"] == 1
                and f["text"] != "" and f["event_wer"] != ""
            ]
            if not fs:
                continue
            pairs = [compute_text(f["text"], f["corrected_event_stt"]) for f in fs]
            wer = combine(r.wer for r in pairs)
            cer = combine(r.cer for r in pairs)
            empty_items = sum(1 for f in fs if not str(f["corrected_event_stt"]).strip())
            glued_lines = sum(1 for f in fs if f["glued_line"] == 1)
            text_lines.append(
                f"PoC{poc:<3}{tipo:<14}{len(fs):>4}{100 * wer.sample_rate:>8.1f}%{100 * cer.sample_rate:>8.1f}%"
                f"{empty_items:>6} ({100 * empty_items / len(fs):.0f}%)"
                + (f"   ({glued_lines} con la línea anterior pegada, quitada)" if glued_lines else "")
            )
    return "\n".join(text_lines) + "\n"


def _p(v: list[float], q: float) -> str:
    if not v:
        return "—"
    v = sorted(v)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return f"{v[lo] + (v[hi] - v[lo]) * (k - lo):,.0f}"


def summary(rows: list[dict[str, Any]]) -> str:
    text_lines = ["Campaña de interrupciones — resumen por PoC", ""]
    for poc in sorted({f["poc"] for f in rows}):
        of = [f for f in rows if f["poc"] == poc]
        failed_calls = len({f["clave"] for f in of if f.get("estado") == "roja"})
        text_lines.append(
            f"PoC{poc} ({len({f['clave'] for f in of})} llamadas"
            + (f", {failed_calls} roja(s) incluida(s)" if failed_calls else "") + ")"
        )
        for tipo, what in (("interrupcion", "se calló (esperado)"),
                          ("apoyo", "se calló (falsa detención)"),
                          ("control", "se calló solo (tasa de base)")):
            fs = [f for f in of if f["tipo"] == tipo]
            fired_list = [f for f in fs if f["fired"] == 1 and f["stopped"] != ""]
            det = [f for f in fired_list if f["stopped"] == 1]
            sample_rate = f"{len(det)}/{len(fired_list)}" + (f" ({100 * len(det) / len(fired_list):.0f}%)" if fired_list else "")
            linea = f"  {tipo:<13} eventos {len(fs):>3} · disparados {len(fired_list):>3} · {what}: {sample_rate}"
            if tipo == "interrupcion":
                t = [float(f["stop_time_ms"]) for f in det]
                r = [float(f["reply_latency_ms"]) for f in det if f["reply_latency_ms"] != ""]
                linea += (f"\n    tiempo de detención p50 {_p(t, .5)} · p75 {_p(t, .75)} · p95 {_p(t, .95)} ms"
                          f"\n    latencia de réplica p50 {_p(r, .5)} ms (n={len(r)})")
            # Pausa y retome (arnés desde el 5 de octubre de 2026; vacío antes).
            resumed = [f for f in det if f.get("endless_pause") == 0 and f.get("pause_duration_ms") != ""]
            no_end = [f for f in det if f.get("endless_pause") == 1]
            if resumed or no_end:
                pause = [float(f["pause_duration_ms"]) for f in resumed]
                returns = [float(f["speaks_again_ms"]) for f in resumed]
                linea += (f"\n    retomó {len(resumed)}/{len(det)} · pausa p50 {_p(pause, .5)} ms"
                          f" · vuelve a hablar p50 {_p(returns, .5)} ms"
                          f" · no volvió antes de la línea siguiente {len(no_end)}")
            text_lines.append(linea)
        text_lines.append("")
    return "\n".join(text_lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campana", type=Path, required=True, help="carpeta de la campaña del lote")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--include-failed", action="store_true",
                   help="suma los eventos de las llamadas rojas (ver el docstring)")
    args = p.parse_args(argv)
    rows = campaign_rows(args.campana, include_failed=args.include_failed)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "interruptions_events.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows) + "\n" + what_the_stt_heard(rows)
    (args.output / "interruptions_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"{len(rows)} eventos escritos en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
