"""Cifras de la sección de interrupciones del reporte, como macros de LaTeX.

Igual que `analysis_report.py`: ningún número de la sección se teclea a mano.
Lee de `docs/report/data/` y escribe `docs/report/latex/generated/interruptions.tex`.

    uv run --project analisis python -m shared.metrics.interruptions_report

**Campaña que se reporta: `campana_interrupciones_v3/`** (3 repeticiones × 5
escenarios × 3 PoC): PoC1 y PoC2 de la campaña del 5 de octubre de 2026 y PoC3
repetida el 9 de octubre con la API pública en μ-law a 8 kHz. La PoC3 del 5 de
octubre (versión alpha) queda en `campana_interrupciones_v2/` y sale en el
reporte por `alpha_reporte.py`. Las anteriores —el piloto del 30 de septiembre
(`interrupciones_con_rojas/`, `interrupciones_poc3/`) y la primera campaña
(`campana_interrupciones/`)— quedan como antecedente: de ellas solo salen
cuántas llamadas tuvieron.

Fuentes, todas dentro de `campana_interrupciones_v3/` salvo donde se dice:

- `interrupciones_eventos.csv` (`shared.metrics.export_interruptions
  --include-failed`). **Población:** todas las llamadas, `ok` y `rojas` (salen
  rojas por criterios de latencia, no de esta medición), sin los eventos del
  último turno actuado de la llamada (`turno_final = 1`, la despedida). Las
  tres PoC se miden con el arnés: la voz de PoC3 se escucha en tramos de
  20 ms, como la de PoC1 y PoC2.
- `interruption_pauses.csv` (`validation.interruption_pauses`): la misma
  regla de detención aplicada a la grabación de dos canales. Solo se usa para
  contar en cuántos eventos coinciden la grabación y el arnés.
- `hoja_retomes_autor.csv` + `hoja_retomes_clave.csv` (`validation.resumption_sheet_audio`):
  qué hizo el agente al retomar tras un apoyo (C, R, N, T, ?), según la
  calificación del autor (12 de octubre de 2026), hecha oyendo el recorte de
  audio de cada retome y leyendo su transcripción, con la columna de la clase vacía.
  La original está en `hoja_retomes_calificada_autor.xlsx`; la propuesta del
  asistente de IA queda archivada en `hoja_retomes_A.csv` y no se usa.
- Validación con agente que se calla a tiempo conocido:
  `validacion_interrupciones.csv` (PoC1), `validacion_interrupciones_poc2.csv`
  (PoC2) y `interruption_campaign_v3/interruption_validation_poc3.csv` (PoC3,
  corrida con el arnés de tramos de 20 ms).
- Transporte de PoC2 justo antes y justo después de la campaña:
  `transporte_antes_v2/` y `transporte_despues_v2/`.
- Red de la validación de latencia (otro día): `generated/results.json`.
- `interruption_agreements.csv` (`validation.interruption_agreements`): qué dijo
  el transportista antes de que el agente registrara el resultado de la llamada.

**Pruebas de hipótesis.** Las hipótesis salieron del piloto y de la campaña del
5 de octubre; esta campaña es una muestra nueva, así que se prueban aquí. Una
familia fija de comparaciones (`COMPARACIONES`), cada una sobre un valor por
llamada (los dos eventos de una clase en una llamada no son independientes).
Las tres PoC se corrieron a la par, una llamada de cada una con el mismo
escenario y la misma repetición, así que todas las comparaciones son pareadas:
rangos con signo (Wilcoxon) sobre las parejas en que las dos llamadas tienen
valor, como la comparación PoC1–PoC2 de la latencia. Bonferroni sobre la familia
completa.

Las macros empiezan con `RInt` (campaña), `RVal` (validación) y `RIntPrueba`
(pruebas). La letra final es la PoC (A, B, C).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np

from shared.metrics.wer import compute_text, combine

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "latex" / "generated"

# PoC1 y PoC2 de la campaña del 5 de octubre y PoC3 repetida el 9 de octubre en
# μ-law, juntos en `campana_interrupciones_v3/` (mismos escenarios y repeticiones).
CAMPANA: Final[str] = "interruption_campaign_v3"
BACKGROUND: Final[dict[str, tuple[str, ...]]] = {
    "Piloto": ("interruptions_with_failed", "interruptions_poc3"),
    "Primera": ("interruption_campaign",),
}
VALIDATION: Final[dict[str, str]] = {
    "A": "interruption_validation.csv",
    "B": "interruption_validation_poc2.csv",
    "C": "interruption_campaign_v3/interruption_validation_poc3.csv",
}
TRANSPORT: Final[dict[str, str]] = {"Antes": "transport_before_v2", "Despues": "transport_after_v2"}

LETTER: Final[dict[str, str]] = {"1": "A", "2": "B", "3": "C"}
KIND: Final[dict[str, str]] = {"interrupcion": "Int", "apoyo": "Apo", "control": "Ctl"}
# C continuó, R repitió, N dijo algo nuevo, T tomó el apoyo como respuesta del
# transportista (por ejemplo, como aceptación), ? no se sabe.
RETOME: Final[dict[str, str]] = {"C": "C", "R": "R", "N": "N", "T": "T", "?": "Q"}
ALFA: Final[float] = 0.05


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def macro(name: str, valor: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{valor}}}"


def num(x: float | None, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    text = f"{abs(x):,.{dec}f}".replace(",", "{,}")
    return f"$-${text}" if x < 0 and round(abs(x), dec) != 0 else text


def valor_p(x: float | None) -> str:
    if x is None:
        return "---"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def leer(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def median_value(v: list[float]) -> float | None:
    return statistics.median(v) if v else None


def quartiles(v: list[float]) -> tuple[float | None, float | None]:
    if not v:
        return None, None
    q1, q3 = np.percentile(v, [25, 75])
    return float(q1), float(q3)


def floats(rows: list[dict[str, str]], col: str) -> list[float]:
    return [float(f[col]) for f in rows if f.get(col) not in ("", None)]


def fraction(a: int, b: int) -> str:
    return f"{a}/{b}"


# --------------------------------------------------------------------------
# Población
# --------------------------------------------------------------------------


def events(data: Path) -> list[dict[str, str]]:
    """Todos los eventos de la campaña, `ok` y `rojas`, sin los de la despedida."""
    return [f for f in leer(data / CAMPANA / "interruptions_events.csv") if f.get("final_turn") != "1"]


def measurable(rows: list[dict[str, str]], poc: str, tipo: str) -> list[dict[str, str]]:
    """Eventos disparados con el agente hablando en la referencia: los que tienen veredicto."""
    return [f for f in rows if f["poc"] == poc and f["tipo"] == tipo
            and f["fired"] == "1" and f["stopped"] != ""]


def they_resumed(halted_ones: list[dict[str, str]]) -> list[dict[str, str]]:
    return [f for f in halted_ones if f.get("endless_pause") == "0"]


def background(data: Path) -> list[str]:
    tex: list[str] = []
    for name, folders in BACKGROUND.items():
        rows = [f for c in folders if (r := data / c / "interruptions_events.csv").is_file() for f in leer(r)]
        for poc, letter in LETTER.items():
            tex.append(macro(f"RInt{name}Llamadas{letter}", len({f["clave"] for f in rows if f["poc"] == poc})))
        tex.append(macro(f"RInt{name}Llamadas", len({f["clave"] for f in rows})))
    return tex


# --------------------------------------------------------------------------
# Campaña
# --------------------------------------------------------------------------


def _poc_block(all_entries: list[dict[str, str]], rows: list[dict[str, str]], poc: str) -> tuple[list[str], dict]:
    letter = LETTER[poc]
    of_poc = [f for f in all_entries if f["poc"] == poc]
    states = {f["clave"]: f.get("estado", "ok") for f in of_poc}
    tex = [
        macro(f"RIntLlamadas{letter}", len(states)),
        macro(f"RIntRojas{letter}", sum(1 for e in states.values() if e == "roja")),
        macro(f"RIntDespedida{letter}", sum(1 for f in of_poc if f.get("final_turn") == "1")),
    ]
    res: dict[str, Any] = {"calls": len(states)}
    for tipo, c in KIND.items():
        med = measurable(rows, poc, tipo)
        det = [f for f in med if f["stopped"] == "1"]
        ret = they_resumed(det)
        pause = floats(ret, "pause_duration_ms")
        q1, q3 = quartiles(pause)
        tex += [
            macro(f"RInt{c}Frac{letter}", fraction(len(det), len(med))),
            macro(f"RInt{c}Retoma{letter}", fraction(len(ret), len(det)) if det else "---"),
            macro(f"RInt{c}NoVuelve{letter}", len(det) - len(ret)),
            macro(f"RInt{c}PauMed{letter}", num(median_value(pause))),
            macro(f"RInt{c}PauQa{letter}", num(q1)),
            macro(f"RInt{c}PauQb{letter}", num(q3)),
        ]
        res[tipo] = {"measurable": len(med), "halted_ones": len(det), "they_resumed": len(ret),
                     "pause_p50": median_value(pause)}
        if tipo == "interrupcion":
            t = floats(det, "stop_time_ms")
            r = floats(det, "reply_latency_ms")
            tq1, tq3 = quartiles(t)
            rq1, rq3 = quartiles(r)
            tex += [
                macro(f"RIntDetMed{letter}", num(median_value(t))),
                macro(f"RIntDetQa{letter}", num(tq1)),
                macro(f"RIntDetQb{letter}", num(tq3)),
                macro(f"RIntRepMed{letter}", num(median_value(r))),
                macro(f"RIntRepQa{letter}", num(rq1)),
                macro(f"RIntRepQb{letter}", num(rq3)),
                macro(f"RIntRepN{letter}", len(r)),
            ]
            res[tipo].update(stop_p50=median_value(t), replica_p50=median_value(r))
    return tex, res


def _stt_block(rows: list[dict[str, str]], poc: str) -> list[str]:
    """WER y CER de lo que el STT oyó del evento, sin la línea anterior pegada.

    Errores sumados sobre las palabras de todas las líneas de la clase.
    """
    letter = LETTER[poc]
    tex: list[str] = []
    for tipo in ("interrupcion", "apoyo"):
        c = KIND[tipo]
        fs = [f for f in rows if f["poc"] == poc and f["tipo"] == tipo and f["fired"] == "1"
              and f.get("text") and f.get("event_wer") not in ("", None)]
        hypothesis = [f.get("corrected_event_stt", f["event_stt"]) for f in fs]
        pairs = [compute_text(f["text"], h) for f, h in zip(fs, hypothesis)]
        tex += [
            macro(f"RInt{c}Wer{letter}", num(100 * combine(r.wer for r in pairs).sample_rate, 1) if pairs else "---"),
            macro(f"RInt{c}Cer{letter}", num(100 * combine(r.cer for r in pairs).sample_rate, 1) if pairs else "---"),
            macro(f"RInt{c}SinTexto{letter}", fraction(sum(1 for h in hypothesis if not h.strip()), len(fs))),
            macro(f"RInt{c}Pegada{letter}", sum(1 for f in fs if f.get("glued_line") == "1")),
        ]
    return tex


def recording_match(data: Path, rows: list[dict[str, str]]) -> list[str]:
    """En cuántos eventos el arnés y la grabación dicen lo mismo sobre "se calló"."""
    ruta = data / CAMPANA / "interruption_pauses.csv"
    if not ruta.is_file():
        return [macro("RIntCoincide", "---")]
    pauses = {(p["clave"], p["turn"]): p for p in leer(ruta)}
    judged = [f for f in rows if f["fired"] == "1" and f["stopped"] != ""
                and (f["clave"], f["turn"]) in pauses]
    equal_values = sum(1 for f in judged
                  if (f["stopped"] == "1") == (pauses[(f["clave"], f["turn"])]["pause_ms"] != ""))
    return [macro("RIntCoincide", fraction(equal_values, len(judged)))]


# --------------------------------------------------------------------------
# Qué hizo el agente al retomar
# --------------------------------------------------------------------------


def classification(data: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Los retomes clasificados, con su PoC y su llamada, sin los de la despedida."""
    folder = data / CAMPANA
    # La calificación del autor sobre la hoja con el antes y el después sacados
    # del audio (`validation.hoja_retomes_audio`).
    sheet = {r["id"]: r for r in leer(folder / "resumption_sheet_author.csv")}
    clave = {r["id"]: r for r in leer(folder / "resumption_sheet_key.csv")}
    alive_ones = {(f["clave"], f["turn"]) for f in rows}
    output = []
    for i, r in sheet.items():
        k = clave[i]
        if (k["clave"], k["turn"]) not in alive_ones:
            continue
        letter = r["classification"].strip().upper()
        if letter not in RETOME:
            raise SystemExit(f"retome {i} sin clasificar o con clave inválida: {letter!r}")
        output.append({"poc": k["poc"], "clave": k["clave"], "classification": letter})
    return output


def resumptions(classified: list[dict[str, str]]) -> list[str]:
    tex: list[str] = []
    for poc, letter in LETTER.items():
        of = [r for r in classified if r["poc"] == poc]
        tex.append(macro(f"RIntRetTotal{letter}", len(of)))
        for k, suffix in RETOME.items():
            tex.append(macro(f"RIntRet{suffix}{letter}", sum(1 for r in of if r["classification"] == k)))
    return tex


# --------------------------------------------------------------------------
# Pruebas de hipótesis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    name: str  # sufijo de las macros
    measured_value: str  # llave de `valores_por_llamada`
    a: str
    b: str
    measured_value_b: str | None = None  # si el lado b usa otra medida (apoyo contra control)


COMPARISONS: Final[tuple[Comparison, ...]] = (
    Comparison("CalloTresUno", "backchannel_went_silent", "3", "1"),
    Comparison("CalloTresDos", "backchannel_went_silent", "3", "2"),
    Comparison("PausaTresUno", "backchannel_pause", "3", "1"),
    Comparison("PausaTresDos", "backchannel_pause", "3", "2"),
    Comparison("PausaUnoDos", "backchannel_pause", "1", "2"),
    Comparison("ApoyoControlTres", "backchannel_went_silent", "3", "3", measured_value_b="control_dropped"),
    Comparison("SigueTresUno", "continuo", "3", "1"),
    Comparison("SigueTresDos", "continuo", "3", "2"),
)


def _per_call(rows: list[dict[str, str]], poc: str, tipo: str,
                 fn: Callable[[list[dict[str, str]]], float | None]) -> dict[str, float]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for f in measurable(rows, poc, tipo):
        groups[f["clave"]].append(f)
    return {k: v for k, g in groups.items() if (v := fn(g)) is not None}


def _went_quiet(g: list[dict[str, str]]) -> float:
    return sum(f["stopped"] == "1" for f in g) / len(g)


def _pause(g: list[dict[str, str]]) -> float | None:
    return median_value(floats(they_resumed([f for f in g if f["stopped"] == "1"]), "pause_duration_ms"))


def values_per_call(rows: list[dict[str, str]], classified: list[dict[str, str]],
                        measured_value: str, poc: str) -> dict[str, float]:
    """Un valor por llamada: la medida resumida sobre los eventos de esa llamada."""
    if measured_value == "backchannel_went_silent":
        return _per_call(rows, poc, "apoyo", _went_quiet)
    if measured_value == "control_dropped":
        return _per_call(rows, poc, "control", _went_quiet)
    if measured_value == "backchannel_pause":
        return _per_call(rows, poc, "apoyo", _pause)
    if measured_value == "continuo":
        # Fracción de retomes que continuaron la frase; los "no se sabe" no cuentan.
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in classified:
            if r["poc"] == poc and r["classification"] != "?":
                groups[r["clave"]].append(r["classification"] == "C")
        return {k: sum(v) / len(v) for k, v in groups.items()}
    raise ValueError(measured_value)


def tests(rows: list[dict[str, str]], classified: list[dict[str, str]]) -> tuple[list[str], dict]:
    from scipy import stats

    umbral = ALFA / len(COMPARISONS)
    tex = [macro("RIntPruebaFamilia", len(COMPARISONS)), macro("RIntPruebaAlfa", f"{umbral:.5f}".rstrip("0"))]
    res: dict[str, Any] = {}
    # La pareja de una llamada es la de las otras PoC con el mismo escenario y
    # la misma repetición (dentro de una PoC, la misma llamada).
    matched_pair = {f["clave"]: (f["scenario"], f["repetition"]) for f in rows}
    for c in COMPARISONS:
        a = {matched_pair[k]: v for k, v in values_per_call(rows, classified, c.measured_value, c.a).items()}
        b = {matched_pair[k]: v for k, v in values_per_call(rows, classified, c.measured_value_b or c.measured_value, c.b).items()}
        commons = sorted(set(a) & set(b))
        p = float(stats.wilcoxon([a[k] for k in commons], [b[k] for k in commons]).pvalue)
        n = str(len(commons))
        passes = p < umbral
        res[c.name] = {"p": p, "n": n, "passes": passes}
        tex += [macro(f"RIntPrueba{c.name}P", valor_p(p)), macro(f"RIntPrueba{c.name}N", n),
                macro(f"RIntPrueba{c.name}Pasa", "sí" if passes else "no")]
    return tex, res


def campana(data: Path) -> tuple[list[str], dict[str, Any]]:
    all_entries = leer(data / CAMPANA / "interruptions_events.csv")
    rows = events(data)
    tex: list[str] = [macro("RIntDespedida", sum(1 for f in all_entries if f.get("final_turn") == "1"))]
    res: dict[str, Any] = {}
    for poc, letter in LETTER.items():
        t, res[letter] = _poc_block(all_entries, rows, poc)
        tex += t + _stt_block(rows, poc)
    tex.append(macro("RIntLlamadas", len({f["clave"] for f in all_entries})))
    tex.append(macro("RIntPegadaOtras", sum(1 for f in rows if f["poc"] != "3" and f["fired"] == "1"
                                            and f.get("text") and f.get("glued_line") == "1")))
    tex += recording_match(data, rows)
    classified = classification(data, rows)
    tex += resumptions(classified)
    t, res["tests"] = tests(rows, classified)
    tex += t
    return tex, res


# --------------------------------------------------------------------------
# Validación y transporte
# --------------------------------------------------------------------------


def _validation(rows: list[dict[str, str]]) -> dict[str, Any]:
    inter = [f for f in rows if f["tipo"] == "interrupcion" and f["fired"] == "1"]
    others = [f for f in rows if f["tipo"] != "interrupcion" and f["fired"] == "1"]
    per_s: list[float] = []
    for s in sorted({f["target_stop_ms"] for f in inter}, key=float):
        e = floats([f for f in inter if f["target_stop_ms"] == s], "stop_error_ms")
        if e:
            per_s.append(statistics.median(e))

    def offset(col: str) -> float | None:
        return median_value(floats(inter, col)) if inter and col in inter[0] else None

    return {
        "detectadas": fraction(sum(1 for f in inter if f["harness_stopped"] == "1"), len(inter)),
        "false_alarms": fraction(sum(1 for f in others if f["harness_stopped"] == "1"), len(others)),
        "err_min": min(per_s) if per_s else None,
        "err_max": max(per_s) if per_s else None,
        "err_median": median_value(floats(inter, "stop_error_ms")),
        "inbound": offset("speech_onset_offset_ms"),
        "ultima": offset("last_speech_offset_ms"),
    }


def validation(data: Path) -> tuple[list[str], dict[str, Any]]:
    tex: list[str] = []
    res: dict[str, Any] = {}
    for letter, file in VALIDATION.items():
        ruta = data / file
        if not ruta.is_file():
            raise SystemExit(f"falta la validación {ruta}")
        v = res[letter] = _validation(leer(ruta))
        tex += [
            macro(f"RValIntDet{letter}", v["detectadas"]),
            macro(f"RValIntFalsas{letter}", v["false_alarms"]),
            macro(f"RValIntErrMin{letter}", num(v["err_min"], 0)),
            macro(f"RValIntErrMax{letter}", num(v["err_max"], 0)),
            macro(f"RValIntErrMed{letter}", num(v["err_median"], 0)),
            macro(f"RValIntIda{letter}", num(v["inbound"], 0)),
            macro(f"RValIntUltima{letter}", num(v["ultima"], 0)),
        ]
    return tex, res


def transport(data: Path, output: Path) -> tuple[list[str], float | None]:
    """Transporte de PoC2 justo antes y justo después de la campaña."""
    tex: list[str] = []
    errors: list[float] = []
    totals: list[float] = []
    for name, folder in TRANSPORT.items():
        ruta = data / folder / "interruption_validation_poc2.csv"
        if not ruta.is_file():
            raise SystemExit(f"falta el transporte {ruta}")
        v = _validation(leer(ruta))
        tex += [
            macro(f"RIntTrans{name}Ida", num(v["inbound"])),
            macro(f"RIntTrans{name}Vuelta", num(v["ultima"])),
            macro(f"RIntTrans{name}Total", num(v["inbound"] + v["ultima"])),
            macro(f"RIntTrans{name}Err", num(v["err_median"])),
        ]
        errors.append(v["err_median"])
        totals.append(v["inbound"] + v["ultima"])
    # La red de la validación de latencia, de otro día, para contrastar.
    ruta_json = output / "results.json"
    network = None
    latency: dict[str, Any] = {}
    if ruta_json.is_file():
        results = json.loads(ruta_json.read_text(encoding="utf-8"))
        network = ((results.get("instruments") or {}).get("validacionb") or {}).get("normal_network")
        latency = results.get("latency") or {}
    if network:
        tex.append(macro("RIntRedLatTotal", num(network["inbound"] + network["outbound"])))
    else:
        tex.append(macro("RIntRedLatTotal", "---"))
    # Lo propio de LiveKit: lo que su transporte pasa del de PoC1 (que no tiene
    # red) después de quitarle el viaje por la red. La red de la noche de la
    # campaña no se midió, así que se calcula con cada viaje medido otro día y
    # se publica el rango; por debajo de cero no queda nada propio.
    l1, l2 = latency.get("1") or {}, latency.get("2") or {}
    parts = [l1.get("inbound_p50"), l1.get("outbound_p50"), l2.get("inbound_p50"), l2.get("outbound_p50")]
    networks = ([network["inbound"] + network["outbound"]] if network else []) + list(totals)
    if None not in parts and networks:
        excess = parts[2] + parts[3] - parts[0] - parts[1]
        own_ones = [excess - r for r in networks]
        tex += [
            macro("RPropioLkMax", num(max(0.0, max(own_ones)))),
            macro("RPropioLkMin", num(max(0.0, min(own_ones)))),
        ]
    else:
        tex += [macro("RPropioLkMax", "---"), macro("RPropioLkMin", "---")]
    err = statistics.mean(errors)
    tex.append(macro("RIntTransErrProm", num(err)))
    return tex, err


# --------------------------------------------------------------------------


AGREEMENT_KINDS: dict[str, str] = {
    "acepto": "Acepto", "apoyo": "Apoyo", "linea_incompleta": "Incompleta",
    "anticipado": "Anticipado", "sin_acuerdo": "Sin",
}


def agreements(data: Path) -> list[str]:
    """Qué precedió al registro del resultado (`validation.interruption_agreements`).

    `RIntAcu<Clase><L>` cuenta llamadas por clase; `RIntAcuRegistrados<L>`, las
    que cerraron con acuerdo, y `RIntAcuNoAceptados<L>`, las que lo registraron
    sin que el transportista lo aceptara (apoyo, línea incompleta o anticipado).
    """
    ruta = data / CAMPANA / "interruption_agreements.csv"
    if not ruta.is_file():
        return []
    rows = leer(ruta)
    tex: list[str] = []
    for poc, L in LETTER.items():
        of = [f for f in rows if f["poc"] == poc]
        for kind, name in AGREEMENT_KINDS.items():
            tex.append(macro(f"RIntAcu{name}{L}", sum(f["kind"] == kind for f in of)))
        tex.append(macro(f"RIntAcuRegistrados{L}", sum(f["kind"] != "sin_acuerdo" for f in of)))
        tex.append(macro(f"RIntAcuNoAceptados{L}",
                         sum(f["kind"] in ("apoyo", "linea_incompleta", "anticipado") for f in of)))
    return tex


def pauses_and_tool(data: Path) -> list[str]:
    """PoC3: cuántas de sus pausas coinciden con una consulta a la herramienta.

    `validation.pauses_and_tool`. `RIntApoHerrC`: veces que se calló ante un
    apoyo con una consulta cerca; `RIntApoSinHerrC`: apoyos sin consulta cerca, y
    `RIntApoSinHerrCalloC`, en cuántos de esos se calló. Igual con `Ctl`.
    """
    ruta = data / CAMPANA / "pauses_and_tool.csv"
    if not ruta.is_file():
        return []
    rows = leer(ruta)
    tex: list[str] = []
    for tipo, name in (("apoyo", "Apo"), ("control", "Ctl")):
        of = [f for f in rows if f["tipo"] == tipo]
        sin = [f for f in of if f["query_nearby"] == "0"]
        tex += [
            macro(f"RInt{name}HerrC", sum(1 for f in of if f["stopped"] == "1" and f["query_nearby"] == "1")),
            macro(f"RInt{name}SinHerrC", len(sin)),
            macro(f"RInt{name}SinHerrCalloC", sum(1 for f in sin if f["stopped"] == "1")),
        ]
    return tex


def generate(data: Path, output: Path) -> tuple[list[str], dict[str, Any]]:
    tex = ["% GENERADO por shared/metrics/interruptions_report.py — no editar a mano."]
    tex += background(data)
    t, res_c = campana(data)
    tex += t
    t, res_v = validation(data)
    tex += t
    t, err_b = transport(data, output)
    tex += t
    # Tiempo de detención sin lo que agrega el camino de cada PoC: el transporte
    # de PoC2 (promedio de antes y después de la campaña) y el puente de PoC3
    # (su validación con el arnés de tramos de 20 ms). PoC1 no agrega nada medible.
    det_b = res_c["B"]["interrupcion"]["stop_p50"]
    det_c = res_c["C"]["interrupcion"]["stop_p50"]
    err_c = res_v["C"]["err_median"]
    tex.append(macro("RIntDetSinTransB", num(det_b - err_b) if det_b is not None else "---"))
    tex.append(macro("RIntDetSinPuenteC", num(det_c - err_c) if det_c is not None and err_c is not None else "---"))
    tex += agreements(data)
    tex += pauses_and_tool(data)
    return tex, {"campana": res_c, "validacion": res_v, "transport_error_b": err_b}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args(argv)
    tex, res = generate(args.data, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "interruptions.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"{len(tex) - 1} macros en {args.output / 'interruptions.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
