"""Las cifras y figuras del reporte, calculadas desde `docs/reporte/datos/`.

Es el único camino de los números al documento: lee los CSV que escriben
`exportar_para_reporte` y `exportar_wer`, y genera
`docs/reporte/latex/generado/resultados.tex`, un archivo de macros (`\\R...`) y
de cajas de boxplot que `main.tex` incluye con `\\input`. Ningún número de la
sección de resultados se teclea a mano.

## Población

Solo las llamadas en estado `ok` (`llamadas.csv`), en las tres PoC por igual:
84 pares de PoC1 y PoC2, y 82 de las 90 llamadas de PoC3. Las 8 restantes de
PoC3 no pasaron las revisiones de confiabilidad de la campaña, igual que las
llamadas de PoC1 y PoC2 que quedaron fuera de los 84 pares.

## Estadística

- **Resumen:** mediana (p50) y percentil 95 (p95) de la latencia por
  intervención.
- **Incertidumbre:** intervalo de confianza del 95 % (IC95) por bootstrap: se
  sortean llamadas completas con reemplazo, 10,000 veces (semilla 20260923), y
  el IC95 va del percentil 2.5 al 97.5 de las 10,000 medianas. Se sortean
  llamadas y no respuestas porque las respuestas de una misma llamada no son
  independientes. PoC1 contra PoC2 se sortea por par (mismo escenario y
  repetición), porque la campaña los corrió alternados.
- **Relevancia práctica:** una diferencia se compara contra un margen de 200 ms.
- **Pruebas de hipótesis:** Kruskal-Wallis entre las tres PoC y Wilcoxon por
  pares, con corrección de Bonferroni, sobre la mediana de cada llamada. Como
  medida del tamaño del efecto se publica la probabilidad de que una llamada de
  una PoC sea más rápida que una de la otra.
- **Robustez:** cuánto cambia la mediana si se quita una llamada cualquiera, y
  sin el 1 % de respuestas más lentas.

## Transporte de salida

La marca de salida de cada PoC es una por respuesta: el primer audio que suelta.
El exportador la resta del inicio de la respuesta con contenido, y cuando la
respuesta empieza con un sonido corto (un «claro», un relleno) y sigue tras una
pausa, esa resta se lleva el sonido y la pausa, que no son red. Aquí se mide
contra el primer sonido que oyó el arnés (`vuelta_al_primer_sonido`), y la
latencia sin transporte se recalcula con esa salida. El CSV queda como lo
escribió el exportador.

Uso, desde la raíz del repo, con el entorno fijo del análisis (`analisis/`):

    uv run --project analisis python -m shared.metrics.analisis_reporte
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "latex" / "generated"

REPLICAS: Final[int] = 10_000
SEED: Final[int] = 20260923
MARGIN_MS: Final[float] = 200.0
POCS: Final[tuple[str, ...]] = ("1", "2", "3")
LETTER: Final[dict[str, str]] = {"1": "A", "2": "B", "3": "C"}  # para nombres de macro
PAIR_NAME: Final[dict[str, str]] = {"12": "UnoDos", "13": "UnoTres", "23": "DosTres"}
ROMANOS: Final[dict[int, str]] = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

# El TTFT de PoC1 es bimodal: sin herramienta ~670 ms, con herramienta ~1,700 ms
# (la función y la segunda petición al modelo quedan dentro). El valle está en
# 1,000-1,250 ms (`docs/CONTRATO_METRICAS.md` §2.7.1). El de PoC2 se detiene en
# el primer token aunque sea una llamada a función, así que nunca lo contiene.
TTFT_VALLEY_POC1_MS: Final[float] = 1_100.0

# La espera de fin de turno de PoC2 tiene dos caminos: ~577 ms cuando el
# detector acústico pasa su umbral y ~2,500 ms cuando no se convence y agota la
# espera máxima (`docs/comparativo_resultados.md`, hallazgo de PoC2). No hay
# turnos entre 1,500 y 2,250 ms, así que 1,500 separa los dos modos.
END_OF_TURN_VALLEY_MS: Final[float] = 1_500.0

# Un sonido del agente que llega más de esto antes de su respuesta con contenido
# es un «sonido previo» (un «claro», un relleno). Es el umbral de la tabla del
# primer sonido.
PRIOR_SOUND_MS: Final[float] = 60.0

# Holgura al decidir si la marca de salida es la del primer sonido: dos tramos
# de 20 ms, la resolución con que el arnés fecha la voz.
MARK_TOLERANCE_MS: Final[float] = 40.0

# Lo que se escribe en la tabla mientras falten los datos de un bloque.
PENDING: Final[str] = "\\textit{pend.}"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    poc: str
    clave: str
    scenario: int
    repetition: int
    observed_value: float
    first_sound: float | None
    inbound: float | None
    outbound: float | None
    corrected: float | None
    with_rt: int | None
    tool_result: int | None = None
    spoke_before_result: int | None = None
    ms_speaking_without_result: float | None = None
    # `en_vivo`, `instrumento_externo` o `recuperado` (ver el exportador).
    origin: str | None = None
    # La salida tal como viene en el CSV, y qué se hizo con ella
    # (`vuelta_al_primer_sonido`).
    raw_outbound: float | None = None
    outbound_state: str | None = None
    # Algún evento de herramienta en toda la respuesta (fin de voz -> línea siguiente).
    tool_in_response: int | None = None
    # PoC1 y PoC2: la misma respuesta medida sobre la grabación (ver el exportador).
    reconstructed: float | None = None
    clip_id: str | None = None
    turn: str | None = None


def outbound_to_first_sound(
    observed_value: float, first_sound: float | None, outbound: float | None, origin: str | None
) -> tuple[float | None, str]:
    """El transporte de salida medido contra el primer sonido que oyó el arnés.

    `vuelta` va de la marca de salida del framework al inicio de la respuesta
    con contenido. Si antes sonó algo (`hueco` > `PRIOR_SOUND_MS`), la marca es
    la de ese primer sonido, y la vuelta trae el sonido y la pausa: se descuenta
    el hueco. Si al descontarlo la marca queda después del primer sonido (más
    allá de la tolerancia), no es la de ningún sonido de esta respuesta y la
    salida queda sin dato. Los turnos `recuperado` ya se midieron contra el
    primer sonido.

    Devuelve la salida y su estado: `directa`, `corregida`, `sin_marca` o
    `sin_dato`.
    """
    if outbound is None:
        return None, "no_value"
    if origin == "recuperado" or first_sound is None:
        return outbound, "directa"
    hole = observed_value - first_sound
    if hole <= PRIOR_SOUND_MS:
        return outbound, "directa"
    rest = outbound - hole
    if rest >= -MARK_TOLERANCE_MS:
        return round(rest, 1), "corrected"
    return None, "no_mark"


def _num(text: str | None) -> float | None:
    return float(text) if text not in ("", None) else None


def _int(text: str | None) -> int | None:
    v = _num(text)
    return None if v is None else int(v)


def calls_ok(ruta: Path) -> set[str]:
    with ruta.open(encoding="utf-8") as f:
        return {
            r["clave"] for r in csv.DictReader(f)
            if r.get("estado") == "ok" and not r.get("reason_if_omitted")
        }


def read_lines(ruta: Path, valid_ones: set[str] | None = None) -> list[Line]:
    output = []
    with ruta.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if valid_ones is not None and r["clave"] not in valid_ones:
                continue
            observed_value = float(r["observed_latency_ms"])
            primer = _num(r.get("first_sound_latency_ms"))
            inbound = _num(r["ida_ms"])
            raw = _num(r["outbound_ms"])
            origin = r.get("origin") or None
            outbound, estado = outbound_to_first_sound(observed_value, primer, raw, origin)
            output.append(Line(
                poc=r["poc"], clave=r["clave"],
                scenario=int(r["scenario"]), repetition=int(r["repetition"]),
                observed_value=observed_value,
                first_sound=primer,
                inbound=inbound, outbound=outbound,
                # Con la salida contra el primer sonido; el redondeo es el del exportador.
                corrected=(
                    round(observed_value - inbound - outbound, 1)
                    if inbound is not None and outbound is not None else None
                ),
                with_rt=_int(r["with_round_trip"]),
                tool_result=_int(r.get("tool_result")),
                spoke_before_result=_int(r.get("spoke_before_result")),
                ms_speaking_without_result=_num(r.get("ms_speaking_without_result")),
                origin=origin,
                raw_outbound=raw,
                outbound_state=estado,
                tool_in_response=_int(r.get("tool_in_response")),
                reconstructed=_num(r.get("reconstructed_latency_ms")),
                clip_id=r.get("clip_id") or None,
                turn=r.get("turn") or None,
            ))
    return output


def per_call(rows: Sequence[Line], valor) -> list[np.ndarray]:
    """Los valores de cada llamada, en arreglos, descartando los `None`."""
    groups: dict[str, list[float]] = defaultdict(list)
    for f in rows:
        v = valor(f)
        if v is not None:
            groups[f.clave].append(v)
    return [np.array(v) for v in groups.values() if v]


def pairs_per_scenario(
    rows: Sequence[Line], valor
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Las llamadas de PoC1 y PoC2 emparejadas por (escenario, repetición)."""
    cells: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in rows:
        if f.poc in ("1", "2"):
            v = valor(f)
            if v is not None:
                cells[(f.scenario, f.repetition)][f.poc].append(v)
    return [
        (np.array(c["1"]), np.array(c["2"]))
        for c in cells.values() if c.get("1") and c.get("2")
    ]


# --------------------------------------------------------------------------
# Bootstrap por llamada
# --------------------------------------------------------------------------


def _resample(rng: np.random.Generator, groups: Sequence[np.ndarray]) -> np.ndarray:
    """Llamadas completas, sorteadas con reemplazo."""
    idx = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in idx])


def ci_median(groups: Sequence[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    reps = np.array([np.median(_resample(rng, groups)) for _ in range(REPLICAS)])
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def ci_difference(a, b, rng) -> tuple[float, float]:
    reps = np.array([
        np.median(_resample(rng, a)) - np.median(_resample(rng, b))
        for _ in range(REPLICAS)
    ])
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def ci_paired_difference(pairs: Sequence[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, float]:
    """Se sortea el PAR entero (las dos llamadas juntas)."""
    reps = np.empty(REPLICAS)
    n = len(pairs)
    for k in range(REPLICAS):
        idx = rng.integers(0, n, n)
        reps[k] = (
            np.median(np.concatenate([pairs[i][0] for i in idx]))
            - np.median(np.concatenate([pairs[i][1] for i in idx]))
        )
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def obs_and_outputs_per_call(
    rows: Sequence[Line],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Por llamada, todas sus latencias observadas y las salidas que se midieron.

    Se queda con las llamadas que tienen al menos una salida medida.
    """
    obs: dict[str, list[float]] = defaultdict(list)
    outputs: dict[str, list[float]] = defaultdict(list)
    for f in rows:
        obs[f.clave].append(f.observed_value)
        if f.outbound is not None:
            outputs[f.clave].append(f.outbound)
    return [(np.array(obs[c]), np.array(outputs[c])) for c in obs if outputs[c]]


def no_output(calls: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mediana de la latencia observada menos mediana del transporte de salida.

    Las dos medianas son de la PoC entera: todas sus respuestas y todas sus
    salidas medidas, igual que en la tabla de latencia.
    """
    obs = np.concatenate([o for o, _ in calls])
    outputs = np.concatenate([s for _, s in calls])
    return float(np.median(obs) - np.median(outputs))


def ci_difference_no_exit(a, b, rng) -> tuple[float, float]:
    """Se sortean llamadas enteras, con sus respuestas y sus salidas juntas."""
    def resample(calls):
        return [calls[i] for i in rng.integers(0, len(calls), len(calls))]

    reps = np.array([
        no_output(resample(a)) - no_output(resample(b))
        for _ in range(REPLICAS)
    ])
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def verdict(low: float, alto: float) -> str:
    """Lectura del IC95 de una diferencia contra el margen de 200 ms.

    Si el intervalo incluye el cero pero cabe completo dentro de ±200 ms, la
    diferencia puede ser cero y en ningún caso llega al margen: las dos PoC son
    equivalentes en la práctica.
    """
    if low <= 0 <= alto:
        if -MARGIN_MS <= low and alto <= MARGIN_MS:
            return "equivalentes en la práctica"
        return "no se distingue"
    if -MARGIN_MS <= low and alto <= MARGIN_MS:
        return "distinta de cero, menor que 200 ms"
    if alto < -MARGIN_MS or low > MARGIN_MS:
        return "distinta de cero y mayor que 200 ms"
    return "distinta de cero; no se sabe si pasa de 200 ms"


# --------------------------------------------------------------------------
# Boxplots (pgfplots `boxplot prepared`)
# --------------------------------------------------------------------------


def box(values: Sequence[float]) -> dict[str, float]:
    v = np.sort(np.asarray(values, dtype=float))
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    span = q3 - q1
    inside = v[(v >= q1 - 1.5 * span) & (v <= q3 + 1.5 * span)]
    return {
        "lower whisker": float(inside.min()), "lower quartile": float(q1),
        "median": float(med), "upper quartile": float(q3),
        "upper whisker": float(inside.max()), "n": int(v.size),
    }


def box_tex(c: dict[str, float]) -> str:
    return (
        f"lower whisker={c['lower whisker']:.1f}, lower quartile={c['lower quartile']:.1f}, "
        f"median={c['median']:.1f}, upper quartile={c['upper quartile']:.1f}, "
        f"upper whisker={c['upper whisker']:.1f}"
    )


# --------------------------------------------------------------------------
# Formato
# --------------------------------------------------------------------------


def ms(x: float | None, decimals: int = 0) -> str:
    """Número con coma de millares, para LaTeX (`2{,}609`)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    text = f"{x:,.{decimals}f}".replace(",", "{,}")
    # Signo menos tipográfico: el guion de texto se ve como un guion, no como un menos.
    return "$-$" + text[1:] if text.startswith("-") else text


def pct(x: float, decimals: int = 1) -> str:
    return f"{100 * x:.{decimals}f}"


def valor_p(x: float) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def p_value_text(x: float) -> str:
    """El valor p con su relación, para usarlo dentro de una frase."""
    return "$p<0.001$" if x < 0.001 else f"$p={x:.3f}$"


def macro(name: str, valor: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{valor}}}"


def high_percentiles(prefix: str, values: Sequence[float]) -> list[str]:
    """El p75 y el p95 de una etapa, para acompañar a su mediana en la tabla por capa."""
    if not len(values):
        return [macro(f"{prefix}Qtres", "---"), macro(f"{prefix}Noventaycinco", "---")]
    q3, p95 = np.percentile(values, [75, 95])
    return [macro(f"{prefix}Qtres", ms(float(q3), 0)),
            macro(f"{prefix}Noventaycinco", ms(float(p95), 0))]


def box_macro(name: str, values: Sequence[float]) -> str:
    """`\\Nombre{estilo}` dibuja la caja dentro de un `axis` de pgfplots.

    Los valores fuera de los bigotes van como coordenadas: pgfplots los dibuja como
    puntos sueltos si el estilo les da marca, y no se ven con `mark=none`.
    """
    c = box(values)
    outside = " ".join(f"(0,{v:.1f})" for v in sorted(float(x) for x in values)
                     if v < c["lower whisker"] or v > c["upper whisker"])
    return (
        f"\\newcommand{{\\{name}}}[1]{{\\addplot+[#1, boxplot prepared={{{box_tex(c)}}}] "
        f"coordinates {{{outside}}};}}"
    )


# --------------------------------------------------------------------------
# Latencia
# --------------------------------------------------------------------------


def tools(
    of: dict[str, list[Line]], with_tool, rng, suffix: str,
) -> tuple[dict, list[str]]:
    """Latencia con y sin herramienta por PoC, con la partición `with_tool`."""
    tex: list[str] = []
    summary: dict = {}
    for p in POCS:
        L = suffix + LETTER[p]
        with_f = [f for f in of[p] if with_tool(f) == 1]
        without_f = [f for f in of[p] if with_tool(f) == 0]
        having = [f.observed_value for f in with_f]
        sin = [f.observed_value for f in without_f]
        g_with = per_call(with_f, lambda f: f.observed_value)
        g_without = per_call(without_f, lambda f: f.observed_value)
        lo, hi = ci_difference(g_with, g_without, rng)
        slo, shi = ci_median(g_without, rng)
        row = {
            "fraction": len(having) / (len(having) + len(sin)),
            "with_p50": float(np.median(having)), "no_p50": float(np.median(sin)),
            "no_ci": [slo, shi],
            "no_ric": float(np.subtract(*np.percentile(sin, [75, 25]))),
            "delta": float(np.median(having) - np.median(sin)), "delta_ci": [lo, hi],
            "n_with": len(having), "n_without": len(sin),
        }
        summary[p] = row
        tex += [
            macro(f"RRtFrac{L}", pct(row["fraction"])),
            macro(f"RRtNCon{L}", str(len(having))), macro(f"RRtNSin{L}", str(len(sin))),
            macro(f"RRtCon{L}", ms(row["with_p50"], 1)), macro(f"RRtSin{L}", ms(row["no_p50"], 1)),
            macro(f"RRtSinLo{L}", ms(slo, 1)), macro(f"RRtSinHi{L}", ms(shi, 1)),
            macro(f"RRtSinRic{L}", ms(row["no_ric"], 0)),
            macro(f"RRtDelta{L}", ms(row["delta"], 1)),
            macro(f"RRtDeltaEnt{L}", ms(row["delta"], 0)),
            macro(f"RRtDeltaLo{L}", ms(lo, 1)), macro(f"RRtDeltaHi{L}", ms(hi, 1)),
            box_macro(f"RCajaRtCon{L}", having),
            box_macro(f"RCajaRtSin{L}", sin),
        ]
    deltas = [summary[p]["delta"] for p in POCS]
    tex += [macro(f"RRtDeltaMin{suffix}", ms(min(deltas) / 1000, 1)),
            macro(f"RRtDeltaMax{suffix}", ms(max(deltas) / 1000, 1))]
    return summary, tex


def analyze_latency(rows: list[Line], rng) -> tuple[list[str], dict]:
    tex: list[str] = []
    summary: dict = {"latency": {}, "differences": {}, "round_trip": {}, "scenarios": {},
                     "first_sound": {}}
    of = {p: [f for f in rows if f.poc == p] for p in POCS}

    # --- Las tres cifras: observada, transporte, corregida ---
    for p in POCS:
        L = LETTER[p]
        obs = np.array([f.observed_value for f in of[p]])
        lo, hi = ci_median(per_call(of[p], lambda f: f.observed_value), rng)
        inbound_legs = [f.inbound for f in of[p] if f.inbound is not None]
        outbounds = [f.outbound for f in of[p] if f.outbound is not None]
        corr = [f.corrected for f in of[p] if f.corrected is not None]
        q1, q3 = np.percentile(obs, [25, 75])
        row = {
            "calls": len({f.clave for f in of[p]}), "n": int(obs.size),
            "p50": float(np.median(obs)), "ci": [lo, hi], "p95": float(np.percentile(obs, 95)),
            "q1": float(q1), "q3": float(q3),
            "inbound_p50": float(np.median(inbound_legs)) if inbound_legs else None,
            "outbound_p50": float(np.median(outbounds)) if outbounds else None,
            "outbound_ric": float(np.subtract(*np.percentile(outbounds, [75, 25]))) if outbounds else None,
            "corrected_p50": float(np.median(corr)) if corr else None,
        }
        tex += [
            macro(f"RLlamadas{L}", str(row["calls"])),
            macro(f"RN{L}", ms(row["n"])),
            macro(f"RObsMed{L}", ms(row["p50"], 1)),
            macro(f"RObsLo{L}", ms(lo, 1)), macro(f"RObsHi{L}", ms(hi, 1)),
            # Cuánto se aleja del p50 el extremo más lejano de su IC95.
            macro(f"RObsMargen{L}", ms(max(row["p50"] - lo, hi - row["p50"]), 0)),
            macro(f"RObsNoventaycinco{L}", ms(row["p95"], 1)),
            macro(f"RObsQuno{L}", ms(q1, 0)), macro(f"RObsQtres{L}", ms(q3, 0)),
            macro(f"RObsRic{L}", ms(q3 - q1, 0)),
            macro(f"RIda{L}", ms(row["inbound_p50"], 1)),
            macro(f"RVuelta{L}", ms(row["outbound_p50"], 1)),
            macro(f"RVueltaRic{L}", ms(row["outbound_ric"], 1)),
            *high_percentiles(f"RIda{L}", inbound_legs), *high_percentiles(f"RVuelta{L}", outbounds),
            box_macro(f"RCajaObs{L}", obs),
            # Enteros, para la tabla por capa.
            macro(f"RObsMedEnt{L}", ms(row["p50"], 0)),
            macro(f"RObsNoventaycincoEnt{L}", ms(row["p95"], 0)),
            macro(f"RIdaEnt{L}", ms(row["inbound_p50"], 0)),
            macro(f"RVueltaEnt{L}", ms(row["outbound_p50"], 0)),
            # Cuántas salidas se midieron contra el primer sonido y cuántas quedaron sin dato.
            macro(f"RVueltaCorregidas{L}", str(sum(f.outbound_state == "corrected" for f in of[p]))),
            macro(f"RVueltaSinMarca{L}", str(sum(f.outbound_state == "no_mark" for f in of[p]))),
        ]
        if inbound_legs and outbounds:
            tex.append(macro(f"RTransporte{L}", ms(row["inbound_p50"] + row["outbound_p50"], 0)))
        if corr:
            clo, chi = ci_median(per_call(of[p], lambda f: f.corrected), rng)
            row["corrected_ci"] = [clo, chi]
            row["corrected_p95"] = float(np.percentile(corr, 95))
            # Lo que el framework ve de sí mismo se queda corto por esto.
            underestimates = row["p50"] - row["corrected_p50"]
            tex += [
                macro(f"RCorrMed{L}", ms(row["corrected_p50"], 1)),
                macro(f"RCorrLo{L}", ms(clo, 1)), macro(f"RCorrHi{L}", ms(chi, 1)),
                macro(f"RCorrNoventaycinco{L}", ms(row["corrected_p95"], 1)),
                macro(f"RSubestima{L}", ms(underestimates, 0)),
                macro(f"RSubestimaPct{L}", pct(underestimates / row["p50"], 0)),
                box_macro(f"RCajaCorr{L}", corr),
            ]
        summary["latency"][p] = row

    # --- Diferencias entre PoC, con su IC95 y la lectura contra 200 ms ---
    g = {p: per_call(of[p], lambda f: f.observed_value) for p in POCS}
    pairs = pairs_per_scenario(rows, lambda f: f.observed_value)
    intervals = {
        "12": ci_paired_difference(pairs, rng),
        "13": ci_difference(g["1"], g["3"], rng),
        "23": ci_difference(g["2"], g["3"], rng),
    }
    lat = summary["latency"]
    points = {"12": lat["1"]["p50"] - lat["2"]["p50"], "13": lat["1"]["p50"] - lat["3"]["p50"],
              "23": lat["2"]["p50"] - lat["3"]["p50"]}
    for k, (lo, hi) in intervals.items():
        n = PAIR_NAME[k]
        summary["differences"][k] = {"difference": points[k], "ci": [lo, hi], "verdict": verdict(lo, hi)}
        tex += [
            macro(f"RDif{n}", ms(points[k], 1)),
            # Mismo redondeo que la tabla de diferencias, para que el texto y la
            # tabla den la misma cifra.
            macro(f"RDifAbs{n}", ms(abs(points[k]), 1)),
            macro(f"RDifLo{n}", ms(lo, 1)), macro(f"RDifHi{n}", ms(hi, 1)),
            macro(f"RDifVeredicto{n}", verdict(lo, hi)),
        ]
    tex.append(macro("RPares", str(len(pairs))))
    tex.append(macro("RNTotal", ms(sum(len(of[p]) for p in POCS))))
    # Cuántas llamadas por escenario quedaron en la población de análisis.
    scenario_pairs = defaultdict(int)
    for f in {(f.scenario, f.repetition) for f in of["1"]} & {(f.scenario, f.repetition) for f in of["2"]}:
        scenario_pairs[f[0]] += 1
    scenario_calls = defaultdict(set)
    for f in of["3"]:
        scenario_calls[f.scenario].add(f.clave)
    tex += [
        macro("RParesEscMin", str(min(scenario_pairs.values()))),
        macro("RParesEscMax", str(max(scenario_pairs.values()))),
        macro("RLlamadasEscMinC", str(min(len(v) for v in scenario_calls.values()))),
        macro("RLlamadasEscMaxC", str(max(len(v) for v in scenario_calls.values()))),
    ]

    # Sin transporte: PoC1 contra PoC2 (las dos cifras completas) y PoC1
    # contra PoC3 descontándole a PoC3 solo lo que se pudo medir (la salida).
    pairs_c = pairs_per_scenario(rows, lambda f: f.corrected)
    if pairs_c:
        lo, hi = ci_paired_difference(pairs_c, rng)
        punto = lat["1"]["corrected_p50"] - lat["2"]["corrected_p50"]
        summary["differences"]["12_corregida"] = {"difference": punto, "ci": [lo, hi], "verdict": verdict(lo, hi)}
        tex += [
            macro("RDifCorrUnoDos", ms(punto, 1)),
            macro("RDifCorrLoUnoDos", ms(lo, 1)), macro("RDifCorrHiUnoDos", ms(hi, 1)),
            macro("RDifCorrVeredictoUnoDos", verdict(lo, hi)),
        ]
    # PoC1 contra PoC3 sin el camino de salida. A PoC3 no se le puede medir el
    # de entrada, así que solo se quita la salida, y a las dos igual: a la
    # mediana de todas sus respuestas se le resta la mediana de su salida, como
    # dice la nota de la tabla de diferencias.
    g1 = obs_and_outputs_per_call(of["1"])
    g3 = obs_and_outputs_per_call(of["3"])
    if g1 and g3:
        lo, hi = ci_difference_no_exit(g1, g3, rng)
        punto = no_output(g1) - no_output(g3)
        summary["differences"]["13_sin_salida"] = {"difference": punto, "ci": [lo, hi],
                                                   "verdict": verdict(lo, hi)}
        tex += [
            macro("RDifSinSalidaUnoTres", ms(punto, 1)),
            macro("RDifSinSalidaLoUnoTres", ms(lo, 1)), macro("RDifSinSalidaHiUnoTres", ms(hi, 1)),
            macro("RDifSinSalidaAbsUnoTres", ms(abs(punto), 0)),
            macro("RDifSinSalidaVeredictoUnoTres", verdict(lo, hi)),
        ]
        for p, g in (("1", g1), ("3", g3)):
            tex.append(macro(f"RSinSalida{LETTER[p]}", ms(no_output(g), 0)))
    # Cuánto le ahorra a PoC1 correr en la misma máquina, contra lo medido en PoC3.
    if lat["1"]["outbound_p50"] is not None and lat["3"]["outbound_p50"] is not None:
        tex.append(macro("RVentajaVueltaUnoTres", ms(lat["3"]["outbound_p50"] - lat["1"]["outbound_p50"], 0)))

    # --- Llamada a herramientas ---
    # La definición publicada: hubo herramienta en TODA la respuesta (del fin de
    # voz a la línea siguiente). La anterior solo miraba hasta la primera voz
    # del agente, y en PoC3 dejaba como "sin herramienta" las respuestas en que
    # GPT-Live llamó a la herramienta después de empezar a hablar. La anterior
    # se sigue calculando, con el mismo generador y en el mismo orden, para no
    # correr los intervalos que vienen después; la nueva usa uno propio.
    previous, t_prev = tools(of, lambda f: f.with_rt, rng, "Prev")
    summary["previous_round_trip"] = previous
    summary["round_trip"], t_rt = tools(
        of, lambda f: f.tool_in_response, np.random.default_rng(SEED + 3), "",
    )
    tex += t_prev + t_rt

    # En PoC3: ¿el resultado llegó cuando el agente ya hablaba?
    with_result = [f for f in of["3"] if f.tool_result == 1]
    if with_result:
        spoke = [f for f in with_result if f.spoke_before_result == 1]
        mss = [f.ms_speaking_without_result for f in spoke if f.ms_speaking_without_result is not None]
        summary["tool_poc3"] = {"with_result": len(with_result), "spoke_before": len(spoke),
                                      "ms_p50": float(np.median(mss)) if mss else None}
        tex += [
            macro("RHayHerr", "1"),
            macro("RHerrN", str(len(with_result))), macro("RHerrHablo", str(len(spoke))),
            macro("RHerrFrac", pct(len(spoke) / len(with_result), 0)),
            macro("RHerrMs", ms(float(np.median(mss)), 0) if mss else "---"),
        ]
    else:
        tex += [macro("RHayHerr", "0")] + [
            macro(f"RHerr{campo}", PENDING) for campo in ("N", "Hablo", "Frac", "Ms")
        ]

    # Cota: en las respuestas que arrancaron antes de tener el resultado de la
    # herramienta, la racha que el arnés tomó como respuesta no podía traer ese
    # dato. Se mide hasta que llegó el resultado, que es lo más temprano que pudo
    # llegar el contenido que lo usa. La misma regla en las tres PoC: en PoC1 y
    # PoC2 la función corre en el proceso del agente y su hora es la del
    # resultado. Generador aparte, para no correr los intervalos de después.
    def until_result(f: Line) -> float:
        if f.spoke_before_result == 1 and f.ms_speaking_without_result is not None:
            return f.observed_value + f.ms_speaking_without_result
        return f.observed_value

    rng_hr = np.random.default_rng(SEED + 2)
    g_hr = {p: per_call(of[p], until_result) for p in POCS}
    med_hr = {p: float(np.median(np.concatenate(g_hr[p]))) for p in POCS}
    intervals_hr = {
        "12": ci_paired_difference(pairs_per_scenario(rows, until_result), rng_hr),
        "13": ci_difference(g_hr["1"], g_hr["3"], rng_hr),
        "23": ci_difference(g_hr["2"], g_hr["3"], rng_hr),
    }
    for p in POCS:
        tex += [
            macro(f"RHastaResMed{LETTER[p]}", ms(med_hr[p], 1)),
            macro(f"RHastaResN{LETTER[p]}", str(sum(f.spoke_before_result == 1 for f in of[p]))),
        ]
    for k, (lo, hi) in intervals_hr.items():
        n = PAIR_NAME[k]
        punto = med_hr[k[0]] - med_hr[k[1]]
        summary["differences"][f"{k}_until_result"] = {
            "difference": punto, "ci": [lo, hi], "verdict": verdict(lo, hi),
        }
        tex += [
            macro(f"RDifHastaRes{n}", ms(punto, 1)),
            macro(f"RDifHastaResLo{n}", ms(lo, 1)), macro(f"RDifHastaResHi{n}", ms(hi, 1)),
            macro(f"RDifHastaResVeredicto{n}", verdict(lo, hi)),
        ]

    # --- Por escenario (descriptivo) ---
    for e in range(1, 6):
        for p in POCS:
            vals = [f.observed_value for f in of[p] if f.scenario == e]
            if not vals:
                continue
            summary["scenarios"].setdefault(str(e), {})[p] = {"n": len(vals), "p50": float(np.median(vals))}
            tex += [
                macro(f"REsc{ROMANOS[e]}{LETTER[p]}", ms(float(np.median(vals)), 0)),
                macro(f"REscN{ROMANOS[e]}{LETTER[p]}", ms(len(vals))),
                box_macro(f"RCajaEsc{ROMANOS[e]}{LETTER[p]}", vals),
            ]

    # --- Primer sonido ---
    for p in POCS:
        L = LETTER[p]
        pairs_ps = [(f.observed_value, f.first_sound) for f in of[p] if f.first_sound is not None]
        if not pairs_ps:
            summary["first_sound"][p] = None
            tex += [macro(f"RPrimerMed{L}", "---"), macro(f"RPrimerN{L}", "0"),
                    macro(f"RPrimerLo{L}", "---"), macro(f"RPrimerHi{L}", "---"),
                    macro(f"RPrimerAdel{L}", "---"), macro(f"RPrimerFrac{L}", "---")]
            continue
        primer = [ps for _, ps in pairs_ps]
        leads = [o - ps for o, ps in pairs_ps if o - ps > PRIOR_SOUND_MS]
        lo, hi = ci_median(per_call([f for f in of[p] if f.first_sound is not None],
                                        lambda f: f.first_sound), rng)
        summary["first_sound"][p] = {
            "n": len(pairs_ps), "p50": float(np.median(primer)), "ci": [lo, hi],
            "with_lead": len(leads),
            "lead_p50_if_any": float(np.median(leads)) if leads else None,
        }
        tex += [
            macro(f"RPrimerN{L}", str(len(pairs_ps))),
            macro(f"RPrimerMed{L}", ms(float(np.median(primer)), 1)),
            macro(f"RPrimerLo{L}", ms(lo, 1)), macro(f"RPrimerHi{L}", ms(hi, 1)),
            macro(f"RPrimerFrac{L}", pct(len(leads) / len(pairs_ps))),
            macro(f"RPrimerAdel{L}", ms(float(np.median(leads)), 0) if leads else "---"),
        ]
    ps = summary["first_sound"]
    if ps.get("1") and ps.get("3"):
        tex.append(macro("RPrimerDifUnoTres", ms(ps["1"]["p50"] - ps["3"]["p50"], 0)))
    # Las diferencias al primer sonido, con el mismo bootstrap por llamada que
    # las de la respuesta con contenido, para poder ponerlas lado a lado.
    # Generador aparte: sacar números del compartido correría los intervalos
    # que se calculan después (WER), y esas cifras ya están publicadas.
    if all(ps.get(p) for p in POCS):
        rng_ps = np.random.default_rng(SEED + 1)
        g_ps = {p: per_call(of[p], lambda f: f.first_sound) for p in POCS}
        intervals_ps = {
            "12": ci_paired_difference(pairs_per_scenario(rows, lambda f: f.first_sound), rng_ps),
            "13": ci_difference(g_ps["1"], g_ps["3"], rng_ps),
            "23": ci_difference(g_ps["2"], g_ps["3"], rng_ps),
        }
        for k, (lo, hi) in intervals_ps.items():
            n = PAIR_NAME[k]
            punto = ps[k[0]]["p50"] - ps[k[1]]["p50"]
            summary["differences"][f"{k}_first_sound"] = {
                "difference": punto, "ci": [lo, hi], "verdict": verdict(lo, hi),
            }
            tex += [
                macro(f"RPrimerDif{n}Ic", ms(punto, 1)),
                macro(f"RPrimerDifLo{n}", ms(lo, 1)), macro(f"RPrimerDifHi{n}", ms(hi, 1)),
                macro(f"RPrimerDifVeredicto{n}", verdict(lo, hi)),
            ]
    # Turnos de PoC3 que el instrumento externo dejaba sin número y que se
    # miden con la regla del arnés (el exportador los marca `recuperado`).
    rec = [f for f in of["3"] if f.origin == "recuperado"]
    with_filler = [f for f in rec if f.first_sound is not None and f.observed_value - f.first_sound > PRIOR_SOUND_MS]
    summary["recovered_items"] = {"n": len(rec), "with_filler": len(with_filler)}
    tex += [
        macro("RRecuperadosC", str(len(rec))),
        macro("RRecuperadosPctC", pct(len(rec) / len(of["3"]), 0) if of["3"] else "---"),
        macro("RRecuperadosRellenoC", str(len(with_filler))),
        macro("RRecuperadosRapidosC", str(len(rec) - len(with_filler))),
        macro("RRecuperadosMedC", ms(float(np.median([f.observed_value for f in rec])), 0) if rec else "---"),
        macro("RRecuperadosPrimerMedC",
              ms(float(np.median([f.first_sound for f in with_filler])), 0) if with_filler else "---"),
        macro("RRecuperadosContenidoMedC",
              ms(float(np.median([f.observed_value for f in with_filler])), 0) if with_filler else "---"),
    ]
    return tex, summary


# --------------------------------------------------------------------------
# Pruebas de hipótesis
# --------------------------------------------------------------------------


def medians_per_call(rows: list[Line]) -> dict[str, dict[tuple[int, int, str], float]]:
    groups: dict[str, dict[str, list[Line]]] = {p: defaultdict(list) for p in POCS}
    for f in rows:
        groups[f.poc][f.clave].append(f)
    return {
        p: {(fs[0].scenario, fs[0].repetition, c): float(np.median([f.observed_value for f in fs]))
            for c, fs in groups[p].items()}
        for p in POCS
    }


def prob_faster(a: np.ndarray, b: np.ndarray) -> float:
    """Probabilidad de que un valor de `a` sea menor que uno de `b` (empates a medias)."""
    smaller = sum(int((x < b).sum()) for x in a)
    tie = sum(int((x == b).sum()) for x in a)
    return (smaller + 0.5 * tie) / (a.size * b.size)


def nonparametric_tests(rows: list[Line]) -> tuple[list[str], dict]:
    """Kruskal-Wallis y Wilcoxon con Bonferroni, sobre la mediana de cada llamada.

    Las pruebas de rangos suponen observaciones independientes, y las respuestas
    de una misma llamada no lo son. Por eso la unidad es la llamada.

    - Kruskal-Wallis: ¿las tres PoC vienen de la misma distribución?
    - PoC1 contra PoC2: Wilcoxon de rangos con signo sobre los pares.
    - PoC3 contra cada una: Wilcoxon de suma de rangos (Mann-Whitney), porque
      PoC3 no tiene pareja.
    - Bonferroni: cada valor p por pares se multiplica por 3.

    Tamaño del efecto: probabilidad de que la primera PoC sea más rápida que la
    segunda (en la pareada, la fracción de pares en que lo fue).
    """
    from scipy import stats

    med = medians_per_call(rows)
    m = {p: np.array(list(v.values())) for p, v in med.items()}
    h, p_kw = stats.kruskal(m["1"], m["2"], m["3"])
    summary: dict = {"n": {p: int(m[p].size) for p in POCS},
                     "kruskal_wallis": {"H": float(h), "p": float(p_kw)}, "comparisons": {}}

    pair1 = {(e, r): v for (e, r, _), v in med["1"].items()}
    pair2 = {(e, r): v for (e, r, _), v in med["2"].items()}
    commons = sorted(set(pair1) & set(pair2))
    d = np.array([pair1[k] - pair2[k] for k in commons])
    w = stats.wilcoxon(d)
    cmp = {"12": {"prueba": "rangos con signo", "p": float(w.pvalue),
                   "prob": float((d < 0).mean() + 0.5 * (d == 0).mean()), "n": len(commons)}}
    for k, (a, b) in {"13": ("1", "3"), "23": ("2", "3")}.items():
        u = stats.mannwhitneyu(m[a], m[b], alternative="two-sided")
        cmp[k] = {"prueba": "suma de rangos", "p": float(u.pvalue),
                   "prob": prob_faster(m[a], m[b]), "n": int(m[a].size + m[b].size)}
    for c in cmp.values():
        c["p_bonferroni"] = min(1.0, 3 * c["p"])
    summary["comparisons"] = cmp

    tex = [macro("RKwH", f"{h:.1f}"), macro("RKwP", valor_p(p_kw)),
           macro("RKwPTxt", p_value_text(p_kw)),
           macro("RKwN", str(sum(x.size for x in m.values())))]
    for k, c in cmp.items():
        n = PAIR_NAME[k]
        tex += [
            macro(f"RWP{n}", valor_p(c["p"])),
            macro(f"RWPBonf{n}", valor_p(c["p_bonferroni"])),
            macro(f"RWPBonf{n}Txt", p_value_text(c["p_bonferroni"])),
            macro(f"RWProb{n}", pct(c["prob"], 0)),
            macro(f"RWN{n}", str(c["n"])),
        ]
    return tex, summary


def single_call_influence(rows: list[Line]) -> tuple[list[str], dict]:
    """Cuánto mueve una sola llamada las medianas y sus diferencias.

    Se repite el cálculo quitando una llamada distinta cada vez y se registra el
    cambio máximo de la mediana de cada PoC. Una diferencia entre dos PoC cambia,
    a lo más, la suma de lo que cambian sus dos medianas.
    """
    calls: dict[str, dict[str, list[float]]] = {p: defaultdict(list) for p in POCS}
    for f in rows:
        calls[f.poc][f.clave].append(f.observed_value)

    def median_without(p: str, outside: str | None) -> float:
        return float(np.median([v for c, vs in calls[p].items() if c != outside for v in vs]))

    base = {p: median_without(p, None) for p in POCS}
    change = {p: max(abs(median_without(p, c) - base[p]) for c in calls[p]) for p in POCS}
    diff = {k: change[k[0]] + change[k[1]] for k in PAIR_NAME}
    cut = {}
    for p in POCS:
        v = np.sort([x for vs in calls[p].values() for x in vs])
        cut[p] = float(np.median(v[: int(np.floor(v.size * 0.99))]))
    tex = []
    for p in POCS:
        tex += [macro(f"RJackMax{LETTER[p]}", ms(change[p], 1)),
                macro(f"RRecorte{LETTER[p]}", ms(cut[p], 1))]
    tex += [macro(f"RJackDif{PAIR_NAME[k]}", ms(v, 1)) for k, v in diff.items()]
    tex.append(macro("RJackDifMax", ms(max(diff.values()), 0)))
    return tex, {"change_max_median": change, "change_max_difference": diff,
                 "median_without_slowest_1pct": cut}


# --------------------------------------------------------------------------
# Manejo de turnos
# --------------------------------------------------------------------------


def analyze_calls(ruta: Path) -> tuple[list[str], dict]:
    """Conteos de manejo de turnos y duración, por PoC, desde `llamadas.csv`.

    - Falsos fines de turno = cierres del detector − intervenciones oídas. Un
      cierre es, en Pipecat, cada `UserStoppedSpeakingFrame` del analizador de
      turno; en LiveKit, cada turno del usuario que el framework confirmó
      (`livekit_closures` en el exportador). En PoC3 no existen: el modelo no
      publica sus cierres.
    - El agente habló encima del transportista: solapes medidos sobre la
      grabación de dos canales, igual en las tres PoC.
    """
    with ruta.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                 if r.get("estado") == "ok" and not r["reason_if_omitted"]]
    tex: list[str] = []
    summary: dict = {}
    for p in POCS:
        L = LETTER[p]
        ls = [r for r in rows if r["poc"] == p]
        heard_values = sum(int(r["heard_lines"] or 0) for r in ls)
        closures = sum(int(r["detector_closures"] or 0) for r in ls)
        over = [int(r["agent_talked_over"]) for r in ls if r["agent_talked_over"] != ""]
        # Guardado, no citado: si se incluye lo decide la dirección del estudio.
        mutes = [int(r["turns_without_response"]) for r in ls if r.get("turns_without_response") not in (None, "")]
        duration = [float(r["duration_s"]) for r in ls if r["duration_s"]]
        has_closings = heard_values > 0
        summary[p] = {
            "calls": len(ls), "heard_values": heard_values if has_closings else None,
            "closures": closures if has_closings else None,
            "false_ends": (closures - heard_values) if has_closings else None,
            "fragmentation": (closures / heard_values) if has_closings else None,
            "over": sum(over), "calls_with_talk_over": sum(1 for e in over if e > 0),
            "duration_p50_s": float(np.median(duration)) if duration else None,
            "sin_respuesta": sum(mutes) if mutes else None,
            "calls_without_response": sum(1 for m in mutes if m > 0) if mutes else None,
        }
        r = summary[p]
        tex += [
            macro(f"RCierres{L}", ms(r["closures"]) if has_closings else "no lo publica"),
            macro(f"RFalsos{L}", ms(r["false_ends"]) if has_closings else "no lo publica"),
            macro(f"RFrag{L}", f"{r['fragmentation']:.2f}" if has_closings else "---"),
            macro(f"ROidas{L}", ms(heard_values) if has_closings else "---"),
            macro(f"REncima{L}", str(r["over"])),
            *[macro(f"REncimaEsc{ROMANOS[e]}{L}",
                    str(sum(int(x["agent_talked_over"] or 0) for x in ls if int(x["scenario"]) == e)))
              for e in range(1, 6)],
            macro(f"RLlamadasEncima{L}", str(r["calls_with_talk_over"])),
            macro(f"RDuracion{L}", f"{r['duration_p50_s']:.0f}"),
            macro(f"RSinRespuesta{L}", str(r["sin_respuesta"]) if mutes else "---"),
            macro(f"RLlamadasSinRespuesta{L}", str(r["calls_without_response"]) if mutes else "---"),
        ]
    return tex, summary


# --------------------------------------------------------------------------
# Latencia por etapa (PoC1 y PoC2)
# --------------------------------------------------------------------------


def analyze_stages(ruta: Path, valid_ones: set[str]) -> tuple[list[str], dict]:
    """Mediana de cada etapa del pipeline, en las respuestas que la tienen.

    El primer token del LLM no se mide igual en los dos frameworks
    (`CONTRATO_METRICAS.md` §2.7.1): se publica solo sin herramienta. En PoC1 eso
    es el modo rápido de su distribución bimodal; en PoC2, todas sus muestras.
    """
    with ruta.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["clave"] in valid_ones]
    tex: list[str] = []
    summary: dict = {}
    for p in ("1", "2"):
        L = LETTER[p]
        fs = [r for r in rows if r["poc"] == p]

        def values(campo: str) -> list[float]:
            return [v for v in (_num(r[campo]) for r in fs) if v is not None]

        ttft = values("llm_ttft_ms")
        no_tool = [v for v in ttft if v < TTFT_VALLEY_POC1_MS] if p == "1" else ttft
        end = values("end_of_turn_ms")
        faster = [v for v in end if v <= END_OF_TURN_VALLEY_MS]
        slow = [v for v in end if v > END_OF_TURN_VALLEY_MS]
        self_reported = values("full_turn_ms")
        row = {
            "samples": len(fs),
            # La latencia que el framework reporta de sí mismo, con su propio reloj.
            "self_reported_p50": float(np.median(self_reported)),
            "end_of_turn": float(np.median(end)),
            "fast_end_of_turn": float(np.median(faster)) if faster else None,
            "slow_end_of_turn": float(np.median(slow)) if slow else None,
            "slow_fraction": len(slow) / len(end) if end else None,
            "stt": float(np.median(values("stt_ms"))),
            "llm_without_tool": float(np.median(no_tool)),
            "tts": float(np.median(values("tts_ttfb_ms"))),
        }
        summary[p] = row
        tex += [
            macro(f"REtapaN{L}", ms(len(fs))),
            macro(f"RPropia{L}", ms(row["self_reported_p50"], 1)),
            macro(f"REtapaFin{L}", ms(row["end_of_turn"], 0)),
            macro(f"REtapaFinRapida{L}", ms(row["fast_end_of_turn"], 0)),
            macro(f"REtapaFinLenta{L}", ms(row["slow_end_of_turn"], 0)),
            macro(f"REtapaFinLentaPct{L}", pct(row["slow_fraction"] or 0.0, 0)),
            macro(f"REtapaStt{L}", ms(row["stt"], 0)),
            macro(f"REtapaLlm{L}", ms(row["llm_without_tool"], 0)),
            macro(f"REtapaTts{L}", ms(row["tts"], 0)),
            *high_percentiles(f"REtapaFin{L}", end),
            *high_percentiles(f"REtapaStt{L}", values("stt_ms")),
            *high_percentiles(f"REtapaLlm{L}", no_tool),
            *high_percentiles(f"REtapaTts{L}", values("tts_ttfb_ms")),
        ]
    return tex, summary


def coordinates_by_layer(summary: dict) -> list[str]:
    """Las barras de la figura por capa, en números crudos (pgfplots no lee `{,}`).

    `\\RCoordCapas<L>` son pares `(ms, fila)`; la fila 7 es la capa de arriba.
    PoC3 solo tiene las dos capas que se miden desde fuera.
    """
    rows = {"entry": 7, "end": 6, "stt": 5, "llm": 4, "herr": 3, "tts": 2, "output": 1}
    output = []
    for p in POCS:
        lat, rt = summary["latency"][p], summary["round_trip"][p]
        values = {"herr": rt["delta"], "output": lat["outbound_p50"]}
        if p in summary.get("stages", {}):
            e = summary["stages"][p]
            values.update(entry=lat["inbound_p50"], end=e["end_of_turn"], stt=e["stt"],
                           llm=e["llm_without_tool"], tts=e["tts"])
        points = " ".join(f"({v:.1f},{rows[k]})" for k, v in values.items() if v is not None)
        output.append(macro(f"RCoordCapas{LETTER[p]}", points))
    return output


def pending_stages() -> list[str]:
    """Las macros de la tabla por capa, mientras no exista `etapas.csv`."""
    return [
        macro(f"REtapa{campo}{L}", PENDING)
        for L in ("A", "B")
        for campo in ("N", "Fin", "FinRapida", "FinLenta", "FinLentaPct", "Stt", "Llm", "Tts")
    ] + [
        macro(f"RPropia{campo}{L}", PENDING)
        for L in ("A", "B") for campo in ("", "Falta", "FaltaPct")
    ]


# --------------------------------------------------------------------------
# Reconocimiento
# --------------------------------------------------------------------------


def analyze_wer(ruta: Path, valid_ones: set[str], rng) -> tuple[list[str], dict]:
    """WER, CER y error en montos por PoC, desde `wer_llamadas.csv`.

    Las tasas son agregadas: errores sumados entre unidades de la referencia. El
    IC95 es bootstrap por llamada sobre ese cociente.
    """
    from scipy import stats

    with ruta.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                 if not r["reason_if_omitted"] and r["clave"] in valid_ones]
    tex: list[str] = []
    summary: dict = {}
    families = {"words": "Wer", "characters": "Cer", "amounts": "Montos"}

    def arrays(fs: list[dict], prefix: str) -> tuple[np.ndarray, np.ndarray]:
        n = np.array([int(r[f"{prefix}_reference"]) for r in fs], dtype=float)
        e = np.array([int(r[f"{prefix}_s"]) + int(r[f"{prefix}_d"]) + int(r[f"{prefix}_i"])
                      for r in fs], dtype=float)
        return e, n

    per_poc = {p: [r for r in rows if r["poc"] == p] for p in POCS}
    for p in POCS:
        fs = per_poc[p]
        if not fs:
            continue
        L = LETTER[p]
        summary[p] = {"calls": len(fs)}
        tex.append(macro(f"RWerLlamadas{L}", str(len(fs))))
        # Llamadas cuya última línea no cuenta: se cerraron antes de que llegara su
        # transcripción (`exportar_wer.HOLGURA_MINIMA_SEG`).
        tex.append(macro(f"RWerFuera{L}", str(sum(1 for r in fs if r.get("last_line_outside") == "1"))))
        for prefix, name in families.items():
            e, n = arrays(fs, prefix)
            sample_rate = e.sum() / n.sum()
            reps = np.empty(REPLICAS)
            for k in range(REPLICAS):
                idx = rng.integers(0, len(fs), len(fs))
                reps[k] = e[idx].sum() / n[idx].sum()
            lo, hi = np.percentile(reps, [2.5, 97.5])
            summary[p][prefix] = {"sample_rate": float(sample_rate), "ci": [float(lo), float(hi)],
                                   "errors": int(e.sum()), "units": int(n.sum())}
            tex += [
                macro(f"R{name}{L}", f"{100 * sample_rate:.2f}"),
                macro(f"R{name}Lo{L}", f"{100 * lo:.2f}"),
                macro(f"R{name}Hi{L}", f"{100 * hi:.2f}"),
                macro(f"R{name}N{L}", ms(float(n.sum()))),
                macro(f"R{name}Err{L}", ms(float(e.sum()))),
            ]
    if all(p in summary for p in POCS):
        for p in ("1", "2"):
            reason = summary[p]["words"]["sample_rate"] / summary["3"]["words"]["sample_rate"]
            summary[p]["times_pocc"] = reason
            tex.append(macro(f"RWerVeces{LETTER[p]}", f"{reason:.1f}"))

    for e_ in range(1, 6):
        for p in POCS:
            fs = [r for r in per_poc[p] if int(r["scenario"]) == e_]
            if fs:
                e, n = arrays(fs, "words")
                em, nm = arrays(fs, "amounts")
                tex += [
                    macro(f"RWerEsc{ROMANOS[e_]}{LETTER[p]}", f"{100 * e.sum() / n.sum():.1f}"),
                    macro(f"RMontosEsc{ROMANOS[e_]}{LETTER[p]}",
                          f"{100 * em.sum() / nm.sum():.1f}" if nm.sum() else "---"),
                    macro(f"RMontosEscN{ROMANOS[e_]}{LETTER[p]}", str(int(nm.sum()))),
                    macro(f"RMontosEscErr{ROMANOS[e_]}{LETTER[p]}", str(int(em.sum()))),
                    # Por qué cambia el número de montos entre PoC: cuántas llamadas
                    # hay en el escenario y hasta qué línea llegó cada una.
                    macro(f"RWerEscLlamadas{ROMANOS[e_]}{LETTER[p]}", str(len(fs))),
                    macro(f"RWerEscLineas{ROMANOS[e_]}{LETTER[p]}",
                          f"{np.mean([int(r['turns_reference']) for r in fs]):.1f}"),
                ]

    def call_rate(fs: list[dict], prefix: str) -> tuple[list[dict], np.ndarray]:
        """El error de cada llamada. En montos solo cuentan las llamadas que tienen alguno:
        los escenarios 1 y 4 no tienen montos."""
        e, n = arrays(fs, prefix)
        having = n > 0
        return [r for r, c in zip(fs, having) if c], e[having] / n[having]

    def effect_r(p_raw: float, n_obs: int) -> float:
        """r = Z/sqrt(N), con Z el de la prueba de dos colas y N las llamadas comparadas
        (en la pareada, el número de pares), como en Peres (2026)."""
        return float(stats.norm.isf(p_raw / 2) / np.sqrt(n_obs))

    # La misma familia de pruebas para cada tipo de error, sobre el error de cada llamada:
    # Kruskal-Wallis entre las tres y, por par, Wilcoxon (PoC1 contra PoC2, pareadas por
    # escenario y repetición) o suma de rangos (contra PoC3), con Bonferroni sobre los
    # tres pares y el tamaño del efecto r.
    for prefix, name in families.items():
        rates = {p: call_rate(per_poc[p], prefix) for p in POCS if per_poc[p]}
        if len(rates) < 3:
            continue
        w = {p: v for p, (_, v) in rates.items()}
        h, p_kw = stats.kruskal(w["1"], w["2"], w["3"])
        n_kw = sum(len(v) for v in w.values())
        tex += [macro(f"R{name}KwH", f"{h:.1f}"), macro(f"R{name}KwP", valor_p(p_kw)),
                macro(f"R{name}KwPTxt", p_value_text(p_kw)),
                # Épsilon cuadrado de Kruskal-Wallis: H / (N - 1).
                macro(f"R{name}KwEpsilon", f"{h / (n_kw - 1):.2f}")]
        tex += [macro(f"R{name}Pruebas{LETTER[p]}", str(len(w[p]))) for p in POCS]
        pair1 = {(r["scenario"], r["repetition"]): v for r, v in zip(rates["1"][0], w["1"])}
        pair2 = {(r["scenario"], r["repetition"]): v for r, v in zip(rates["2"][0], w["2"])}
        commons = sorted(set(pair1) & set(pair2))
        tests = {
            "UnoDos": (stats.wilcoxon([pair1[k] - pair2[k] for k in commons]).pvalue, len(commons)),
            "UnoTres": (stats.mannwhitneyu(w["1"], w["3"], alternative="two-sided").pvalue,
                        len(w["1"]) + len(w["3"])),
            "DosTres": (stats.mannwhitneyu(w["2"], w["3"], alternative="two-sided").pvalue,
                        len(w["2"]) + len(w["3"])),
        }
        detail = summary.setdefault("tests_per_call", {})[prefix] = {
            "kruskal_wallis": {"H": float(h), "p": float(p_kw), "N": n_kw}}
        for pair, (pval, n_obs) in tests.items():
            p_bonf = min(1.0, 3 * float(pval))
            r = effect_r(float(pval), n_obs)
            tex += [macro(f"R{name}WPBonf{pair}", valor_p(p_bonf)),
                    macro(f"R{name}WPBonf{pair}Txt", p_value_text(p_bonf)),
                    macro(f"R{name}Efecto{pair}", f"{r:.2f}")]
            detail[pair] = {"p_bonferroni": p_bonf, "r": r, "N": n_obs}
        if prefix == "words":
            summary["kruskal_wallis"] = {"H": float(h), "p": float(p_kw)}
            summary["wilcoxon_bonferroni"] = {pair: d["p_bonferroni"] for pair, d in detail.items()
                                              if pair != "kruskal_wallis"}
    return tex, summary


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    valid_ones = calls_ok(args.data / "calls.csv")
    rows = read_lines(args.data / "lines.csv", valid_ones)
    rng = np.random.default_rng(SEED)
    tex = [
        "% GENERADO por shared/metrics/analysis_report.py — no editar a mano.",
        f"% {REPLICAS} réplicas bootstrap, semilla {SEED}.",
    ]
    t, summary = analyze_latency(rows, rng)
    tex += t
    for name, (t, r) in {
        "turns": analyze_calls(args.data / "calls.csv"),
        "tests": nonparametric_tests(rows),
        "influence": single_call_influence(rows),
    }.items():
        tex += t
        summary[name] = r
    stages_path = args.data / "stages.csv"
    if stages_path.is_file():
        t, summary["stages"] = analyze_stages(stages_path, valid_ones)
        tex += t + [macro("RHayEtapas", "1")]
        # Cuánto se queda corta la cifra que el framework reporta de sí mismo.
        for p in ("1", "2"):
            obs = summary["latency"][p]["p50"]
            missing = obs - summary["stages"][p]["self_reported_p50"]
            tex += [macro(f"RPropiaFalta{LETTER[p]}", ms(missing, 0)),
                    macro(f"RPropiaFaltaPct{LETTER[p]}", pct(missing / obs, 0))]
        tex += coordinates_by_layer(summary)
    else:
        tex += pending_stages() + [macro("RHayEtapas", "0")]
    ruta_wer = args.data / "wer_calls.csv"
    if ruta_wer.is_file():
        t, summary["wer"] = analyze_wer(ruta_wer, valid_ones, rng)
        tex += t + [macro("RHayWer", "1")]
    else:
        tex.append(macro("RHayWer", "0"))

    # Robustez de la comparación (segunda revisión): generadores propios.
    from shared.metrics.robustness_report import generate as robustness

    t, summary["robustness"] = robustness(rows, args.data, args.output, valid_ones)
    tex += t

    # Poblaciones, validación del arnés, sensibilidad, audios y guiones.
    from shared.metrics.report_instruments import generate

    t, summary["instruments"] = generate(args.data, args.output, valid_ones)
    tex += t
    # El transporte de PoC2 contra lo que la validación midió como red pura: lo
    # que sobra no es internet (detector de voz a la entrada; arranque del
    # audio antes del primer sonido audible, a la salida).
    network = (summary["instruments"].get("validacionb") or {}).get("normal_network")
    lat1, lat2 = summary["latency"]["1"], summary["latency"]["2"]
    if network and lat2.get("inbound_p50") is not None and lat2.get("outbound_p50") is not None:
        no_network = lat2["inbound_p50"] + lat2["outbound_p50"] - network["inbound"] - network["outbound"]
        tex += [
            macro("RIdaNoRedB", ms(lat2["inbound_p50"] - network["inbound"], 0)),
            macro("RVueltaNoRedB", ms(lat2["outbound_p50"] - network["outbound"], 0)),
            macro("RTransporteNoRedB", ms(no_network, 0)),
            # Lo que el transporte de PoC2 que no es red pasa del transporte de PoC1,
            # que tampoco tiene red.
            macro("RNoRedDifUnoDos", ms(no_network - lat1["inbound_p50"] - lat1["outbound_p50"], 0)),
        ]
    else:
        tex += [macro(n, "---") for n in ("RIdaNoRedB", "RVueltaNoRedB", "RTransporteNoRedB",
                                          "RNoRedDifUnoDos")]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (args.output / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
