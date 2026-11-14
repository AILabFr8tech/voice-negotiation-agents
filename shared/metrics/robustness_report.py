"""Robustez de la comparación: reconstrucción, mezcla de escenarios y de líneas, excluidas.

Responde a la segunda revisión del reporte (R5, R12 y R20). Cada función
devuelve las macros que escribe en `resultados.tex`; las que sortean usan su
propio generador, para no correr los intervalos ya publicados.

- `reconstruccion`: PoC1 y PoC2 medidas también sobre la grabación, con la
  regla de los turnos recuperados de PoC3, contra su medición en vivo.
- `scenario_mix`: las diferencias con PoC3 si las dos PoC tuvieran la
  mezcla de escenarios de los pares.
- `by_line` y `branch_frequency`: la latencia con el mismo peso por línea
  y qué línea oyó cada PoC en los turnos con ramas.
- `wer_per_scenario`: el WER con el mismo peso por escenario.
- `with_excluded`: tasa de fallas y latencia con todas las llamadas corridas.
- `comprension`: qué monto entendió el agente en la autocorrección de E5-T4 (R6).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from shared.metrics.analysis_report import (
    LETTER,
    PAIR_NAME,
    POCS,
    REPLICAS,
    SEED,
    Line,
    ci_difference,
    read_lines,
    macro,
    ms,
    pct,
    verdict,
)

# Una línea sin latencia en la sensibilidad con las excluidas: lo más que el
# arnés espera una respuesta (tabla de parámetros del arnés).
CENSORING_MS: Final[float] = 25_000.0
# Para decir que la reconstrucción coincide con la medición en vivo: una
# ventana de energía, 20 ms.
COINCIDE_MS: Final[float] = 20.0
# El corte de la campaña de PoC1 y PoC2 (las primeras 90 de cada una).
TOPE_CAMPANA200: Final[int] = 180


def reconstruction(rows: list[Line]) -> list[str]:
    """PoC1 y PoC2 medidas sobre la grabación contra su medición en vivo."""
    tex: list[str] = []
    for p in ("1", "2"):
        L = LETTER[p]
        pairs = [(f.observed_value, f.reconstructed) for f in rows if f.poc == p and f.reconstructed is not None]
        if not pairs:
            tex += [macro(f"RRecon{c}{L}", "---") for c in ("N", "Total", "Med", "ObsMed", "DifMed", "Coincide")]
            continue
        obs = np.array([o for o, _ in pairs])
        rec = np.array([r for _, r in pairs])
        diff = rec - obs
        tex += [
            macro(f"RReconN{L}", str(len(pairs))),
            macro(f"RReconTotal{L}", str(sum(f.poc == p for f in rows))),
            macro(f"RReconMed{L}", ms(float(np.median(rec)), 1)),
            macro(f"RReconObsMed{L}", ms(float(np.median(obs)), 1)),
            macro(f"RReconDifMed{L}", ms(float(np.median(diff)), 1)),
            macro(f"RReconCoincide{L}", pct(float(np.mean(np.abs(diff) <= COINCIDE_MS)), 0)),
        ]
    return tex


def _calls_per_scenario(rows: Sequence[Line]) -> dict[int, list[np.ndarray]]:
    groups: dict[str, list[Line]] = defaultdict(list)
    for f in rows:
        groups[f.clave].append(f)
    output: dict[int, list[np.ndarray]] = defaultdict(list)
    for fs in groups.values():
        output[fs[0].scenario].append(np.array([f.observed_value for f in fs]))
    return output


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    acc = (np.cumsum(w) - w / 2) / w.sum()
    return float(np.interp(0.5, acc, v))


def scenario_mix(rows: list[Line]) -> list[str]:
    """Las diferencias con PoC3 con la mezcla de escenarios de los pares.

    PoC3 quedó con 14/17/18/18/15 llamadas por escenario y los pares con
    18/18/15/18/15. La cifra puntual pesa cada llamada por (llamadas de los
    pares en su escenario) / (llamadas de su PoC en ese escenario); el
    intervalo sortea llamadas dentro de cada escenario, para las dos PoC con el
    número de llamadas de los pares.
    """
    rng = np.random.default_rng(SEED + 5)
    per_scen = {p: _calls_per_scenario([f for f in rows if f.poc == p]) for p in POCS}
    quota = {e: len(v) for e, v in per_scen["1"].items()}

    def point(p: str) -> float:
        v, w = [], []
        for e, n in quota.items():
            groups = per_scen[p][e]
            for g in groups:
                v += list(g)
                w += [n / len(groups)] * len(g)
        return _weighted_median(np.array(v), np.array(w))

    def drawn(p: str) -> float:
        parts = []
        for e, n in quota.items():
            groups = per_scen[p][e]
            parts += [groups[i] for i in rng.integers(0, len(groups), n)]
        return float(np.median(np.concatenate(parts)))

    tex: list[str] = []
    for k in ("13", "23"):
        a, b = k[0], k[1]
        punto = point(a) - point(b)
        reps = np.array([drawn(a) - drawn(b) for _ in range(REPLICAS)])
        lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
        n = PAIR_NAME[k]
        tex += [
            macro(f"RMezclaDif{n}", ms(punto, 1)),
            macro(f"RMezclaDifLo{n}", ms(lo, 1)), macro(f"RMezclaDifHi{n}", ms(hi, 1)),
            macro(f"RMezclaDifVeredicto{n}", verdict(lo, hi)),
        ]
    return tex


def _medians_per_line(calls: list[list[tuple[str, float]]], text_lines: list[str]) -> np.ndarray:
    """La mediana de cada línea (NaN si en esta muestra nadie la oyó)."""
    per_clip: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        for clip, valor in call:
            per_clip[clip].append(valor)
    return np.array([float(np.median(per_clip[c])) if per_clip.get(c) else np.nan for c in text_lines])


FUNCTIONS: Final[dict[str, str]] = {
    "proponer_precio": "Proponer", "ofrecer_compensacion_no_monetaria": "Compensa",
    "registrar_acuerdo": "Registra", "finalizar_sin_acuerdo": "Finaliza",
}


def tools_per_poc(ruta: Path) -> list[str]:
    """Cuántas veces llamó cada PoC a cada herramienta (`validation.tools_per_call`)."""
    with ruta.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tex: list[str] = []
    for p in POCS:
        of = [f for f in rows if f["poc"] == p]
        tex.append(macro(f"RFuncTotal{LETTER[p]}", str(sum(int(f["calls"]) for f in of))))
        for function, name in FUNCTIONS.items():
            tex.append(macro(f"RFunc{name}{LETTER[p]}",
                             str(sum(int(f["calls"]) for f in of if f["function"] == function))))
    return tex


def unauthorized(ruta: Path) -> list[str]:
    """Compensaciones dichas sin la herramienta y ofertas sobre el precio máximo
    (`validation.unauthorized`)."""
    with ruta.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tex: list[str] = []
    for p in POCS:
        of = [f for f in rows if f["poc"] == p]
        on_top = [float(f["offer_above_max"]) for f in of if f["offer_above_max"]]
        tex += [
            macro(f"RAutLlamadas{LETTER[p]}", str(len(of))),
            macro(f"RAutCompDichas{LETTER[p]}", str(sum(bool(f["compensations_stated"]) for f in of))),
            macro(f"RAutCompSinHerr{LETTER[p]}", str(sum(bool(f["stated_without_the_tool"]) for f in of))),
            macro(f"RAutSobreMax{LETTER[p]}", str(len(on_top))),
            macro(f"RAutSobreMaxMonto{LETTER[p]}", ms(max(on_top)) if on_top else "---"),
        ]
    return tex


def by_line(rows: list[Line]) -> list[str]:
    """La latencia con el mismo peso para cada línea del guion.

    Por las ramas, cada PoC oyó una mezcla distinta de líneas. Se toma la
    mediana de cada línea, por su audio, en cada PoC, y se comparan las líneas
    que oyeron las tres, cada una con el mismo peso: el promedio y la mediana
    de las diferencias línea por línea. Los intervalos sortean llamadas
    completas de cada PoC, como el resto del reporte.
    """
    rng = np.random.default_rng(SEED + 8)
    calls: dict[str, list[list[tuple[str, float]]]] = {}
    for p in POCS:
        g: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for f in rows:
            if f.poc == p and f.clip_id:
                g[f.clave].append((f.clip_id, f.observed_value))
        calls[p] = list(g.values())
    heard_values = [{c for ll in calls[p] for c, _ in ll} for p in POCS]
    text_lines = sorted(set.intersection(*heard_values))
    medians = {p: _medians_per_line(calls[p], text_lines) for p in POCS}
    tex = [macro("RLineaN", str(len(text_lines)))]
    for k, n in PAIR_NAME.items():
        a, b = k[0], k[1]
        diff = medians[a] - medians[b]
        reps_mean, reps_median = [], []
        for _ in range(REPLICAS):
            ma = _medians_per_line([calls[a][i] for i in rng.integers(0, len(calls[a]), len(calls[a]))], text_lines)
            mb = _medians_per_line([calls[b][i] for i in rng.integers(0, len(calls[b]), len(calls[b]))], text_lines)
            d = ma - mb
            reps_mean.append(np.nanmean(d))
            reps_median.append(np.nanmedian(d))
        plo, phi = np.percentile(reps_mean, [2.5, 97.5])
        mlo, mhi = np.percentile(reps_median, [2.5, 97.5])
        tex += [
            macro(f"RLineaDif{n}", ms(float(np.mean(diff)), 0)),
            macro(f"RLineaDifLo{n}", ms(float(plo), 0)), macro(f"RLineaDifHi{n}", ms(float(phi), 0)),
            macro(f"RLineaDifVeredicto{n}", verdict(float(plo), float(phi))),
            macro(f"RLineaDifMed{n}", ms(float(np.median(diff)), 0)),
            macro(f"RLineaDifMedLo{n}", ms(float(mlo), 0)), macro(f"RLineaDifMedHi{n}", ms(float(mhi), 0)),
            macro(f"RLineaDifMedVeredicto{n}", verdict(float(mlo), float(mhi))),
            macro(f"RLineaMasRapida{n}", str(int((diff < 0).sum()))),
        ]
    return tex


RE_CLIP: Final[re.Pattern[str]] = re.compile(r"e(\d)_t(\d)(?:_(alt\d))?$")


def branch_frequency(rows: list[Line], destination: Path) -> list[str]:
    """Qué línea oyó cada PoC en los turnos con ramas, entre sus respuestas medidas.

    Escribe `ramas_por_poc.tex`: un renglón de tabla por variante.
    """
    turns: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for f in rows:
        m = RE_CLIP.match(f.clip_id or "")
        if m:
            turns[f"E{m.group(1)}-T{m.group(2)}"][m.group(3) or "base"][f.poc] += 1
    with_branches = {t: v for t, v in turns.items() if any(k != "base" for k in v)}
    table_rows = []
    for turn in sorted(with_branches):
        variants = with_branches[turn]
        totals = {p: sum(variants[v][p] for v in variants) for p in POCS}
        for v in sorted(variants, key=lambda x: (x != "base", x)):
            cells = [
                f"{pct(variants[v][p] / totals[p], 0)}\\,\\%" if totals[p] else "---"
                for p in POCS
            ]
            name = "base" if v == "base" else v.replace("alt", "alternativa ")
            table_rows.append(f"{turn} & {name} & " + " & ".join(cells) + "\\\\")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "branches_per_poc.tex").write_text(
        "% GENERADO por shared/metrics/robustez_reporte.py — no editar a mano.\n"
        + "\n".join(table_rows) + "\n", encoding="utf-8")
    return [macro("RRamasTurnos", str(len(with_branches)))]


def wer_per_scenario(ruta: Path, valid_ones: set[str]) -> list[str]:
    """El WER con el mismo peso para cada escenario.

    Las palabras de cada escenario se suman aparte y se promedian las cinco
    tasas. El WER no se puede dar por línea: se calcula alineando la llamada
    completa (ver `export_wer`).
    """
    with ruta.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["clave"] in valid_ones and not r.get("reason_if_omitted")]
    tex: list[str] = []
    for p in POCS:
        rates = []
        for e in range(1, 6):
            fs = [r for r in rows if r["poc"] == p and int(r["scenario"]) == e]
            ref = sum(int(r["words_reference"]) for r in fs)
            err = sum(int(r["words_s"]) + int(r["words_d"]) + int(r["words_i"]) for r in fs)
            if ref:
                rates.append(err / ref)
        tex.append(macro(f"RWerEsc{LETTER[p]}", f"{100 * float(np.mean(rates)):.1f}" if rates else "---"))
    return tex


def _runs(pob: list[dict], p: str) -> list[dict]:
    """Las corridas de la campaña de una PoC, antes de cualquier criterio."""
    return [
        r for r in pob
        if r["poc"] == p and not (r["campana"] == "campana200" and int(r["indice"]) > TOPE_CAMPANA200)
        # Las repeticiones de PoC3 no son corridas nuevas: la tasa de fallas usa
        # los primeros intentos.
        and r["campana"] != "campana_poc3_pcmu_reintentos"
    ]


def with_excluded(data: Path) -> tuple[list[str], dict]:
    """Tasa de fallas y latencia con todas las llamadas corridas, también las excluidas.

    Las líneas que se quedaron sin latencia cuentan como `CENSORING_MS`, lo más que
    el arnés espera una respuesta: es el peor caso para la PoC que falla.
    """
    tex: list[str] = []
    summary: dict = {}
    with (data / "populations.csv").open(encoding="utf-8") as fh:
        pob = list(csv.DictReader(fh))
    calls: dict[str, dict] = {}
    for name in ("calls.csv", "excluded_calls.csv"):
        ruta = data / name
        if ruta.is_file():
            with ruta.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    calls[r["clave"]] = r
    all_items = read_lines(data / "lines.csv")
    ruta_ex = data / "excluded_lines.csv"
    if ruta_ex.is_file():
        all_items += read_lines(ruta_ex)
    by_key: dict[str, list[float]] = defaultdict(list)
    for f in all_items:
        by_key[f.clave].append(f.observed_value)

    rng = np.random.default_rng(SEED + 6)
    groups: dict[str, list[np.ndarray]] = {}
    for p in POCS:
        L = LETTER[p]
        runs = _runs(pob, p)
        failed_calls = [r for r in runs if r["estado"] != "ok"]
        g = []
        censored = 0
        for r in runs:
            row = calls.get(r["clave"])
            if row is None or row.get("reason_if_omitted"):
                continue
            values = list(by_key.get(r["clave"], []))
            missing_items = max(0, int(row["played_lines"] or 0) - len(values))
            censored += missing_items
            values += [CENSORING_MS] * missing_items
            if values:
                g.append(np.array(values))
        groups[p] = g
        med = float(np.median(np.concatenate(g)))
        summary[p] = {"runs": len(runs), "failed_calls": len(failed_calls), "calls": len(g),
                      "censored": censored, "p50": med}
        tex += [
            macro(f"RTasaFallas{L}", pct(len(failed_calls) / len(runs))),
            macro(f"RSensExclMed{L}", ms(med, 1)),
            macro(f"RSensExclLlamadas{L}", str(len(g))),
            macro(f"RSensExclCensuradas{L}", str(censored)),
        ]
    for k, n in PAIR_NAME.items():
        lo, hi = ci_difference(groups[k[0]], groups[k[1]], rng)
        punto = summary[k[0]]["p50"] - summary[k[1]]["p50"]
        summary[f"diff_{k}"] = {"difference": punto, "ci": [lo, hi], "verdict": verdict(lo, hi)}
        tex += [
            macro(f"RSensExclDif{n}", ms(punto, 1)),
            macro(f"RSensExclDifLo{n}", ms(lo, 1)), macro(f"RSensExclDifHi{n}", ms(hi, 1)),
            macro(f"RSensExclDifVeredicto{n}", verdict(lo, hi)),
        ]
    return tex, summary


def overlaps(ruta: Path) -> list[str]:
    """Los solapes de voz clasificados (`validation.classified_overlaps`)."""
    with ruta.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tex: list[str] = []
    for p in POCS:
        of = [f for f in rows if f["poc"] == p]
        tex += [
            macro(f"RSolapes{LETTER[p]}", str(len(of))),
            macro(f"RSolapesRetomo{LETTER[p]}", str(sum(f["kind"] == "retomo_en_su_pausa" for f in of))),
            macro(f"RSolapesMitad{LETTER[p]}", str(sum(f["kind"] == "arranco_a_mitad_de_la_linea" for f in of))),
        ]
    resumption_events = [int(f["entered_after_line_start_ms"]) for f in rows if f["kind"] == "retomo_en_su_pausa"]
    half = [int(f["entered_after_line_start_ms"]) for f in rows if f["kind"] != "retomo_en_su_pausa"]
    tex += [
        macro("RSolapesRetomoMax", ms(max(resumption_events), 0) if resumption_events else "---"),
        macro("RSolapesMitadMin", ms(min(half), 0) if half else "---"),
    ]
    return tex


def requests(ruta: Path, ruta_prompt: Path) -> list[str]:
    """El censo de los encargos de PoC3 (`validation.request_census`)."""
    with ruta.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tokens = [int(f["input_tokens"]) for f in rows if f["input_tokens"]]
    tex = [
        macro("RCensoLlamadas", str(len({f["clave"] for f in rows}))),
        macro("RCensoEncargos", str(len(rows))),
        macro("RCensoIguales", str(sum(f["ratio"] == "equal" for f in rows))),
        macro("RCensoParte", str(sum(f["ratio"] == "parte" for f in rows))),
        macro("RCensoOtra", str(sum(f["ratio"] not in ("equal", "parte") for f in rows))),
        macro("RCensoTokensMed", ms(float(np.median(tokens)), 0) if tokens else "---"),
        macro("RCensoTokensMin", ms(min(tokens), 0) if tokens else "---"),
        macro("RCensoTokensMax", ms(max(tokens), 0) if tokens else "---"),
    ]
    # El primer encargo de cada llamada: si los tokens crecen con el tiempo que
    # lleva la llamada, el delegado recibe la conversación acumulada.
    first_items: dict[str, tuple[float, int]] = {}
    for f in rows:
        if not f["input_tokens"]:
            continue
        t = float(f["call_second"])
        if f["clave"] not in first_items or t < first_items[f["clave"]][0]:
            first_items[f["clave"]] = (t, int(f["input_tokens"]))
    if len(first_items) > 2:
        x = np.array([t for t, _ in first_items.values()])
        y = np.array([k for _, k in first_items.values()], dtype=float)
        pending, base = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        tex += [
            macro("RCensoPrimeros", str(len(first_items))),
            macro("RCensoTokensBase", ms(float(base), 0)),
            macro("RCensoTokensPorSeg", f"{pending:.1f}"),
            macro("RCensoTokensR", f"{r:.3f}"),
        ]
    delegate_path = ruta.with_name("delegate_responses.csv")
    if delegate_path.is_file():
        with delegate_path.open(encoding="utf-8") as fh:
            e3 = [r for r in csv.DictReader(fh) if r["scenario"] == "3"]
        tex.append(macro("RCensoTarifaJusta", str(sum(r["fair_tariff"] == "1" for r in e3))))
        tex.append(macro("RCensoLlamadasEscIII", str(len(e3))))
    if ruta_prompt.is_file():
        with ruta_prompt.open(encoding="utf-8") as fh:
            prompt = [int(r["first_call_tokens"]) for r in csv.DictReader(fh)]
        tex.append(macro("RPromptPocUno", ms(float(np.median(prompt)), 0) if prompt else "---"))
    return tex


def phone_tail(rows: list[Line], ruta: Path) -> list[str]:
    """Dos efectos de cómo el arnés oyó a PoC3 (`validation.phone_tail_poc3`).

    - `Fila`: si el teléfono reprodujera en fila, cada respuesta sonaría
      `adelanto_en_fila_ms` más tarde (va a favor de PoC3 en lo publicado).
    - `SinDescarte`: contando las rachas que el arnés descartó en vivo por cortas
      y que en la grabación duran 0.6 s o más (va en contra de PoC3).

    Para cada uno, la mediana de PoC3 y sus diferencias con PoC1 y PoC2.
    """
    with ruta.open(encoding="utf-8") as fh:
        per_turn = {(r["clave"], r["turn"]): r for r in csv.DictReader(fh)}
    rng = np.random.default_rng(SEED + 7)
    of = {p: [f for f in rows if f.poc == p] for p in POCS}

    def groups(fs: list[Line], valor) -> list[np.ndarray]:
        g: dict[str, list[float]] = defaultdict(list)
        for f in fs:
            g[f.clave].append(valor(f))
        return [np.array(v) for v in g.values()]

    def in_row(f: Line) -> float:
        r = per_turn.get((f.clave, str(f.turn)))
        return f.observed_value + float(r["lead_in_row_ms"] or 0.0) if r else f.observed_value

    def no_discard(f: Line) -> float:
        r = per_turn.get((f.clave, str(f.turn)))
        return min(f.observed_value, float(r["latency_without_discard_ms"])) if r else f.observed_value

    tex: list[str] = []
    for name, valor in (("Fila", in_row), ("SinDescarte", no_discard)):
        g3 = groups(of["3"], valor)
        med3 = float(np.median(np.concatenate(g3)))
        tex.append(macro(f"RCola{name}MedC", ms(med3, 1)))
        for p in ("1", "2"):
            gp = groups(of[p], lambda f: f.observed_value)
            lo, hi = ci_difference(gp, g3, rng)
            punto = float(np.median(np.concatenate(gp))) - med3
            n = PAIR_NAME[p + "3"]
            tex += [
                macro(f"RCola{name}Dif{n}", ms(punto, 1)),
                macro(f"RCola{name}DifLo{n}", ms(lo, 1)), macro(f"RCola{name}DifHi{n}", ms(hi, 1)),
                macro(f"RCola{name}DifVeredicto{n}", verdict(lo, hi)),
            ]
    leads = np.array([float(r["lead_in_row_ms"] or 0.0) for r in per_turn.values()])
    tex += [
        macro("RColaFilaPNoventaycinco", ms(float(np.percentile(leads, 95)), 0)),
        macro("RColaFilaMax", ms(float(leads.max()), 0)),
        macro("RColaFilaMasDeMedio", str(int((leads > 500).sum()))),
        macro("RColaSinDescarteN", str(sum(
            float(r["latency_without_discard_ms"]) < float(r["latency_ms"]) for r in per_turn.values()))),
    ]
    return tex


def comprehension(ruta: Path) -> list[str]:
    """Qué monto entendió el agente en E5-T4 (`validation.amount_comprehension`)."""
    with ruta.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tex: list[str] = []
    for p in POCS:
        of = [f for f in rows if f["poc"] == p]
        tex += [
            macro(f"RCompN{LETTER[p]}", str(len(of))),
            macro(f"RCompStt{LETTER[p]}", str(sum(f["transcript_with_corrected"] == "1" for f in of))),
            macro(f"RCompVisible{LETTER[p]}", str(sum(f["reading"] in ("corregido", "retractado") for f in of))),
            macro(f"RCompCorregido{LETTER[p]}", str(sum(f["reading"] == "corregido" for f in of))),
            macro(f"RCompRetractado{LETTER[p]}", str(sum(f["reading"] == "retractado" for f in of))),
            macro(f"RCompRetractadoSttMal{LETTER[p]}", str(sum(
                f["reading"] == "retractado" and f["transcript_with_corrected"] == "0" for f in of))),
        ]
    return tex


def generate(rows: list[Line], data: Path, output: Path, valid_ones: set[str]) -> tuple[list[str], dict]:
    """Todas las macros de robustez, en un orden fijo."""
    tex = reconstruction(rows) + scenario_mix(rows) + by_line(rows)
    tex += branch_frequency(rows, output)
    if (data / "tools_per_call.csv").is_file():
        tex += tools_per_poc(data / "tools_per_call.csv")
    if (data / "unauthorized.csv").is_file():
        tex += unauthorized(data / "unauthorized.csv")
    if (data / "overlaps.csv").is_file():
        tex += overlaps(data / "overlaps.csv")
    if (data / "requests.csv").is_file():
        tex += requests(data / "requests.csv", data / "prompt_poc1.csv")
    if (data / "phone_tail_poc3.csv").is_file():
        tex += phone_tail(rows, data / "phone_tail_poc3.csv")
    if (data / "amount_comprehension.csv").is_file():
        tex += comprehension(data / "amount_comprehension.csv")
    ruta_wer = data / "wer_calls.csv"
    if ruta_wer.is_file():
        tex += wer_per_scenario(ruta_wer, valid_ones)
    t, summary = with_excluded(data)
    return tex + t, {"with_excluded": summary}
