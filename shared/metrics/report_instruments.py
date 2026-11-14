"""Cifras y tablas de las secciones de Instrumentos, Poblaciones y Apéndices.

Lo llama `analysis_report.main` y escribe, igual que el resto de las cifras,
macros de LaTeX: ningún conteo de exclusiones, ningún resultado de la validación
del arnés y ningún dato de los audios se teclea a mano en el `.tex`.

Fuentes, todas en `docs/report/data/`:

- `poblaciones.csv`: llamadas corridas, criterios fallidos y la regla de pareja
  (lo escribe `validation/local_inventory.py`);
- `validacion_arnes.csv` y `validacion_arnes_resumen.txt`: el agente de retardo
  conocido (`validation/validate_harness.py`);
- `deepgram_sensitivity.csv` (`validation/deepgram_sensitivity.py`);
- `fin_de_voz_clips.csv` (`validation/clip_figures.py`);
- `speech_onset.csv` (`validation/speech_onset.py`);
- `entorno.txt` (`validation/local_inventory.py`);

Las líneas del guion y las reglas de rama salen de
`shared/test_scenarios/scenarios.yaml`, las del escenario de interrupciones de
`interrupciones.yaml` y las correcciones de la referencia de
`shared/metrics/audio_reference.py`; la traducción al inglés de esas
líneas, para el artículo, de `traduccion_en.yaml`. Las tablas de los guiones no
copian texto a mano.
"""

from __future__ import annotations

import csv
import math
import platform
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
SCRIPT: Final[Path] = RAIZ / "shared" / "test_scenarios" / "scenarios.yaml"
INTERRUPTIONS: Final[Path] = RAIZ / "shared" / "test_scenarios" / "interrupciones.yaml"
TRANSLATION: Final[Path] = RAIZ / "shared" / "test_scenarios" / "translation_en.yaml"
POOL_CAP: Final[int] = 180

LETTER: Final[dict[str, str]] = {"1": "A", "2": "B", "3": "C"}

CRITERIA: Final[dict[str, str]] = {
    "clips_not_played": "Clips",
    "stt_missed_some_clip": "Stt",
    "insufficient_turns": "Turnos",
    "turn_without_response": "SinResp",
    "response_without_audio": "SinAudio",
}

CONDITIONS: Final[dict[str, str]] = {
    "retardo_300": "Tres",
    "retardo_1000": "Mil",
    "retardo_2500": "DosMil",
    "nivel_-6": "Seis",
    "nivel_-35": "Treinta",
    "arranque_suave": "Suave",
}


def macro(name: str, valor: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{valor}}}"


def num(x: float, dec: int = 1) -> str:
    """Número con coma de millares y `$-$` para los negativos, como el resto."""
    if isinstance(x, float) and math.isnan(x):
        return "---"
    text = f"{abs(x):,.{dec}f}".replace(",", "{,}")
    return f"$-${text}" if x < 0 and round(abs(x), dec) != 0 else text


def pending(what: str) -> str:
    return f"\\textcolor{{red}}{{[PENDIENTE: {what}]}}"


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tex(text: str) -> str:
    """Escapa texto libre para LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
    }
    return "".join(replacements.get(c, c) for c in text)


# --------------------------------------------------------------------------
# Poblaciones
# --------------------------------------------------------------------------


def analyze_populations(ruta: Path) -> tuple[list[str], dict]:
    """Corridas, exclusiones por criterio y por la regla de las tres PoC, y analizadas.

    La población de cada PoC son sus primeras 90 llamadas: en `campana200` las de
    índice ≤ 180 (alternadas, 90 de cada una) y en `campana_poc3_pcmu` las 90.
    """
    all_items = leer_csv(ruta)
    # Las repeticiones de PoC3 entran en las analizadas, no en las corridas ni en
    # las fallas (enmienda del 9 de octubre).
    retries = [f for f in all_items if f["campana"] == "campana_poc3_pcmu_reintentos"]
    rows = [
        f for f in all_items
        if f["campana"] == "campana_poc3_pcmu"
        or (f["campana"] == "campana200" and int(f["indice"]) <= POOL_CAP)
    ]
    output: list[str] = []
    summary: dict[str, Any] = {}
    for poc, L in LETTER.items():
        fs = [f for f in rows if f["poc"] == poc]
        per_scen = Counter(f["scenario"] for f in fs)
        failed_calls = [f for f in fs if f["estado"] != "ok"]
        # Regla de las tres PoC: `en_la_poza` ya la aplica (`inventario_local`).
        # `pareja` son las que pasaron sus criterios y quedaron fuera porque
        # otra PoC falló en ese escenario y repetición.
        matched_pair = [f for f in fs if f["estado"] == "ok" and f["in_the_pool"] != "1"]
        analyzed = [f for f in fs + [r for r in retries if r["poc"] == poc]
                      if f["estado"] == "ok" and f["in_the_pool"] == "1"]
        retried = [r for r in retries if r["poc"] == poc]
        per_criterion = {c: sum(1 for f in failed_calls if f.get(c) == "1") for c in CRITERIA}
        failures_per_call = sorted(
            (sum(f.get(c) == "1" for c in CRITERIA) for f in failed_calls), reverse=True)
        several = sum(1 for n in failures_per_call if n > 1)
        reviews = sum(1 for f in failed_calls if (f.get("failed_poc3_reviews") or "").strip())
        scen = sorted(per_scen.values())
        analyzed_scen = Counter(f["scenario"] for f in analyzed)
        summary[poc] = {
            "runs": len(fs), "per_scenario": dict(per_scen), "failed_calls": len(failed_calls),
            "per_criterion": per_criterion, "matched_pair": len(matched_pair), "several": several,
            "reviews": reviews, "analyzed": len(analyzed),
            "analyzed_by_scenario": dict(analyzed_scen),
        }
        output += [
            macro(f"RCorridas{L}", len(fs)),
            macro(f"RCorridasEsc{L}", scen[0] if scen and scen[0] == scen[-1] else pending("corridas por escenario desiguales")),
            macro(f"RRojas{L}", len(failed_calls)),
            macro(f"RExclPareja{L}", len(matched_pair)),
            macro(f"RExcluidas{L}", len(failed_calls) + len(matched_pair)),
            macro(f"RAnalizadas{L}", len(analyzed)),
            macro(f"RReintentos{L}", len(retried)),
            macro(f"RReintentosOk{L}", sum(r["estado"] == "ok" for r in retried)),
            macro(f"RRojasVarios{L}", several),
            # Cuántas fallas suman los renglones por criterio, y cuántas tuvo cada
            # llamada que falló más de uno (de mayor a menor), para el pie del cuadro.
            macro(f"RCritSuma{L}", sum(per_criterion.values())),
            macro(f"RRojasVariosDetalle{L}",
                  " y ".join(str(n) for n in failures_per_call if n > 1) or "0"),
            macro(f"RRevisiones{L}", reviews),
            *[macro(f"RCrit{n}{L}", per_criterion[c]) for c, n in CRITERIA.items()],
            *[macro(f"RAnalizadasEsc{r}{L}", analyzed_scen.get(str(e), 0))
              for e, r in zip(range(1, 6), ("I", "II", "III", "IV", "V"))],
        ]
    return output, summary


# --------------------------------------------------------------------------
# Validación del arnés
# --------------------------------------------------------------------------


def analyze_validation(
    ruta: Path, summary_file_path: Path, prefix: str = "RVal"
) -> tuple[list[str], dict]:
    """La validación del arnés por un camino: `RVal` es PoC1, `RValB` PoC2, `RValC` PoC3."""
    rows = leer_csv(ruta)
    output: list[str] = []
    summary: dict[str, Any] = {}

    def floats(fs: list[dict], col: str) -> list[float]:
        return [float(f[col]) for f in fs if f.get(col) not in (None, "")]

    for cond, n in CONDITIONS.items():
        fs = [f for f in rows if f["condition"] == cond]
        err = floats(fs, "error_ms")
        eos = floats(fs, "end_of_speech_error_ms")
        resp = floats(fs, "response_error_ms")
        frase = floats(fs, "error_contra_frase_ms")
        r = {
            "turns": len(fs), "measured_items": len(err), "lost": len(fs) - len(err),
            "bias": statistics.fmean(err) if err else math.nan,
            "max_abs": max(abs(e) for e in err) if err else math.nan,
            "sd": statistics.stdev(err) if len(err) > 1 else math.nan,
            "eos": statistics.fmean(eos) if eos else math.nan,
            "resp": statistics.fmean(resp) if resp else math.nan,
            "frase": statistics.fmean(frase) if frase else math.nan,
            "median_value": statistics.median(err) if err else math.nan,
        }
        summary[cond] = r
        output += [
            macro(f"{prefix}{n}Turnos", r["turns"]),
            macro(f"{prefix}{n}Medidos", r["measured_items"]),
            macro(f"{prefix}{n}Perdidos", r["lost"]),
            macro(f"{prefix}{n}Sesgo", num(r["bias"])),
            macro(f"{prefix}{n}Max", num(r["max_abs"])),
            macro(f"{prefix}{n}Desv", num(r["sd"])),
            macro(f"{prefix}{n}Fdv", num(r["eos"])),
            macro(f"{prefix}{n}Resp", num(r["resp"])),
            macro(f"{prefix}{n}Frase", num(r["frase"])),
            macro(f"{prefix}{n}Mediana", num(r["median_value"])),
        ]
    eos = floats(rows, "end_of_speech_error_ms")
    err = floats(rows, "error_ms")
    resp = floats(rows, "response_error_ms")
    lost = [f for f in rows if f.get("reported_latency_ms") in (None, "")]
    # Los dos lados del error con la voz a nivel normal (las tres condiciones de
    # retardo, a −20 dBFS), en mediana: por un camino con red son la ida y la
    # vuelta del audio, y la mediana no se deja llevar por los pocos turnos en
    # que el arnés tomó una racha posterior.
    normals = [f for f in rows if f["condition"].startswith("retardo_")]
    inbound = [-v for v in floats(normals, "end_of_speech_error_ms")]
    outbound = floats(normals, "response_error_ms")
    runs = len({(f["condition"], f["scenario"]) for f in rows})
    # Turnos en que el arnés tomó otra racha: a más de 500 ms de la mediana de
    # su condición. La mediana de cada condición no se mueve por ellos.
    outliers = 0
    for cond in CONDITIONS:
        e = floats([f for f in rows if f["condition"] == cond], "error_ms")
        if e:
            m = statistics.median(e)
            outliers += sum(1 for x in e if abs(x - m) > 500.0)
    output.append(macro(f"{prefix}Atipicos", outliers))
    if inbound and outbound:
        summary["normal_network"] = {"inbound": statistics.median(inbound), "outbound": statistics.median(outbound)}
    output += [
        macro(f"{prefix}Turnos", len(rows)),
        macro(f"{prefix}Corridas", runs),
        macro(f"{prefix}FdvMedia", num(statistics.fmean(eos)) if eos else "---"),
        macro(f"{prefix}FdvMin", num(min(eos)) if eos else "---"),
        macro(f"{prefix}FdvMax", num(max(eos)) if eos else "---"),
        macro(f"{prefix}ErrMaxAbs", num(max(abs(e) for e in err)) if err else "---"),
        macro(f"{prefix}RespMedia", num(statistics.fmean(resp)) if resp else "---"),
        macro(f"{prefix}Medidos", len(err)),
        macro(f"{prefix}IdaNormal", num(statistics.median(inbound)) if inbound else "---"),
        macro(f"{prefix}VueltaNormal", num(statistics.median(outbound)) if outbound else "---"),
        macro(f"{prefix}RedNormal",
              num(statistics.median(inbound) + statistics.median(outbound)) if inbound and outbound else "---"),
        macro(f"{prefix}Perdidos", len(lost)),
        macro(f"{prefix}PerdidosClips", ", ".join(sorted({tex(f["clip_id"]) for f in lost})) or "---"),
    ]
    # La voz del agente falso y la máquina salen del resumen que escribió la
    # propia corrida, no de lo que uno recuerde.
    voice, system = "", ""
    if summary_file_path.is_file():
        for linea in summary_file_path.read_text(encoding="utf-8").splitlines():
            if linea.startswith("voz del agente falso:"):
                voice = linea.split(":", 1)[1].strip()
            elif linea.startswith("system:"):
                system = linea.split(":", 1)[1].strip()
    output += [
        macro(f"{prefix}Voz", tex(voice) if voice else pending("voz del agente falso")),
        macro(f"{prefix}Sistema", tex(system) if system else pending("sistema de la validación")),
    ]
    return output, summary


# --------------------------------------------------------------------------
# Sensibilidad de Deepgram
# --------------------------------------------------------------------------


def analyze_sensitivity(ruta: Path) -> tuple[list[str], dict]:
    from scipy import stats

    rows = leer_csv(ruta)
    output: list[str] = []
    summary: dict[str, Any] = {}

    def sample_rate(fs: list[dict], prefix: str) -> float:
        n = sum(int(f[f"{prefix}_reference"]) for f in fs)
        return sum(int(f[f"{prefix}_errors"]) for f in fs) / n

    import json

    for poc in ("1", "2"):
        L = LETTER[poc]
        fs = [f for f in rows if f["poc"] == poc]
        fil_hyp = sum(sum(json.loads(f["fillers_hypothesis"]).values()) for f in fs)
        fil_ref = sum(sum(json.loads(f["muletillas_reference"]).values()) for f in fs)
        r = {
            "calls": len(fs), "masked": sum(int(f["masked"]) for f in fs),
            "wer": sample_rate(fs, "words"), "wer_sm": sample_rate(fs, "sm_words"),
            "cer": sample_rate(fs, "characters"), "cer_sm": sample_rate(fs, "sm_characters"),
            "amounts": sample_rate(fs, "amounts"), "amounts_sm": sample_rate(fs, "sm_amounts"),
            "fillers_hyp": fil_hyp, "fillers_ref": fil_ref,
        }
        summary[poc] = r
        output += [
            macro(f"RSensLlamadas{L}", r["calls"]),
            macro(f"RSensEnmasc{L}", r["masked"]),
            macro(f"RSensWer{L}", f"{100 * r['wer']:.2f}"),
            macro(f"RSensWerSm{L}", f"{100 * r['wer_sm']:.2f}"),
            macro(f"RSensCer{L}", f"{100 * r['cer']:.2f}"),
            macro(f"RSensCerSm{L}", f"{100 * r['cer_sm']:.2f}"),
            macro(f"RSensMontos{L}", f"{100 * r['amounts']:.2f}"),
            macro(f"RSensMontosSm{L}", f"{100 * r['amounts_sm']:.2f}"),
            macro(f"RSensMuletHip{L}", fil_hyp),
            macro(f"RSensMuletRef{L}", fil_ref),
            macro(f"RSensBaja{L}", f"{100 * (r['wer'] - r['wer_sm']):.2f}"),
        ]

    def p_bonf(prefix: str) -> float:
        def by(poc: str) -> dict[tuple[str, str], float]:
            return {
                (f["scenario"], f["repetition"]):
                    int(f[f"{prefix}_errors"]) / max(1, int(f[f"{prefix}_reference"]))
                for f in rows if f["poc"] == poc
            }
        a, b = by("1"), by("2")
        commons = sorted(set(a) & set(b))
        return min(1.0, 3 * float(stats.wilcoxon([a[k] - b[k] for k in commons]).pvalue))

    p0, p1 = p_bonf("words"), p_bonf("sm_words")
    summary["p_bonferroni"] = {"original": p0, "no_fillers": p1}
    output += [
        macro("RSensPBonf", f"{p0:.3f}"),
        macro("RSensPBonfSm", f"{p1:.3f}"),
        macro("RSensDifWer", f"{100 * (summary['1']['wer'] - summary['2']['wer']):.2f}"),
        macro("RSensDifWerSm", f"{100 * (summary['1']['wer_sm'] - summary['2']['wer_sm']):.2f}"),
    ]
    return output, summary


# --------------------------------------------------------------------------
# Distancia del primer sonido audible a la voz plena
# --------------------------------------------------------------------------


def analyze_onset(ruta: Path) -> tuple[list[str], dict]:
    if not ruta.is_file():
        missing = pending("distancia del primer sonido audible a la voz plena en las grabaciones de la campaña")
        return (
            [macro("RHayArranque", "0")]
            + [macro(f"RArr{q}{L}", missing) for L in LETTER.values() for q in ("Med", "Pv", "Pn", "N")]
            + [macro("RArrDifUnoTres", missing), macro("RArrDifUnoDos", missing)],
            {},
        )
    rows = leer_csv(ruta)
    output = [macro("RHayArranque", "1")]
    summary: dict[str, Any] = {}
    for poc, L in LETTER.items():
        d = np.array([float(f["distance_ms"]) for f in rows if f["poc"] == poc])
        if d.size == 0:
            output += [macro(f"RArr{q}{L}", pending(f"sin grabaciones de PoC{poc}")) for q in ("Med", "Pv", "Pn", "N")]
            continue
        p25, p50, p75 = np.percentile(d, [25, 50, 75])
        summary[poc] = {"n": int(d.size), "p25": p25, "p50": p50, "p75": p75}
        output += [
            macro(f"RArrMed{L}", num(p50, 0)), macro(f"RArrPv{L}", num(p25, 0)),
            macro(f"RArrPn{L}", num(p75, 0)), macro(f"RArrN{L}", f"{d.size:,}".replace(",", "{,}")),
        ]
    if "1" in summary and "3" in summary:
        output.append(macro("RArrDifUnoTres", num(summary["1"]["p50"] - summary["3"]["p50"], 0)))
    if "1" in summary and "2" in summary:
        output.append(macro("RArrDifUnoDos", num(summary["1"]["p50"] - summary["2"]["p50"], 0)))
    return output, summary


# --------------------------------------------------------------------------
# Los audios y los guiones (apéndices)
# --------------------------------------------------------------------------


def reproducible_ones(data: dict) -> list[tuple[dict, dict, str, str]]:
    """`(escenario, turno, clip_id, linea)` de cada línea que algún guion reproduce."""
    output = []
    for scen in data["scenarios"]:
        for turn in scen["turns"]:
            base = f"e{scen['id']}_{str(turn['id']).lower()}"
            output.append((scen, turn, base, turn["linea"]))
            alternates = turn.get("alternates") or []
            seen_items = set()
            for rama in turn.get("branches") or []:
                if rama["linea"] in alternates and rama["linea"] not in seen_items:
                    seen_items.add(rama["linea"])
                    k = alternates.index(rama["linea"]) + 1
                    output.append((scen, turn, f"{base}_alt{k}", rama["linea"]))
    return output


def condition_in_words(rama: dict) -> str:
    if rama.get("by_default"):
        return "ninguna de las anteriores (rama predeterminada)"
    if "reached_amount" in rama:
        return (f"el agente dijo {int(rama['reached_amount']):,} o más en algún momento "
                "de la llamada").replace(",", "{,}")
    if "turn_amount" in rama:
        return (f"el agente dijo {int(rama['turn_amount']):,} o más en su última "
                "response").replace(",", "{,}")
    if rama.get("compensation_in_turn"):
        return "el agente ofreció una compensación en su última respuesta"
    if rama.get("compensation_in_call"):
        return "el agente ofreció alguna compensación en algún momento de la llamada"
    if "words" in rama:
        return "aparece alguna de estas frases: " + ", ".join(rama["words"])
    if "no_words" in rama:
        return "no aparece ninguna de estas frases: " + ", ".join(rama["no_words"])
    return pending("condición no reconocida")


def short_condition(rama: dict) -> str:
    """La condición de una rama en pocas palabras, para la tabla de cada escenario."""
    if rama.get("by_default"):
        return "en cualquier otro caso"
    if "reached_amount" in rama:
        return f"si el agente ya dijo {num(int(rama['reached_amount']), 0)} o más"
    if "turn_amount" in rama:
        return f"si el agente dijo {num(int(rama['turn_amount']), 0)} o más en su respuesta"
    if rama.get("compensation_in_turn"):
        return "si el agente ofreció una compensación en su respuesta"
    if rama.get("compensation_in_call"):
        return "si el agente ofreció una compensación en algún momento de la llamada"
    return condition_in_words(rama)


def _branch_clip(base: str, alternates: list[str], rama: dict) -> str:
    """La base es `eN_tM`; la alternativa en la posición k, `eN_tM_altk`."""
    if rama["linea"] not in alternates:
        return base
    return f"{base}_alt{alternates.index(rama['linea']) + 1}"


def short_condition_in(rama: dict) -> str:
    """`short_condition` en inglés, para las tablas del artículo."""
    if rama.get("by_default"):
        return "in any other case"
    if "reached_amount" in rama:
        return f"if the agent has already said {num(int(rama['reached_amount']), 0)} or more"
    if "turn_amount" in rama:
        return f"if the agent said {num(int(rama['turn_amount']), 0)} or more in its reply"
    if rama.get("compensation_in_turn"):
        return "if the agent offered a compensation in its reply"
    if rama.get("compensation_in_call"):
        return "if the agent offered a compensation at any point in the call"
    if "words" in rama:
        return "if any of these phrases appears: " + tex(", ".join(rama["words"]))
    if "no_words" in rama:
        return "if none of these phrases appears: " + tex(", ".join(rama["no_words"]))
    return pending("condición no reconocida")


def _what_sounds(clip: str, linea: str, corrections: Any) -> str:
    """El texto del audio, con asterisco donde difiere del guion."""
    corr = corrections.get(clip)
    return tex(corr.audio) + "\\textsuperscript{*}" if corr else tex(linea)


def _scenario_rows(scen: dict, corrections: Any, condition: Any,
                         translation: dict[str, str] | None = None) -> list[str]:
    """Turno, cuándo suena y lo que dice el audio; con `traduccion`, también su traducción."""
    rows = []
    for turn in scen["turns"]:
        base = f"e{scen['id']}_{str(turn['id']).lower()}"
        branches = turn.get("branches") or []
        alternates = turn.get("alternates") or []
        # En un turno con ramas, una línea por rama y en el orden en que se revisan.
        options = (
            [(_branch_clip(base, alternates, r), r["linea"], condition(r)) for r in branches]
            if branches else [(base, turn["linea"], "---")]
        )
        for k, (clip, linea, when) in enumerate(options):
            cells = [str(turn["id"]) if k == 0 else "", when, _what_sounds(clip, linea, corrections)]
            if translation is not None:
                cells.append(tex(translation[clip]))
            rows.append(" & ".join(cells) + "\\\\")
    return rows


def tables_by_scenario(data: dict, destination: Path, corrections: Any) -> None:
    """Las líneas de cada escenario: cuándo suenan y lo que dice el audio.

    Escribe `guion_e<N>.tex` por escenario y `guion_interrupciones.tex`, solo
    los renglones; la tabla y el porqué de cada escenario están en `main.tex`.
    Para el artículo en inglés escribe además `guion_en_e<N>.tex` y
    `guion_en_interrupciones.tex`, con la condición en inglés y la traducción de
    `traduccion_en.yaml` junto al audio, que es lo único que sonó.
    """
    import yaml

    translation = yaml.safe_load(TRANSLATION.read_text(encoding="utf-8"))["clips"]
    header = "% GENERADO por shared/metrics/instrumentos_reporte.py — no editar a mano.\n"
    for scen in data["scenarios"]:
        (destination / f"script_e{scen['id']}.tex").write_text(
            header + "\n".join(_scenario_rows(scen, corrections, short_condition)) + "\n",
            encoding="utf-8")
        (destination / f"script_en_e{scen['id']}.tex").write_text(
            header + "\n".join(
                _scenario_rows(scen, corrections, short_condition_in, translation)) + "\n",
            encoding="utf-8")

    kinds = {"interrupcion": "Interrupción", "apoyo": "Apoyo verbal"}
    kinds_in = {"interrupcion": "Interruption", "apoyo": "Backchannel"}
    rows, rows_in, anterior = [], [], None
    for c in yaml.safe_load(INTERRUPTIONS.read_text(encoding="utf-8"))["clips"]:
        new = c["tipo"] != anterior
        rows.append(f"{kinds[c['tipo']] if new else ''} & {tex(c['text'])}\\\\")
        rows_in.append(
            f"{kinds_in[c['tipo']] if new else ''} & {tex(c['text'])} & {tex(translation[c['id']])}\\\\")
        anterior = c["tipo"]
    (destination / "interruption_script.tex").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8")
    (destination / "script_en_interruptions.tex").write_text(
        header + "\n".join(rows_in) + "\n", encoding="utf-8")


def script_tables(destination: Path) -> tuple[list[str], dict]:
    import yaml

    from shared.metrics.audio_reference import CORRECTIONS

    data = yaml.safe_load(SCRIPT.read_text(encoding="utf-8"))
    text_lines = reproducible_ones(data)

    rules: list[str] = []
    for scen in data["scenarios"]:
        for turn in scen["turns"]:
            branches = turn.get("branches") or []
            base = f"e{scen['id']}_{str(turn['id']).lower()}"
            alternates = turn.get("alternates") or []
            for k, rama in enumerate(branches, start=1):
                clip = _branch_clip(base, alternates, rama)
                # Lo que suena es el audio: donde difiere del guion, el texto
                # del audio con asterisco, igual que en las tablas por escenario.
                rules.append(
                    f"E{scen['id']}-{turn['id']} & {k} & {condition_in_words(rama)} & "
                    f"\\texttt{{{tex(clip)}}}: {_what_sounds(clip, rama['linea'], CORRECTIONS)}\\\\"
                )
            if branches:
                rules.append("\\addlinespace")

    corrected_ones = [c for c in CORRECTIONS.values() if c.clip_id in {clip for _, _, clip, _ in text_lines}]
    notes = [f"\\texttt{{{tex(c.clip_id)}}}: el guion decía «{tex(c.script)}»" for c in corrected_ones]
    # El motivo dice si el recorte se llevó una palabra o si la grabación se
    # desvió del guion (ver referencia_del_audio.CORRECCIONES).
    per_cut = sum(1 for c in corrected_ones if c.motivo.startswith("el recorte"))
    destination.mkdir(parents=True, exist_ok=True)
    tables_by_scenario(data, destination, CORRECTIONS)
    (destination / "script_notes.tex").write_text(
        "% GENERADO por shared/metrics/instrumentos_reporte.py — no editar a mano.\n"
        + "\\begin{itemize}[leftmargin=1.2cm,itemsep=0pt]\n"
        + "\n".join(f"\\item {n}" for n in notes) + "\n\\end{itemize}\n", encoding="utf-8")
    (destination / "branches.tex").write_text(
        "% GENERADO por shared/metrics/instrumentos_reporte.py — no editar a mano.\n"
        + "\n".join(rules[:-1] if rules and rules[-1] == "\\addlinespace" else rules) + "\n",
        encoding="utf-8")
    with_branches = sorted({f"E{e['id']}-{t['id']}" for e in data["scenarios"] for t in e["turns"] if t.get("branches")})
    return (
        [macro("RGuionLineas", len(text_lines)),
         macro("RGuionBase", sum(1 for _, _, c, _ in text_lines if "_alt" not in c)),
         macro("RGuionAlt", sum(1 for _, _, c, _ in text_lines if "_alt" in c)),
         macro("RGuionTurnosConRamas", len(with_branches)),
         macro("RGuionTurnosConRamasLista", ", ".join(with_branches)),
         macro("RGuionCorregidas", len(notes)),
         macro("RGuionCorregidasRecorte", per_cut),
         macro("RGuionCorregidasDesvio", len(corrected_ones) - per_cut)],
        {"text_lines": len(text_lines), "turns_with_branches": with_branches},
    )


def clip_table(ruta: Path, destination: Path) -> tuple[list[str], dict]:
    rows = leer_csv(ruta)
    table_rows = []
    for f in rows:
        sha = f["sha256_ulaw"]
        table_rows.append(
            f"\\texttt{{{tex(f['clip_id'])}}} & {float(f['duration_s']):.2f} & "
            f"{float(f['end_of_speech_s']):.2f} & \\texttt{{{sha[:32]}}}\\newline\\texttt{{{sha[32:]}}}\\\\"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "clips.tex").write_text(
        "% GENERADO por shared/metrics/instrumentos_reporte.py — no editar a mano.\n"
        + "\n".join(table_rows) + "\n", encoding="utf-8")
    dur = [float(f["duration_s"]) for f in rows]
    cola = [float(f["cola_seg"]) for f in rows]
    import json

    manifest = json.loads((RAIZ / "shared" / "test_scenarios" / "clips" / "manifest.json").read_text(encoding="utf-8"))
    ids = {f["clip_id"] for f in rows}
    rms = [c["speech_rms_dbfs"] for c in manifest["clips"] if c["id"] in ids]
    rates = {c["origin_sample_rate"] for c in manifest["clips"] if c["id"] in ids}
    # La grabación telefónica de referencia de la calibración: sus dos canales,
    # medidos con el mismo criterio de 25 dB bajo el pico.
    calibration = json.loads((RAIZ / "shared" / "audio" / "level_calibration.json").read_text(encoding="utf-8"))
    channel_list = [calibration["measurement"][k]["speech_rms_dbfs"] for k in ("canal_0", "canal_1")]
    target = float(calibration["calibration_target"]["valor_provisional_dbfs"])
    return (
        [macro("RClipsN", len(rows)),
         macro("RClipsCoinciden", sum(int(f["coincide"]) for f in rows)),
         macro("RClipsShaOk", sum(int(f["verified_sha256"]) for f in rows)),
         macro("RClipsDurMin", f"{min(dur):.2f}"), macro("RClipsDurMax", f"{max(dur):.2f}"),
         macro("RClipsColaMax", num(1000 * max(cola), 0)),
         macro("RClipsRmsMin", num(min(rms))), macro("RClipsRmsMax", num(max(rms))),
         macro("RClipsRmsMediana", num(statistics.median(rms))),
         macro("RClipsDentroUno", sum(abs(x - target) <= 1.0 for x in rms)),
         macro("RRefCanalAlto", num(max(channel_list))), macro("RRefCanalBajo", num(min(channel_list))),
         macro("RClipsEntreRef", sum(min(channel_list) <= x <= max(channel_list) for x in rms)),
         macro("RClipsTasaOrigen", num(next(iter(rates)) / 1000, 0) if len(rates) == 1 else pending("tasa de origen"))],
        {"n": len(rows)},
    )


# --------------------------------------------------------------------------
# Entorno
# --------------------------------------------------------------------------


def analyze_environment(ruta: Path) -> list[str]:
    """Versiones del análisis (este proceso) y de la exportación (la PC de las pruebas)."""
    import scipy

    from shared.metrics.analysis_report import REPLICAS, SEED

    output = [
        macro("RReplicas", f"{REPLICAS:,}".replace(",", "{,}")),
        macro("RSemilla", SEED),
        macro("RAnalisisPython", platform.python_version()),
        macro("RAnalisisNumpy", np.__version__),
        macro("RAnalisisScipy", scipy.__version__),
    ]
    values: dict[str, str] = {}
    if ruta.is_file():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if ":" in linea and not linea.startswith(" "):
                k, v = linea.split(":", 1)
                values[k.strip()] = v.strip()
            elif "=" in linea:
                k, v = linea.strip().split("=", 1)
                values[k] = v
    def v(clave: str, what: str) -> str:
        return tex(values[clave]) if values.get(clave) else pending(what)
    cpu = values.get("cpu", "").split("·")[0].strip()
    cores = values.get("cpu", "").split("núcleos lógicos:")[-1].strip() if "núcleos" in values.get("cpu", "") else ""
    output += [
        macro("REntSistema", v("system", "sistema operativo")),
        macro("REntCpu", tex(cpu) if cpu else pending("CPU")),
        macro("REntNucleos", tex(cores) if cores else pending("núcleos")),
        macro("REntRam", v("ram_gb", "RAM")),
        macro("REntPython", v("python", "versión de Python")),
        macro("REntNumpy", v("numpy", "versión de numpy")),
        macro("REntScipy", v("scipy", "versión de scipy")),
        macro("REntVozElevenlabs", v("ELEVENLABS_VOICE_ID", "voz de ElevenLabs")),
        macro("REntModeloLive", v("OPENAI_LIVE_MODEL", "modelo de GPT-Live")),
    ]
    return output


# --------------------------------------------------------------------------


def min_latency(ruta: Path, valid_ones: set[str]) -> list[str]:
    """La latencia más baja medida en vivo: dice si el punto ciego pesó.

    El punto ciego es de la regla en vivo, que solo toma voz del agente posterior
    al final del archivo del transportista. PoC3 se mide sobre la grabación desde
    el fin de voz y no lo tiene, así que sus turnos no cuentan aquí.
    """
    if not ruta.is_file():
        return [macro("RObsMin", pending("latencia mínima"))]
    values = [
        float(f["observed_latency_ms"]) for f in leer_csv(ruta)
        if f["clave"] in valid_ones and f.get("observed_latency_ms")
        and f.get("origin", "en_vivo") in ("en_vivo", "")
        and (f.get("origin") or f.get("poc") in ("1", "2"))
    ]
    return [macro("RObsMin", num(min(values), 0))]


def interruption_clip_table(destination: Path) -> tuple[list[str], dict]:
    """Los 12 audios del escenario de interrupciones, para el apéndice.

    Salen de su manifiesto (`interruption_clips/manifest.json`) y la huella se
    vuelve a calcular sobre cada archivo `.ulaw`. Los `.ulaw` no viajan en el
    repo (son la voz de una persona): sin ellos la huella no se puede rehacer y
    `RClipsIntShaOk` sale `---`, en vez de tumbar el análisis entero.
    """
    import hashlib
    import json

    folder = RAIZ / "shared" / "test_scenarios" / "interruption_clips"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audio_files = [folder / f"{c['id']}.ulaw" for c in manifest["clips"]]
    with_audio = all(a.is_file() for a in audio_files)
    table_rows, verified = [], 0
    for c, audio in zip(manifest["clips"], audio_files):
        sha = c["sha256_ulaw"]
        if with_audio:
            verified += hashlib.sha256(audio.read_bytes()).hexdigest() == sha
        table_rows.append(
            f"\\texttt{{{tex(c['id'])}}} & {float(c['duration_s']):.2f} & "
            f"{float(c['end_of_speech_s']):.2f} & \\texttt{{{sha[:32]}}}\\newline\\texttt{{{sha[32:]}}}\\\\"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "interruption_clips.tex").write_text(
        "% GENERADO por shared/metrics/instrumentos_reporte.py — no editar a mano.\n"
        + "\n".join(table_rows) + "\n", encoding="utf-8")
    n = len(manifest["clips"])
    sha_ok = verified if with_audio else None
    return (
        [macro("RClipsIntN", n), macro("RClipsIntShaOk", "---" if sha_ok is None else sha_ok)],
        {"n": n, "sha_ok": sha_ok},
    )


def analyze_direct_validation(ruta: Path) -> tuple[list[str], dict]:
    """El reanálisis de la validación de PoC3 (`validation.reanalysis_validation_poc3`).

    Por condición: el error del método directo (el de los turnos recuperados) en
    todos los turnos, y el de los dos métodos sin los turnos en que el modelo
    falso dejó cola de la respuesta anterior (`cola_vieja`).
    """
    rows = leer_csv(ruta)
    output: list[str] = []
    summary: dict[str, Any] = {}

    def errors(fs: list[dict], col: str, cleaned: bool) -> list[float]:
        return [float(f[col]) for f in fs if f[col] not in ("", None)
                and (not cleaned or f["old_tail"] == "0")]

    for cond, n in CONDITIONS.items():
        fs = [f for f in rows if f["condition"] == cond]
        d, dl = errors(fs, "direct_error_ms", False), errors(fs, "direct_error_ms", True)
        pl = errors(fs, "bridge_error_ms", True)
        summary[cond] = {"turns": len(fs), "direct_value": len(d)}
        output += [
            macro(f"RValD{n}Medidos", len(d)),
            macro(f"RValD{n}Mediana", num(statistics.median(d)) if d else "---"),
            macro(f"RValD{n}Limpio", num(statistics.median(dl)) if dl else "---"),
            macro(f"RValD{n}NLimpio", len(dl)),
            macro(f"RValC{n}Limpio", num(statistics.median(pl)) if pl else "---"),
        ]
    output += [
        macro("RValDColaVieja", sum(f["old_tail"] == "1" for f in rows)),
        macro("RValDTurnos", len(rows)),
    ]
    return output, summary


def generate(data: Path, destination: Path, valid_ones: set[str] | None = None) -> tuple[list[str], dict]:
    """Todo lo de este módulo. Lo que falte sale como PENDIENTE visible."""
    output: list[str] = min_latency(data / "lines.csv", valid_ones or set())
    summary: dict[str, Any] = {}
    sources = {
        "poblaciones": (data / "populations.csv", lambda r: analyze_populations(r)),
        "validacion": (data / "harness_validation.csv",
                       lambda r: analyze_validation(r, data / "harness_validation_summary.txt")),
        "validacionb": (data / "harness_validation_poc2.csv",
                        lambda r: analyze_validation(
                            r, data / "harness_validation_poc2_summary.txt", "RValB")),
        "validacionc": (data / "harness_validation_poc3.csv",
                        lambda r: analyze_validation(
                            r, data / "harness_validation_poc3_summary.txt", "RValC")),
        "validaciond": (data / "harness_validation_poc3_direct.csv", analyze_direct_validation),
        "sensibilidad": (data / "deepgram_sensitivity.csv", lambda r: analyze_sensitivity(r)),
        "clips": (data / "end_of_speech_clips.csv", lambda r: clip_table(r, destination)),
    }
    for name, (ruta, fn) in sources.items():
        if ruta.is_file():
            t, r = fn(ruta)
            output += t
            summary[name] = r
            output.append(macro(f"RHay{name.capitalize()}", "1"))
        else:
            output.append(macro(f"RHay{name.capitalize()}", "0"))
    t, summary["scripts"] = script_tables(destination)
    output += t
    t, summary["interruption_clips"] = interruption_clip_table(destination)
    output += t
    t, summary["arranque"] = analyze_onset(data / "speech_onset.csv")
    output += t
    output += analyze_environment(data / "environment.txt")
    return output, summary
