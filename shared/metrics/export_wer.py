"""Exporta a CSV los conteos de error de reconocimiento por llamada (WER, CER, montos).

Cierra `docs/wer_reconciliacion_pending.md`: las cifras de la etapa 4 no se
podían rehacer desde el repo. Este módulo fija **un** cálculo, reproducible, y
es el que cita el reporte.

## La decisión de método: el WER se calcula por llamada, no por turno

La referencia (lo que el transportista dijo) está cortada por turnos del guion;
la hipótesis (lo que transcribió cada sistema) está cortada por turnos de cada
reconocedor, y los cortes no coinciden: PoC1 parte una intervención en ~1.9
pedazos, PoC2 en 1.0, y PoC3 entrega fragmentos que siguen la cadencia del
habla. Emparejar turno por turno obliga a adivinar qué pedazo va con qué línea,
y ese emparejamiento fue justo lo que falló en 23 llamadas de PoC3.

Aquí no se empareja nada: por cada llamada se une **todo** lo que el
transportista dijo y **todo** lo que el sistema transcribió de él, y la
distancia de edición alinea las dos secuencias completas. Es el WER "del canal"
y trata igual a las tres PoC, porque no depende de cómo cada una corta turnos.

## De dónde sale cada lado

- **Referencia:** `linea_reproducida` de la bitácora del arnés, turno por turno,
  con las seis correcciones de `audio_reference.clip_reference`, y sin
  la **cola muerta** (`clips_reference.classify_tail`: clips que sonaron
  cuando la conversación ya había terminado).
- **Hipótesis:** las entradas `rol == "transportista"` de la `transcripcion` del
  JSON de la llamada. **Nunca** el campo `stt` de la bitácora: en PoC2 repite
  resultados interinos y en PoC3 se guardó con la regla de unión rota
  (`docs/hallazgo_ensamblado_stt_poc3.md`).
- **Hipótesis de PoC3:** la del JSON, con los espacios del censo
  (`census_spacing`). El JSON del puente parte el turno del transportista
  cuando un fragmento del agente llega a media palabra (`"Dé"` + agente +
  `"jame"`), y al unir los turnos con espacio queda `"de jame"`. El censo trae
  los fragmentos tal como los mandó la API, con su espacio adentro. Se usa solo
  hasta donde llega el JSON: lo que el censo trae de más al final (fragmentos
  que el puente ya no guardó) lo decide la regla de la última línea, igual en
  las tres PoC.

La normalización y la alineación son las de `shared/metrics/wer.py`, las mismas
de todo el proyecto.

## La última línea, si la llamada se cerró antes de que llegara su transcripción

Si la llamada terminó menos de `MIN_SLACK_S` después de que la última
línea del transportista terminó de sonar, la transcripción de esa línea no tuvo
tiempo de llegar completa: el cierre la corta, no el reconocedor. Esa línea sale
de la referencia y, de la hipótesis, el pedazo que alcanzó a llegar
(`trim_tail`). La holgura se mide igual en las tres PoC, sobre la grabación
de auditoría del arnés: del fin del audio de la línea (ubicado por correlación en
el canal del transportista) al fin de la grabación, que es el fin de la llamada.
El umbral sale de PoC3, la que transcribe más lento: casi todas sus llamadas
que se cerraron a menos de 4 s tienen la última línea cortada, y todas las que
se cerraron después la tienen completa.

## Poblaciones

Las de la latencia, con la regla de las tres PoC: los pares de `campana200` para
PoC1 y PoC2, y las llamadas `ok` de `campana_poc3_pcmu` para PoC3.

Uso, desde la raíz del repo (tarda ~10-15 min: la alineación por carácter de
una llamada completa toma unos 3 s):

    uv run --project poc3_gpt_live python -m shared.metrics.export_wer
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any, Final, NamedTuple

from shared.metrics.export_for_report import (
    CLIPS_DEFAULT,
    LOTE_DEFAULT,
    DEFAULT_OUTPUT,
    read_progress,
    read_summary,
    campaign200_pairs,
)
from shared.metrics.reference import CARRIER_ROLE
from shared.metrics.clips_reference import classify_tail, log_turns
from shared.metrics.audio_reference import clip_reference
from shared.metrics.text import tokenize_text
from shared.metrics.wer import compute_amounts, compute_text

MIN_SLACK_S: Final[float] = 4.0
CLIP_START_S: Final[float] = 1.5

COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition",
    "turns_reference", "dead_tail_turns",
    "words_reference", "words_s", "words_d", "words_i",
    "characters_reference", "characters_s", "characters_d", "characters_i",
    "amounts_reference", "amounts_s", "amounts_d", "amounts_i",
    "last_line_slack_s", "last_line_outside",
    "reason_if_omitted",
]


def last_line_slack(wav: Path, clip_id: str, clips: Path) -> float | None:
    """Segundos entre el fin del audio de la línea y el fin de la grabación."""
    import numpy as np

    from shared.metrics import external_latency_poc3 as ext3
    from shared.metrics.export_for_report import _a_8khz

    ruta = clips / f"{clip_id}.ulaw"
    if not wav.is_file() or not ruta.is_file():
        return None
    audio, sample_rate = ext3.leer_wav(wav)
    a8, t8 = _a_8khz(audio, sample_rate)
    clip = ext3.ulaw_a_pcm16(ruta.read_bytes()).astype(np.float64)
    # Se ubica por su principio: si la llamada se cerró mientras la línea sonaba,
    # el final no está en la grabación y la holgura sale negativa.
    start = clip[: int(CLIP_START_S * t8)]
    i, pico, distant = ext3.clip_position(a8[:, 0].astype(np.float64), start)
    if pico < ext3.MIN_CORRELATION or pico < distant * 1.5:
        return None
    return round((len(a8) - (i + len(clip))) / t8, 2)


def trim_tail(hypothesis: str, linea: str) -> str:
    """La hipótesis sin el pedazo final que corresponde al principio de `linea`.

    Busca, entre las últimas palabras, el tramo que empieza como la línea y más
    se le parece (proporción de coincidencia de al menos 0.6; a igual
    proporción, el más largo). Si no hay ninguno, la línea no alcanzó a
    transcribirse y la hipótesis queda igual.
    """
    import difflib

    h = tokenize_text(hypothesis)
    f = tokenize_text(linea)
    if not h or not f:
        return " ".join(h)
    # Primero por la primera palabra de la línea; si el reconocedor la cambió, por la segunda.
    for k, first in enumerate(f[:2]):
        best: tuple[float, int] | None = None
        for s in range(max(0, len(h) - len(f) - 3), len(h)):
            cola = h[s:]
            if cola[0] != first:
                continue
            r = difflib.SequenceMatcher(None, cola, f[k:k + len(cola)], autojunk=False).ratio()
            if r >= 0.6 and (best is None or r > best[0]):
                best = (r, s)
        if best is not None:
            return " ".join(h[:best[1]])
    return " ".join(h)


def census_transcript(census: Path) -> str:
    """Los fragmentos de entrada de PoC3, pegados crudos: el espacio viene dentro."""
    parts = []
    for linea in census.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        c = json.loads(linea)
        if c.get("type") == "session.input_transcript.delta":
            delta = (c.get("event") or {}).get("delta")
            if isinstance(delta, str):
                parts.append(delta)
    return " ".join("".join(parts).split())


def census_spacing(json_text: str, census_text: str) -> str:
    """El texto del JSON con los espacios del censo.

    Sin espacios, el JSON es un prefijo del censo (lo mismo, menos los
    fragmentos finales que el puente no guardó). Se corta el censo donde se
    acaba el JSON.
    """
    n = len("".join(json_text.split()))
    if not "".join(census_text.split()).startswith("".join(json_text.split())):
        return json_text
    seen_items = 0
    for i, ch in enumerate(census_text):
        if seen_items == n:
            return census_text[:i].strip()
        if not ch.isspace():
            seen_items += 1
    return census_text


class Texts(NamedTuple):
    reference: str
    hypothesis: str
    turns: int
    dead_tail_turns: int
    last_line_outside: bool


def json_transcript(call: dict[str, Any]) -> str:
    """Lo que el sistema transcribió del transportista, en el JSON de la llamada."""
    return " ".join(
        str(e.get("text", "")).strip()
        for e in call.get("transcript") or []
        if e.get("role") == CARRIER_ROLE and str(e.get("text", "")).strip()
    )


def call_texts(
    event_log: dict[str, Any], call: dict[str, Any], slack: float | None = None,
    hypothesis: str | None = None,
) -> Texts:
    """La referencia y la hipótesis de una llamada completa, listas para comparar.

    `holgura`: segundos entre el fin de la última línea y el fin de la llamada
    (`last_line_slack`); con menos de `MIN_SLACK_S`, esa línea
    no cuenta. `hipotesis`: el texto del censo (PoC3); de él se toman los
    espacios del JSON (`census_spacing`).
    """
    alive_ones, excluded_clips = classify_tail(log_turns(event_log))
    of_json = json_transcript(call)
    hypothesis = of_json if hypothesis is None else census_spacing(of_json, hypothesis)
    outside = slack is not None and slack < MIN_SLACK_S and len(alive_ones) > 1
    if outside:
        hypothesis = trim_tail(hypothesis, clip_reference(alive_ones[-1].clip_id, alive_ones[-1].linea))
        alive_ones = alive_ones[:-1]
    reference = " ".join(clip_reference(t.clip_id, t.linea) for t in alive_ones)
    return Texts(reference, hypothesis, len(alive_ones), len(excluded_clips), outside)


def call_slack(event_log: dict[str, Any], artifacts: dict[str, Any]) -> float | None:
    """La holgura de la última línea que cuenta, sobre la grabación de auditoría."""
    alive_ones, _ = classify_tail(log_turns(event_log))
    if not alive_ones:
        return None
    return last_line_slack(
        Path(artifacts.get("auditoria_wav") or ""), alive_ones[-1].clip_id, CLIPS_DEFAULT,
    )


def call_counts(
    event_log: dict[str, Any], call: dict[str, Any], slack: float | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Los conteos de error de una llamada completa (`call_texts`). Solo números."""
    t = call_texts(event_log, call, slack, hypothesis)
    text = compute_text(t.reference, t.hypothesis)
    amounts = compute_amounts(t.reference, t.hypothesis).alignment
    w, c = text.wer, text.cer
    return {
        "turns_reference": t.turns,
        "dead_tail_turns": t.dead_tail_turns,
        "words_reference": text.words_reference,
        "words_s": w.substitutions, "words_d": w.deleted, "words_i": w.insertions,
        "characters_reference": c.hits + c.substitutions + c.deleted,
        "characters_s": c.substitutions, "characters_d": c.deleted, "characters_i": c.insertions,
        "amounts_reference": amounts.hits + amounts.substitutions + amounts.deleted,
        "amounts_s": amounts.substitutions, "amounts_d": amounts.deleted, "amounts_i": amounts.insertions,
        "last_line_slack_s": slack, "last_line_outside": int(t.last_line_outside),
    }


def _row(row: dict[str, Any], counts: dict[str, int] | None, motivo: str = "") -> dict[str, Any]:
    base = {
        "poc": row.get("poc"), "clave": row.get("clave"),
        "scenario": row.get("scenario"), "repetition": row.get("repetition"),
        "reason_if_omitted": motivo,
    }
    return {**base, **(counts or {})}


def export(folder: Path, runs: list[dict[str, Any]], json_key: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warning_list: list[str] = []
    for k, row in enumerate(runs, start=1):
        clave = str(row["clave"])
        if k % 10 == 0 or k == len(runs):
            # La alineación por carácter tarda ~3 s por llamada: sin esto la
            # corrida parece colgada durante minutos.
            print(f"  {folder.name}: {k}/{len(runs)} llamadas", flush=True)
        summary = read_summary(Path(row.get("_folder", folder)), str(row.get("batch_key", clave)))
        art = (summary or {}).get("artifacts") or {}
        event_log = Path(art["event_log"]) if art.get("event_log") else None
        destination = next((Path(art[k]) for k in json_key if art.get(k)), None)
        if event_log is None or destination is None or not event_log.is_file() or not destination.is_file():
            rows.append(_row(row, None, "falta la bitácora o el JSON de la llamada"))
            warning_list.append(f"{clave}: falta la bitácora o el JSON de la llamada")
            continue
        try:
            log_data = json.loads(event_log.read_text(encoding="utf-8"))
            census = Path(art.get("census") or "")
            counts = call_counts(
                log_data, json.loads(destination.read_text(encoding="utf-8")),
                call_slack(log_data, art),
                census_transcript(census) if census.is_file() else None,
            )
            rows.append(_row(row, counts))
        except Exception as exc:  # una corrida rota no tumba la exportación
            rows.append(_row(row, None, f"error: {exc}"))
            warning_list.append(f"{clave}: {exc}\n{traceback.format_exc(limit=2)}")
    return rows, warning_list


def summary_per_poc(rows: list[dict]) -> list[str]:
    text_lines = ["poc  llamadas  palabras   WER      CER      montos(n)  error_montos"]
    for poc in (1, 2, 3):
        fs = [f for f in rows if int(f["poc"]) == poc and not f["reason_if_omitted"]]
        if not fs:
            text_lines.append(f"{poc:<4} 0")
            continue

        def sample_rate(prefix: str) -> float:
            n = sum(f[f"{prefix}_reference"] for f in fs)
            e = sum(f[f"{prefix}_s"] + f[f"{prefix}_d"] + f[f"{prefix}_i"] for f in fs)
            return e / n if n else float("nan")

        words = sum(f["words_reference"] for f in fs)
        amounts = sum(f["amounts_reference"] for f in fs)
        text_lines.append(
            f"{poc:<4} {len(fs):>8}  {words:>8}  {sample_rate('words'):.4f}   "
            f"{sample_rate('characters'):.4f}   {amounts:>9}  {sample_rate('amounts'):.4f}"
        )
    return text_lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    from shared.metrics.campaigns import CAMPAIGN_POC3, valid_combinations, poc3_for_analysis

    # Regla de las tres PoC, la misma de la latencia.
    valid_ones = valid_combinations(args.lote)
    c200 = args.lote / "campana200"
    f12, a12 = export(c200, campaign200_pairs(read_progress(c200), valid_ones),
                        ("pipeline_json", "worker_json"))
    c3 = args.lote / CAMPAIGN_POC3
    # El primer intento, o su repetición si falló (enmienda del 9 de octubre).
    ok3 = [{**f, "_folder": str(c)} for c, f in poc3_for_analysis(args.lote)
           if f.get("estado") == "ok" and (int(f["scenario"]), int(f["repetition"])) in valid_ones]
    f3, a3 = export(c3, ok3, ("bridge_json",))
    rows, warning_list = f12 + f3, a12 + a3

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "wer_calls.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    text_lines = summary_per_poc(rows)
    (args.output / "verification_wer.txt").write_text(
        "\n".join(text_lines + ["", f"warning_list: {len(warning_list)}", *warning_list]) + "\n", encoding="utf-8"
    )
    print("\n".join(text_lines))
    print(f"\n{len(rows)} llamadas, {len(warning_list)} avisos · escrito en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
