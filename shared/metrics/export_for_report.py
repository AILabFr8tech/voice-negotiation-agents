"""Exporta a CSV los números de la campaña que el reporte necesita para sus figuras.

Los JSON por llamada no viajan en el repo (traen habla transcrita y pesan), así
que las gráficas del reporte —boxplots por PoC y por escenario, intervalos por
bootstrap— no se pueden hacer desde el repo. Este módulo corre en la máquina
donde están los artefactos y escribe **solo números**: ni texto del agente ni
texto del transportista.

No mide nada nuevo. Reusa los tres módulos que ya produjeron las cifras
published_ones:

- `return_path.leer` — latencia observada, ida y vuelta de PoC1 y PoC2;
- `external_latency_poc3.call_turns` — PoC3 con el mismo instrumento;
  los turnos que ese instrumento deja sin número se recuperan con la regla del
  arnés (`recovered_turns`), que es la que ya mide PoC1 y PoC2;
- `round_trips` — qué intervenciones llevan una llamada a herramienta dentro.

Las poblaciones son las declaradas en `docs/populations_and_preregistrations.md` §1:
84 pares de `campana200` (índice ≤ 180, las dos corridas `ok`) y las 90
corridas de `campana_poc3`. Al terminar imprime una verificación contra las
cifras publicadas (606 / 585 / 592 intervenciones); si no cuadra, el CSV no se
debe usar hasta saber por qué.

Uso, desde la raíz del repo:

    uv run --project poc3_gpt_live --with scipy python -m shared.metrics.export_for_report

Salida en `docs/report/data/`: `intervenciones.csv`, `llamadas.csv`, `etapas.csv`,
`intervenciones_excluidas.csv` y `llamadas_excluidas.csv` (las corridas de PoC1 y
PoC2 que no entraron a los pares) y `verificacion.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.metrics import return_path as rp
from shared.metrics import external_latency_poc3 as ext3
from shared.carrier_simulator.player import MIN_SUBSTANTIVE_SPEECH_S
from shared.metrics import round_trips as rt

RAIZ: Final[Path] = Path(__file__).resolve().parents[2]
LOTE_DEFAULT: Final[Path] = RAIZ / "diagnostics" / "lote"
CLIPS_DEFAULT: Final[Path] = RAIZ / "shared" / "test_scenarios" / "clips"
DEFAULT_OUTPUT: Final[Path] = RAIZ / "docs" / "report" / "data"

# El corte de la poza de campana200: de la 181 en adelante murió la llave de
# ElevenLabs (`docs/metricas/campana200/README.md`).
TOPE_CAMPANA200: Final[int] = 180

# Lo publicado, para la verificación final (`docs/estado_para_el_reporte.md` §1).
EXPECTED: Final[dict[int, dict[str, float]]] = {
    1: {"calls": 84, "lines": 606, "p50": 2609.5, "with_rt": 0.573},
    2: {"calls": 84, "lines": 585, "p50": 3081.0, "with_rt": 0.586},
    3: {"calls": 90, "lines": 592, "p50": 2464.2, "with_rt": 0.297},
}

LINE_COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "turn", "clip_id",
    "observed_latency_ms", "first_sound_latency_ms", "ida_ms", "outbound_ms",
    "corrected_latency_ms", "with_round_trip", "fragmentation",
    # Solo PoC3: si el resultado de una herramienta llegó dentro de la ventana
    # de la intervención y, de ser así, si el agente ya estaba hablando.
    "tool_result", "spoke_before_result", "ms_speaking_without_result",
    # De dónde sale el número: `en_vivo` (PoC1 y PoC2, el arnés durante la
    # llamada), `instrumento_externo` (PoC3, `latencia_externa_poc3`) o
    # `recuperado` (PoC3, turnos que el instrumento externo dejó sin número y
    # que se miden con la regla del arnés; ver `turnos_recuperados`).
    "origin",
    # Si hubo algún evento de herramienta en TODA la respuesta: del fin de voz
    # al inicio de la línea siguiente. `con_round_trip` solo mira hasta la
    # primera voz del agente, y en PoC3 eso deja fuera las herramientas que
    # GPT-Live llama después de empezar a hablar.
    "tool_in_response",
    # Veces que el detector de fin de turno dio por terminada la línea (PoC1 y
    # PoC2). Ver `cierres_de_livekit`.
    "detector_closures",
    # PoC1 y PoC2: la misma latencia medida sobre la grabación con la regla de
    # los turnos recuperados de PoC3, para comprobar que la reconstrucción
    # reproduce la medición en vivo.
    "reconstructed_latency_ms",
]
# Una fila por muestra del pipeline que terminó en respuesta (PoC1 y PoC2). Es la
# materia prima del desglose por etapa; PoC3 no tiene etapas separables.
STAGE_COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "indice",
    "full_turn_ms", "end_of_turn_ms", "stt_ms", "llm_ttft_ms", "tts_ttfb_ms",
]
STAGE_FIELDS: Final[tuple[str, ...]] = (
    "full_turn_ms", "end_of_turn_ms", "stt_ms", "llm_ttft_ms", "tts_ttfb_ms",
)
CALL_COLUMNS: Final[list[str]] = [
    "poc", "clave", "scenario", "repetition", "estado", "duration_s",
    "heard_lines", "lines_with_latency", "fragmentation",
    "failed_criteria", "turns_with_discard", "agent_talked_over",
    "round_trips_llm_n", "cost_usd", "first_sound", "reason_if_omitted",
    # Turnos de PoC3 recuperados con la regla del arnés (vacío en PoC1 y PoC2).
    "recovered_turns",
    # Líneas del guion a media llamada que se quedaron sin respuesta con
    # contenido: sin número, y con algún turno posterior que sí lo tiene (el
    # agente siguió en la llamada). No cuenta el cierre. Igual en las 3 PoC.
    "turns_without_response",
    # Líneas que el arnés reprodujo (una marca por línea). Con las que tienen
    # número dan las que se quedaron sin latencia, también en las excluidas.
    "played_lines",
    # Suma de `cierres_del_detector` de la llamada (PoC1 y PoC2).
    "detector_closures",
]


def _no_mayor(primer: float | None, observed_value: float) -> float | None:
    """El primer sonido no puede llegar después de la respuesta sustantiva.

    Si llega (la respuesta se fechó en vivo y el primer sonido sobre el WAV, con
    ~2 ms de ancla y 20 ms de ventana de diferencia), manda la observada.
    """
    if primer is None:
        return None
    return round(min(primer, observed_value), 1)


def _primero(row: dict[str, Any], *keys: str) -> float | None:
    """El primer valor numérico entre `claves`, o `None`."""
    for clave in keys:
        valor = row.get(clave)
        if isinstance(valor, (int, float)):
            return float(valor)
    return None


# Tolerancia al buscar el primer sonido: la ventana de energía es de 20 ms y el
# ancla tiene ~2 ms de error típico, así que un arranque que cae hasta tres
# ventanas antes del fin de voz sigue siendo "después" dentro de la resolución.
FIRST_SOUND_SLACK_S: Final[float] = 0.060


def marks_from_lines(
    block: list[dict[str, Any]], clips: Path
) -> list[dict[str, Any]]:
    """Las marcas de un turno reconstruidas desde `por_intervencion`.

    PoC2 manda sus marcas por el canal de datos de LiveKit y no escribe
    `ceros_*.jsonl`. Cada intervención ya trae `end_of_speech` (época) y su
    `clip_id`, y el manifiesto del banco dice en qué segundo del clip termina la
    voz (`end_of_speech_s`), así que `start_of_clip = end_of_speech - end_of_speech_s`
    es exacto: es la misma medición con la que el arnés fechó el cero.
    """
    manifest = clips / "manifest.json"
    if not manifest.is_file():
        return []
    end_in_clip = {
        c["id"]: float(c["end_of_speech_s"])
        for c in json.loads(manifest.read_text(encoding="utf-8")).get("clips", [])
        if c.get("end_of_speech_s") is not None
    }
    marks = []
    for i in block:
        clip_id, end = i.get("clip_id"), i.get("end_of_speech")
        if not clip_id or end is None or clip_id not in end_in_clip:
            continue
        marks.append({
            "turn": i.get("turn"),
            "clip_id": clip_id,
            "start_of_clip": float(end) - end_in_clip[clip_id],
            "measured_end_of_speech": float(end),
        })
    return marks


def _a_8khz(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Lleva el WAV a 8 kHz, la tasa del banco, para que el ancla correlacione."""
    if sample_rate == 8000:
        return audio, sample_rate
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(8000, sample_rate)
    channel_list = [
        resample_poly(audio[:, k].astype(np.float64), 8000 // g, sample_rate // g)
        for k in range(audio.shape[1])
    ]
    return np.column_stack(channel_list), 8000


def first_sound_per_turn(
    marks: list[dict[str, Any]], wav: Path | None, clips: Path
) -> dict[str, float]:
    """Del fin de voz del clip al PRIMER sonido del agente, sin filtro alguno.

    La latencia publicada (`line_latency_ms`) mide hasta la primera
    racha **sustantiva**: descarta las de menos de 0.6 s y los apoyos verbales.
    Ésta no descarta nada: es lo que el transportista oye primero, aunque sea
    "un segundo…" o un "mm". Se mide igual en las tres PoC —mismo WAV de
    auditoría del arnés, misma ancla por correlación contra los clips, misma
    regla de rachas— así que las dos cifras se pueden comparar lado a lado.

    Devuelve `{turno: ms}`; vacío si falta el WAV o no se pudo anclar.
    """
    if not marks or wav is None or not wav.is_file():
        return {}
    audio, sample_rate = _a_8khz(*ext3.leer_wav(wav))
    anchor = ext3.anchor_on(marks, audio[:, 0], sample_rate, clips_path=clips)
    if anchor is None:
        return {}
    streaks = sorted(
        start + anchor.displacement
        for start, _ in ext3.speech_streaks(ext3.envelope(audio[:, 1], sample_rate))
    )
    ordered = sorted(marks, key=lambda m: m["measured_end_of_speech"])
    output: dict[str, float] = {}
    for k, mark in enumerate(ordered):
        cero = mark["measured_end_of_speech"]
        tope = (
            ordered[k + 1].get("start_of_clip", float("inf"))
            if k + 1 < len(ordered) else float("inf")
        )
        primero = next(
            (r for r in streaks if cero - FIRST_SOUND_SLACK_S <= r < tope), None
        )
        if primero is not None:
            output[str(mark["turn"])] = round((primero - cero) * 1000.0, 1)
    return output


@dataclass(frozen=True)
class RecoveredTurn:
    """Un turno de PoC3 medido con la regla del arnés, en época."""

    turn: str
    clip_id: str | None
    end_of_speech: float
    first_sound: float
    first_speech_heard: float
    """Inicio de la primera racha sustantiva: la misma definición que PoC1 y PoC2."""

    @property
    def line_latency_ms(self) -> float:
        return round((self.first_speech_heard - self.end_of_speech) * 1000.0, 1)

    @property
    def first_sound_latency_ms(self) -> float:
        return round((self.first_sound - self.end_of_speech) * 1000.0, 1)


def discarded_fragments(event_log: Path | None, displacement: float) -> list[tuple[float, float]]:
    """`(inicio, fin)` en época de lo que el arnés descartó en vivo como no-respuesta."""
    if event_log is None or not event_log.is_file():
        return []
    data = json.loads(event_log.read_text(encoding="utf-8"))
    output = []
    for turn in data.get("turns", []):
        for f in turn.get("discarded_fragments") or []:
            start = float(f["start"]) + displacement
            output.append((start, start + float(f.get("duration_ms") or 0.0) / 1000.0))
    return output


def recovered_turns(
    marks: list[dict[str, Any]],
    wav: Path | None,
    clips: Path,
    *,
    measured_items: set[str],
    event_log: Path | None,
) -> list[RecoveredTurn]:
    """Los turnos que el instrumento externo de PoC3 dejó sin número, medidos por el arnés.

    El instrumento externo necesita una muestra del puente por turno y solo busca
    la respuesta en los 2 s que siguen al acuse del puente. Eso deja fuera dos
    clases de turno que PoC1 y PoC2 sí miden: la respuesta más rápida que los
    800 ms con que el puente cierra el turno del transportista, y el relleno
    ("mm, déjame ver") seguido de más de 2 s de pausa antes del contenido.

    Aquí se miden con la regla con que el arnés midió a PoC1 y PoC2 en vivo:
    la primera racha de al menos `MIN_SUBSTANTIVE_SPEECH_S` después del fin
    de voz, sin contar lo que el propio arnés descartó en vivo (apoyo verbal o
    ruido corto, en la bitácora). El primer sonido es la primera racha, sin
    filtro. Todo sobre el WAV de auditoría, con la misma ancla y la misma
    envolvente que `first_sound_per_turn`.
    """
    if not marks or wav is None or not wav.is_file():
        return []
    audio, sample_rate = _a_8khz(*ext3.leer_wav(wav))
    anchor = ext3.anchor_on(marks, audio[:, 0], sample_rate, clips_path=clips)
    if anchor is None:
        return []
    shift = anchor.displacement
    env = ext3.envelope(audio[:, 1], sample_rate)
    all_items = [(i + shift, f + shift) for i, f in ext3.speech_streaks(env)]
    # Lo descartado en vivo se borra de la envolvente (una ventana de margen por
    # lado) y las rachas se vuelven a armar: lo que queda es lo que el arnés
    # habría podido aceptar como respuesta.
    clean = env.copy()
    for start, end in discarded_fragments(event_log, shift):
        a = max(0, int(np.floor((start - shift) / ext3.WINDOW_S)) - 1)
        b = int(np.ceil((end - shift) / ext3.WINDOW_S)) + 1
        clean[a:b] = 0.0
    substantive_ones = [
        (i + shift, f + shift) for i, f in ext3.speech_streaks(clean)
        if (f - i) >= MIN_SUBSTANTIVE_SPEECH_S
    ]
    ordered = sorted(marks, key=lambda m: m["measured_end_of_speech"])
    output: list[RecoveredTurn] = []
    for k, mark in enumerate(ordered):
        turn = str(mark["turn"])
        if turn in measured_items:
            continue
        cero = float(mark["measured_end_of_speech"])
        tope = (
            float(ordered[k + 1].get("start_of_clip", float("inf")))
            if k + 1 < len(ordered) else float("inf")
        )
        in_window = lambda r: cero - FIRST_SOUND_SLACK_S <= r[0] < tope  # noqa: E731
        primero = next((r for r in all_items if in_window(r)), None)
        substantive = next((r for r in substantive_ones if in_window(r)), None)
        if primero is None or substantive is None:
            continue
        output.append(RecoveredTurn(
            turn=turn, clip_id=mark.get("clip_id"), end_of_speech=cero,
            first_sound=min(primero[0], substantive[0]),
            first_speech_heard=substantive[0],
        ))
    return output


def no_response_midcall(turns: list[str], with_number: set[str]) -> int:
    """Turnos sin número seguidos de alguno que sí lo tiene, en orden del guion.

    Si después de un turno mudo el agente vuelve a contestar, no era el cierre
    de la llamada: se quedó sin responder a esa línea.
    """
    ultimo = max((k for k, t in enumerate(turns) if t in with_number), default=-1)
    return sum(1 for k, t in enumerate(turns) if k < ultimo and t not in with_number)


def _marks_of(zeros: Path | None) -> list[dict[str, Any]]:
    return ext3.marks_de(zeros) if zeros is not None and zeros.is_file() else []


def read_progress(folder: Path) -> list[dict[str, Any]]:
    ruta = folder / "progress.jsonl"
    if not ruta.is_file():
        raise SystemExit(f"no encuentro {ruta}")
    return [
        json.loads(l)
        for l in ruta.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def read_summary(folder: Path, clave: str) -> dict[str, Any] | None:
    ruta = folder / "summaries" / f"{clave}.json"
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def first_sound_state(
    marks: list[dict[str, Any]], wav: Path | None, primer: dict[str, float]
) -> str:
    """Por qué una llamada tiene o no la latencia al primer sonido."""
    if wav is None or not wav.is_file():
        return "sin WAV de auditoría"
    if not marks:
        return "sin marcas de tiempo"
    if not primer:
        return "no se pudo anclar"
    return "ok"


def call_row(
    row: dict[str, Any], summary: dict[str, Any] | None, motivo: str = "", first_sound: str = "",
    *, recovered_items: int | None = None, sin_respuesta: int | None = None,
    text_lines: int | None = None, closures: int | None = None,
) -> dict[str, Any]:
    r = summary or {}
    inter = r.get("lines") or {}
    return {
        "poc": row.get("poc"),
        "clave": row.get("clave"),
        "scenario": row.get("scenario"),
        "repetition": row.get("repetition"),
        "estado": row.get("estado"),
        "duration_s": r.get("duration_s", row.get("duration_s")),
        "heard_lines": inter.get("heard_lines"),
        "lines_with_latency": inter.get("with_latency"),
        "fragmentation": r.get("fragmentation"),
        "failed_criteria": r.get("failed_criteria"),
        "turns_with_discard": len(r.get("turns_with_discard") or []),
        "agent_talked_over": r.get("agent_interruptions"),
        "round_trips_llm_n": (r.get("llm_round_trips") or {}).get("n"),
        "cost_usd": r.get("cost_usd"),
        "first_sound": first_sound,
        "reason_if_omitted": motivo,
        "recovered_turns": recovered_items,
        "turns_without_response": sin_respuesta,
        "played_lines": text_lines,
        "detector_closures": closures,
    }


def campaign200_pairs(
    rows: list[dict[str, Any]], valid_ones: set[tuple[int, int]] | None = None
) -> list[dict[str, Any]]:
    """Las corridas de los pares: índice ≤ 180 y las dos `ok`.

    Con `validas`, además, solo los escenarios y repeticiones que pasaron en las
    tres PoC (`campanas.valid_combinations`).
    """
    per_cell: dict[tuple[Any, Any], dict[int, dict[str, Any]]] = {}
    for f in rows:
        if int(f.get("indice", 0)) > TOPE_CAMPANA200 or f.get("estado") != "ok":
            continue
        if valid_ones is not None and (int(f["scenario"]), int(f["repetition"])) not in valid_ones:
            continue
        per_cell.setdefault((f["scenario"], f["repetition"]), {})[int(f["poc"])] = f
    return [
        run
        for cell in per_cell.values()
        if set(cell) == {1, 2}
        for run in cell.values()
    ]


def excluded_from_campaign200(
    rows: list[dict[str, Any]], valid_ones: set[tuple[int, int]] | None = None
) -> list[dict[str, Any]]:
    """Las corridas de las primeras 180 que no entraron a los pares.

    Las que fallaron algún criterio y su pareja (regla de pareja). Sirven para
    el análisis de sensibilidad con las llamadas excluidas.
    """
    in_pairs = {f["clave"] for f in campaign200_pairs(rows, valid_ones)}
    return [
        f for f in rows
        if int(f.get("indice", 0)) <= TOPE_CAMPANA200 and f["clave"] not in in_pairs
    ]


def next_clip(marks: list[dict[str, Any]]) -> dict[str, float]:
    """`{turno: inicio de la línea siguiente}`: el fin de la ventana de la respuesta."""
    ordered = sorted(marks, key=lambda m: m["measured_end_of_speech"])
    return {
        str(m["turn"]): (
            float(ordered[k + 1].get("start_of_clip", float("inf")))
            if k + 1 < len(ordered) else float("inf")
        )
        for k, m in enumerate(ordered)
    }


def livekit_closures(worker_json: Path, marks: list[dict[str, Any]]) -> dict[str, int]:
    """Cuántas veces LiveKit dio por terminada cada línea del transportista.

    El bloque `por_intervencion` de PoC2 junta en un solo cero todos los
    mensajes del usuario que preceden a una respuesta, así que su
    `eventos_del_pipeline` vale 1 siempre. El evento comparable con el cierre
    del analizador de Pipecat (`UserStoppedSpeakingFrame`) es cada turno del
    usuario que LiveKit confirmó: cada entrada de `fragmentos_usuario`. Se
    asigna a la línea en cuya ventana (del inicio del clip al inicio del
    siguiente) cae su `stopped_speaking_at`.
    """
    data = json.loads(worker_json.read_text(encoding="utf-8"))
    windows = []
    tope = next_clip(marks)
    for m in marks:
        windows.append((str(m["turn"]), float(m["start_of_clip"]), tope[str(m["turn"])]))
    output = {t: 0 for t, _, _ in windows}
    for turn in (data.get("livekit_extra") or {}).get("per_turn") or []:
        if turn.get("is_opening"):
            continue
        for f in turn.get("user_fragments") or []:
            end = f.get("stopped_speaking_at") if isinstance(f, dict) else None
            if not isinstance(end, (int, float)):
                continue
            for t, a, b in windows:
                if a <= end < b:
                    output[t] += 1
                    break
    return output


# Lo que se le da a una llamada de PoC2 antes de su primera línea y después de
# la última al buscar sus herramientas en el registro de la campaña. Entre dos
# llamadas de PoC2 corre siempre una de PoC1, de un par de minutos.
CALL_MARGIN_S: Final[float] = 30.0


@dataclass(frozen=True)
class _Window:
    turn: str
    end_of_speech: float
    first_speech_heard: float


def export_run_12(
    row: dict[str, Any], folder: Path, clips: Path, *, offset: float,
    poc2_calls: list[rt.FunctionCall],
) -> tuple[list[dict], dict, list[str], list[dict]]:
    """Una corrida de PoC1 o PoC2: sus intervenciones, su fila de llamada, avisos y etapas."""
    clave = str(row["clave"])
    poc = int(row["poc"])
    summary = read_summary(folder, clave)
    art = (summary or {}).get("artifacts") or {}
    source = art.get("zero_crossing") if poc == 1 else art.get("worker_json")
    if summary is None or not source or not Path(source).is_file():
        return [], call_row(row, summary, "falta el resumen o su artefacto"), [
            f"{clave}: falta el resumen o su artefacto"], []
    lines: list[dict] = []
    stages: list[dict] = []
    try:
        run = rp.leer(Path(source)) or {}
        inbound_of = {str(i["turn"]): i.get("ida_ms") for i in run.get("per_line") or []}
        outbound_of = {str(v["turn"]): v["return_path_ms"] for v in run.get("detail") or []}
        if poc == 1:
            functions = rt.poc1_calls(folder / "calls" / f"{clave}.log", offset_s=offset)
        else:
            functions = poc2_calls
        dates = [f.when for f in functions]
        label = summary.get("label") or row.get("label")
        candidates = [
            Path(source).with_name(f"zeros_{label}.jsonl"),
            *( [Path(art["event_log"]).with_name(f"zeros_{label}.jsonl")] if art.get("event_log") else [] ),
        ]
        zeros = next((c for c in candidates if c.is_file()), None)
        wav = Path(art["auditoria_wav"]) if art.get("auditoria_wav") else None
        event_log = Path(art["event_log"]) if art.get("event_log") else None
        block = (summary.get("lines") or {}).get("per_line") or []
        marks = _marks_of(zeros) or marks_from_lines(block, clips)
        primer = first_sound_per_turn(marks, wav, clips)
        tope = next_clip(marks)
        # El registro de herramientas de PoC2 es uno para toda la campaña: se
        # queda con lo de esta llamada, para que la ventana de la última línea
        # (que no tiene línea siguiente) no alcance la llamada que sigue.
        if marks:
            since = min(float(m["start_of_clip"]) for m in marks) - CALL_MARGIN_S
            until = max(float(m["measured_end_of_speech"]) for m in marks) + CALL_MARGIN_S
            functions = [f for f in functions if since <= f.when <= until]
            dates = [f.when for f in functions]
        closures = (
            {str(i.get("turn", "")): _primero(i, "pipeline_events", "fragmentation") for i in block}
            if poc == 1 else livekit_closures(Path(source), marks)
        )
        reconstructed_items = {
            r.turn: r.line_latency_ms
            for r in recovered_turns(marks, wav, clips, measured_items=set(), event_log=event_log)
        }
        # En PoC1 y PoC2 la función corre en el mismo proceso que el agente: la
        # hora del registro (`Calling function` en Pipecat, la línea `tool …`
        # del worker de LiveKit) es también la hora de su resultado.
        windows = [
            _Window(str(i.get("turn", "")), float(i["end_of_speech"]), float(i["agent_first_speech"]))
            for i in block
            if i.get("end_of_speech") is not None and i.get("agent_first_speech") is not None
        ]
        with_data_point = [f.when for f in functions if f.name in rt.FUNCTIONS_WITH_DATA]
        tool = results_per_turn(windows, marks, with_data_point) if dates else {}
        call_json = art.get("pipeline_json") or art.get("worker_json")
        if call_json and Path(call_json).is_file():
            stages.extend(call_stages(Path(call_json), row))
        for i in block:
            # PoC1 y PoC2 reescriben `por_intervencion` con sus propias
            # llaves (`marcas.bloque_de_intervenciones`); el cruce
            # compartido usa `latencia_ms`. Se aceptan las dos.
            obs = _primero(i, "line_latency_ms", "latency_ms")
            if obs is None:
                continue
            turn = str(i.get("turn", ""))
            inbound = _primero(i, "speech_bias_ms")
            if inbound is None:
                inbound = inbound_of.get(turn)
            outbound = outbound_of.get(turn)
            start, end = i.get("end_of_speech"), i.get("agent_first_speech")
            with_rt = (
                None if start is None or end is None
                else int(any(start < t <= end for t in dates))
            )
            occurred, spoke, ms_without = tool.get(turn, (None, None, None))
            lines.append({
                "poc": poc, "clave": clave,
                "scenario": row["scenario"], "repetition": row["repetition"],
                "turn": turn, "clip_id": i.get("clip_id"),
                "observed_latency_ms": obs,
                "first_sound_latency_ms": _no_mayor(primer.get(turn), obs),
                "ida_ms": inbound, "outbound_ms": outbound,
                "corrected_latency_ms": (
                    round(obs - inbound - outbound, 1)
                    if inbound is not None and outbound is not None else None
                ),
                "with_round_trip": with_rt,
                "fragmentation": _primero(i, "pipeline_events", "fragmentation"),
                "tool_result": occurred,
                "spoke_before_result": spoke,
                "ms_speaking_without_result": ms_without,
                "origin": "en_vivo",
                "tool_in_response": (
                    None if start is None
                    else int(any(start < t < tope.get(turn, float("inf")) for t in dates))
                ),
                "detector_closures": closures.get(turn),
                "reconstructed_latency_ms": reconstructed_items.get(turn),
            })
        order = [str(i.get("turn", "")) for i in block]
        with_number = {
            str(i.get("turn", "")) for i in block
            if _primero(i, "line_latency_ms", "latency_ms") is not None
        }
        closings_total = [c for c in closures.values() if c is not None]
        call = call_row(
            row, summary, first_sound=first_sound_state(marks, wav, primer),
            sin_respuesta=no_response_midcall(order, with_number),
            text_lines=len(marks), closures=int(sum(closings_total)) if closings_total else None,
        )
        return lines, call, [], stages
    except Exception as exc:  # una corrida rota no tumba la exportación
        return [], call_row(row, summary, f"error: {exc}"), [
            f"{clave}: {exc}\n{traceback.format_exc(limit=2)}"], []


def call_stages(ruta_json: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Las muestras de latencia por etapa que publicó el pipeline, sin la apertura.

    Solo las que terminaron en respuesta (`full_turn_ms` con número): con
    `stop_secs = 0.2` PoC1 cierra casi dos turnos por intervención, y los que el
    framework absorbió sin contestar no tienen LLM ni TTS que medir.
    """
    data = json.loads(ruta_json.read_text(encoding="utf-8"))
    output = []
    for m in data.get("latencies") or []:
        if not isinstance(m, dict) or m.get("is_opening"):
            continue
        if not isinstance(m.get("full_turn_ms"), (int, float)):
            continue
        output.append({
            "poc": row.get("poc"), "clave": row.get("clave"),
            "scenario": row.get("scenario"), "repetition": row.get("repetition"),
            "indice": m.get("indice"),
            **{c: _primero(m, c) for c in STAGE_FIELDS},
        })
    return output


def export_campaign200(
    folder: Path, clips: Path, *, excluded_calls: bool = False,
    valid_ones: set[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[dict], list[str], list[dict]]:
    """Los 84 pares, o —con `excluidas`— las corridas de las primeras 180 que no entraron."""
    rows = read_progress(folder)
    worker_log = folder / "worker.log"
    offset = rt.time_offset(worker_log)
    poc2_calls = rt.worker_calls(worker_log)

    lines: list[dict] = []
    calls: list[dict] = []
    warning_list: list[str] = []
    stages: list[dict] = []
    runs = (excluded_from_campaign200(rows, valid_ones) if excluded_calls
                else campaign200_pairs(rows, valid_ones))
    for row in runs:
        i, l, a, e = export_run_12(
            row, folder, clips, offset=offset, poc2_calls=poc2_calls,
        )
        lines += i
        calls.append(l)
        warning_list += a
        stages += e
    return lines, calls, warning_list, stages


def results_per_turn(
    turns: list[Any], marks: list[dict[str, Any]], results: list[float]
) -> dict[str, tuple[int, int | None, float | None]]:
    """Si el resultado de una herramienta llegó en la ventana de cada intervención.

    La ventana va del fin de voz del clip al inicio del clip siguiente. Si llegó
    alguno, se mira el primero: si llegó **después** de que el agente empezó a
    hablar, el agente habló sin tenerlo, y se registra cuánto tiempo
    (`docs/hallazgo_round_trip.md` §3.4). Devuelve `{turno: (hubo, hablo, ms)}`.
    """
    ordered = sorted(marks, key=lambda m: m["measured_end_of_speech"])
    following = {
        str(m["turn"]): (
            ordered[k + 1].get("start_of_clip", float("inf"))
            if k + 1 < len(ordered) else float("inf")
        )
        for k, m in enumerate(ordered)
    }
    output: dict[str, tuple[int, int | None, float | None]] = {}
    for t in turns:
        tope = following.get(str(t.turn), float("inf"))
        in_window = [r for r in results if t.end_of_speech < r < tope]
        if not in_window:
            output[str(t.turn)] = (0, None, None)
            continue
        primero = min(in_window)
        spoke = primero > t.first_speech_heard
        output[str(t.turn)] = (
            1, int(spoke), round((primero - t.first_speech_heard) * 1000.0, 1) if spoke else 0.0
        )
    return output


def export_campaign_poc3(
    folder: Path | None, clips: Path, *, runs: list[tuple[Path, dict]] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Las corridas de PoC3: las de `carpeta`, o los pares `(carpeta, fila)` de `corridas`.

    Una repetición trae en `clave` su clave de análisis (`…-bis`) y en
    `clave_del_lote` la de sus archivos.
    """
    lines: list[dict] = []
    calls: list[dict] = []
    warning_list: list[str] = []
    if runs is None:
        runs = [(folder, f) for f in read_progress(folder)]
    for folder, row in runs:
        if int(row.get("poc", 3)) != 3:
            continue
        clave = str(row["clave"])
        sid = row.get("label")
        summary = read_summary(folder, str(row.get("batch_key", clave)))
        art = (summary or {}).get("artifacts") or {}
        event_log = Path(art["event_log"]) if art.get("event_log") else None
        live_bridge = Path(art["bridge_json"]) if art.get("bridge_json") else None
        census = Path(art["census"]) if art.get("census") else None
        harness = event_log.parent if event_log else None
        wav = Path(art["auditoria_wav"]) if art.get("auditoria_wav") else (
            harness / f"auditoria_{sid}.wav" if harness else None
        )
        zeros = harness / f"zeros_{sid}.jsonl" if harness else None
        missing_items = [
            n for n, r in (("live_bridge", live_bridge), ("wav", wav), ("zeros", zeros))
            if r is None or not r.is_file()
        ]
        if not sid or summary is None or missing_items:
            motivo = f"missing_items: {', '.join(missing_items) or 'summary'}"
            calls.append(call_row(row, summary, motivo))
            warning_list.append(f"{clave}: {motivo}")
            continue
        try:
            turns, _, report = ext3.call_turns(
                call_sid=sid,
                scenario=row["scenario"],
                zeros_path=zeros,
                bridge_json_path=live_bridge,
                wav_path=wav,
                path_of_log=event_log,
                clips_path=clips,
            )
            marks = _marks_of(zeros)
            primer = first_sound_per_turn(marks, wav, clips)
            dates: list[float] = []
            results: list[float] = []
            if census is not None and census.is_file():
                data = json.loads(live_bridge.read_text(encoding="utf-8"))
                anchor = (data.get("clock") or {}).get("anchor_wall_s")
                if anchor is not None:
                    events = rt.poc3_calls(census, float(anchor))
                    dates = [f.when for f in events]
                    # Solo los resultados que la respuesta necesita (ver
                    # `rt.FUNCIONES_CON_DATO`).
                    results = [
                        f.when for f in events
                        if f.is_result and f.function in rt.FUNCTIONS_WITH_DATA
                    ]
            recovered_items = recovered_turns(
                marks, wav, clips, measured_items={str(t.turn) for t in turns}, event_log=event_log,
            ) if turns else []
            acknowledgements = {
                str(m["turn"]): mu["t_ack_wall"]
                for m, mu in ext3.pair(
                    marks, ext3.axis_samples(json.loads(live_bridge.read_text(encoding="utf-8")))
                )
            }
            tool = results_per_turn([*turns, *recovered_items], marks, results) if dates else {}
            tope = next_clip(marks)

            def in_the_response(t: Any) -> int | None:
                if not dates:
                    return None
                end = tope.get(str(t.turn), float("inf"))
                return int(any(t.end_of_speech < c < end for c in dates))
            for t in turns:
                occurred, spoke, ms_without = tool.get(str(t.turn), (None, None, None))
                lines.append({
                    "poc": 3, "clave": clave,
                    "scenario": row["scenario"], "repetition": row["repetition"],
                    "turn": t.turn, "clip_id": t.clip_id,
                    "observed_latency_ms": t.line_latency_ms,
                    "first_sound_latency_ms": _no_mayor(
                        primer.get(str(t.turn)), t.line_latency_ms
                    ),
                    # PoC3 no publica `sesgo_de_voz`: la ida no tiene medida
                    # equivalente, y por eso no hay cifra corregida.
                    "ida_ms": None,
                    "outbound_ms": round(t.return_path_ms, 1),
                    "corrected_latency_ms": None,
                    "with_round_trip": (
                        int(any(t.end_of_speech < c <= t.first_speech_heard for c in dates))
                        if dates else None
                    ),
                    "fragmentation": None,
                    "tool_result": occurred,
                    "spoke_before_result": spoke,
                    "ms_speaking_without_result": ms_without,
                    "origin": "instrumento_externo",
                    "tool_in_response": in_the_response(t),
                })
            for t in recovered_items:
                occurred, spoke, ms_without = tool.get(str(t.turn), (None, None, None))
                acknowledgement = acknowledgements.get(str(t.turn))
                # El camino de vuelta solo vale si el acuse del puente es el de
                # este primer sonido; si el puente no dio muestra, no hay.
                outbound = (
                    round((t.first_sound - acknowledgement) * 1000.0, 1)
                    if acknowledgement is not None
                    and -ext3.ACKNOWLEDGEMENT_SLACK_S <= t.first_sound - acknowledgement
                    <= ext3.RETURN_PATH_MAX_S
                    else None
                )
                lines.append({
                    "poc": 3, "clave": clave,
                    "scenario": row["scenario"], "repetition": row["repetition"],
                    "turn": t.turn, "clip_id": t.clip_id,
                    "observed_latency_ms": t.line_latency_ms,
                    "first_sound_latency_ms": t.first_sound_latency_ms,
                    "ida_ms": None,
                    "outbound_ms": outbound,
                    "corrected_latency_ms": None,
                    "with_round_trip": (
                        int(any(t.end_of_speech < c <= t.first_speech_heard for c in dates))
                        if dates else None
                    ),
                    "fragmentation": None,
                    "tool_result": occurred,
                    "spoke_before_result": spoke,
                    "ms_speaking_without_result": ms_without,
                    "origin": "recuperado",
                    "tool_in_response": in_the_response(t),
                })
            order = [str(m["turn"]) for m in sorted(marks, key=lambda m: m["measured_end_of_speech"])]
            with_number = {str(t.turn) for t in turns} | {t.turn for t in recovered_items}
            motivo = "" if turns else f"sin turnos: {report.get('motivo', '')}"
            calls.append(call_row(
                row, summary, motivo, first_sound=first_sound_state(marks, wav, primer),
                recovered_items=len(recovered_items),
                sin_respuesta=no_response_midcall(order, with_number) if turns else None,
                text_lines=len(marks),
            ))
        except Exception as exc:
            calls.append(call_row(row, summary, f"error: {exc}"))
            warning_list.append(f"{clave}: {exc}\n{traceback.format_exc(limit=2)}")
    return lines, calls, warning_list


def verify(lines: list[dict], calls: list[dict] | None = None) -> list[str]:
    text_lines = ["poc  llamadas(esp)  intervenciones(esp)  p50_ms(esp)  with_rt(esp)  veredicto"]
    # La verificación es contra lo publicado, que no traía los recuperados.
    published_ones = [i for i in lines if i.get("origin") != "recuperado"]
    for poc, exp in EXPECTED.items():
        lat = [i["observed_latency_ms"] for i in published_ones if i["poc"] == poc]
        rts = [i["with_round_trip"] for i in published_ones if i["poc"] == poc and i["with_round_trip"] is not None]
        n_calls = len({i["clave"] for i in published_ones if i["poc"] == poc})
        p50 = float(np.percentile(lat, 50)) if lat else float("nan")
        frac = (sum(rts) / len(rts)) if rts else float("nan")
        cuadra = (
            n_calls == exp["calls"]
            and len(lat) == exp["lines"]
            and abs(p50 - exp["p50"]) <= 1.0
            and abs(frac - exp["with_rt"]) <= 0.005
        )
        text_lines.append(
            f"{poc:<4} {n_calls:>4} ({exp['calls']:.0f})     "
            f"{len(lat):>5} ({exp['lines']:.0f})        "
            f"{p50:>8.1f} ({exp['p50']})  {frac:.3f} ({exp['with_rt']})  "
            f"{'CUADRA' if cuadra else 'NO CUADRA'}"
        )
    text_lines += ["", "primer sonido (sin filtrar) contra la respuesta sustantiva:",
               "poc  with_data_point/lines  adelanto_p50_ms  con_adelanto>60ms"]
    for poc in EXPECTED:
        rows = [i for i in lines if i["poc"] == poc]
        pairs = [
            (i["observed_latency_ms"], i["first_sound_latency_ms"])
            for i in rows if i.get("first_sound_latency_ms") is not None
        ]
        leads = [o - p for o, p in pairs]
        p50 = float(np.percentile(leads, 50)) if leads else float("nan")
        large = sum(1 for a in leads if a > 60.0)
        text_lines.append(f"{poc:<4} {len(pairs):>5}/{len(rows):<5}             {p50:>8.1f}         {large}")
    recovered_ones = [i for i in lines if i.get("origin") == "recuperado"]
    text_lines += [
        "",
        f"turnos de PoC3 recuperados con la regla del arnés: {len(recovered_ones)}",
    ]
    if recovered_ones:
        v = sorted(i["observed_latency_ms"] for i in recovered_ones)
        p = sorted(i["first_sound_latency_ms"] for i in recovered_ones)
        text_lines.append(
            f"  contenido p50 {np.percentile(v, 50):.1f} ms (mín {v[0]:.1f}, máx {v[-1]:.1f})"
            f" · primer sonido p50 {np.percentile(p, 50):.1f} ms"
        )
    if calls:
        text_lines += ["", "turnos sin respuesta a media llamada (no cuenta el cierre):"]
        for poc in EXPECTED:
            of = [l for l in calls if l["poc"] == poc and l.get("turns_without_response") is not None]
            n = sum(int(l["turns_without_response"]) for l in of)
            having = sum(1 for l in of if int(l["turns_without_response"]) > 0)
            text_lines.append(f"{poc:<4} {n} turnos en {having} de {len(of)} llamadas")
        text_lines += ["", "estado del primer sonido por llamada:"]
        for poc in EXPECTED:
            count: dict[str, int] = {}
            for l in calls:
                if l["poc"] == poc:
                    clave = l.get("first_sound") or "no se intentó"
                    count[clave] = count.get(clave, 0) + 1
            text_lines.append(f"{poc:<4} " + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())))
    return text_lines


def write_csv(ruta: Path, columns: list[str], rows: list[dict]) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT,
                   help="carpeta que contiene campana200/ y campana_poc3/")
    p.add_argument("--clips", type=Path, default=CLIPS_DEFAULT,
                   help="carpeta con los .ulaw del banco (para anclar PoC3)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    from shared.metrics.campaigns import (
        CAMPAIGN_POC3, CAMPAIGN_POC3_RETRIES, valid_combinations, poc3_for_analysis,
    )

    # Regla de las tres PoC: entra el escenario y repetición que pasó en las tres.
    valid_ones = valid_combinations(args.lote)
    i12, l12, a12, stages = export_campaign200(args.lote / "campana200", args.clips, valid_ones=valid_ones)
    # PoC3: el primer intento, o su repetición si falló (enmienda del 9 de
    # octubre). Los primeros intentos reemplazados van con las excluidas.
    runs3 = poc3_for_analysis(args.lote)
    repeated = {
        (int(f["scenario"]), int(f["repetition"]))
        for c, f in runs3 if c.name == CAMPAIGN_POC3_RETRIES
    }
    replaced = [
        (args.lote / CAMPAIGN_POC3, f) for f in read_progress(args.lote / CAMPAIGN_POC3)
        if f.get("estado") != "ok" and (int(f["scenario"]), int(f["repetition"])) in repeated
    ]
    i3, l3, a3 = export_campaign_poc3(None, args.clips, runs=runs3)
    ir3, lr3, ar3 = export_campaign_poc3(None, args.clips, runs=replaced)
    a3 += ar3
    ix, lx, ax, _ = export_campaign200(args.lote / "campana200", args.clips, excluded_calls=True, valid_ones=valid_ones)
    # Las de PoC3 que pasaron sus criterios pero cuyo escenario y repetición
    # falló en PoC1 o PoC2 van con las excluidas, como las de PoC1 y PoC2.
    outside = {
        str(l["clave"]) for l in l3
        if l.get("estado") == "ok" and (int(l["scenario"]), int(l["repetition"])) not in valid_ones
    }
    ix += ir3
    lx += lr3
    ix += [i for i in i3 if str(i["clave"]) in outside]
    lx += [l for l in l3 if str(l["clave"]) in outside]
    i3 = [i for i in i3 if str(i["clave"]) not in outside]
    l3 = [l for l in l3 if str(l["clave"]) not in outside]
    lines, calls, warning_list = i12 + i3, l12 + l3, a12 + a3 + ax

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "lines.csv", LINE_COLUMNS, lines)
    write_csv(args.output / "calls.csv", CALL_COLUMNS, calls)
    write_csv(args.output / "stages.csv", STAGE_COLUMNS, stages)
    # Las corridas de PoC1 y PoC2 que no entraron a los pares (las de PoC3 ya
    # van en los archivos de arriba, con su estado en `poblaciones.csv`).
    write_csv(args.output / "excluded_lines.csv", LINE_COLUMNS, ix)
    write_csv(args.output / "excluded_calls.csv", CALL_COLUMNS, lx)
    verification = verify(lines, calls)
    text = "\n".join(verification + ["", f"warning_list: {len(warning_list)}", *warning_list])
    (args.output / "verification.txt").write_text(text + "\n", encoding="utf-8")

    print("\n".join(verification))
    print(f"\n{len(lines)} intervenciones, {len(calls)} llamadas, {len(warning_list)} avisos")
    print(f"escrito en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
