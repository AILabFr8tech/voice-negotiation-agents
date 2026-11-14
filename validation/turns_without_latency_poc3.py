"""The PoC3 turns that ended up without a latency, extracted to a folder for listening.

PoC3 latency is measured with the same external instrument as PoC1 and PoC2
(`shared.metrics.external_latency_poc3`), but each turn also needs a sample
of the **bridge's** wall axis. The bridge does not consider the carrier's turn
finished until 800 ms of silence, so when the agent starts earlier, that turn
comes out with no sample and no number. The validation with the fake model
showed that these are precisely the fast-response turns: at 300 ms none was
measured and at 1,000 ms almost half were lost, even though the harness did
hear the response in all of them.

This script changes no figure. For every call in the PoC3 campaign it finds
the turns without a latency and, for each one:

1. Cuts the audio from the harness audit recording, in stereo:
   **carrier on the left, agent on the right**. The cut starts a little
   before the carrier's line (to hear what the agent was doing when the
   harness started it) and ends when the next line starts, or where the
   recording ends if it is the last turn. That way the agent's response is
   complete and is not cut by the script.
2. Computes two figures with the harness rule, without asking the bridge for
   anything: the agent's **first sound** and the **first substantive streak**
   (at least 0.6 s, not discarded live as a backchannel). It is the same
   definition PoC1 and PoC2 use.
3. Records why it was lost: the bridge gave no sample for that turn, or it
   did but no agent streak fell next to its acknowledgement.

Writes to `--output` the `audio_files/` folder, an `indice.csv` with one row
per turn and blank columns to annotate what is heard, and a `LEEME.txt`.

Runs on the test PC, which is where the recordings are:

    uv run --project poc3_gpt_live --with scipy python -m validation.turns_without_latency_poc3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path
from statistics import median
from typing import Any, Final

import numpy as np

from shared.carrier_simulator.player import MIN_SUBSTANTIVE_SPEECH_S
from shared.metrics import external_latency_poc3 as ext3
from shared.metrics.export_for_report import (
    CLIPS_DEFAULT,
    FIRST_SOUND_SLACK_S,
    LOTE_DEFAULT,
    _a_8khz,
    read_progress,
    read_summary,
)

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = RAIZ / "diagnostics" / "turns_without_latency_poc3"

# The cut starts at whichever is earlier: `ANTES_SEG` before the end of speech,
# or `BEFORE_CLIP_S` before the carrier's line starts (to hear whether the
# agent was still talking when the harness played it).
ANTES_SEG: Final[float] = 4.0
BEFORE_CLIP_S: Final[float] = 3.0
# It ends `AFTER_THE_NEXT_S` after the next line starts; on the last turn,
# where the recording ends. Never beyond `AFTER_CAP_S` from the end of speech
# (the harness safety cap is 25 s).
AFTER_THE_NEXT_S: Final[float] = 1.0
AFTER_CAP_S: Final[float] = 30.0
# A last turn whose recording ends before this with no agent speech is the
# close of the call, not a missing response.
COLA_MINIMA_SEG: Final[float] = 3.0

# An agent streak that was discarded live falls at most this far from the one
# seen in the WAV (the same tolerance the external instrument uses).
DISCARD_TOLERANCE_S: Final[float] = 0.10

COLUMNS: Final[list[str]] = [
    "file", "clave", "scenario", "repetition", "turn", "clip_id",
    "carrier_line", "motivo",
    "first_sound_latency_ms", "first_sound_duration_ms",
    "harness_rule_latency_ms", "agent_audible_at_end_of_speech",
    "end_of_speech_second_in_audio", "recording_seconds_after_end_of_speech",
    "last_turn", "agent_was_speaking_at_clip_start", "agent_pause_before_clip_ms",
    # To be filled in while listening to the audio.
    "what_the_agent_said", "tipo", "notes",
]

LEEME: Final[str] = f"""\
Turnos de PoC3 sin latencia
===========================

Cada archivo de audios/ es un turno de la campaña de PoC3 que no tiene número de
latencia. Están en estéreo:

  - izquierda: el transportista (el clip que reprodujo el arnés)
  - derecha:   el agente

El transportista termina de hablar en el segundo que dice la columna
`segundo_del_fin_de_voz_en_el_audio`. Lo que suena después en el canal derecho
es la respuesta del agente. Cada audio termina cuando arranca la línea
siguiente del transportista (o donde termina la grabación, en el último turno),
así que la respuesta queda completa.

Qué anotar en indice.csv, al oír cada archivo:

  que_dijo_el_agente  lo que dice el agente al arrancar, tal cual se oye
  tipo                una de estas:
                        respuesta  contesta con contenido desde el principio
                        relleno    abre con "déjame ver", "un segundo", "mm",
                                   "ajá"... y después contesta
                        encima     arranca antes de que el transportista termine
                        nada       no se oye respuesta del agente
                        otro       cualquier otra cosa (explicar en notas)
  notas               lo que haga falta

Columnas que ya vienen llenas (las calcula el script, no se tocan):

  latencia_primer_sonido_ms   del fin de voz al primer sonido del agente,
                              sea lo que sea
  latencia_regla_del_arnes_ms del fin de voz a la primera racha de al menos
                              0.6 s: la misma regla con que se mide la latencia
                              de PoC1 y PoC2
  agente_sonaba_al_fin_de_voz 1 si el agente ya estaba sonando cuando el
                              transportista terminó
  agente_hablaba_al_empezar_el_clip
                              1 si el agente estaba sonando cuando el arnés
                              arrancó la línea del transportista
  pausa_del_agente_antes_del_clip_ms
                              en ese caso, cuánto había durado el silencio del
                              agente antes de retomar. El arnés pasa a la línea
                              siguiente tras 2 s de silencio del agente
  ultimo_turno                1 si es la última línea del guion
  segundos_de_grabacion_tras_el_fin_de_voz
                              cuánto sigue la grabación; si es poco y el agente
                              no habló, la llamada ya se estaba cerrando
  motivo                      por qué el método de PoC3 no le dio número
"""


def _manifest(clips: Path) -> dict[str, str]:
    """`{clip_id: text}` from the clip bank's manifest."""
    ruta = clips / "manifest.json"
    if not ruta.is_file():
        return {}
    data = json.loads(ruta.read_text(encoding="utf-8"))
    return {c["id"]: c.get("text", "") for c in data.get("clips", [])}


def _write_wav(ruta: Path, audio: np.ndarray, sample_rate: int) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.ascontiguousarray(audio, dtype="<i2").tobytes())


def lost_turns(
    *,
    call_sid: str,
    scenario: int,
    zeros: Path,
    live_bridge: Path,
    wav: Path,
    event_log: Path | None,
    clips: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The turns of a call that the external instrument left without a number.

    Returns `(turns, report)`. Each turn carries its figures and the audio
    segment to cut (`cutoff`, in samples of the original WAV).
    """
    measured_items, _, report = ext3.call_turns(
        call_sid=call_sid, scenario=scenario, zeros_path=zeros,
        bridge_json_path=live_bridge, wav_path=wav,
        path_of_log=event_log, clips_path=clips,
    )
    if "motivo" in report:
        # The whole call was left out (clock, axis or anchor): these are not
        # fast turns, and the report already counts it as a call without latency.
        return [], {"motivo": f"el instrumento externo no midió la llamada: {report['motivo']}"}
    with_number = {str(t.turn) for t in measured_items}
    marks = ext3.marks_de(zeros)
    if not marks:
        return [], {"motivo": "sin marcas"}

    bridge_data = json.loads(live_bridge.read_text(encoding="utf-8"))
    with_sample = {
        str(m["turn"]) for m, _ in ext3.pair(marks, ext3.axis_samples(bridge_data))
    }

    audio, sample_rate = ext3.leer_wav(wav)
    a8, t8 = _a_8khz(audio, sample_rate)
    anchor = ext3.anchor_on(marks, a8[:, 0], t8, clips_path=clips)
    if anchor is None:
        return [], {"motivo": "no se pudo anclar"}
    shift = anchor.displacement
    streaks = sorted(
        (start + shift, end + shift)
        for start, end in ext3.speech_streaks(ext3.envelope(a8[:, 1], t8))
    )
    descartes: list[float] = []
    if event_log is not None and event_log.is_file():
        descartes = ext3.discards_of(json.loads(event_log.read_text(encoding="utf-8")), shift)

    ordered = sorted(marks, key=lambda m: m["measured_end_of_speech"])
    output: list[dict[str, Any]] = []
    for k, mark in enumerate(ordered):
        turn = str(mark["turn"])
        if turn in with_number:
            continue
        cero = float(mark["measured_end_of_speech"])
        tope = (
            float(ordered[k + 1].get("start_of_clip", float("inf")))
            if k + 1 < len(ordered) else float("inf")
        )
        after = [r for r in streaks if cero - FIRST_SOUND_SLACK_S <= r[0] < tope]
        primero = after[0] if after else None
        substantive = next(
            (
                r for r in after
                if (r[1] - r[0]) >= MIN_SUBSTANTIVE_SPEECH_S
                and not any(abs(r[0] - d) <= DISCARD_TOLERANCE_S for d in descartes)
            ),
            None,
        )
        was_playing = any(
            start < cero - FIRST_SOUND_SLACK_S and end > cero for start, end in streaks
        )
        clip_start = float(mark.get("start_of_clip", cero))
        # What the agent was doing when the line started: a streak that crosses
        # it or starts inside it, and the silence that preceded it.
        over = next((r for r in streaks if r[0] < cero and r[1] > clip_start), None)
        pause = None
        if over is not None:
            priors = [r for r in streaks if r[1] <= over[0]]
            if priors:
                pause = round((over[0] - priors[-1][1]) * 1000.0, 0)
        ultimo = k + 1 == len(ordered)
        wav_end = shift + len(audio) / sample_rate
        cola = wav_end - cero
        from_t = min(cero - ANTES_SEG, clip_start - BEFORE_CLIP_S)
        until_t = min(
            tope + AFTER_THE_NEXT_S if not ultimo else wav_end,
            cero + AFTER_CAP_S,
        )
        since = max(0, int(round((from_t - shift) * sample_rate)))
        until = min(len(audio), int(round((until_t - shift) * sample_rate)))
        if primero is None and ultimo and cola < COLA_MINIMA_SEG:
            motivo = (
                "último turno: la grabación termina a los "
                f"{cola:.1f} s del fin de voz, sin voz del agente (cierre de la llamada)"
            )
        elif primero is None:
            motivo = "el agente no habló antes de la línea siguiente"
        elif turn in with_sample:
            motivo = "hubo muestra del puente, pero ninguna racha del agente junto a su acuse"
        else:
            motivo = (
                "el puente no dio muestra: el agente arrancó antes de que el "
                "puente cerrara el turno del transportista (800 ms de silencio)"
            )
        output.append({
            "turn": turn,
            "clip_id": mark.get("clip_id"),
            "motivo": motivo,
            "first_sound_latency_ms": (
                round((primero[0] - cero) * 1000.0, 1) if primero else None
            ),
            "first_sound_duration_ms": (
                round((primero[1] - primero[0]) * 1000.0, 1) if primero else None
            ),
            "harness_rule_latency_ms": (
                round((substantive[0] - cero) * 1000.0, 1) if substantive else None
            ),
            "agent_audible_at_end_of_speech": int(was_playing),
            "agent_was_speaking_at_clip_start": int(over is not None),
            "agent_pause_before_clip_ms": pause,
            "last_turn": int(ultimo),
            "recording_seconds_after_end_of_speech": round(cola, 2),
            "end_of_speech_second_in_audio": round(
                (cero - shift) - since / sample_rate, 2
            ),
            "cutoff": (since, until),
        })
    return output, {
        "marks": len(marks),
        "with_number": len(with_number),
        "no_number": len(output),
    }


def _artifacts(folder: Path, row: dict[str, Any]) -> dict[str, Path | None] | None:
    """A call's paths, resolved the same way as in the exporter."""
    summary = read_summary(folder, str(row["clave"]))
    sid = row.get("label")
    art = (summary or {}).get("artifacts") or {}
    event_log = Path(art["event_log"]) if art.get("event_log") else None
    live_bridge = Path(art["bridge_json"]) if art.get("bridge_json") else None
    harness = event_log.parent if event_log else None
    wav = Path(art["auditoria_wav"]) if art.get("auditoria_wav") else (
        harness / f"auditoria_{sid}.wav" if harness else None
    )
    zeros = harness / f"zeros_{sid}.jsonl" if harness else None
    if not sid or any(r is None or not r.is_file() for r in (live_bridge, wav, zeros)):
        return None
    return {"event_log": event_log, "live_bridge": live_bridge, "wav": wav, "zeros": zeros}


def _summary(rows: list[dict[str, Any]], reports: list[dict[str, Any]]) -> str:
    marks = sum(i.get("marks", 0) for i in reports)
    having = sum(i.get("with_number", 0) for i in reports)
    text_lines = [
        "Turnos de PoC3 sin latencia",
        f"llamadas revisadas: {len(reports)}"
        f" (sin poder revisar: {sum(1 for i in reports if 'motivo' in i)})",
        f"turns: {marks} · con número: {having} · sin número: {len(rows)}",
        "",
        "por motivo:",
    ]
    reasons: dict[str, int] = {}
    for f in rows:
        reasons[f["motivo"]] = reasons.get(f["motivo"], 0) + 1
    text_lines += [f"  {n:>4}  {m}" for m, n in sorted(reasons.items(), key=lambda x: -x[1])]
    over = [f for f in rows if f["agent_was_speaking_at_clip_start"]]
    pauses = [float(f["agent_pause_before_clip_ms"]) for f in over
              if f["agent_pause_before_clip_ms"] is not None]
    text_lines += [
        "",
        f"el agente ya sonaba al fin de voz: {sum(f['agent_audible_at_end_of_speech'] for f in rows)}",
        f"el agente hablaba cuando arrancó la línea del transportista: {len(over)}"
        + (f" (pausa previa: mín {min(pauses):.0f} · mediana {median(pauses):.0f}"
           f" · máx {max(pauses):.0f} ms)" if pauses else ""),
    ]
    for campo, name in (
        ("first_sound_latency_ms", "primer sonido"),
        ("harness_rule_latency_ms", "regla del arnés (racha ≥ 0.6 s)"),
    ):
        v = [float(f[campo]) for f in rows if f[campo] is not None]
        if v:
            text_lines.append(
                f"{name}: n={len(v)} · mín {min(v):.0f} · mediana {median(v):.0f}"
                f" · máx {max(v):.0f} ms"
            )
        else:
            text_lines.append(f"{name}: sin valores")
    return "\n".join(text_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", type=Path, default=LOTE_DEFAULT,
                   help="folder that contains campana_poc3_pcmu/")
    p.add_argument("--clips", type=Path, default=CLIPS_DEFAULT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="folder for the audio files and the index (not committed)")
    p.add_argument("--all-items", action="store_true",
                   help="also include calls that did not end in ok status")
    args = p.parse_args(argv)

    folder = args.lote / "campana_poc3_pcmu"
    texts = _manifest(args.clips)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    progress = [
        f for f in read_progress(folder)
        if int(f.get("poc", 3)) == 3 and (args.all_items or f.get("estado") == "ok")
    ]
    for k, row in enumerate(progress, start=1):
        if k % 10 == 0 or k == len(progress):
            print(f"  {k}/{len(progress)} calls", flush=True)
        clave = str(row["clave"])
        paths = _artifacts(folder, row)
        if paths is None:
            reports.append({"clave": clave, "motivo": "falta un artefacto"})
            continue
        try:
            turns, report = lost_turns(
                call_sid=str(row["label"]), scenario=int(row["scenario"]),
                clips=args.clips, **paths,
            )
        except Exception as exc:  # a broken call does not stop the others
            reports.append({"clave": clave, "motivo": f"error: {exc}"})
            continue
        reports.append({"clave": clave, **report})
        if not turns:
            continue
        audio, sample_rate = ext3.leer_wav(paths["wav"])  # type: ignore[arg-type]
        for t in turns:
            since, until = t.pop("cutoff")
            name = f"{len(rows) + 1:03d}_{clave}_{t['turn']}_{t['clip_id']}.wav"
            _write_wav(args.output / "audio_files" / name, audio[since:until], sample_rate)
            rows.append({
                "file": name, "clave": clave,
                "scenario": row["scenario"], "repetition": row.get("repetition"),
                "carrier_line": texts.get(str(t["clip_id"]), ""),
                "what_the_agent_said": "", "tipo": "", "notes": "",
                **t,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel opens the accented characters correctly.
    with (args.output / "indice.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (args.output / "LEEME.txt").write_text(LEEME, encoding="utf-8")
    text = _summary(rows, reports)
    (args.output / "summary.txt").write_text(text, encoding="utf-8")
    problems = [i for i in reports if "motivo" in i]
    if problems:
        (args.output / "calls_no_revisadas.txt").write_text(
            "\n".join(f"{i['clave']}: {i['motivo']}" for i in problems) + "\n", encoding="utf-8"
        )
    print("\n" + text)
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
