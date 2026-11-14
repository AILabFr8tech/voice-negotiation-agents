"""The agent's pause at each interruption event, measured from the audio, with no manual rating.

The harness metric (`stopped`: agent silence ≥ 0.7 s within 3 s of the
carrier's speech) says **whether** the agent stopped, not **for how long**. A
brief pause followed by "I pick up where I was" counts the same as "I cut off,
wait and build another response", and in PoC3 the controls (where nobody
talks over it) came out 6 of 8 "stopped" because of their natural pauses
between sentences. What separates the two behaviors is the **duration** of
that pause, and that can be measured from the audio:

- **pause**: the agent's first silence of ≥ 0.7 s that starts within the 3 s
  following the reference (the same stop definition as the harness,
  `STOP_SILENCE_S`, `STOP_WINDOW_S`), and how long it lasts;
- **speaks again**: when that pause ends, counted from when the carrier
  finished (in the control, from the reference).

The **controls** are each PoC's base rate: how much it pauses when nobody is
talking to it. A backchannel that does not derail the agent leaves a pause
similar to the control's; one that makes it cut off and answer something else
lengthens it.

**The pause duration does not depend on the transport**: PoC2's network delay
shifts the start and end of the pause equally. "Speaks again" does include
it, because it is counted from an instant on the harness side.

Same instrument for all three: the agent channel of the audit WAV
(`shared/carrier_simulator/auditoria.py`), 20 ms windows and the harness
speech threshold (`RMS_WITH_SPEECH`). That channel is timestamped as each frame
arrives: it is good for pauses of hundreds of milliseconds, not for resolving
individual milliseconds. In PoC3 the bridge delivers 100 ms blocks, so their
edges have up to one block of granularity.

**This was PoC3's stop measurement while the harness timestamped each ~100 ms
bridge block on arrival**, because that way up to one block is added to a
pause and a natural pause can look like a stop. Since the harness listens to
PoC3 in 20 ms segments, the report measures all three PoCs with the harness,
and this measurement only counts in how many events the recording and the
harness agree (`shared/metrics/interruptions_report.py`). The rule is the same
(silence ≥ 0.7 s that starts within 3 s), but on 20 ms windows of the audio.
That is why the CSV also carries:

- **audio_stop_ms**: the last segment with speech before the pause minus the
  reference (the harness's `stop_time_ms` convention);
- **speaks_again_ms**, which in an interruption is the reply latency with the
  same definition as the harness (first sound after the stop minus the end of
  the carrier's speech);
- **max_silence_ms**: the agent's longest silence that starts within the
  window, however long it lasts. It shows how close to the threshold an event
  with no stop came.

In PoC1 and PoC2 the audio arrives in 20 ms segments and the harness and this
measurement agree; there the harness's `stopped`, fixed before running,
prevails. The pause duration remains a descriptive measure defined after
seeing the data, and is reported as such.

    uv run --project poc3_gpt_live python -m validation.interruption_pauses \\
        --campana diagnostics/lote/interrupciones --campana diagnostics/lote/interrupciones_poc3 \\
        --include-failed --output docs/report/data

Writes `interruption_pauses.csv` (one row per event) and
`interruption_pauses_summary.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio.wav import read_wav_stereo
from shared.carrier_simulator.interruptions import (
    NO_REPLY_WAIT_S,
    STOP_SILENCE_S,
    STOP_WINDOW_S,
)
from shared.carrier_simulator.observed import RMS_WITH_SPEECH

WINDOW_S: Final[float] = 0.020
KINDS: Final[tuple[str, ...]] = ("control", "apoyo", "interrupcion")

COLUMNS: Final[list[str]] = [
    "poc", "estado", "clave", "scenario", "turn", "tipo", "clip_id",
    "metric_stopped", "pause_ms", "endless_pause", "speaks_again_ms", "audio_stop_ms",
    "max_silence_ms",
]


def frames_with_speech(canal: np.ndarray, sample_rate: int) -> np.ndarray:
    """One boolean per 20 ms window: does it cross the harness speech threshold?"""
    n = max(1, int(round(sample_rate * WINDOW_S)))
    completed = len(canal) // n
    if not completed:
        return np.zeros(0, dtype=bool)
    blocks = canal[: completed * n].astype(np.float64).reshape(completed, n)
    return np.sqrt((blocks**2).mean(axis=1)) >= RMS_WITH_SPEECH


def measure_in_audio(
    with_voice: np.ndarray, *, reference: float, end_of_speech: float, until: float
) -> dict[str, Any]:
    """The agent's stop around an event, on 20 ms windows.

    - `pause_ms`: the first silence of ≥ `STOP_SILENCE_S` that starts
      within `STOP_WINDOW_S` of `reference`. `None` if there was none
      (it kept talking, apart from short gaps between words): it did not stop.
    - `last_speech`: start of the last segment with speech before that pause (s).
    - `stop_ms`: `last_speech` minus `reference`.
    - `no_end`: the pause reached `until` without the agent speaking again;
      its duration is then a lower bound.
    - `resumes_ms`: the end of the pause minus `end_of_speech`. `None` if there
      was no pause or it did not end.
    - `max_silence_ms`: the longest silence that starts within the window, of
      any duration (up to the pause, if there was one).

    Instants in seconds from the start of the WAV.
    """
    a = max(0, int(reference / WINDOW_S))
    tope = min(len(with_voice), int(until / WINDOW_S))
    minimum = int(round(STOP_SILENCE_S / WINDOW_S))
    last_onset = a + int(round(STOP_WINDOW_S / WINDOW_S))
    longest = 0
    k = a
    while k < min(tope, last_onset):
        if with_voice[k]:
            k += 1
            continue
        end = k
        while end < tope and not with_voice[end]:
            end += 1
        longest = max(longest, end - k)
        if end - k >= minimum:
            priors = np.flatnonzero(with_voice[:k])
            ultima = float(priors[-1]) * WINDOW_S if priors.size else None
            no_end = end >= tope
            return {
                "pause_ms": round((end - k) * WINDOW_S * 1000.0, 1),
                "last_speech": ultima,
                "stop_ms": None if ultima is None else round((ultima - reference) * 1000.0, 1),
                "no_end": no_end,
                "resumes_ms": None if no_end else round((end * WINDOW_S - end_of_speech) * 1000.0, 1),
                "max_silence_ms": round(longest * WINDOW_S * 1000.0, 1),
            }
        k = end
    return {"pause_ms": None, "last_speech": None, "stop_ms": None, "no_end": False, "resumes_ms": None,
            "max_silence_ms": round(longest * WINDOW_S * 1000.0, 1)}


def measure_pause(
    with_voice: np.ndarray, *, reference: float, end_of_speech: float, until: float
) -> tuple[float | None, bool, float | None]:
    """`(pause_ms, no_end, resumes_ms)`; see `measure_in_audio`."""
    m = measure_in_audio(with_voice, reference=reference, end_of_speech=end_of_speech, until=until)
    return m["pause_ms"], m["no_end"], m["resumes_ms"]


def rows_of(campana: Path, *, include_failed: bool) -> tuple[list[dict[str, Any]], int]:
    """One row per fired event with an audit WAV, and how many lacked one."""
    valid_items = ("ok", "roja") if include_failed else ("ok",)
    rows: list[dict[str, Any]] = []
    no_wav = 0
    cache: dict[str, np.ndarray] = {}
    for linea in (campana / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        p = json.loads(linea)
        if p.get("estado") not in valid_items:
            continue
        ruta = campana / "summaries" / f"{p['clave']}.json"
        if not ruta.is_file():
            continue
        summary = json.loads(ruta.read_text(encoding="utf-8"))
        events = [e for e in summary.get("events") or []
                   if e.get("tipo") in KINDS and e.get("fired") and e.get("t_speech_onset") is not None]
        wav = (summary.get("artifacts") or {}).get("auditoria_wav")
        if not wav or not Path(wav).is_file():
            no_wav += len(events)
            continue
        if wav not in cache:
            _, agent, sample_rate = read_wav_stereo(wav)
            cache[wav] = frames_with_speech(agent, sample_rate)
        with_voice = cache[wav]
        for ev in events:
            reference = float(ev["t_speech_onset"])
            end_of_speech = float(ev.get("t_end_of_speech") or reference)
            m = measure_in_audio(
                with_voice, reference=reference, end_of_speech=end_of_speech,
                until=end_of_speech + NO_REPLY_WAIT_S,
            )
            pause, no_end, returns = m["pause_ms"], m["no_end"], m["resumes_ms"]
            rows.append({
                "poc": p.get("poc"), "estado": p.get("estado"), "clave": p.get("clave"),
                "scenario": p.get("scenario"), "turn": ev.get("turn"), "tipo": ev["tipo"],
                "clip_id": ev.get("clip_id") or "",
                "metric_stopped": "" if ev.get("stopped") is None else int(bool(ev["stopped"])),
                "pause_ms": "" if pause is None else pause,
                "endless_pause": int(no_end),
                "speaks_again_ms": "" if returns is None else returns,
                "audio_stop_ms": "" if m["stop_ms"] is None else m["stop_ms"],
                "max_silence_ms": m["max_silence_ms"],
            })
    return rows, no_wav


def _span(v: list[float]) -> str:
    if not v:
        return "—"
    return f"{statistics.median(v):,.0f} [{min(v):,.0f}, {max(v):,.0f}]"


def summary(rows: list[dict[str, Any]], *, no_wav: int = 0) -> str:
    text_lines = [
        "Pausa del agente en los eventos de interrupción (medida del audio, mediana [mín, máx] en ms)",
        "Pausa = primer silencio ≥ 0.7 s que empieza dentro de 3 s de la voz del transportista,",
        "la misma definición de detención del arnés. Descriptiva, definida después de ver los datos.",
        "",
        f"{'':6}{'tipo':<14}{'n':>3}  {'con pausa':<10}{'pause':<26}{'sin volver':<12}vuelve a hablar tras el clip",
    ]
    for poc in sorted({str(f["poc"]) for f in rows}):
        for tipo in KINDS:
            fs = [f for f in rows if str(f["poc"]) == poc and f["tipo"] == tipo]
            if not fs:
                continue
            having = [f for f in fs if f["pause_ms"] != ""]
            pauses = [float(f["pause_ms"]) for f in having if not f["endless_pause"]]
            no_end = sum(1 for f in having if f["endless_pause"])
            returns = [float(f["speaks_again_ms"]) for f in having if f["speaks_again_ms"] != ""]
            failed_calls = sum(1 for f in fs if f["estado"] == "roja")
            text_lines.append(
                f"PoC{poc:<3}{tipo:<14}{len(fs):>3}  {f'{len(having)}/{len(fs)}':<10}{_span(pauses):<26}"
                f"{no_end:<12}{_span(returns)}" + (f"   ({failed_calls} de llamadas rojas)" if failed_calls else "")
            )
        text_lines.append("")
    if no_wav:
        text_lines.append(f"{no_wav} evento(s) sin WAV de auditoría: no entran.")
    return "\n".join(text_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campana", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-failed", action="store_true")
    args = p.parse_args(argv)

    rows: list[dict[str, Any]] = []
    no_wav = 0
    for c in args.campana:
        f, s = rows_of(c, include_failed=args.include_failed)
        rows.extend(f)
        no_wav += s
    text = summary(rows, no_wav=no_wav)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "interruption_pauses.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (args.output / "interruption_pauses_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"{len(rows)} events written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
