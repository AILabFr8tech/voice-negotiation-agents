"""Timeline of the call in which an ``ok'' said over the agent closed the deal.

In call 0004-poc1-e2-r1 of the interruption scenario (PoC1, Pipecat), the
carrier asked for 18,000, the agent started offering 16,500, an ``Ok.'' was
played over it and the agent recorded the agreement. This script timestamps each
piece in seconds since the start of the line preceding the backchannel:

- the carrier's two lines and the backchannel, with their speech (harness marks);
- what the agent said before and after the backchannel (call summary; the end of
  its response comes from the agent channel of the audit recording);
- when the agent received the transcript of the line and of the backchannel
  (call log);
- when it called `registrar_acuerdo` (Pipecat log, in local time) and when it
  hung up.

What was heard from the agent comes from the resumption sheet rated by the author.
Writes `backchannel_timeline.csv` to `--output`. Reads the call's artifacts,
which are not shipped in the repo.

    python -m validation.backchannel_timeline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.metrics.campaigns import INTERRUPTIONS_POC1_POC2
from shared.metrics.export_for_report import LOTE_DEFAULT, read_summary

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
SHEET: Final[Path] = DATA / "interruption_campaign_v3"
CLAVE: Final[str] = "0004-poc1-e2-r1"
TURN: Final[str] = "T6"
# Same rule as the harness for the agent's speech (harness parameter table).
SEGMENT_S: Final[float] = 0.02
UMBRAL_DBFS: Final[float] = -44.0
GAP_S: Final[float] = 0.3
# Tolerance for considering the recording aligned with the harness marks.
TOLERANCE_S: Final[float] = 0.15

COLUMNS: Final[list[str]] = ["clave", "que", "start_s", "end_s", "text"]

RE_HARNESS = re.compile(r"^\s*\[\s*([\d.]+)s\]\s+(.*)$")
RE_LINEA = re.compile(r"^(T\d+) \(\d+/\d+\) (\S+) \(([\d.]+) s\)")
RE_PIPECAT = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) \|")
RE_LABEL = re.compile(r"(\d{8}-\d{6})$")


def harness_lines(log: list[str]) -> list[tuple[float, str]]:
    return [(float(m.group(1)), m.group(2)) for m in map(RE_HARNESS.match, log) if m]


def run_zero(harness: list[tuple[float, str]], marks: list[dict[str, Any]]) -> float:
    """Epoch of the log's zero: each line's start as epoch minus its second in the log."""
    start = {m["turn"]: float(m["start_of_clip"]) for m in marks}
    subtractions = [start[m.group(1)] - t for t, text in harness if (m := RE_LINEA.match(text)) and m.group(1) in start]
    return statistics.median(subtractions)


def to_epoch(local_time: str, label: str, cero: float) -> float:
    """Pipecat log local time to epoch; the time zone comes from the room label."""
    local_zero = datetime.strptime(RE_LABEL.search(label).group(1), "%Y%m%d-%H%M%S")
    huso = round((local_zero.replace(tzinfo=timezone.utc).timestamp() - cero) / 1800) * 1800
    t = datetime.strptime(local_time, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
    return t - huso


def agent_streaks(ruta_wav: Path) -> list[tuple[float, float]]:
    """Speech streaks of the agent channel (channel 1) in the audit recording."""
    with wave.open(str(ruta_wav)) as w:
        sample_rate, channel_list = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, channel_list)[:, 1]
    n = int(SEGMENT_S * sample_rate)
    segments = audio[: len(audio) // n * n].astype(np.float64).reshape(-1, n)
    rms = np.sqrt(np.mean(segments**2, axis=1))
    voice = 20 * np.log10(np.maximum(rms, 1e-9) / 32768.0) >= UMBRAL_DBFS
    streaks: list[list[float]] = []
    for i in np.flatnonzero(voice):
        t = i * SEGMENT_S
        if streaks and t - streaks[-1][1] < GAP_S:
            streaks[-1][1] = t + SEGMENT_S
        else:
            streaks.append([t, t + SEGMENT_S])
    return [(a, b) for a, b in streaks]


def what_was_heard(clave: str, turn: str) -> dict[str, str]:
    ids = [r["id"] for r in csv.DictReader((SHEET / "resumption_sheet_key.csv").open(encoding="utf-8-sig"))
           if r["clave"] == clave and r["turn"] == turn]
    rows = [r for r in csv.DictReader((SHEET / "resumption_sheet_author.csv").open(encoding="utf-8-sig")) if r["id"] in ids]
    if len(rows) != 1:
        raise SystemExit(f"expected one rated resumption for {clave} {turn}, found {len(rows)}")
    return rows[0]


def timeline(res: dict[str, Any], log: list[str], marks: list[dict[str, Any]],
                    event_log: dict[str, Any], streaks: list[tuple[float, float]],
                    ear: dict[str, str]) -> list[dict[str, Any]]:
    harness = harness_lines(log)
    cero = run_zero(harness, marks)
    per_turn = {m["turn"]: m for m in marks}
    sig = f"T{int(TURN[1:]) + 1}"
    linea, following = per_turn[TURN], per_turn[sig]
    origin = float(linea["start_of_clip"]) - cero
    event = next(e for e in res["events"] if e["turn"] == TURN and e["tipo"] == "apoyo")
    inter = next(i for i in res["lines"]["per_line"] if i["turn"] == TURN)
    onset = float(inter["agent_first_speech"]) - cero
    goes_quiet = float(event["last_speech_before_stopping"])
    returns = float(event["t_end_of_speech"]) + float(event["speaks_again_ms"]) / 1000

    # The recording and the marks share the same zero: checked against the agent's onset.
    near = min(streaks, key=lambda r: abs(r[0] - onset))
    if abs(near[0] - onset) > TOLERANCE_S:
        raise SystemExit(f"the recording does not match the marks: {near[0]:.3f} vs {onset:.3f}")
    after = min(streaks, key=lambda r: abs(r[0] - returns))
    if abs(after[0] - returns) > TOLERANCE_S:
        raise SystemExit(f"could not find the agent's return in the recording ({returns:.3f})")

    transcript = [(t, text.split(None, 1)[1].strip("'")) for t, text in harness
                     if text.startswith("nosotros") and origin <= t < onset]
    backchannel_stt = next(t for t, text in harness if text.startswith("nosotros") and t >= float(event["t_speech_onset"]))
    hangs_up = next(t for t, text in harness if text.startswith("el agente colgó"))
    record = None
    for l in log:
        m = RE_PIPECAT.match(l)
        if m and "Calling function [registrar_acuerdo:" in l:
            monto = re.search(r"'monto_final': (\d+)", l).group(1)
            record = (to_epoch(m.group(1), res["label"], cero) - cero, monto)
    if record is None or not goes_quiet < record[0] < after[1]:
        raise SystemExit(f"the agreement record does not fall between the backchannel and the agent's return: {record}")

    texts = {t["turn"]: t["played_line"] for t in event_log["turns"]}
    rows = [
        ("linea", 0.0, float(linea["measured_end_of_speech"]) - float(linea["start_of_clip"]), texts[TURN]),
        ("transcripcion_de_la_linea", transcript[-1][0] - origin, None, " ".join(t for _, t in transcript)),
        ("agente_antes", onset - origin, goes_quiet - origin, ear["before_event"]),
        ("apoyo", float(event["t_speech_onset"]) - origin, float(event["t_end_of_speech"]) - origin, event["text"]),
        ("transcripcion_del_apoyo", backchannel_stt - origin, None, event["event_stt"]),
        ("registrar_acuerdo", record[0] - origin, None, f"monto_final={record[1]}"),
        ("agente_despues", after[0] - origin, after[1] - origin, ear["after_the_event"]),
        ("linea_siguiente", float(following["start_of_clip"]) - cero - origin,
         float(following["measured_end_of_speech"]) - cero - origin, texts[sig]),
        ("cuelga", hangs_up - origin, None, ""),
    ]
    return [{"clave": CLAVE, "que": q, "start_s": round(a, 3), "end_s": "" if b is None else round(b, 3),
             "text": t} for q, a, b, t in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lote", type=Path, default=LOTE_DEFAULT / INTERRUPTIONS_POC1_POC2)
    ap.add_argument("--output", type=Path, default=DATA)
    args = ap.parse_args(argv)

    res = read_summary(args.lote, CLAVE)
    if res is None:
        raise SystemExit(f"summary for {CLAVE} not found in {args.lote}")
    art = res["artifacts"]
    cero_json = json.loads(Path(art["zero_crossing"]).read_text(encoding="utf-8"))
    marks = [json.loads(l)["mark"] for l in Path(cero_json["cobertura"]["file"]).read_text(encoding="utf-8").splitlines()
              if l.strip() and json.loads(l).get("tipo") == "mark"]
    log = (args.lote / "calls" / f"{CLAVE}.log").read_text(encoding="utf-8").splitlines()
    event_log = json.loads(Path(art["event_log"]).read_text(encoding="utf-8"))
    rows = timeline(res, log, marks, event_log, agent_streaks(Path(art["auditoria_wav"])),
                            what_was_heard(CLAVE, TURN))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "backchannel_timeline.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for f in rows:
        print(f"{f['que']:26s} {f['start_s']:>8} {f['end_s']!s:>8}  {f['text'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
