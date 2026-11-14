"""Resumption sheet with the "before" taken from the audio, not from the text sent for synthesis.

The previous sheet (`validation.resumption_sheet`) took what the agent said
before the backchannel from the text of its response. In PoC1 and PoC2 that
text is the **full** text sent to the TTS: if the backchannel cut the agent off
mid-sentence, the text carries words that were never played, and on resuming it
would look as if it "repeated" what the carrier never heard. Here the "before"
is what actually played: the agent channel of the harness recording, from the
end of the carrier's line to the agent's last sound before going silent,
transcribed with Deepgram (`nova-3`, Spanish). The "after" also comes from the
audio: from when the agent speaks again until the carrier's next line plays
(or 20 s). It is taken from the audio and not from the model's text because
that text has two defects that change the classification: in PoC1 and PoC2 it
starts with the rest of the cut-off sentence, which was synthesized but not
played, glued to the new message; and in PoC3 it starts at the instant of the
backchannel, so it repeats what the agent managed to say before going silent.
The model's text is kept in `resumption_sheet_antes.csv` for reference.

Same rules as the previous sheet: the backchannels in which the agent stopped
and spoke again, in random order and without the PoC. Writes, to `--output`:

- `resumption_sheet.xlsx`: for rating (column `classification`: C, R, N or ?),
  with a criteria tab.
- `resumption_sheet_key.csv`: the PoC, the call and the turn of each row.
- `resumption_sheet_antes.csv`: the audio segment of each row and its transcript.

Uses the Deepgram key from `.env`, like PoC1 and PoC2; does not print it. Costs
cents (a few minutes of audio).

    poc1_pipecat/.venv/Scripts/python.exe -m validation.resumption_sheet_audio \\
        --events docs/report/data/interruption_campaign_v3/interruptions_events.csv \\
        --output docs/report/data/interruption_campaign_v3
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import wave
import zipfile
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

import numpy as np

from shared.audio import read_wav_stereo
from validation.resumption_sheet import SEED, resumptions

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
LOTE: Final[Path] = RAIZ / "diagnostics" / "lote"
# Campaign of each PoC in the interruption scenario.
CAMPAIGN_OF: Final[dict[str, str]] = {
    "1": "interruptions_full_v2", "2": "interruptions_full_v2", "3": "interruptions_poc3_pcmu",
}
UMBRAL_RMS: Final[float] = 200.0  # the harness's: −44 dBFS
COLA_SEG: Final[float] = 0.3  # keeps the decay of the last syllable
URL_DEEPGRAM: Final[str] = (
    "https://api.deepgram.com/v1/listen?model=nova-3&language=es&punctuate=true&smart_format=false"
)

CRITERIA: Final[list[tuple[str, str]]] = [
    ("C — continuó", "Siguió la frase o la idea donde la dejó, sin volver a decir lo que ya había "
                    "dicho. Un tropiezo de una o dos palabras cuenta como continuar."),
    ("R — repitió", "Volvió a decir una oferta, una cláusula de tres palabras o más, o el arranque de "
                   "la frase que ya había dicho, igual o con otras palabras."),
    ("N — nueva", "Dejó la frase y arrancó otro mensaje con contenido distinto. Repetir solo un "
                 "reconocimiento (\"entiendo\", \"Carlos\") no cuenta como repetir."),
    ("? — no se sabe", "El texto no alcanza para decidir."),
    ("Antes del evento", "Lo que de verdad sonó del agente antes de callarse (transcrito del audio)."),
    ("Después del evento", "Lo que sonó del agente desde que volvió a hablar hasta la línea siguiente "
                           "del transportista (transcrito del audio)."),
]


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def _rms_per_segment(canal: np.ndarray, sample_rate: int) -> np.ndarray:
    n = int(sample_rate * 0.02)
    x = np.asarray(canal[: len(canal) // n * n], dtype=np.float64).reshape(-1, n)
    return np.sqrt((x ** 2).mean(axis=1))


def segment_before(harness: np.ndarray, agent: np.ndarray, sample_rate: int, event: dict[str, Any]) -> tuple[float, float]:
    """`(start, end)` in seconds: from the end of the carrier's line to the agent's last sound."""
    rms = _rms_per_segment(harness, sample_rate)
    k = int((float(event["t_clip_start"]) - 0.05) / 0.02)
    while k > 0 and rms[k] < UMBRAL_RMS:
        k -= 1
    start = (k + 1) * 0.02
    end = float(event["last_speech_before_stopping"]) + COLA_SEG
    return start, end


AFTER_CAP_S: Final[float] = 20.0


def segment_after(harness: np.ndarray, sample_rate: int, event: dict[str, Any]) -> tuple[float, float]:
    """`(start, end)`: from when the agent speaks again to the carrier's next line."""
    start = float(event["last_speech_before_stopping"]) + float(event["pause_duration_ms"]) / 1000.0
    rms = _rms_per_segment(harness, sample_rate)
    k = int(start / 0.02) + 1
    tope = min(len(rms), int((start + AFTER_CAP_S) / 0.02))
    while k < tope and rms[k] < UMBRAL_RMS:
        k += 1
    return max(0.0, start - 0.05), k * 0.02


def wav_mono(samples: np.ndarray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def transcribe(wav: bytes, key: str) -> str:
    import httpx

    r = httpx.post(URL_DEEPGRAM, content=wav, timeout=60.0,
                   headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    r.raise_for_status()
    alternates = r.json()["results"]["channels"][0]["alternatives"]
    return (alternates[0].get("transcript") or "").strip() if alternates else ""


# --------------------------------------------------------------------------
# Excel without dependencies: an .xlsx is a zip of XML
# --------------------------------------------------------------------------


def _sheet_xml(rows: list[list[str]], widths: list[int]) -> str:
    def col(i: int) -> str:
        return chr(ord("A") + i)

    table_rows = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{col(i)}{r}" t="inlineStr" s="{1 if r == 1 else 2}"><is><t xml:space="preserve">'
            f"{escape(str(v))}</t></is></c>"
            for i, v in enumerate(row)
        )
        table_rows.append(f'<row r="{r}">{cells}</row>')
    columns = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{a}" customWidth="1"/>' for i, a in enumerate(widths))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{columns}</cols><sheetData>{''.join(table_rows)}</sheetData></worksheet>"
    )


def write_xlsx(ruta: Path, sheets: list[tuple[str, list[list[str]], list[int]]]) -> None:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1))
            + "</Types>"))
        z.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"))
        z.writestr("xl/workbook.xml", (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
            + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, (n, _, _) in enumerate(sheets, 1))
            + "</sheets></workbook>"))
        z.writestr("xl/_rels/workbook.xml.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1))
            + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>"))
        z.writestr("xl/styles.xml", (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="{ns}">'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="3"><xf/>'
            '<xf fontId="1" applyFont="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
            '<xf applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>'
            "</styleSheet>"))
        for i, (_, rows, widths) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows, widths))


# --------------------------------------------------------------------------
# The sheet
# --------------------------------------------------------------------------


def event_of(poc: str, clave: str, turn: str) -> tuple[dict[str, Any], Path]:
    """The log event and the audit WAV of a call in the interruption scenario."""
    folder = LOTE / CAMPAIGN_OF[poc]
    summary = json.loads((folder / "summaries" / f"{clave}.json").read_text(encoding="utf-8"))
    art = summary.get("artifacts") or {}
    event_log = Path(art["event_log"])
    sid = summary.get("label") or event_log.stem.removeprefix("log_")
    wav = Path(art.get("auditoria_wav") or event_log.with_name(f"auditoria_{sid}.wav"))
    for t in json.loads(event_log.read_text(encoding="utf-8"))["turns"]:
        if t.get("turn") == turn and t.get("event"):
            return t["event"], wav
    raise SystemExit(f"{clave} {turn}: the log does not contain the event")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=SEED + 3)
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env", override=False)
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is missing from .env")

    with args.events.open(encoding="utf-8-sig", newline="") as f:
        rows = resumptions(list(csv.DictReader(f)), kinds=("apoyo",))
    random.Random(args.seed).shuffle(rows)
    args.output.mkdir(parents=True, exist_ok=True)

    sheet = [["id", "over_the_agent", "antes_del_evento (lo que sonó)", "despues_del_evento (lo que sonó)",
             "classification", "note"]]
    key_rows, before_rows = [], []
    audio_files: dict[Path, tuple[np.ndarray, np.ndarray, int]] = {}
    for n, e in enumerate(rows, start=1):
        ident = f"R{n:03d}"
        event, wav = event_of(e["poc"], e["clave"], e["turn"])
        if wav not in audio_files:
            audio_files[wav] = read_wav_stereo(wav)
        harness, agent, sample_rate = audio_files[wav]
        start, end = segment_before(harness, agent, sample_rate, event)
        text = transcribe(wav_mono(agent[int(start * sample_rate): int(end * sample_rate)], sample_rate), key)
        d_start, d_end = segment_after(harness, sample_rate, event)
        after = transcribe(wav_mono(agent[int(d_start * sample_rate): int(d_end * sample_rate)], sample_rate), key)
        print(f"{ident} PoC{e['poc']}: {text[:50]} || {after[:60]}", flush=True)
        sheet.append([ident, e.get("text", ""), text, after, "", ""])
        key_rows.append({"id": ident, "poc": e["poc"], "clave": e["clave"], "turn": e["turn"],
                            "estado": e.get("estado", "")})
        before_rows.append({"id": ident, "start_s": round(start, 2), "end_s": round(end, 2),
                            "heard_before": text, "before_model_text": e.get("agent_before_event", ""),
                            "after_start_s": round(d_start, 2), "after_end_s": round(d_end, 2),
                            "after_heard": after,
                            "after_model_text": e.get("agent_after_event", "")})

    criteria = [["Clave", "Qué significa"], *[[a, b] for a, b in CRITERIA]]
    write_xlsx(args.output / "resumption_sheet.xlsx",
                  [("resumptions", sheet, [7, 18, 60, 60, 13, 30]), ("criteria", criteria, [22, 90])])
    for name, csv_rows in (("resumption_sheet_key.csv", key_rows), ("resumption_sheet_antes.csv", before_rows)):
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"\n{len(rows)} resumptions · written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
