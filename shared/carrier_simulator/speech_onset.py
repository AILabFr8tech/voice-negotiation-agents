"""La latencia de INICIO de cada turno, calculada después de la corrida.

## Para qué sirve, y por qué no es el sesgo de fin

El sesgo que publican los dos arneses cuelga del **fin** de la voz del clip, y
ahí dentro viven dos cosas a la vez: el camino del audio y el endpointing del
VAD (los 576 ms que LiveKit resta, los 192 que pipecat espera, el desajuste de
criterio). Separar una de otra mirando solo el fin no se puede.

El **inicio** no lleva endpointing adentro. Entre que suena la primera muestra
con voz del clip y que el VAD marca el arranque del habla solo hay camino. Así
que:

    start_latency = estimated_speech_start - (start_of_clip + cabeza)

donde `cabeza` es el silencio con que arranca el clip, medido con el **mismo**
umbral con que se midió su fin de voz.

Con eso se contestó de dónde venía la deriva del sesgo de PoC2 dentro de una
call: la latencia de inicio crecía igual, o sea que era camino y no cero.
Ver `docs/METRICS_CONTRACT.md` §2.2.2.

## Por qué se calcula aquí y no en la corrida

Porque la cabeza del clip es una medición **del clip**, y el lugar de esas es el
manifiesto — junto a `end_of_speech_s`, con el umbral que las produjo. Medirla en
vivo sería DSP a media llamada, y medirla en cada PoC por su cuenta arriesgaría
que dos arneses llegaran a números distintos sobre el mismo audio. Mientras no
haya que volver a preparar los clips, esto lo hace después y sobre el mismo
archivo, que da exactamente el mismo número: contra las siete marcas de la
corrida `20260917-152841`, el `measured_end_of_speech` recalculado aquí reprodujo el
de la corrida con diferencia de 0.0 ms.

## Uso

    python -m shared.carrier_simulator.speech_onset <json de la corrida>

Lee el JSON de cualquiera de las dos PoC: busca el bloque `per_turn` y entiende
las dos formas de nombrar el arranque estimado.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.audio.level import DROP_FOR_LEVEL_DB, windows_with_speech
from shared.carrier_simulator.script import RUTA_CLIPS_DEFAULT, load_clip

# Cómo nombra cada PoC el instante en que su VAD dijo "empezó a hablar".
# PoC2 reexporta el campo de LiveKit; PoC1 lo nombra por lo que es.
ONSET_KEYS: tuple[str, ...] = (
    "started_speaking_at",
    "estimated_speech_start",
)


@dataclass(frozen=True)
class TurnStart:
    """La latencia de inicio de un turno, con lo que hizo falta para calcularla."""

    turn: str
    clip_id: str
    t_in_call: float
    head_ms: float
    latency_ms: float
    umbral_dbfs: float
    # El fin de voz recalculado contra el que publicó la corrida. Es el testigo
    # de que este módulo mide con el mismo criterio: si no diera ~0, la cabeza
    # tampoco sería comparable con nada.
    end_disagreement_ms: float | None

    def a_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "clip_id": self.clip_id,
            "t_in_call_s": round(self.t_in_call, 3),
            "clip_head_ms": round(self.head_ms, 1),
            "start_latency_ms": round(self.latency_ms, 1),
            "speech_threshold_dbfs": round(self.umbral_dbfs, 2),
            "end_disagreement_ms": (
                None
                if self.end_disagreement_ms is None
                else round(self.end_disagreement_ms, 1)
            ),
        }


def speech_edges(
    samples: np.ndarray,
    sample_rate: int,
    *,
    umbral_dbfs: float | None = None,
    drop_db: float = DROP_FOR_LEVEL_DB,
) -> tuple[float, float, float]:
    """`(inicio_seg, end_s, umbral_dbfs)` de la voz dentro del clip.

    `umbral_dbfs` se pasa cuando la corrida lo dejó escrito: usar el mismo
    número —y no re-derivarlo del pico— es lo que hace que la cabeza sea
    comparable con el `measured_end_of_speech` que ya viajó en la marca.
    """
    with_voice, per_window, umbral = windows_with_speech(
        samples, sample_rate, umbral_dbfs=umbral_dbfs, drop_db=drop_db
    )
    duration = len(samples) / float(sample_rate)
    if with_voice.size == 0:
        return 0.0, duration, umbral
    start = int(with_voice[0]) * per_window / float(sample_rate)
    end = min(len(samples), (int(with_voice[-1]) + 1) * per_window) / float(sample_rate)
    return start, end, umbral


def _estimated_onset(row: dict[str, Any]) -> float | None:
    for key in ONSET_KEYS:
        valor = row.get(key)
        if isinstance(valor, (int, float)):
            return float(valor)
    return None


def rows_per_turn(data: Any) -> list[dict[str, Any]]:
    """Encuentra el `per_turn` del JSON, venga de PoC1 o de PoC2.

    Se busca en vez de pedir la ruta exacta porque las dos PoC lo cuelgan en
    lugares distintos (`livekit_extra.arnes` en una, la raíz en la otra) y esa
    diferencia no vale un parámetro de línea de comandos.
    """
    if isinstance(data, dict):
        for key, valor in data.items():
            if key == "per_turn" and isinstance(valor, list):
                return [f for f in valor if isinstance(f, dict)]
            found_one = rows_per_turn(valor)
            if found_one:
                return found_one
    elif isinstance(data, list):
        for element in data:
            found_one = rows_per_turn(element)
            if found_one:
                return found_one
    return []


def compute(
    rows: list[dict[str, Any]], *, clips: Path | str = RUTA_CLIPS_DEFAULT
) -> list[TurnStart]:
    """La latencia de inicio de cada turno emparejado que traiga su clip."""
    clips = Path(clips)
    output: list[TurnStart] = []
    cero: float | None = None

    for row in rows:
        if not row.get("paired", True):
            continue
        clip_id = row.get("clip_id")
        start_of_clip = row.get("start_of_clip")
        onset = _estimated_onset(row)
        if not clip_id or not isinstance(start_of_clip, (int, float)):
            continue
        if onset is None:
            continue

        ruta = clips / f"{clip_id}.ulaw"
        if not ruta.is_file():
            ruta = clips / f"{clip_id}.wav"
        if not ruta.is_file():
            continue

        samples, _ = load_clip(ruta, publish_sample_rate=8000)
        published_threshold = row.get("speech_threshold_dbfs")
        head_s, end_s, umbral = speech_edges(
            samples,
            8000,
            umbral_dbfs=(
                float(published_threshold)
                if isinstance(published_threshold, (int, float))
                else None
            ),
        )

        published_end = row.get("measured_end_of_speech")
        disagreement = (
            (float(published_end) - (float(start_of_clip) + end_s)) * 1000.0
            if isinstance(published_end, (int, float))
            else None
        )

        if cero is None:
            cero = float(start_of_clip)
        output.append(
            TurnStart(
                turn=str(row.get("harness_turn") or row.get("turn") or "?"),
                clip_id=str(clip_id),
                t_in_call=float(start_of_clip) - cero,
                head_ms=head_s * 1000.0,
                latency_ms=(onset - (float(start_of_clip) + head_s)) * 1000.0,
                umbral_dbfs=umbral,
                end_disagreement_ms=disagreement,
            )
        )
    return output


def usable(
    turns: list[TurnStart],
) -> tuple[list[TurnStart], list[TurnStart]]:
    """Separa los turnos medibles de los que traen un arranque imposible.

    Una latencia de inicio **negativa** diría que el VAD marcó el arranque del
    habla antes de que sonara la primera muestra con voz del clip, y eso no
    puede pasar. Lo que pasó es que la marca no corresponde a ese clip: el VAD
    ya venía disparado desde antes, por el ruido de confort o por la cola de la
    racha anterior.

    Se apartan en vez de promediarse, por lo mismo que `harness_block` aparta
    los sesgos negativos: adentro se llevan el p50 a cualquier lado y el número
    resultante no mide nada.
    """
    return (
        [t for t in turns if t.latency_ms >= 0.0],
        [t for t in turns if t.latency_ms < 0.0],
    )


def summary(turns: list[TurnStart]) -> dict[str, Any]:
    """Los cinco números y la pendiente contra el tiempo de llamada.

    La pendiente es el punto: una latencia de inicio que **crece** dentro de la
    llamada es camino, no endpointing, porque aquí no hay endpointing que
    crecer. Ver el docstring del módulo.

    Los turnos con arranque imposible no entran al resumen; se cuentan aparte.
    """
    measurable, impossible_cases = usable(turns)
    if not measurable:
        return {"n": 0, "impossible_onsets": len(impossible_cases)}
    turns = measurable
    values = [t.latency_ms for t in turns]
    timings = [t.t_in_call for t in turns]
    ordered_items = sorted(values)
    data: dict[str, Any] = {
        "n": len(values),
        "impossible_onsets": len(impossible_cases),
        "min_ms": round(ordered_items[0], 1),
        "p50_ms": round(statistics.median(ordered_items), 1),
        "max_ms": round(ordered_items[-1], 1),
    }
    disagreements = [
        abs(t.end_disagreement_ms)
        for t in turns
        if t.end_disagreement_ms is not None
    ]
    if disagreements:
        data["end_disagreement_max_ms"] = round(max(disagreements), 1)
    if len(values) >= 3 and len(set(timings)) > 1:
        # Regresión simple: cuántos ms de latencia agrega cada segundo de
        # llamada. Con menos de 3 turnos no significa nada y no se publica.
        media_t = statistics.fmean(timings)
        media_v = statistics.fmean(values)
        num = sum((t - media_t) * (v - media_v) for t, v in zip(timings, values))
        den = sum((t - media_t) ** 2 for t in timings)
        if den > 0:
            data["slope_ms_per_s"] = round(num / den, 2)
    return data


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="shared.carrier_simulator.speech_onset",
        description=(
            "La latencia de inicio de cada turno de una corrida del arnés. No "
            "lleva endpointing adentro, así que separa camino de cero."
        ),
    )
    arg_parser.add_argument("json", help="el JSON de la corrida (PoC1 o PoC2)")
    arg_parser.add_argument(
        "--clips",
        default=str(RUTA_CLIPS_DEFAULT),
        help="carpeta de los clips preparados",
    )
    args = arg_parser.parse_args(argv)

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rows = rows_per_turn(data)
    if not rows:
        print(f"[FALLA] {args.json} no trae ningún bloque `per_turn`")
        return 1

    turns = compute(rows, clips=Path(args.clips))
    if not turns:
        print(
            "[FALLA] ningún turno tenía a la vez clip, inicio del clip y "
            "arranque estimado. ¿Están los clips preparados?"
        )
        return 1

    print(f"\nLatencia de inicio — {Path(args.json).name}\n")
    print(f"  {'turn':<6}{'t (s)':>8}{'cabeza ms':>11}{'LATENCIA ms':>13}{'dif fin':>9}")
    print("  " + "-" * 45)
    for t in turns:
        diff = "—" if t.end_disagreement_ms is None else f"{t.end_disagreement_ms:.1f}"
        print(
            f"  {t.turn:<6}{t.t_in_call:>8.1f}{t.head_ms:>11.1f}"
            f"{t.latency_ms:>13.1f}{diff:>9}"
        )

    summary_data = summary(turns)
    impossible_cases = summary_data.get("impossible_onsets", 0)
    if not summary_data["n"]:
        print(
            f"\n  [!] los {impossible_cases} turnos traen un arranque IMPOSIBLE "
            "(latencia negativa): la marca del VAD no corresponde al clip.\n"
            "      El VAD venía disparado desde antes — ruido de confort o la "
            "cola de la racha\n      anterior. En esta PoC la latencia de inicio "
            "no se puede leer; el sesgo de FIN sí."
        )
        return 1
    print(
        f"\n  n {summary_data['n']} · min {summary_data['min_ms']} · "
        f"p50 {summary_data['p50_ms']} · max {summary_data['max_ms']} ms"
    )
    if impossible_cases:
        print(
            f"\n  [!] {impossible_cases} turno(s) con arranque IMPOSIBLE (latencia "
            "negativa), fuera del resumen:\n"
            "      el VAD marcó el habla antes de que sonara la voz del clip, "
            "así que esa marca\n      no es de este clip. Con varios, el p50 de "
            "arriba tampoco es de fiar."
        )
    if "slope_ms_per_s" in summary_data:
        print(
            f"  pending: {summary_data['slope_ms_per_s']:+.2f} ms de "
            "latencia por segundo de llamada"
        )
    if summary_data.get("end_disagreement_max_ms", 0.0) > 1.0:
        print(
            f"\n  [!] el fin de voz recalculado difiere hasta "
            f"{summary_data['end_disagreement_max_ms']} ms del publicado: "
            "los clips en disco no son los que se reprodujeron"
        )
        return 1
    print(
        "\n  No lleva endpointing adentro: si crece con la llamada, la deriva\n"
        "  es del camino del audio y no del cero. Ver CONTRATO_METRICAS §2.2.2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
