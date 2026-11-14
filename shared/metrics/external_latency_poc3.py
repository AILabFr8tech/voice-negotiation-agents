"""PoC3 medida con el instrumento externo de PoC1 y PoC2.

PoC1 y PoC2 publican `line_latency_ms` —del fin de voz del arnés al
primer audio del agente, **las dos puntas fechadas por el arnés**
(`shared/carrier_simulator/cruce.py`)—. PoC3 no la produce: su arnés pasa
`intervenciones=None` a propósito (`poc3_gpt_live/app/harness.py:809`) y lo que
publica es `full_turn_ms`, que mide el **puente** por energía local.

Este módulo la reconstruye sin correr ninguna llamada, desde artefactos que ya
están en disco. Cierra la asimetría A1. Detalle y resultados en
`docs/instrumento_de_latency_poc3.md`.

## Las dos puntas

- **El cero** es `measured_end_of_speech` de `ceros_<sid>.jsonl`, en época. Es
  exactamente el mismo campo del mismo archivo que usan PoC1 y PoC2.
- **El acuse** es la primera voz del agente **oída por el arnés**, que vive en el
  canal 1 del WAV de auditoría.

## Por qué ese WAV sí sirve para milisegundos

Hay **dos** grabadoras en PoC3 y sólo una tiene el cabezal monótono:

- `poc3_gpt_live/diagnostics/audit_recorder.py` la llama el **puente**
  (`telefonia.py:779`) y sí lleva cabezal: 1.7–1.9 s de corrimiento sobre 90 s.
  Su canal del agente está en el eje de lo que el transportista *oiría*.
- `shared/carrier_simulator/auditoria.py` la llama el **arnés**, vía
  `Observado.annotate_audio` (`observado.py:231`), fechando en
  `time.time() - observado.inicio`. `_to_timeline` coloca cada trama en su
  instante exacto y no corre nada.

Los `auditoria_ARNES-*.json` de la campaña llevan el esquema de la compartida
—sin `cabezal`, sin `speech_onsets`—, o sea que los escribió el arnés. La
advertencia del propio artefacto ("sirve para oír quién habló primero, NO para
medir milisegundos") apunta al jitter del transporte; aquí ese jitter **es lo
que se quiere medir**, porque es el camino de vuelta y PoC1 y PoC2 lo llevan
dentro de su número igual.

## El ancla

El WAV está en el eje de la corrida y las marcas en época: falta un escalar por
llamada. Sale de correlacionar los bytes μ-law **del manifiesto** contra el
canal 0, que es lo que el arnés publicó. Los picos salen en 0.9998–0.9999 y la
dispersión entre clips es la cuantización de trama de 20 ms, así que la mediana
sobre los 8 clips deja el ancla con ~2 ms de error típico.

**No se ancla por índice**: un clip con pausa interna (`e3_t6`) produce dos
arranques y corre el pareo. Por eso se correlaciona la forma de onda entera.

## La misma regla, no sólo el mismo eje

`TurnoObservado.t_agent_first_speech` no es la primera trama con energía: es
el inicio de la primera **racha** posterior al cero que sobrevive a
`_streak_verdict`. Se replican las tres constantes del módulo compartido
—se importan, no se copian— y además se saltan las rachas que la corrida anotó
como descartadas en la bitácora, que es lo que atrapa el apoyo verbal que la
duración sola deja pasar.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import numpy as np

from shared.carrier_simulator.observed import (
    STREAK_HANGOVER_S,
    RMS_WITH_SPEECH,
)
from shared.carrier_simulator.player import MIN_SUBSTANTIVE_SPEECH_S

# Ventana de energía: la trama del transporte. No es una perilla — es el tamaño
# del paquete que el arnés recibe, así que medir más fino no aporta resolución.
WINDOW_S: Final[float] = 0.020

# Tolerancias del ancla. La dispersión esperada entre clips es la cuantización
# de trama (20 ms); 30 ms deja margen sin admitir un ancla equivocada, que
# fallaría por segundos y no por milisegundos.
ANCHOR_TOLERANCE_S: Final[float] = 0.030
MIN_CLIPS_TO_ANCHOR: Final[int] = 4
MIN_CORRELATION: Final[float] = 0.5

# Holgura hacia atrás al buscar el acuse oído: el acuse del puente se sella al
# salir la trama y la ventana de energía está cuantizada, así que la racha
# correspondiente puede empezar hasta tres ventanas antes.
ACKNOWLEDGEMENT_SLACK_S: Final[float] = 0.060

# Por encima de esto, la racha pertenece a otra intervención y no al turno que
# se está midiendo.
RETURN_PATH_MAX_S: Final[float] = 2.0


@dataclass(frozen=True)
class ExternalTurn:
    """Un turno de PoC3 con las dos puntas fechadas por el arnés."""

    call_sid: str
    scenario: int
    turn: str
    clip_id: str | None
    end_of_speech: float
    """Cero del turno: `measured_end_of_speech` de la marca, en época."""
    first_speech_heard: float
    """Acuse: primera racha sustantiva del agente en el canal 1, en época."""
    bridge_acknowledgement: float
    """`t_acuse_pared`: cuando el puente soltó la trama hacia el transporte."""
    full_turn_ms: float | None
    """Lo que publicó el puente por energía local, para poder restar."""

    @property
    def line_latency_ms(self) -> float:
        """La magnitud de PoC1 y PoC2, con la misma definición."""
        return round((self.first_speech_heard - self.end_of_speech) * 1000.0, 1)

    @property
    def return_path_ms(self) -> float:
        """Del acuse del puente a que el arnés lo oiga."""
        return round((self.first_speech_heard - self.bridge_acknowledgement) * 1000.0, 1)

    @property
    def until_bridge_acknowledgement_ms(self) -> float:
        """Del cero externo al acuse del puente: la `d` del discriminante.

        **No es el término del cero.** Es la latencia entera menos el camino de
        vuelta, o sea lo que mediría PoC3 si fechara su acuse donde lo fecha
        hoy pero contara desde el fin de voz real del clip. La diferencia
        contra `full_turn_ms` sí es el término del cero.
        """
        return round((self.bridge_acknowledgement - self.end_of_speech) * 1000.0, 1)


@dataclass(frozen=True)
class Anchor:
    """El escalar que lleva el eje de la corrida al eje de época, y su calidad."""

    displacement: float
    clips_used: int
    evaluated_clips: int
    dispersion_ms: float
    standard_error_ms: float
    min_correlation: float


# --------------------------------------------------------------------------
# μ-law, sin depender del puente
# --------------------------------------------------------------------------

def _tabla_ulaw() -> np.ndarray:
    """La tabla de 256 entradas, armada una vez.

    Se arma aquí en vez de importarla de `poc3_gpt_live` porque este módulo es
    compartido y no debe depender de una PoC. El resultado es idéntico: es el
    estándar G.711, no una implementación con criterio.
    """
    raw_value = np.arange(256, dtype=np.uint8)
    x = ~raw_value
    sign = (x & 0x80) != 0
    exponent = (x >> 4) & 0x07
    mantissa = x & 0x0F
    magnitude = (((mantissa.astype(np.int32) << 1) + 33) << exponent) - 33
    return np.where(sign, -magnitude, magnitude).astype(np.int16)


_TABLA: Final[np.ndarray] = _tabla_ulaw()


def ulaw_a_pcm16(raw_value: bytes) -> np.ndarray:
    return _TABLA[np.frombuffer(raw_value, dtype=np.uint8)]


# --------------------------------------------------------------------------
# Energía y rachas
# --------------------------------------------------------------------------

def envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS por ventana de `WINDOW_S`."""
    per_window = max(1, int(sample_rate * WINDOW_S))
    n = samples.size // per_window
    if n == 0:
        return np.zeros(0)
    blocks = samples[: n * per_window].astype(np.float64).reshape(n, per_window)
    return np.sqrt((blocks ** 2).mean(axis=1))


def speech_streaks(env: np.ndarray) -> list[tuple[float, float]]:
    """`(inicio, fin)` de cada racha, con el hangover del módulo compartido.

    Es la misma regla de agrupamiento que `Observado.annotate_audio`: una ventana
    con energía extiende la racha abierta si el último sonido cae dentro del
    hangover, y si no abre una nueva.
    """
    with_voice = np.flatnonzero(env >= RMS_WITH_SPEECH)
    if with_voice.size == 0:
        return []
    output: list[tuple[float, float]] = []
    start = ultimo = int(with_voice[0])
    for i in with_voice[1:]:
        if (int(i) - ultimo) * WINDOW_S > STREAK_HANGOVER_S:
            output.append((start * WINDOW_S, ultimo * WINDOW_S + WINDOW_S))
            start = int(i)
        ultimo = int(i)
    output.append((start * WINDOW_S, ultimo * WINDOW_S + WINDOW_S))
    return output


# --------------------------------------------------------------------------
# El ancla
# --------------------------------------------------------------------------

def clip_position(canal: np.ndarray, clip: np.ndarray) -> tuple[int, float, float]:
    """Dónde está `clip` dentro de `canal`, por correlación normalizada.

    Devuelve `(muestra_de_inicio, pico, mejor_pico_lejano)`. El segundo valor
    lejano sirve para exigir que el pico sea un pico y no una meseta: el canal 0
    lleva el clip literal, así que un pico sano sale por encima de 0.999 y el
    resto de la correlación muy por debajo.
    """
    from scipy.signal import fftconvolve

    x = canal.astype(np.float64)
    y = clip - clip.mean()
    numerator = fftconvolve(x, y[::-1], mode="valid")
    cumulative = np.concatenate(([0.0], np.cumsum(x * x)))
    energy = cumulative[y.size:] - cumulative[: -y.size]
    denominator = np.sqrt(np.maximum(energy, 1e-9) * float((y * y).sum()))
    r = numerator / denominator
    i = int(np.argmax(r))
    far = np.abs(np.arange(r.size) - i) > 400  # 50 ms a 8 kHz
    return i, float(r[i]), float(r[far].max()) if far.any() else 0.0


def anchor_on(
    marks: Sequence[dict[str, Any]],
    published_channel: np.ndarray,
    sample_rate: int,
    *,
    clips_path: Path,
) -> Anchor | None:
    """El desplazamiento que lleva el eje de la corrida a época.

    `None` si los clips no concuerdan entre sí: es preferible perder la llamada
    a anclarla mal, porque un ancla equivocada falla por segundos y se vería
    como una latencia absurda en vez de como un error de método.
    """
    estimates: list[float] = []
    peaks: list[float] = []
    for mark in marks:
        clip_id = mark.get("clip_id")
        if not clip_id:
            continue
        ruta = clips_path / f"{clip_id}.ulaw"
        if not ruta.exists():
            continue
        clip = ulaw_a_pcm16(ruta.read_bytes()).astype(np.float64)
        if clip.size < sample_rate // 2:  # menos de medio segundo no ancla nada
            continue
        i, pico, distant = clip_position(published_channel, clip)
        if pico < MIN_CORRELATION or pico < distant * 1.5:
            continue
        estimates.append(mark["start_of_clip"] - i / sample_rate)
        peaks.append(pico)

    if len(estimates) < MIN_CLIPS_TO_ANCHOR:
        return None
    values = np.array(estimates)
    median_value = float(np.median(values))
    good = np.abs(values - median_value) <= ANCHOR_TOLERANCE_S
    if int(good.sum()) < MIN_CLIPS_TO_ANCHOR:
        return None
    displacement = float(np.median(values[good]))
    residues = np.abs(values[good] - displacement)
    return Anchor(
        displacement=displacement,
        clips_used=int(good.sum()),
        evaluated_clips=len(estimates),
        dispersion_ms=float(residues.max() * 1000.0),
        standard_error_ms=float(
            np.std(values[good]) / np.sqrt(good.sum()) * 1000.0
        ),
        min_correlation=float(min(peaks)),
    )


# --------------------------------------------------------------------------
# Lectura de artefactos
# --------------------------------------------------------------------------

def marks_de(zeros_path: Path) -> list[dict[str, Any]]:
    """Las marcas del arnés con su `measured_end_of_speech`."""
    output = []
    for linea in zeros_path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        on_top = json.loads(linea)
        if on_top.get("tipo") != "mark":
            continue
        mark = on_top["mark"]
        if mark.get("measured_end_of_speech") is not None:
            output.append(mark)
    return output


def axis_samples(bridge_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Las muestras de régimen del eje de pared, con sus dos instantes."""
    axis = bridge_json.get("gpt_live_extra", {}).get("wall_axis", {})
    if axis.get("fault") is not None:
        return []
    return [
        m
        for m in axis.get("samples", [])
        if not m.get("is_opening")
        and m.get("t_ack_wall") is not None
        and m.get("t_zero_wall") is not None
    ]


def discards_of(event_log: dict[str, Any], anchor: float) -> list[float]:
    """Inicios, en época, de las rachas que la corrida descartó en vivo.

    Sirve para dos cosas: saltarlas al elegir el acuse (es lo que hace la regla
    de producción) y **validar el ancla**, porque cada una tiene que caer sobre
    una racha detectada en el WAV.
    """
    output = []
    for turn in event_log.get("turns", []):
        for fragment in turn.get("discarded_fragments", []) or []:
            output.append(fragment["start"] + anchor)
    return output


def leer_wav(ruta: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(ruta), "rb") as w:
        raw_value = w.readframes(w.getnframes())
        channel_list, sample_rate = w.getnchannels(), w.getframerate()
    return np.frombuffer(raw_value, dtype="<i2").reshape(-1, channel_list), sample_rate


# --------------------------------------------------------------------------
# El cruce
# --------------------------------------------------------------------------

def pair(
    marks: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Cada marca con la muestra cuyo cero le queda más cerca.

    Por cercanía y no por índice: cuando un turno no produjo muestra, el pareo
    posicional corre todo lo que sigue y aparecen `d` de decenas de segundos.
    La tolerancia de 1.5 s es holgada para el desfase real (~150 ms) y estrecha
    frente al hueco entre dos turnos del guion (7–25 s).
    """
    output = []
    used: set[int] = set()
    for mark in marks:
        end_of_speech = mark["measured_end_of_speech"]
        best: int | None = None
        distance: float | None = None
        for i, sample in enumerate(samples):
            if i in used:
                continue
            d = abs(sample["t_zero_wall"] - end_of_speech)
            if distance is None or d < distance:
                distance, best = d, i
        if best is None or distance is None or distance > 1.5:
            continue
        used.add(best)
        output.append((mark, samples[best]))
    return output


def call_turns(
    *,
    call_sid: str,
    scenario: int,
    zeros_path: Path,
    bridge_json_path: Path,
    wav_path: Path,
    path_of_log: Path | None,
    clips_path: Path,
) -> tuple[list[ExternalTurn], Anchor | None, dict[str, Any]]:
    """Los turnos de una llamada de PoC3, medidos por fuera.

    Devuelve `(turnos, ancla, informe)`. El informe lleva la validación del
    ancla contra los descartes, que es lo que permite afirmar que el eje quedó
    bien puesto sin tener que creerle a la correlación.
    """
    live_bridge = json.loads(bridge_json_path.read_text(encoding="utf-8"))
    if not live_bridge.get("clock", {}).get("interpretable"):
        return [], None, {"motivo": "reloj no interpretable"}
    samples = axis_samples(live_bridge)
    if not samples:
        return [], None, {"motivo": "sin eje de pared"}
    marks = marks_de(zeros_path)
    if not marks:
        return [], None, {"motivo": "sin marcas"}

    audio, sample_rate = leer_wav(wav_path)
    anchor = anchor_on(marks, audio[:, 0], sample_rate, clips_path=clips_path)
    if anchor is None:
        return [], None, {"motivo": "no se pudo anclar"}

    streaks = [
        (start + anchor.displacement, end + anchor.displacement)
        for start, end in speech_streaks(envelope(audio[:, 1], sample_rate))
    ]
    descartes: list[float] = []
    if path_of_log is not None and path_of_log.exists():
        descartes = discards_of(
            json.loads(path_of_log.read_text(encoding="utf-8")),
            anchor.displacement,
        )
    over_a_streak = sum(
        1
        for d in descartes
        if any(start - 0.10 <= d <= end + 0.10 for start, end in streaks)
    )

    turns: list[ExternalTurn] = []
    without_acknowledgement = 0
    for mark, sample in pair(marks, samples):
        acknowledgement = sample["t_ack_wall"]
        chosen: float | None = None
        for start, end in streaks:
            if end < acknowledgement - ACKNOWLEDGEMENT_SLACK_S:
                continue
            if (end - start) < MIN_SUBSTANTIVE_SPEECH_S:
                continue
            if any(abs(start - d) <= 0.10 for d in descartes):
                continue
            chosen = start
            break
        if chosen is None or not (
            -ACKNOWLEDGEMENT_SLACK_S
            <= chosen - acknowledgement
            <= RETURN_PATH_MAX_S
        ):
            without_acknowledgement += 1
            continue
        turns.append(
            ExternalTurn(
                call_sid=call_sid,
                scenario=scenario,
                turn=mark["turn"],
                clip_id=mark.get("clip_id"),
                end_of_speech=mark["measured_end_of_speech"],
                first_speech_heard=chosen,
                bridge_acknowledgement=acknowledgement,
                full_turn_ms=sample.get("full_turn_ms"),
            )
        )
    report = {
        "marks": len(marks),
        "measured_turns": len(turns),
        "no_usable_acknowledgement": without_acknowledgement,
        "descartes": len(descartes),
        "discards_over_a_streak": over_a_streak,
        "anchor": anchor.__dict__,
    }
    return turns, anchor, report


def campaign_turns(
    *,
    progress: Path,
    harness_folder: Path,
    recordings_folder: Path,
    clips_path: Path,
) -> tuple[list[ExternalTurn], list[dict[str, Any]]]:
    """Recorre un `progreso.jsonl` de PoC3 y devuelve todos sus turnos."""
    all_entries: list[ExternalTurn] = []
    reports: list[dict[str, Any]] = []
    for linea in progress.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        row = json.loads(linea)
        sid = row["label"]
        paths = {
            "zeros_path": harness_folder / f"zeros_{sid}.jsonl",
            "bridge_json_path": recordings_folder / f"call_{sid}.json",
            "wav_path": harness_folder / f"auditoria_{sid}.wav",
            "path_of_log": harness_folder / f"event_log_{sid}.json",
        }
        if not all(
            r.exists() for k, r in paths.items() if k != "path_of_log"
        ):
            reports.append({"call_sid": sid, "motivo": "falta un artefacto"})
            continue
        turns, _, report = call_turns(
            call_sid=sid,
            scenario=row["scenario"],
            clips_path=clips_path,
            **paths,
        )
        all_entries.extend(turns)
        reports.append({"call_sid": sid, **report})
    return all_entries, reports


def percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    """p50/p95/p99 con interpolación lineal, como el resto de `shared/metrics/`."""
    v = sorted(values)
    if not v:
        return {"n": 0, "p50": None, "p95": None, "p99": None}

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        low = int(k)
        alto = min(low + 1, len(v) - 1)
        return round(v[low] + (v[alto] - v[low]) * (k - low), 1)

    return {"n": len(v), "min": q(0.0), "p50": q(0.5), "p95": q(0.95),
            "p99": q(0.99), "max": q(1.0)}
