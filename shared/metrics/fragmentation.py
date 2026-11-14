"""¿La fragmentación es del framework o del estímulo? Se decide después y gratis.

## La pregunta

La tasa de fragmentación (§2.2.5 del contrato) dice cuántos turnos cierra el
pipeline por cada intervención que oye: 1.57–1.62 en PoC1, 1.00 en PoC2. Se
reporta como métrica comparativa —cuán finito corta el VAD de cada framework con
su perilla congelada—, y ese enunciado solo se sostiene si los cortes son del
VAD.

Si resulta que los cortes caen **en las pausas internas de los clips**, la
fragmentación mide otra cosa: cómo habló quien grabó el estímulo, filtrado por
la perilla de cada framework. Sigue siendo comparable —los dos oyen el mismo
audio— pero el enunciado del reporte cambia, y con él lo que se puede concluir
sobre la conversación real, donde nadie habla como el clip.

## Cómo se contesta

Los dos datos ya existen y no cuesta nada cruzarlos:

- **Las pausas de cada clip.** Un hueco de silencio *entre* dos tramos de voz,
  medido con el **mismo umbral** con el que se midió su fin de voz —el
  `speech_threshold_dbfs` que viaja en la marca de cada turno—. No se re-deriva del
  pico: usar otro número haría incomparables estas pausas con todo lo demás.
- **Dónde cortó el VAD.** Cada evento del pipeline que **no** es el último de su
  intervención es un corte prematuro: el VAD dio por terminado el turno y el
  transportista siguió hablando. Su `fin` ya es el cero honesto, así que su
  distancia al inicio del clip dice en qué segundo del clip cortó.

Si cada corte prematuro cae en una pausa, la fragmentación es del estímulo.

Hacen falta las dos mitades. **Dónde** cortó dice si el estímulo puso el hueco;
**cuántas** de las pausas disponibles se cobraron dice si el estímulo basta para
predecir el número. Y sin nivel de azar ninguna de las dos significa nada: estos
clips tienen tanto silencio interno que un punto cualquiera de su tramo hablado
cae en una pausa el 57% de las veces, así que "cayó en una pausa" solo es un
hallazgo comparado contra eso.

## Un confound que hay que declarar

Las pausas se miden sobre **el clip**, y lo que el VAD oyó fue el clip **más el
ruido de confort** que el arnés publica de fondo sin parar
(`COMFORT_NOISE_PEAK`). Durante una pausa interna el cable no lleva silencio:
lleva ruido. Esa es la explicación más probable de por qué solo una fracción de
las pausas se vuelve corte —Silero sigue viendo algo— y significa que el número
exacto de cortes depende también de una constante del arnés, no solo de la
perilla del framework. La constante es la misma en las 3 PoC, así que la
comparación se sostiene; el valor absoluto no es el que daría una llamada real.

## El contrafactual que hace decisiva la respuesta

Contar cuántas de esas mismas pausas superan `min_silence_duration` de LiveKit
(0.55 s → 18 ventanas → 576 ms) y cuántas superan `stop_secs` de pipecat (0.2 s
→ 6 ventanas → 192 ms). Si las pausas de los clips viven entre esos dos
umbrales, la diferencia 1.57 contra 1.00 no es "un VAD corta más fino" sino
**exactamente la perilla, contra el mismo audio** — que es un enunciado más
fuerte y más honesto, y se puede escribir con número.

## Uso

    python -m shared.metrics.fragmentacion <cero_*.json> [...]

Devuelve 1 si no pudo cruzar nada.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio.level import windows_with_speech
from shared.carrier_simulator.cero import ZeroMark
from shared.carrier_simulator.cruce import EstimatedZero, Line, group
from shared.carrier_simulator.script import RUTA_CLIPS_DEFAULT, load_clip

# La pausa mínima que cuenta. 200 ms no es arbitrario: es el `stop_secs` de
# pipecat, o sea el hueco más corto que su VAD puede llegar a tomar por fin de
# turno. Por debajo de eso ningún framework de la comparativa puede cortar.
MIN_PAUSE_S: Final[float] = 0.2

# Las perillas congeladas de cada framework, ya cuantizadas a la ventana de
# Silero (512 muestras a 16 kHz = 32 ms), que es como las aplica de verdad.
SILERO_WINDOW_S: Final[float] = 0.032
STOP_SECS_POC1: Final[float] = 0.2
MIN_SILENCE_POC2: Final[float] = 0.55


def effective_threshold_of_pipecat(stop_secs: float) -> float:
    """Lo que pipecat espera de verdad. **Redondea**, no trunca ni sube.

    `pipecat/audio/vad/vad_analyzer.py`: `self._vad_stop_frames =
    round(self._params.stop_secs / vad_frames_per_sec)`. Cuenta tramas, y el
    redondeo de 6.25 baja a 6: con `stop_secs=0.2` espera **192 ms**, menos que
    lo configurado.
    """
    return round(stop_secs / SILERO_WINDOW_S) * SILERO_WINDOW_S


def effective_threshold_of_livekit(min_silence: float) -> float:
    """Lo que LiveKit espera de verdad. **Sube** a la ventana siguiente.

    `livekit/plugins/silero/vad.py`: acumula `silence_threshold_duration +=
    window_duration` y dispara con `>= min_silence_duration`. No cuenta tramas
    contra un entero redondeado: compara duración, así que hacen falta 18
    ventanas para pasar de 0.55 s y el umbral real es **576 ms**.

    Las dos perillas se cuantizan distinto, y en direcciones contrarias. No es
    un detalle de implementación que se pueda promediar: ensancha la banda que
    separa a los dos frameworks de [200, 550) a [192, 576) ms.
    """
    windows = -(-min_silence // SILERO_WINDOW_S)  # techo, sin importar math
    return windows * SILERO_WINDOW_S


EFFECTIVE_THRESHOLD_POC1: Final[float] = effective_threshold_of_pipecat(STOP_SECS_POC1)
EFFECTIVE_THRESHOLD_POC2: Final[float] = effective_threshold_of_livekit(MIN_SILENCE_POC2)

# Resolución con la que se barre el clip para calcular el nivel de azar. Es la
# misma ventana con la que se midieron las pausas, así que no introduce una
# rejilla nueva.
SWEEP_STEP_S: Final[float] = 0.01


def inside_the_pause(offset: float, pause: Pause, bias_s: float) -> bool:
    """Si un cero del pipeline en `offset` es atribuible a esa pausa.

    La ventana no es simétrica y no debería serlo. Por la izquierda no hay
    slack: el cero honesto dice "aquí dejó de haber voz", y no puede ser
    **antes** de que el silencio empiece — un cero anterior al inicio de la
    pausa es un corte a media palabra, que es justo lo que hay que distinguir.
    Por la derecha se le suma el sesgo de la corrida (la mediana de su
    `sesgo_de_voz_ms`), porque un cero prematuro llega por el mismo camino que
    uno final y paga el mismo retraso. Sin esa corrección, un corte
    perfectamente puesto se leería como desalineado.
    """
    return pause.start <= offset <= pause.end + bias_s


@dataclass(frozen=True)
class Pause:
    """Un hueco de silencio entre dos tramos de voz, en segundos del clip."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def a_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start, 3),
            "end_s": round(self.end, 3),
            "duration_ms": round(self.duration * 1000.0, 1),
        }


def internal_pauses(
    samples: np.ndarray,
    sample_rate: int,
    *,
    umbral_dbfs: float | None = None,
    min_s: float = MIN_PAUSE_S,
) -> list[Pause]:
    """Los huecos de silencio **entre** dos tramos de voz del clip.

    Internas por construcción: cada pausa sale de dos ventanas con voz
    consecutivas en el tiempo pero no contiguas, así que tiene voz a los dos
    lados. La cabeza y la cola del clip no son pausas — la cola es justamente el
    fin de voz, y contarla haría que todo clip "terminara en pausa".
    """
    with_voice, per_window, _ = windows_with_speech(samples, sample_rate, umbral_dbfs=umbral_dbfs)
    if with_voice.size < 2:
        return []
    step = per_window / float(sample_rate)
    pauses: list[Pause] = []
    for anterior, following in zip(with_voice[:-1], with_voice[1:]):
        if following - anterior <= 1:
            continue
        pause = Pause(
            start=(int(anterior) + 1) * step, end=int(following) * step
        )
        if pause.duration >= min_s:
            pauses.append(pause)
    return pauses


@dataclass(frozen=True)
class Cut:
    """Un corte prematuro del VAD, situado dentro del clip que lo provocó."""

    turn: str
    clip_id: str
    # En qué segundo del clip cortó el VAD.
    offset_seg: float
    # La pausa a la que se le atribuye —la que lo contiene, o la más cercana si
    # ninguna lo contiene— y a qué distancia de su inicio quedó.
    pause: Pause | None
    distance_s: float | None
    in_a_pause: bool = False

    def a_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "clip_id": self.clip_id,
            "cut_at_s": round(self.offset_seg, 3),
            "pause": self.pause.a_dict() if self.pause else None,
            "distance_to_start_ms": (
                None if self.distance_s is None else round(self.distance_s * 1000.0, 1)
            ),
            "in_a_pause": self.in_a_pause,
        }


def _attribute(
    pauses: list[Pause], offset: float, bias_s: float
) -> tuple[Pause | None, float | None, bool]:
    """A qué pausa se le atribuye el corte, y si de verdad cayó dentro.

    Primero se busca una que lo **contenga**; solo si ninguna lo hace se reporta
    la más cercana, para poder ver de cuánto falló. La distancia se mide contra
    el inicio de la pausa y no contra su centro: el cero honesto es "aquí dejó
    de haber voz", así que un corte bien puesto cae donde la pausa empieza, y
    contra el centro una pausa larga se vería desalineada estando perfecta.
    """
    if not pauses:
        return None, None, False
    for pause in pauses:
        if inside_the_pause(offset, pause, bias_s):
            return pause, offset - pause.start, True
    nearby = min(pauses, key=lambda p: abs(offset - p.start))
    return nearby, offset - nearby.start, False


def chance_fraction(
    pauses: list[Pause],
    *,
    speech_onset: float,
    end_of_speech: float,
    bias_s: float,
    step: float = SWEEP_STEP_S,
) -> float | None:
    """Qué parte del clip cae dentro de una pausa, barriendo su tramo hablado.

    Es el nivel de azar, y sin él el resultado no dice nada: si las pausas
    cubrieran media grabación, "el corte cayó en una pausa" sería cierto la
    mitad de las veces poniendo los cortes al azar. Se barre en rejilla y no con
    números aleatorios a propósito — una medición no debería depender de una
    semilla, y la rejilla exhaustiva da la respuesta exacta.
    """
    if end_of_speech <= speech_onset:
        return None
    # Índice entero y no `t += paso`: acumular flotantes desvía la rejilla y se
    # come el último punto —sobre [0, 2] a 10 ms daba 200 puntos en vez de 201—,
    # que es justo el error que haría irreproducible un número que se publica.
    total = int((end_of_speech - speech_onset) / step) + 1
    inside = sum(
        1
        for i in range(total)
        if any(
            inside_the_pause(speech_onset + i * step, p, bias_s) for p in pauses
        )
    )
    return inside / total if total else None


def premature_cuts(
    lines: list[Line],
    pauses_per_clip: dict[str, list[Pause]],
    *,
    bias_s: float = 0.0,
) -> list[Cut]:
    """Sitúa dentro de su clip cada evento que no fue el último de su intervención.

    El último evento **no** es un corte prematuro: es el fin de la intervención,
    y ahí el transportista sí se calló. Los anteriores son los que partieron la
    intervención en dos, y son los únicos que la pregunta necesita.
    """
    cortes: list[Cut] = []
    for line in lines:
        events = [c for c in line.events if c.end is not None]
        if len(events) < 2:
            continue
        events.sort(key=lambda c: c.end or 0.0)
        pauses = pauses_per_clip.get(line.clip_id, [])
        for event in events[:-1]:
            offset = float(event.end) - line.mark.start_of_clip  # type: ignore[arg-type]
            pause, distance, inside = _attribute(pauses, offset, bias_s)
            cortes.append(
                Cut(
                    turn=line.turn,
                    clip_id=line.clip_id,
                    offset_seg=offset,
                    pause=pause,
                    distance_s=distance,
                    in_a_pause=inside,
                )
            )
    return cortes


def pause_census(analyzed_ones: dict[str, AnalyzedClip]) -> dict[str, Any]:
    pauses_per_clip = {k: v.pauses for k, v in analyzed_ones.items()}
    return _census(pauses_per_clip)


def _census(pauses_per_clip: dict[str, list[Pause]]) -> dict[str, Any]:
    """Cuántas pausas hay, y cuántas pasan la perilla de cada framework.

    Es el contrafactual: si casi todas las pausas caen entre los dos umbrales,
    la diferencia de fragmentación es la perilla y no el VAD.
    """
    all_items = [p for pauses in pauses_per_clip.values() for p in pauses]
    durations = sorted(p.duration for p in all_items)
    # La banda que separa a las dos PoC: pausas largas para que pipecat las
    # tome por fin de turno y cortas para que LiveKit no. Es el contrafactual.
    only_poc1 = [d for d in durations if EFFECTIVE_THRESHOLD_POC1 <= d < EFFECTIVE_THRESHOLD_POC2]
    both = [d for d in durations if d >= EFFECTIVE_THRESHOLD_POC2]
    return {
        "clips": len(pauses_per_clip),
        "pauses": len(all_items),
        "clips_with_any": sum(1 for p in pauses_per_clip.values() if p),
        "duration_ms": {
            "min": round(durations[0] * 1000.0, 1) if durations else None,
            "p50": (
                round(durations[len(durations) // 2] * 1000.0, 1) if durations else None
            ),
            "max": round(durations[-1] * 1000.0, 1) if durations else None,
        },
        "effective_threshold_poc1_ms": round(EFFECTIVE_THRESHOLD_POC1 * 1000.0, 1),
        "effective_threshold_poc2_ms": round(EFFECTIVE_THRESHOLD_POC2 * 1000.0, 1),
        # No se publica un "cortarían en PoC1": con el mínimo de detección en
        # 200 ms, por encima de los 192 efectivos, serían todas por definición y
        # el número no diría nada.
        "in_the_band_only_poc1": len(only_poc1),
        "above_the_poc2_threshold": len(both),
        "how_to_read": (
            "una pausa en la banda [192, 576) ms es un corte que pipecat puede "
            "hacer y LiveKit no, sobre exactamente el mismo audio. Si ahí está "
            "la mayoría, la diferencia de fragmentación es la perilla"
        ),
    }


# --------------------------------------------------------------------------
# Lectura de una corrida
# --------------------------------------------------------------------------


def jsonl_marks(ruta: Path) -> list[ZeroMark]:
    """Las marcas que el arnés publicó, leídas de su `.jsonl`."""
    marks: list[ZeroMark] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        on_top = json.loads(linea)
        if on_top.get("tipo") != "mark":
            continue
        marks.append(ZeroMark.from_dict(on_top["mark"]))
    return marks


def path_of_the_marks(crossing_path: Path) -> Path:
    """`cero_<etiqueta>.json` → `ceros_<etiqueta>.jsonl`, que es su pareja."""
    label = crossing_path.stem.removeprefix("cero_")
    return crossing_path.with_name(f"zeros_{label}.jsonl")


@dataclass(frozen=True)
class AnalyzedClip:
    """Las pausas de un clip y el tramo hablado en el que pueden caer."""

    clip_id: str
    pauses: list[Pause]
    speech_onset: float
    end_of_speech: float


def analyze_clips(
    clips: list[str],
    *,
    folder: Path = RUTA_CLIPS_DEFAULT,
    thresholds: dict[str, float] | None = None,
    min_s: float = MIN_PAUSE_S,
) -> dict[str, AnalyzedClip]:
    """Las pausas internas de cada clip, con el umbral con que se midió su fin.

    `umbrales` viene de las marcas de la corrida. Sin él se re-deriva del pico,
    que da otro número y deja de ser comparable con el `measured_end_of_speech`.
    """
    from shared.audio import TELEPHONY_SAMPLE_RATE
    from shared.carrier_simulator.speech_onset import speech_edges

    thresholds = thresholds or {}
    analyzed_ones: dict[str, AnalyzedClip] = {}
    for clip_id in clips:
        ruta = folder / f"{clip_id}.ulaw"
        if not ruta.is_file():
            ruta = folder / f"{clip_id}.wav"
        if not ruta.is_file():
            continue
        samples, _ = load_clip(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        umbral = thresholds.get(clip_id)
        start, end, _ = speech_edges(samples, TELEPHONY_SAMPLE_RATE, umbral_dbfs=umbral)
        analyzed_ones[clip_id] = AnalyzedClip(
            clip_id=clip_id,
            pauses=internal_pauses(
                samples, TELEPHONY_SAMPLE_RATE, umbral_dbfs=umbral, min_s=min_s
            ),
            speech_onset=start,
            end_of_speech=end,
        )
    return analyzed_ones


def cross_one_run(
    crossing_path: Path, *, folder: Path = RUTA_CLIPS_DEFAULT
) -> dict[str, Any] | None:
    """Cruza los cortes de una corrida contra las pausas de los clips que sonaron."""
    if not crossing_path.is_file():
        return None
    cruce = json.loads(crossing_path.read_text(encoding="utf-8"))
    marks_path = path_of_the_marks(crossing_path)
    if not marks_path.is_file():
        return None
    marks = jsonl_marks(marks_path)
    zeros = [
        EstimatedZero(indice=int(c["indice"]), start=c.get("start"), end=c.get("end"))
        for c in cruce.get("zeros_of_pipeline", [])
    ]
    if not marks or not zeros:
        return None

    lines, _ = group(marks, zeros)
    thresholds = {m.clip_id: m.speech_threshold_dbfs for m in marks if m.speech_threshold_dbfs}
    analyzed_ones = analyze_clips(
        sorted({m.clip_id for m in marks}), folder=folder, thresholds=thresholds
    )
    # El sesgo de esta corrida: un cero prematuro llega por el mismo camino que
    # uno final y paga el mismo retraso. Ver `dentro_de_la_pausa`.
    bias_ms = (cruce.get("speech_bias_ms") or {}).get("median_value")
    bias_s = float(bias_ms) / 1000.0 if bias_ms is not None else 0.0

    pauses_per_clip = {k: v.pauses for k, v in analyzed_ones.items()}
    cortes = premature_cuts(lines, pauses_per_clip, bias_s=bias_s)
    in_pause = [c for c in cortes if c.in_a_pause]

    # El azar se promedia sobre los clips que de verdad se partieron, no sobre
    # todos: el denominador tiene que ser la población de la que salieron los
    # cortes, o el nivel de azar describiría otra cosa.
    split_clips = sorted({c.clip_id for c in cortes})
    random_draws = [
        f
        for clip_id in split_clips
        if (analyzed_one := analyzed_ones.get(clip_id)) is not None
        and (
            f := chance_fraction(
                analyzed_one.pauses,
                speech_onset=analyzed_one.speech_onset,
                end_of_speech=analyzed_one.end_of_speech,
                bias_s=bias_s,
            )
        )
        is not None
    ]
    by_chance = sum(random_draws) / len(random_draws) if random_draws else None

    # Cuántas de las oportunidades que el clip ofrecía se volvieron corte. Es la
    # otra mitad de la respuesta: el estímulo pone las pausas, la perilla decide
    # cuáles se cobran.
    opportunities = sum(
        1
        for clip_id in split_clips
        for p in analyzed_ones.get(clip_id, AnalyzedClip(clip_id, [], 0.0, 0.0)).pauses
        if p.duration >= EFFECTIVE_THRESHOLD_POC1
    )

    return {
        "run": crossing_path.stem,
        "speech_bias_median_ms": bias_ms,
        "lines": len(lines),
        "events": sum(i.fragmentation for i in lines),
        "premature_cuts": len(cortes),
        "cuts_in_a_clip_pause": len(in_pause),
        "chance_fraction": round(by_chance, 3) if by_chance is not None else None,
        "opportunities_in_split_clips": opportunities,
        "detail": [c.a_dict() for c in cortes],
        "pauses_of_clips_that_played": pause_census(analyzed_ones),
    }


def verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Junta las corridas y dice a quién se le atribuye la fragmentación.

    Dos preguntas, y hacen falta las dos. **Dónde** cortó el VAD dice si el
    estímulo puso el hueco; **cuántas** de las pausas disponibles se volvieron
    corte dice si el estímulo basta para predecir el número. Un VAD que corta
    solo en pausas pero en una de cada cuatro no está midiendo el clip: está
    midiendo lo que su perilla hace con el clip.
    """
    cortes = sum(c["premature_cuts"] for c in runs)
    in_pause = sum(c["cuts_in_a_clip_pause"] for c in runs)
    opportunities = sum(c["opportunities_in_split_clips"] for c in runs)
    random_draws = [c["chance_fraction"] for c in runs if c["chance_fraction"]]
    by_chance = sum(random_draws) / len(random_draws) if random_draws else None

    fraction = (in_pause / cortes) if cortes else None
    used = (cortes / opportunities) if opportunities else None

    if fraction is None:
        reading = "no hubo cortes prematuros que cruzar"
    elif by_chance is not None and fraction <= by_chance * 1.5:
        reading = (
            "los cortes caen en pausas no más de lo que caería cualquier punto "
            "al azar: el clip no explica dónde cortó el VAD"
        )
    elif used is not None and used >= 0.8:
        reading = (
            "el VAD corta en las pausas del clip y en casi todas: la "
            "fragmentación la predice el ESTÍMULO, y el reporte no puede "
            "presentarla como una propiedad del framework"
        )
    else:
        reading = (
            "el estímulo pone la OPORTUNIDAD y la perilla decide cuáles se "
            "cobran: los cortes caen en pausas del clip muy por encima del "
            "azar, pero solo en una fracción de las que había. La tasa no es un "
            "artefacto, y tampoco es una propiedad del framework sola — se "
            "reporta como la pareja (perilla, estímulo) que es"
        )
    return {
        "runs": len(runs),
        "premature_cuts": cortes,
        "cuts_in_a_clip_pause": in_pause,
        "paused_fraction": round(fraction, 3) if fraction is not None else None,
        "chance_fraction": round(by_chance, 3) if by_chance is not None else None,
        "opportunities": opportunities,
        "opportunities_taken": (
            round(used, 3) if used is not None else None
        ),
        "verdict": reading,
    }


def script_census(folder: Path = RUTA_CLIPS_DEFAULT) -> dict[str, Any]:
    """Las pausas de **los 49 clips**, no solo de los que sonaron en una corrida.

    Es la caracterización del estímulo entero, y es la que dice qué va a pasar
    en la campaña completa antes de correrla: cuántos cortes de más puede llegar
    a hacer pipecat sobre el mismo audio con el que LiveKit no hará ninguno.
    """
    manifest = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    entries = manifest["clips"]
    thresholds = {
        e["id"]: e["speech_threshold_dbfs"]
        for e in entries
        if e.get("speech_threshold_dbfs") is not None
    }
    analyzed_ones = analyze_clips(
        [e["id"] for e in entries], folder=folder, thresholds=thresholds
    )
    census = pause_census(analyzed_ones)
    census["per_scenario"] = {}
    for entry in entries:
        scenario = str(entry.get("scenario"))
        analyzed_one = analyzed_ones.get(entry["id"])
        if analyzed_one is None:
            continue
        running_count = census["per_scenario"].setdefault(
            scenario, {"clips": 0, "pauses": 0, "only_poc1": 0}
        )
        running_count["clips"] += 1
        running_count["pauses"] += len(analyzed_one.pauses)
        running_count["only_poc1"] += sum(
            1
            for p in analyzed_one.pauses
            if EFFECTIVE_THRESHOLD_POC1 <= p.duration < EFFECTIVE_THRESHOLD_POC2
        )
    return census


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="shared.metrics.fragmentation",
        description=(
            "Cruza los cortes prematuros del VAD contra las pausas internas de "
            "los clips. Offline y gratis: no llama a nadie."
        ),
    )
    arg_parser.add_argument("cruces", nargs="*", help="uno o más `cero_*.json`")
    arg_parser.add_argument(
        "--census",
        action="store_true",
        help="caracteriza los 49 clips del guion y no cruza ninguna corrida",
    )
    arg_parser.add_argument(
        "--clips", default=str(RUTA_CLIPS_DEFAULT), help="carpeta de los clips"
    )
    arg_parser.add_argument("--json", action="store_true", help="salida en JSON")
    args = arg_parser.parse_args(argv)

    folder = Path(args.clips)

    if args.census:
        census = script_census(folder)
        if args.json:
            print(json.dumps(census, ensure_ascii=False, indent=2))
            return 0
        print(
            f"\nLos {census['clips']} clips del guion, pausas internas >= "
            f"{MIN_PAUSE_S * 1000:.0f} ms\n"
        )
        print(
            f"  {census['pauses']} pausas en {census['clips_with_any']} de "
            f"{census['clips']} clips"
        )
        d = census["duration_ms"]
        print(f"  duración          min {d['min']:.0f} · p50 {d['p50']:.0f} · max {d['max']:.0f} ms")
        print(
            f"  solo corta pipecat  {census['in_the_band_only_poc1']:>3}  "
            f"(en [{census['effective_threshold_poc1_ms']:.0f}, "
            f"{census['effective_threshold_poc2_ms']:.0f}) ms)"
        )
        print(
            f"  cortan las dos      {census['above_the_poc2_threshold']:>3}  "
            f"(>= {census['effective_threshold_poc2_ms']:.0f} ms)"
        )
        print("\n  por escenario\n")
        print(f"    {'scen':<5}{'clips':>6}{'pauses':>8}{'solo pipecat':>14}")
        for scenario, running_count in sorted(census["per_scenario"].items()):
            print(
                f"    {scenario:<5}{running_count['clips']:>6}{running_count['pauses']:>8}"
                f"{running_count['only_poc1']:>14}"
            )
        return 0

    if not args.cruces:
        print("[FALLA] hay que dar al menos un `cero_*.json`, o --census")
        return 1

    runs = [
        c
        for c in (cross_one_run(Path(r), folder=folder) for r in args.cruces)
        if c is not None
    ]
    if not runs:
        print("[FALLA] ninguna corrida se pudo cruzar")
        return 1

    resultado = {"runs": runs, "verdict": verdict(runs)}
    if args.json:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return 0

    for run in runs:
        print(f"\n{run['run']}")
        print(
            f"  {run['lines']} intervenciones · "
            f"{run['events']} eventos · "
            f"{run['premature_cuts']} corte(s) prematuro(s), "
            f"{run['cuts_in_a_clip_pause']} en una pausa del clip"
        )
        for cutoff in run["detail"]:
            pause = cutoff["pause"]
            print(
                f"    {cutoff['turn']:<4} {cutoff['clip_id']:<11} "
                f"cortó en {cutoff['cut_at_s']:>6.3f}s · "
                + (
                    f"pausa de {pause['duration_ms']:>6.1f} ms en "
                    f"{pause['start_s']:.3f}s · a {cutoff['distance_to_start_ms']:+7.1f} ms"
                    if pause
                    else "el clip no tiene pausas"
                )
                + ("  [en la pausa]" if cutoff["in_a_pause"] else "  [FUERA]")
            )

    print(f"\n  pausas de los clips (>= {MIN_PAUSE_S * 1000:.0f} ms)")
    for run in runs:
        census = run["pauses_of_clips_that_played"]
        print(
            f"    {run['run'][-15:]:<17} {census['pauses']:>3} pausas en "
            f"{census['clips_with_any']}/{census['clips']} clips · "
            f"{census['in_the_band_only_poc1']:>3} en la banda que solo corta "
            f"pipecat · {census['above_the_poc2_threshold']:>2} también LiveKit"
        )

    final = verdict(runs)
    print(
        f"\n  {final['cuts_in_a_clip_pause']}/{final['premature_cuts']} "
        f"cortes cayeron dentro de una pausa del clip "
        f"({final['paused_fraction']:.0%})"
    )
    if final["chance_fraction"] is not None:
        print(
            f"  por azar caería ahí el {final['chance_fraction']:.0%} de "
            "cualquier punto del tramo hablado"
        )
    if final["opportunities_taken"] is not None:
        print(
            f"  {final['premature_cuts']}/{final['opportunities']} pausas "
            f"disponibles se volvieron corte ({final['opportunities_taken']:.0%})"
        )
    print(f"\n  {final['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
