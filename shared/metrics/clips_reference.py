"""Referencia del WER armada desde los clips reproducidos, sin corrección manual.

Cuando el transportista es una persona actuando, la referencia la tiene que
escribir esa persona: se desvía del guion y solo ella sabe qué dijo. Cuando el
transportista es el arnés reproduciendo un `.ulaw`, **lo que sonó es exactamente
el texto de ese clip**, y la bitácora registra cuál se reprodujo en cada turno,
ramas incluidas. La referencia entonces se deriva, no se transcribe.

Este módulo la deriva. Lo que decide y por qué está en
`docs/METRICS_CONTRACT.md` §2.10; aquí va solo lo que hace falta para leer el
código:

- la **referencia** sale de `linea_reproducida` de la bitácora;
- la **hipótesis** sale de la transcripción del pipeline, **nunca** del `stt`
  por turno de la bitácora: en PoC2 ese campo concatena resultados interinos de
  LiveKit y repite texto;
- dos reglas de exclusión, `classify_tail` y `common_clips`, cada una con su
  prueba;
- lo único que queda manual son las dos notas 1-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from shared.metrics.reference import (
    COUNT_FIELDS,
    SCORE_FIELDS,
    FILE_HYPOTHESIS,
    FREE_HEADER,
    NOTES_HEADER,
    LEFTOVERS_HEADER,
    HYPOTHESIS_PREFIX,
    REFERENCE_PREFIX,
    group_carrier_turns,
)

DEAD_TAIL_REASON: Final[str] = (
    "el clip sonó fuera de la conversación: ni el STT ni el agente reaccionaron"
)
OUT_OF_PAIR_REASON: Final[str] = "la otra PoC del par no reprodujo este clip"

_INSTRUCTIONS: Final[str] = """\
# Referencia del WER DERIVADA de los clips reproducidos.
#
# Las líneas `yo:` NO se corrigen a mano: son el texto exacto del clip que el
# arnés reprodujo, tomado de la bitácora de la corrida. Editarlas rompe la
# correspondencia con el audio que de verdad sonó.
#
# Lo único manual son las dos notas 1-5 del final.
"""


@dataclass(frozen=True)
class PlayedTurn:
    """Un turno tal como la bitácora del arnés lo registró."""

    turn: str
    clip_id: str
    variant: str
    linea: str
    stt: str = ""
    agent: str = ""

    @property
    def had_agent_response(self) -> bool:
        return bool(self.agent.strip())

    @property
    def had_transcript(self) -> bool:
        return bool(self.stt.strip())


@dataclass(frozen=True)
class ExcludedTurn:
    turn: PlayedTurn
    indice: int
    motivo: str


@dataclass
class ClipRun:
    """Una llamada del arnés, lista para producir su referencia."""

    clave: str
    poc: int
    scenario: int
    repetition: str
    call_sid: str = ""
    turns: list[PlayedTurn] = field(default_factory=list)
    excluded_clips: list[ExcludedTurn] = field(default_factory=list)

    @property
    def pair(self) -> tuple[int, str]:
        """Las dos llamadas que comparan el mismo estímulo llevan esta clave."""
        return (self.scenario, self.repetition)

    @property
    def clips(self) -> frozenset[str]:
        return frozenset(t.clip_id for t in self.turns)

    def reference_words(self) -> int:
        return sum(len(t.linea.split()) for t in self.turns)


# --------------------------------------------------------------------------
# Regla 1: la cola muerta
# --------------------------------------------------------------------------


def is_dead_tail(*, had_transcript: bool, had_response: bool) -> bool:
    """La condición de corte, en un solo lugar.

    Vive aparte porque la aplican dos módulos con tipos distintos —este y
    `shared/carrier_simulator/summary.py`, que calcula el WER en vivo— y una
    regla de exclusión que se escribe dos veces se separa una vez.
    """
    return not had_transcript and not had_response


def classify_tail(
    turns: Sequence[PlayedTurn],
) -> tuple[list[PlayedTurn], list[ExcludedTurn]]:
    """Recorta los turnos del final cuyo clip sonó fuera de la conversación.

    El corte exige las **dos** condiciones a la vez: que el STT no transcribiera
    nada **y** que el agente no contestara. Esa conjunción es la regla, no un
    detalle de implementación:

    - sin STT y sin agente, al final: la conversación ya había terminado y el
      arnés reprodujo contra nadie. Contarlo metería borraduras por audio que el
      pipeline nunca estuvo en posición de procesar;
    - **sin STT pero con el agente hablando: es una borradura real y se queda.**
      El STT tuvo su oportunidad y no la aprovechó, que es exactamente lo que el
      WER debe medir.

    Solo se recorta desde el final. Un turno mudo en medio de la conversación no
    es cola: es una falla y tiene que verse como tal.
    """
    end = len(turns)
    while end > 0:
        candidate = turns[end - 1]
        if not is_dead_tail(
            had_transcript=candidate.had_transcript,
            had_response=candidate.had_agent_response,
        ):
            break
        end -= 1

    alive_ones = list(turns[:end])
    excluded_clips = [
        ExcludedTurn(turn=t, indice=i, motivo=DEAD_TAIL_REASON)
        for i, t in enumerate(turns[end:], start=end)
    ]
    return alive_ones, excluded_clips


# --------------------------------------------------------------------------
# Regla 2: el pareo
# --------------------------------------------------------------------------


def common_clips(runs: Sequence[ClipRun]) -> frozenset[str]:
    """Los clips que **todas** las corridas del par reprodujeron.

    Se intersecta por `clip_id`, no por nombre de turno. No es lo mismo: cuando
    el guion se bifurca, las dos PoC pueden llegar al mismo `T5` por ramas
    distintas (`alt1` contra `alt2`), y parear por nombre compararía el WER
    sobre dos textos que no son el mismo.
    """
    if not runs:
        return frozenset()
    common = runs[0].clips
    for c in runs[1:]:
        common &= c.clips
    return common


def match_pairs(runs: Iterable[ClipRun]) -> dict[tuple[int, str], list[ClipRun]]:
    """Agrupa las corridas por (escenario, repetición)."""
    pairs: dict[tuple[int, str], list[ClipRun]] = {}
    for c in runs:
        pairs.setdefault(c.pair, []).append(c)
    return pairs


def trim_to_pair(
    run: ClipRun, commons: frozenset[str]
) -> ClipRun:
    """Deja solo los turnos cuyo clip está en el conjunto común.

    Devuelve una corrida nueva: la original no se toca, para que el conteo sin
    parear siga disponible y la tabla de cobertura se pueda publicar.
    """
    alive_ones: list[PlayedTurn] = []
    outside: list[ExcludedTurn] = []
    for i, t in enumerate(run.turns):
        if t.clip_id in commons:
            alive_ones.append(t)
        else:
            outside.append(ExcludedTurn(turn=t, indice=i, motivo=OUT_OF_PAIR_REASON))
    return ClipRun(
        clave=run.clave,
        poc=run.poc,
        scenario=run.scenario,
        repetition=run.repetition,
        call_sid=run.call_sid,
        turns=alive_ones,
        excluded_clips=[*run.excluded_clips, *outside],
    )


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------


def log_turns(event_log: Mapping[str, Any]) -> list[PlayedTurn]:
    return [
        PlayedTurn(
            turn=str(t.get("turn", "")),
            clip_id=str(t.get("clip_id", "")),
            variant=str(t.get("variant", "")),
            linea=str(t.get("played_line", "") or ""),
            stt=str(t.get("stt", "") or ""),
            agent=str(t.get("agent", "") or ""),
        )
        for t in (event_log.get("turns") or [])
    ]


def read_run(
    event_log: Mapping[str, Any],
    *,
    clave: str,
    poc: int,
    repetition: str,
    call_sid: str = "",
) -> ClipRun:
    """Arma la corrida ya con la cola muerta recortada."""
    alive_ones, excluded_clips = classify_tail(log_turns(event_log))
    return ClipRun(
        clave=clave,
        poc=poc,
        scenario=int(event_log.get("scenario") or 0),
        repetition=str(repetition),
        call_sid=call_sid,
        turns=alive_ones,
        excluded_clips=excluded_clips,
    )


def verify_against_manifest(
    run: ClipRun, manifest: Mapping[str, str]
) -> list[str]:
    """Cruza `linea_reproducida` contra el `texto` del manifiesto de clips.

    Son dos registros independientes del mismo hecho —lo escribió el arnés al
    reproducir, lo escribió el generador de clips al sintetizar— y que coincidan
    es la evidencia de que la referencia derivada es la correcta. Devuelve la
    lista de discrepancias; vacía es lo que tiene que salir.
    """
    problems: list[str] = []
    for t in run.turns:
        expected = manifest.get(t.clip_id)
        if expected is None:
            problems.append(f"{t.turn}: {t.clip_id} no está en el manifiesto")
        elif expected.strip() != t.linea.strip():
            problems.append(
                f"{t.turn}: el manifiesto dice {expected.strip()!r} y la "
                f"bitácora {t.linea.strip()!r}"
            )
    return problems


# --------------------------------------------------------------------------
# Generación del archivo
# --------------------------------------------------------------------------


def generate(
    run: ClipRun,
    *,
    transcript: Sequence[Mapping[str, Any]] = (),
    script_version: str = "",
) -> str:
    """El archivo de referencia, en el mismo formato que `referencia.parsear`.

    La hipótesis se reparte posicionalmente entre los turnos **antes** de
    cualquier exclusión, y por eso `corrida.excluidos` trae el índice original:
    un turno que se cae en medio —el caso de las ramas divergentes— tiene que
    llevarse su grupo del STT con él, o todo lo que viene después queda corrido.
    """
    groups = group_carrier_turns(list(transcript))

    # Reconstruye el orden original para repartir los grupos como se grabaron.
    # Las dos reglas de exclusión numeran sobre rangos que no se pisan —el
    # pareo sobre los turnos que sobrevivieron a la cola, la cola sobre los que
    # van después—, así que los índices juntos cubren `0..total-1` sin repetir.
    excluded_by_index = {e.indice: e for e in run.excluded_clips}
    if len(excluded_by_index) != len(run.excluded_clips):
        raise ValueError("dos exclusiones con el mismo índice: el orden no se puede rehacer")
    total = len(run.excluded_clips) + len(run.turns)
    live_indices = [i for i in range(total) if i not in excluded_by_index]
    alive_by_index = dict(zip(live_indices, run.turns))

    parts: list[str] = [_INSTRUCTIONS]
    parts.append(f"poc: {run.poc}")
    parts.append(f"scenario: {run.scenario}")
    parts.append(f"repetition: {run.repetition}")
    parts.append(f"call_sid: {run.call_sid}")
    parts.append(f"script: {script_version}")
    # Las `stt:` de abajo NO son una copia para leer: llevan las exclusiones
    # repartidas turno por turno, así que son la hipótesis buena. Ver
    # `llamadas._wer_del_canal`.
    parts.append(f"hypothesis: {FILE_HYPOTHESIS}")
    parts.append("")

    for indice in range(total):
        hypothesis = groups[indice] if indice < len(groups) else ""
        outside = excluded_by_index.get(indice)
        if outside is not None:
            parts.append(f"# {outside.turn.turn} EXCLUIDO — {outside.motivo}")
            parts.append(f"#   clip: {outside.turn.clip_id}")
            parts.append(f"#   {REFERENCE_PREFIX} {outside.turn.linea}".rstrip())
            parts.append(f"#   {HYPOTHESIS_PREFIX} {hypothesis}".rstrip())
            parts.append("")
            continue
        turn = alive_by_index[indice]
        parts.append(f"## {turn.turn}")
        parts.append(f"#   clip: {turn.clip_id} ({turn.variant})")
        parts.append(f"{REFERENCE_PREFIX} {turn.linea}".rstrip())
        parts.append(f"{HYPOTHESIS_PREFIX} {hypothesis}".rstrip())
        parts.append("")

    extras = [g for g in groups[total:] if g.strip()]
    if extras:
        parts.append(LEFTOVERS_HEADER)
        parts.append(
            "#   El STT produjo más grupos que turnos reprodujo el arnés: "
            "fragmentación."
        )
        parts.append(
            "#   Se quedan porque son salida real del STT y cuentan para el WER "
            "del canal."
        )
        for i, group in enumerate(extras):
            parts.append(f"## extra{i + 1}")
            parts.append(f"{REFERENCE_PREFIX} ")
            parts.append(f"{HYPOTHESIS_PREFIX} {group}".rstrip())
            parts.append("")

    parts.append(NOTES_HEADER)
    parts.append("#   naturalidad y turn_taking: 1-5, rúbrica en el guion.")
    parts.append("#   Es lo ÚNICO manual de este archivo.")
    for campo in SCORE_FIELDS:
        parts.append(f"{campo}: ")
    for campo in COUNT_FIELDS:
        parts.append(f"{campo}: ")
    parts.append("")
    parts.append(FREE_HEADER)
    parts.append("")

    return "\n".join(parts) + "\n"


def clips_manifest(ruta: Path | str) -> dict[str, str]:
    """`clip_id -> texto`, del manifiesto que dejó el generador de clips."""
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {c["id"]: c["text"] for c in data.get("clips", [])}
