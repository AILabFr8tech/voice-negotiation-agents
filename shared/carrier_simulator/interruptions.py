"""El escenario de interrupciones: el transportista le habla encima al agente.

En el escenario normal el arnés nunca le habla encima al agente: espera a que
lleve 2 s callado (`AGENT_SILENCE_S`). Aquí, en algunos turnos, hace
algo **mientras el agente responde**. Tres clases de evento:

- **interrupción** — suena una línea del transportista que toma la palabra
  ("Espéreme… ¿cuánto dijo?"). Lo esperado: que el agente se calle y conteste.
- **apoyo** — suena un apoyo verbal ("ajá"). Lo esperado: que el agente siga.
- **control** — no suena nada. Sirve para saber con qué frecuencia el agente se
  calla solo en ese mismo momento, porque ya iba a terminar su frase: sin esta
  tasa de base, una "detención" tras un "ajá" no se puede atribuir al "ajá".

## Cuándo se dispara

Cuando el agente lleva `SUSTAINED_SPEECH_S` hablando de corrido en su respuesta
(una racha abierta que empezó después de la línea del transportista). Así el
evento cae siempre a media frase y no sobre un "mm" suelto, y cae igual en las
tres PoC, sin depender de cómo decide cada una su respuesta.

## Qué se mide (todo con el reloj del arnés, como la latencia)

La referencia es el **inicio de la voz** del clip del evento (en el control, el
instante del disparo):

- `stopped` y `tiempo_de_detencion_ms`: el agente se detuvo si su voz se corta
  con un silencio de al menos `STOP_SILENCE_S` que empieza antes de
  `STOP_WINDOW_S` después de la referencia. El tiempo es el último
  sonido antes de ese silencio menos la referencia.
- `latencia_de_replica_ms` (solo interrupción): del fin de la voz del
  transportista al primer sonido del agente después de haberse detenido.
- `duracion_de_la_pausa_ms` y `vuelve_a_hablar_ms` (las tres clases): si se
  detuvo, cuánto duró su silencio (del último
  sonido antes de detenerse al primero después) y cuándo volvió a hablar
  contado desde el fin de la voz del transportista (en el control, desde la
  referencia). En la interrupción, `vuelve_a_hablar_ms` es la latencia de
  réplica. "Se detuvo" dice si hubo un silencio de 0.7 s; esto dice si fue una
  pausa breve tras la que retomó o si cedió la palabra. Si no volvió a hablar
  en `NO_REPLY_WAIT_S` de silencio (en las tres clases),
  `pausa_sin_fin` y la duración es una cota inferior.
- `agent_before_event` y `agent_after_event`: lo que dijo el
  agente en ese turno, partido en la referencia. Sirve para leer si tras el
  evento siguió la misma frase o generó un mensaje nuevo.

Los umbrales se fijan antes de correr. `STOP_SILENCE_S` sale de las
pausas internas del agente medidas en 10 auditorías (741 pausas; p99 468 ms, y
luego un salto a 1,430 ms; ver `reproductor.AGENT_SILENCE_S`): 0.7 s
queda por encima de casi todas las pausas a media frase y por debajo del
siguiente modo.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from shared.carrier_simulator.script import (
    HarnessFailure,
    ScriptStep,
    load_clip_to_publish,
)
from shared.carrier_simulator.observed import (
    STREAK_HANGOVER_S,
    Observed,
    SpeechStreak,
)
from shared.carrier_simulator.player import AGENT_SILENCE_S

INTERRUPTION_CLIPS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "test_scenarios" / "interruption_clips"
)

EVENT_KINDS: Final[tuple[str, ...]] = ("interrupcion", "apoyo", "control")

# Cuánto tiene que llevar hablando el agente, de corrido, para disparar.
SUSTAINED_SPEECH_S: Final[float] = 1.0
# Silencio del agente que cuenta como "se detuvo". Ver el docstring.
STOP_SILENCE_S: Final[float] = 0.7
# Hasta cuándo, desde la referencia, una detención cuenta como respuesta al evento.
STOP_WINDOW_S: Final[float] = 3.0
# Tras una interrupción, si el agente no vuelve a hablar, cuánto silencio se
# espera antes de pasar a la siguiente línea. Más que los 2 s normales: la
# réplica a una interrupción puede tardar lo que tarda una respuesta.
NO_REPLY_WAIT_S: Final[float] = 8.0

# Qué pasa en cada turno elegible, en orden. Dos de cada clase por llamada.
PATRON: Final[tuple[str, ...]] = (
    "interrupcion", "apoyo", "control", "interrupcion", "apoyo", "control",
)


@dataclass
class TurnEvent:
    """Lo que el arnés hace encima de la respuesta del agente en un turno."""

    tipo: str
    clip_id: str | None = None
    text: str = ""
    step: ScriptStep | None = None
    speech_onset_s: float | None = None
    end_of_speech_s: float | None = None


# --------------------------------------------------------------------------
# El plan
# --------------------------------------------------------------------------


def eligible_turns(script: list[ScriptStep]) -> list[str]:
    """Los turnos donde puede caer un evento: ni el primero ni el último.

    El primero contesta al saludo con una frase corta; el último es el cierre,
    donde el agente puede estar colgando. Tampoco los silencios del guion.
    """
    turns = [p.turn for p in script if not p.is_silence]
    return turns[1:-1]


def event_plan(
    scenario: int, repetition: int, turns: list[str], catalog: list[dict[str, Any]]
) -> dict[str, tuple[str, str | None]]:
    """`{turno: (tipo, clip_id)}`, determinista por escenario y repetición.

    El patrón se rota para que cada clase caiga en turnos distintos según la
    corrida, y los clips se rotan para que cada uno se use lo mismo a lo largo
    de la campaña. Mismo escenario y repetición → mismo plan en las tres PoC.
    """
    by_kind = {
        t: [c["id"] for c in catalog if c["tipo"] == t] for t in ("interrupcion", "apoyo")
    }
    for t, ids in by_kind.items():
        if not ids:
            raise HarnessFailure(f"no hay clips de tipo {t!r} en el manifiesto de interrupciones")
    giro = (scenario + repetition) % len(PATRON)
    patron = PATRON[giro:] + PATRON[:giro]
    used_items = {"interrupcion": 0, "apoyo": 0}
    plan: dict[str, tuple[str, str | None]] = {}
    for k, turn in enumerate(turns[: len(patron)]):
        tipo = patron[k]
        if tipo == "control":
            plan[turn] = (tipo, None)
            continue
        ids = by_kind[tipo]
        indice = (2 * (repetition - 1) + scenario + used_items[tipo]) % len(ids)
        used_items[tipo] += 1
        plan[turn] = (tipo, ids[indice])
    return plan


def attach_events(
    script: list[ScriptStep],
    *,
    scenario: int,
    repetition: int,
    folder: Path = INTERRUPTION_CLIPS_PATH,
) -> dict[str, tuple[str, str | None]]:
    """Carga los clips del plan y cuelga cada evento de su paso. Devuelve el plan.

    Se carga todo antes de conectarse, igual que el guion: un clip que falte
    tiene que fallar antes de gastar la llamada. En los turnos que se ramifican
    el evento se cuelga de todas las variantes, porque la rama se elige en vivo.
    """
    from shared.audio.interruptions import load_manifest

    try:
        manifest = load_manifest(folder)
    except FileNotFoundError as error:
        raise HarnessFailure(str(error)) from error
    sample_rate = next((p.sample_rate for p in script if not p.is_silence), None)
    if sample_rate is None:
        raise HarnessFailure("el guion no tiene ningún clip")
    plan = event_plan(scenario, repetition, eligible_turns(script), list(manifest.values()))
    for step in script:
        if step.turn not in plan:
            continue
        tipo, clip_id = plan[step.turn]
        if clip_id is None:
            event = TurnEvent(tipo=tipo)
        else:
            entry = manifest[clip_id]
            ruta = folder / f"{clip_id}.ulaw"
            if not ruta.is_file():
                raise HarnessFailure(f"falta {ruta}: prepara los clips de interrupción")
            samples, published, source, origin = load_clip_to_publish(
                ruta, publish_sample_rate=sample_rate
            )
            event = TurnEvent(
                tipo=tipo, clip_id=clip_id, text=entry["text"],
                step=ScriptStep(
                    turn=step.turn, text=entry["text"], samples=samples,
                    ulaw=published, ulaw_source=source, clip_id=clip_id, origin=origin,
                    sample_rate=sample_rate, end_of_speech_s=entry["end_of_speech_s"],
                ),
                speech_onset_s=float(entry["speech_onset_s"]),
                end_of_speech_s=float(entry["end_of_speech_s"]),
            )
        step.event = event
        for variant in step.variants.values():
            variant.event = event
    return plan


def activate(
    script: list[ScriptStep],
    *,
    scenario: int | None,
    repetition: int | None,
    folder: Path | str = INTERRUPTION_CLIPS_PATH,
) -> dict[str, tuple[str, str | None]]:
    """Lo que llama `--interruptions` en el arnés de cada PoC.

    Exige escenario y repetición porque el plan sale de ellos: sin eso, las tres
    PoC no actuarían los mismos eventos en los mismos turnos.
    """
    if scenario is None or repetition is None:
        raise HarnessFailure("--interruptions necesita --scenario y --repetition")
    plan = attach_events(
        script, scenario=scenario, repetition=repetition, folder=Path(folder)
    )
    print("  escenario de interrupciones:")
    for turn, (tipo, clip) in plan.items():
        print(f"    {turn}: {tipo}{' ' + clip if clip else ''}")
    return plan


# --------------------------------------------------------------------------
# En vivo
# --------------------------------------------------------------------------


async def wait_for_trigger(
    observed: Observed,
    *,
    since: float,
    tope_seg: float,
    closed: asyncio.Event | None = None,
) -> tuple[float, SpeechStreak] | None:
    """El instante en que el agente lleva `SUSTAINED_SPEECH_S` hablando de corrido.

    Devuelve `(instante, racha)` o `None` si se venció el tope o la llamada se
    cerró antes. Solo cuentan rachas que empezaron después de `desde` (el fin
    de la línea del transportista).
    """
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if closed is not None and closed.is_set():
            return None
        streaks = observed.streaks_after(since)
        if streaks:
            ultima = streaks[-1]
            now = observed.now()
            opened = now - ultima.end <= STREAK_HANGOVER_S
            if opened and now - ultima.start >= SUSTAINED_SPEECH_S:
                return now, ultima
        await asyncio.sleep(0.02)
    return None


def measure_event(
    streaks: list[SpeechStreak],
    *,
    reference: float,
    end_of_speech: float,
    tipo: str,
    until: float | None = None,
) -> dict[str, Any]:
    """Las métricas de un evento, sobre las rachas del agente ya observadas.

    `hasta` es el instante en que se mide (cuando el arnés va a soltar la línea
    siguiente): acota la pausa que no terminó.
    """
    actives = [r for r in streaks if r.start <= reference <= r.end + STREAK_HANGOVER_S]
    if not actives:
        # Se calló entre el disparo y el inicio de la voz del clip (unos pocos
        # tramos): cuenta como detenido en 0, con la última voz antes.
        priors = [r for r in streaks if r.end <= reference]
        if not priors:
            return {"hablaba_en_la_reference": False, "stopped": None}
        actives = [priors[-1]]
    end = actives[0].end
    for r in streaks:
        if r.start <= actives[0].start:
            continue
        if r.start - end >= STOP_SILENCE_S:
            break
        end = max(end, r.end)
    stop = end - reference
    stopped = stop <= STOP_WINDOW_S
    data: dict[str, Any] = {
        "hablaba_en_la_reference": bool(actives[0].start <= reference <= actives[0].end + STREAK_HANGOVER_S),
        "stopped": stopped,
        "stop_time_ms": round(stop * 1000.0, 1),
        "last_speech_before_stopping": round(end, 3),
    }
    if stopped:
        replica = next((r for r in streaks if r.start > end + 1e-9), None)
        data["endless_pause"] = replica is None
        if replica is not None:
            data["pause_duration_ms"] = round((replica.start - end) * 1000.0, 1)
            data["speaks_again_ms"] = round((replica.start - end_of_speech) * 1000.0, 1)
        elif until is not None:
            data["pause_duration_ms"] = round(max(0.0, until - end) * 1000.0, 1)
        if tipo == "interrupcion" and replica is not None:
            data["reply_latency_ms"] = round((replica.start - end_of_speech) * 1000.0, 1)
            data["line_duration_ms"] = round(replica.duration * 1000.0, 1)
            data["reply_start"] = round(replica.start, 3)
    return data


async def _wait_after_event(
    observed: Observed,
    *,
    tipo: str,
    reference: float,
    end_of_speech: float,
    tope_seg: float,
    closed: asyncio.Event | None,
) -> str:
    """Cuándo soltar la siguiente línea después del evento.

    **Interrupción:** se espera la réplica — a que el agente vuelva a hablar
    después del clip y termine (2 s de silencio), o `NO_REPLY_WAIT_S` de
    silencio si no replica.

    **Apoyo y control, la misma espera cuando el agente se detuvo:** si se calló
    dentro de la ventana de detención, el arnés espera a que vuelva a hablar
    —para medir cuándo retoma y oír qué dice— y a que termine, o
    `NO_REPLY_WAIT_S` de silencio. Soltar la línea siguiente a los 2 s de
    silencio le hablaría encima a un agente que está por retomar, y "no retomó"
    se confundiría con "no alcanzó a retomar". Si no se detuvo, la regla es la
    de siempre: 2 s de silencio y sigue el guion.
    """
    limit = time.monotonic() + tope_seg
    while time.monotonic() < limit:
        if closed is not None and closed.is_set():
            return "la sala se cerró después del evento"
        silent = observed.agent_silent_since()
        if tipo == "interrupcion":
            replied = any(r.start > end_of_speech for r in observed.speech_streaks)
            if replied and silent is not None and silent >= AGENT_SILENCE_S:
                return "replicó a la interrupción y terminó"
            if not replied and silent is not None and silent >= NO_REPLY_WAIT_S:
                return f"no replicó a la interrupción en {NO_REPLY_WAIT_S:.0f} s de silencio"
        elif silent is not None and silent >= AGENT_SILENCE_S:
            m = measure_event(
                list(observed.speech_streaks), reference=reference, end_of_speech=end_of_speech,
                tipo=tipo, until=observed.now(),
            )
            if not m.get("stopped"):
                return "terminó de hablar"
            if not m.get("endless_pause"):
                return "retomó después de detenerse y terminó"
            if silent >= NO_REPLY_WAIT_S:
                return f"se detuvo y no volvió a hablar en {NO_REPLY_WAIT_S:.0f} s de silencio"
        await asyncio.sleep(0.05)
    return "venció el tope después del evento"


async def act_out_event(
    event: TurnEvent,
    *,
    player: Any,
    observed: Observed,
    trigger: float,
    tope_seg: float,
    closed: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Suelta el clip del evento (o nada, en el control), espera y mide."""
    record: dict[str, Any] = {
        "tipo": event.tipo,
        "clip_id": event.clip_id,
        "text": event.text,
        "fired": True,
        "t_trigger": round(trigger, 3),
    }
    if event.step is not None:
        t_end = await player.play_step(event.step)
        t_beg = observed.t_clip_start
        reference = t_beg + float(event.speech_onset_s or 0.0)
        end_of_speech = t_beg + float(event.end_of_speech_s if event.end_of_speech_s is not None else t_end - t_beg)
        record.update({
            "t_clip_start": round(t_beg, 3),
            "t_clip_end": round(t_end, 3),
            "t_speech_onset": round(reference, 3),
            "t_end_of_speech": round(end_of_speech, 3),
        })
    else:
        reference = end_of_speech = trigger
        record["t_speech_onset"] = round(reference, 3)
    record["advance_reason"] = await _wait_after_event(
        observed, tipo=event.tipo, reference=reference, end_of_speech=end_of_speech,
        tope_seg=tope_seg, closed=closed,
    )
    record.update(measure_event(
        list(observed.speech_streaks), reference=reference, end_of_speech=end_of_speech, tipo=event.tipo,
        until=observed.now(),
    ))
    return record
