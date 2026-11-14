"""Lo que el arnés deja escrito: la bitácora, el reporte en pantalla y el costo.

Las tres cosas tienen que decir lo mismo en las 3 PoC. La bitácora porque es lo
que corrige la referencia del WER, y si cada arnés la escribiera distinto habría
que corregir a mano tres formatos. El reporte y el aviso de costo porque son el
criterio de "esta corrida sirve" y "esta corrida cuesta tanto", y esos no pueden
depender de qué framework se esté probando.

Lo que sí cambia por PoC va por parámetro: quién transcribe, dónde queda el
JSON, qué renglones se gastan. Nada de eso es una decisión de medición.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from shared.carrier_simulator.run import MIN_VALID_TURNS, evaluate_run
from shared.carrier_simulator.script import ScriptStep
from shared.carrier_simulator.observed import Observed


# --------------------------------------------------------------------------
# Bitácora
# --------------------------------------------------------------------------


def event_log_de_la_corrida(
    observed: Observed, label: str, *, scenario: int | None
) -> dict[str, Any]:
    """Lo que el arnés **de verdad** dijo, turno por turno.

    Existe por el árbol de decisión. La referencia del WER se siembra con las
    líneas base del guion (`shared/metrics/reference.py`), y quien actuó
    corrige a mano las que se desviaron. Con el arnés actuando no hay nadie que
    se acuerde: si en T6 se reprodujo `e4_t6_alt1` y la referencia dice la línea
    base, el WER de ese turno mide la distancia entre dos frases distintas y
    sale altísimo sin que nada esté mal.

    Así que el arnés deja escrito qué reprodujo. Es la fuente para corregir la
    referencia de una corrida ramificada, y se escribe **siempre**, también en
    los escenarios lineales, para que el procedimiento sea uno solo.

    `etiqueta` es cómo se llama la corrida en cada PoC: la sala en PoC2, el SID
    de la llamada en PoC1 y PoC3.
    """
    return {
        "room": label,
        "scenario": scenario,
        "what_it_is": (
            "las líneas que el arnés reprodujo de verdad. En un escenario con "
            "ramas NO coinciden con las líneas base del guion, y la referencia "
            "del WER se corrige con esto o mide contra la frase equivocada"
        ),
        "camino": [f"{d.turn}:{d.rama}" for d in observed.decisions],
        "turns": [
            {
                "turn": t.step.turn,
                "clip_id": t.step.clip_id,
                "variant": next(
                    (
                        d.variant
                        for d in observed.decisions
                        if d.turn == t.step.turn
                    ),
                    "base",
                ),
                "played_line": "" if t.step.is_silence else t.step.text,
                "stt": t.stt,
                "agent": t.response,
                # Audio del agente que sonó en este turno y no era su respuesta.
                # Ver `que_son_los_fragmentos_descartados`.
                "discarded_fragments": t.discarded_fragments,
                # Solo en el escenario de interrupciones; en las demás corridas
                # la llave no aparece y la bitácora queda igual que siempre.
                **({"event": t.event} if t.event is not None else {}),
            }
            for t in observed.turns
        ],
        "turns_with_discard": [
            t.step.turn for t in observed.turns if t.out_of_sync
        ],
        "what_the_discarded_fragments_are": (
            "rachas de voz del agente que sonaron después del clip y que el "
            "arnés NO tomó como su respuesta: demasiado cortas para serlo, o "
            "apoyo verbal. Se anotan porque un turno que avanza sobre una de "
            "ellas queda corrido —el arnés habla mientras el agente todavía "
            "piensa— y su acuse cae en el mismo rango que el de un turno sano, "
            "así que sin este registro solo se detecta al oído. Un turno que "
            "aparece en `turnos_con_descarte` hay que mirarlo antes de usar su "
            "latency"
        ),
        "decisions": [d.as_json() for d in observed.decisions],
    }


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportContext:
    """Lo que cambia de PoC a PoC en el reporte. Nada de esto es una métrica."""

    title: str
    # El bloque de arriba, completo y en el orden en que se quiere leer: los
    # renglones genéricos (`linea_de_tramas_publicadas`,
    # `linea_de_tramas_de_audio`) intercalados con los propios de cada PoC —la
    # tasa y el DTX en PoC2, el códec del Media Stream en PoC1—. Van ya
    # formateados porque el orden es cosa de quien lee, no de la medición.
    header: Sequence[str] = ()
    # Quién transcribió lo que publicamos. Sale en el criterio de "el agente nos
    # oyó", que es el que manda a revisar nivel y tasa cuando falla.
    stt_name: str = "el STT"
    # Dónde busca el resultado quien acaba de correr esto.
    footer: Sequence[str] = ()
    # Dónde quedan marcados los turnos mudos en el JSON del worker.
    where_the_mutes_are_marked: str = "el JSON del worker"


def published_frame_line(observed: Observed) -> str:
    """Cuántas tramas salieron y cuántas eran clip. El resto es ruido."""
    return (
        f"  tramas publicadas     {observed.published_frames} "
        f"({observed.clip_frames} de clips, el resto ruido de confort)"
    )


def audio_frame_line(observed: Observed) -> str:
    """Cuántas tramas se recibieron del agente. Llegar no es traer voz."""
    return f"  tramas de audio       {observed.audio_frames}"


def print_report(observed: Observed, context: ReportContext) -> int:
    """Imprime lo observado y devuelve el número de criterios que fallaron."""
    print("\n" + "=" * 70)
    print(context.title)
    print("=" * 70)

    for linea in context.header:
        print(linea)

    if observed.rms:
        with_voice = [r for r in observed.rms if r >= observed.speech_threshold_rms]
        print(
            f"  energía del agente    RMS máx {max(observed.rms):,.0f} | "
            f"{len(with_voice)}/{len(observed.rms)} tramas con voz"
        )

    resultado = evaluate_run(observed)
    print(
        f"\n  turnos alcanzados     {resultado.reached}/{resultado.of_script}"
        f"  · fin: {resultado.motivo}"
    )
    if resultado.no_audio:
        print(
            f"  turnos medibles       {resultado.measurable} "
            f"({len(resultado.no_audio)} con texto y sin audio: "
            f"{', '.join(resultado.no_audio)})"
        )
    print(f"                        {resultado.detail}")

    print(f"\n  turnos del guion ({len(observed.turns)})")
    for turn in observed.turns:
        step = turn.step
        acknowledgement = turn.acknowledgement_ms
        if turn.text_without_audio:
            mark = "MUD"
        elif turn.responded:
            mark = "ok "
        else:
            mark = "SIN"
        print(
            f"\n    [{mark}] {step.turn:<4} {step.clip_id or '(silencio)':<14}"
            + (f" acuse {acknowledgement:>6,.0f} ms" if acknowledgement is not None else " acuse      — ")
            + (" · texto sin audio: fuera del percentil" if turn.text_without_audio else "")
        )
        if step.of_script:
            # Solo se compara contra el guion cuando `texto` ES la línea del
            # guion. En la sonda de `--repetir` no lo es, y marcar diferencias
            # ahí sería ruido que parece un hallazgo.
            print(f"           guion    {step.text!r}")
            if not step.is_silence:
                equal = "=" if turn.stt.strip() == step.text.strip() else "≠"
                print(f"           stt    {equal} {turn.stt!r}")
        elif not step.is_silence:
            print(f"           stt      {turn.stt!r}")
        print(f"           agente   {turn.response!r}" if turn.response else
              "           agente   (no contestó)")
        print(f"           avance   {turn.advance_reason}")

    if observed.decisions:
        print(f"\n  árbol de decisión ({len(observed.decisions)} turno(s))")
        print(
            "    camino    "
            + " → ".join(f"{d.turn}:{d.rama}" for d in observed.decisions)
        )
        for decision in observed.decisions:
            print(
                f"\n    {decision.turn} → {decision.rama} ({decision.variant})"
                + ("  [por defecto]" if decision.by_default else "")
            )
            print(f"           porque   {decision.motivo}")
            for reason in decision.discarded_ones:
                print(f"           descarta {reason}")
            for test in decision.evidence:
                print(f"           oyó      {test}")

    if observed.incidents:
        print("\n  incidentes")
        for incident in observed.incidents:
            print(f"    {incident}")

    # --- Criterios ---
    # Se miden solo sobre los turnos **alcanzados**. Los que quedaron después de
    # que el agente cerró la llamada no se reprodujeron contra nadie, y contarlos
    # como fallas convertiría un cierre normal en tres criterios en rojo — que
    # es lo que pasó el 15 de septiembre y mandó a buscar un bug inexistente.
    acted_out = observed.turns[: resultado.reached]
    spoken_items = [t for t in acted_out if not t.step.is_silence]
    with_stt = [t for t in spoken_items if t.stt]
    answered_items = [t for t in acted_out if t.responded]
    print("\n  criterios")
    criteria = (
        (
            "todos los clips salieron por la pista",
            bool(spoken_items) and all(t.t_clip_end is not None for t in spoken_items),
            "algún clip no se llegó a reproducir",
        ),
        (
            f"el agente OYÓ cada clip: {context.stt_name} los transcribió",
            len(with_stt) == len(spoken_items) and bool(spoken_items),
            f"solo {len(with_stt)} de {len(spoken_items)} clips llegaron al STT. El "
            "agente no está recibiendo nuestro audio, o no lo reconoce como voz: "
            "revisa el nivel del clip y la tasa de publicación",
        ),
        (
            "la corrida alcanzó turnos suficientes",
            resultado.valid,
            f"{resultado.detail}",
        ),
        (
            "cada turno actuado le disparó respuesta al agente",
            len(answered_items) == len(acted_out) and bool(acted_out),
            f"el agente contestó {len(answered_items)} de {len(acted_out)} "
            "turns: oyó pero no contestó, o el fin de turno nunca se disparó",
        ),
        (
            "cada respuesta del agente además se OYÓ",
            not resultado.no_audio,
            f"{len(resultado.no_audio)} turno(s) con texto y sin una sola trama "
            f"con voz ({', '.join(resultado.no_audio)}): el TTS no sonó ahí. "
            "Quedan fuera del percentil y van a confiabilidad; el worker los "
            f"marca en {context.where_the_mutes_are_marked}",
        ),
    )

    faults = 0
    for label, ok, detail in criteria:
        if ok:
            print(f"    [ok]    {label}")
        else:
            faults += 1
            print(f"    [FALLA] {label}")
            print(f"            {detail}")

    if resultado.retry:
        print(
            f"\n  [!] HAY QUE REPETIRLA: {resultado.motivo}."
            "\n      Vuelve a correrla; esta corrida no aporta turnos utilizables."
        )

    for linea in context.footer:
        print(linea)
    return faults


# --------------------------------------------------------------------------
# Costo
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tariffs:
    """Lo que cobra cada renglón. Los valores salen de `docs/costos_pocN.md`.

    El TTS y el LLM se cobran completos a propósito: es una cota alta, no una
    estimación fina.
    """

    per_minute: float = 0.0
    per_call: float = 0.0

    def estimate(self, seconds: float) -> float:
        return (seconds / 60.0) * self.per_minute + self.per_call


@dataclass(frozen=True)
class CostContext:
    """El aviso que se imprime sin `--accept-cost`. Un guardarraíl de CLAUDE.md."""

    title: str
    tariffs: Tariffs
    # (renglón, qué es). El "NO se gasta" es la mitad importante: es donde se
    # lee, negro sobre blanco, que ningún teléfono va a sonar.
    is_spent: Sequence[tuple[str, str]] = ()
    is_not_spent: Sequence[tuple[str, str]] = ()
    # (etiqueta, valor) de lo que identifica la corrida: sala, proyecto, agente.
    fields: Sequence[tuple[str, str]] = field(default_factory=tuple)


def print_cost_warning(
    context: CostContext,
    script: list[ScriptStep],
    seconds: float,
    *,
    retries: int = 0,
) -> None:
    print(f"{context.title}\n")
    if context.is_spent:
        print("  lo que SÍ se gasta")
        for table_row, what_it_is in context.is_spent:
            print(f"    {table_row:<9} {what_it_is}")
    if context.is_not_spent:
        print("\n  lo que NO se gasta")
        for table_row, what_it_is in context.is_not_spent:
            print(f"    {table_row:<9} {what_it_is}")
    print()
    for label, valor in context.fields:
        print(f"  {label:<16} {valor}")

    branched_items = [p for p in script if p.branches_out]
    print(f"  turnos           {len(script)}")
    for step in script:
        which = "(silencio)" if step.is_silence else step.clip_id
        print(f"                   {step.turn:<4} {which:<14} {step.text[:44]!r}")
        for name, variant in sorted(step.variants.items()):
            if name == "base":
                continue
            print(
                f"                     ↳ {name:<6} {variant.clip_id:<14} "
                f"{variant.text[:40]!r}"
            )
    if branched_items:
        print(
            f"  árbol            {len(branched_items)} turno(s) se ramifican "
            f"({', '.join(p.turn for p in branched_items)}): la línea se elige en "
            "vivo\n                   según lo que diga el agente. La duración de "
            "arriba presupuesta la rama más larga"
        )
    print(f"  duración         ~{seconds:.0f} s (cota alta)")
    attempts = 1 + max(0, retries)
    if attempts > 1:
        print(
            f"  reintentos       hasta {retries} más si no alcanza "
            f"{MIN_VALID_TURNS} turnos ({attempts} llamadas en el peor caso)"
        )
    print(
        f"\n  estimado         ~${context.tariffs.estimate(seconds) * attempts:.2f} "
        "USD (cota alta"
        + (f", {attempts} llamadas)" if attempts > 1 else ")")
    )
    print("\nVuelve a correrlo con --accept-cost para ejecutarlo.")
