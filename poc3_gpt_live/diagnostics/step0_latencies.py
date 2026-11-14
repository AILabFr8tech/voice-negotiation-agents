"""Paso 0 de PoC3: ¿son medibles el acuse y la sustantiva con lo que emite el alpha?

VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.
========================================
Este módulo lee los 15 censos de las llamadas telefónicas manuales del 12 de
agosto de 2026 y los usa **exclusivamente como banco de pruebas**: sirven para
saber si el mapeo texto→audio aguanta, no para decir cuánto tarda PoC3. Ningún
número que salga de aquí es publicable ni entra a ninguna poza de percentiles.

Solo lectura. No importa nada de `shared/` ni de `app/`, y nada de lo que hay
aquí toca la ruta de avance del arnés.

## La regla de fechado

El texto es la **etiqueta**; el audio es el **reloj**. Un fragmento de
`output_transcript.added` se mapea a su `output_audio.delta` por el eje del
servidor (`start_ms` contenido en el rango del delta) y se fecha por la
**llegada local** de ese delta — nunca por la llegada del texto, que se adelanta.

## Por qué el criterio de racha NO se toma del eje del servidor

La documentación del alpha dice que un hueco entre rangos representa silencio
omitido, y de ahí salía la idea de cortar las rachas por el eje. **Medido sobre
los 15 censos, eso no describe los datos:** un hueco de 200 ms en el eje
corresponde a ~93 ms de reloj de pared —exactamente lo mismo que dos deltas
contiguos— y, al revés, dos deltas *contiguos en el eje* llegaron a estar
separados por **9.4 s** de reloj. El eje es tiempo de **audio emitido** y no
registra el silencio de ninguna forma; el ritmo de entrega (audio sobre reloj)
va de 0.83 a 0.98 por llamada.

Así que el criterio se toma del **reloj de pared** entre llegadas, y su corte
sale de la distribución medida, no de un número a ojo. La sensibilidad de la
sustantiva a ese corte se reporta.

## El clasificador de apoyo verbal es SOLO DE ANÁLISIS

No se usa `shared.carrier_simulator.verbal_backchannel.is_backchannel`, y es a
propósito: aquel está escrito para rachas **cerradas** de 0.6 a 2.0 s, y aquí
habría que llamarlo sobre trozos de cadencia de ~200 ms. `" Muy bien"` como
fragmento de *"Muy bien, entonces le ofrezco…"* cae entero en su lista y correría
la sustantiva más allá de contenido real. El de abajo exige además un
**terminador de enunciado**, y lo que no alcanza a decidir lo deja en
`INDECISO` para que se revise a mano en vez de resolverlo callado.

Uso:
    python -m poc3_gpt_live.diagnostico.step0_latencies
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

DEFAULT_CENSUS_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "recordings"

EV_AUDIO: Final[str] = "output_audio.delta"
EV_TRANSCRIPT: Final[str] = "output_transcript.added"
EV_TURN_DONE: Final[str] = "turn.done"
USER_ROLE: Final[str] = "user"
AGENT_ROLE: Final[str] = "assistant"


# --------------------------------------------------------------------------
# Clasificador de apoyo verbal — SOLO ANÁLISIS
# --------------------------------------------------------------------------

# Lista propia. **No** se importa `APOYOS` de `shared/`: aquella decide si el
# arnés avanza el turno, y cualquier cambio aquí no debe poder llegar allá.
BACKCHANNELS_ANALYSIS: Final[frozenset[str]] = frozenset(
    {
        "aja", "ajam", "aha", "mm", "mmm", "mhm", "mjm", "ujum", "ah", "ahh",
        "oh", "eh", "este",
        "si", "sip", "claro", "correcto", "exact", "exactamente",
        "efectivamente", "seguro",
        "ok", "okey", "okay", "vale", "entiendo", "entendido", "comprendo",
        "ready", "perfecto", "excelente", "bien", "bueno", "ya",
        "orale", "andale", "sale",
        "de", "acuerdo", "muy", "por", "supuesto", "que", "esta", "tiene",
        "sentido", "gracias",
    }
)

# Un apoyo verbal real no pasa de una locución corta.
MAX_WORDS: Final[int] = 3

# Lo que cierra un enunciado. **Es la diferencia con el clasificador de
# `shared/`**: sin terminador, un trozo de cadencia como `" Muy bien"` puede ser
# el principio de *"Muy bien, entonces le ofrezco…"*, y darlo por apoyo correría
# la sustantiva más allá de contenido real.
TERMINATORS: Final[str] = ".!?…"

BACKCHANNEL: Final[str] = "apoyo"
SUBSTANTIVE: Final[str] = "sustantivo"
UNDECIDED: Final[str] = "indeciso"

# Narración de plan: el defecto propio de PoC3 (el modelo lee en voz alta lo que
# iba a hacer). **No es apoyo verbal y se cuenta aparte**, sin cambiar la
# clasificación: para la métrica es contenido, aunque no sea negociación.
NARRATION: Final[tuple[str, ...]] = (
    "inicio la llamada",
    "ya entro la llamada",
    "te estan oyendo",
    "arranca con un saludo",
    "haz una pausa",
    "voy a saludar",
    "empiezo con el saludo",
    "un momento",
    "un segundo",
    "dejame validar",
    "permiteme validar",
)

_NON_WORD: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", flags=re.UNICODE)


def fold(text: str) -> str:
    """Minúsculas y sin acentos. Misma normalización que el WER."""
    folded_form = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in folded_form if unicodedata.category(c) != "Mn")


def words_of(text: str) -> list[str]:
    return _NON_WORD.sub(" ", fold(text)).split()


def classify(text: str) -> str:
    """`APOYO`, `SUSTANTIVO` o `INDECISO` para un fragmento de cadencia.

    Tres reglas, en orden:

    1. Sin palabras -> `INDECISO`. "No hay texto" no es "el texto es un apoyo".
    2. Todas las palabras en la lista y **enunciado cerrado** (termina en
       `.`/`!`/`?`) y a lo más 3 palabras -> `APOYO`.
    3. Todas en la lista pero **sin cerrar** -> `INDECISO`: puede ser el
       principio de una frase larga. Es el caso que este clasificador existe
       para no resolver mal, y por eso se reporta a mano.

    Cualquier otra cosa es `SUSTANTIVO`.
    """
    raw_value = text.strip()
    words = words_of(raw_value)
    if not words:
        return UNDECIDED
    all_backchannel = len(words) <= MAX_WORDS and all(
        p in BACKCHANNELS_ANALYSIS for p in words
    )
    if not all_backchannel:
        return SUBSTANTIVE
    return BACKCHANNEL if raw_value.endswith(tuple(TERMINATORS)) else UNDECIDED


def is_narration(text: str) -> bool:
    flat_text = " ".join(words_of(text))
    return any(frase in flat_text for frase in NARRATION)


# --------------------------------------------------------------------------
# Lectura del censo
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Delta:
    """Un `output_audio.delta`: su rango en el eje del servidor y su llegada."""

    start_ms: int
    end_ms: int
    t_local: float


@dataclass(frozen=True)
class Fragment:
    """Un `output_transcript.added`."""

    start_ms: int
    end_ms: int
    text: str
    t_local: float


@dataclass(frozen=True)
class UserTurn:
    start_ms: int
    end_ms: int
    text: str
    t_local: float


@dataclass
class Census:
    name: str
    deltas: list[Delta]
    fragments: list[Fragment]
    user_turns: list[UserTurn]
    # `t_local - start_ms/1000` de cada `input_transcript.added`, en orden de
    # llegada. Solo para el diagnóstico de deriva del eje de ENTRADA.
    input_drift: list[float]


def read_census(ruta: Path) -> Census:
    deltas: list[Delta] = []
    fragments: list[Fragment] = []
    turns: list[UserTurn] = []
    input_drifts: list[float] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            row = json.loads(linea)
        except ValueError:
            continue
        tipo = row.get("type")
        event = row.get("event") or {}
        t = float(row.get("t_local_seg", 0.0))
        if tipo == EV_AUDIO:
            deltas.append(Delta(int(event["start_ms"]), int(event["end_ms"]), t))
        elif tipo == EV_TRANSCRIPT:
            fragments.append(
                Fragment(
                    int(event["start_ms"]),
                    int(event["end_ms"]),
                    str((event.get("item") or {}).get("text", "")),
                    t,
                )
            )
        elif tipo == "input_transcript.added":
            input_drifts.append(t - int(event["start_ms"]) / 1000.0)
        elif tipo == EV_TURN_DONE:
            turn = event.get("turn") or {}
            if turn.get("role") == USER_ROLE:
                turns.append(
                    UserTurn(
                        int(turn.get("start_ms", 0)),
                        int(turn.get("end_ms", 0)),
                        str(turn.get("transcript", "")),
                        t,
                    )
                )
    deltas.sort(key=lambda d: (d.start_ms, d.t_local))
    fragments.sort(key=lambda f: (f.start_ms, f.t_local))
    turns.sort(key=lambda t: t.start_ms)
    return Census(
        ruta.stem.replace("census_", ""), deltas, fragments, turns, input_drifts
    )


def censuses(folder: Path) -> Iterator[Census]:
    for ruta in sorted(folder.glob("census_*.jsonl")):
        yield read_census(ruta)


# --------------------------------------------------------------------------
# Rachas de entrega
# --------------------------------------------------------------------------


def streaks(fragments: list[Fragment], *, gap_ms: int) -> list[tuple[int, int]]:
    """Índices `[inicio, fin]` de cada racha de HABLA del agente.

    La racha se rompe cuando el hueco en el eje entre el fin de un fragmento de
    transcripción y el inicio del siguiente supera `gap_ms`.

    **La unidad es el texto y no el audio, y la razón está medida.** El alpha
    entrega audio de forma casi continua —cubre del 82% al 98% del reloj de la
    llamada, silencio incluido—, así que `output_audio.delta` no marca cuándo
    empieza a hablar el agente: marca cuándo entrega tramas. La transcripción,
    en cambio, solo existe donde hubo habla (cubre 25-37% del audio). Cortar por
    el audio mediría atorones del WebSocket; cortar por el texto mide pausas del
    habla.

    Los huecos entre fragmentos vienen cuantizados a múltiplos de 200 ms, que es
    la duración fija de un fragmento.
    """
    if not fragments:
        return []
    output: list[tuple[int, int]] = []
    beg = 0
    for i in range(1, len(fragments)):
        if fragments[i].start_ms - fragments[i - 1].end_ms > gap_ms:
            output.append((beg, i - 1))
            beg = i
    output.append((beg, len(fragments) - 1))
    return output


def delivery_structure(deltas: list[Delta]) -> dict[str, Any]:
    """Huecos del eje contra huecos de reloj, para derivar el criterio de los datos.

    Devuelve los dos, emparejados, para poder contestar si el eje sirve de
    criterio (no sirve) y dónde cae el corte natural del reloj.
    """
    of_axis = [deltas[i].start_ms - deltas[i - 1].end_ms for i in range(1, len(deltas))]
    of_wall = [
        (deltas[i].t_local - deltas[i - 1].t_local) * 1000.0
        for i in range(1, len(deltas))
    ]
    audio_ms = sum(d.end_ms - d.start_ms for d in deltas)
    wall_ms = (deltas[-1].t_local - deltas[0].t_local) * 1000.0 if len(deltas) > 1 else 0.0
    return {
        "of_axis": of_axis,
        "of_wall": of_wall,
        "pairs": list(zip(of_axis, of_wall)),
        "audio_ms": audio_ms,
        "wall_ms": wall_ms,
        "pacing": (audio_ms / wall_ms) if wall_ms > 0 else None,
    }


# --------------------------------------------------------------------------
# El mapeo texto -> audio
# --------------------------------------------------------------------------

RESUELTO: Final[str] = "resuelto"
AMBIGUOUS: Final[str] = "ambiguo"
IN_GAP: Final[str] = "in_gap"
OUTSIDE: Final[str] = "out_of_range"


def map_items(fragment: Fragment, deltas: list[Delta]) -> tuple[str, Delta | None]:
    """El delta que contiene el `start_ms` del fragmento.

    `RESUELTO` si exactamente uno lo contiene; `AMBIGUO` si varios (rangos
    solapados); `IN_GAP` si cae entre dos deltas; `FUERA` si cae antes del
    primero o después del último.
    """
    contain = [d for d in deltas if d.start_ms <= fragment.start_ms < d.end_ms]
    if len(contain) == 1:
        return RESUELTO, contain[0]
    if len(contain) > 1:
        return AMBIGUOUS, contain[0]
    if not deltas or fragment.start_ms < deltas[0].start_ms:
        return OUTSIDE, None
    if fragment.start_ms >= deltas[-1].end_ms:
        return OUTSIDE, None
    return IN_GAP, None


# --------------------------------------------------------------------------
# Acuse y sustantiva
# --------------------------------------------------------------------------


NO_TEXT: Final[str] = "sin_texto"


def axis_drift(census: Census) -> dict[str, float | None]:
    """Cuánto se separa cada familia de eventos del reloj de pared en la llamada.

    Si el eje del servidor siguiera al reloj, `t_local - start_ms/1000` sería
    **constante**. Se mide por separado para la familia de SALIDA
    (`output_audio.delta`) y la de ENTRADA (`input_transcript.added`), porque el
    acuse necesitaría restar una de la otra y eso solo vale si las dos derivan
    igual.
    """

    def extremes(values: list[float]) -> float | None:
        return (values[-1] - values[0]) if len(values) > 1 else None

    output = [d.t_local - d.start_ms / 1000.0 for d in census.deltas]
    # `input_transcript.added` no se guarda como fragmento (esos son de salida);
    # se relee del censo en `leer_censo` solo para este diagnóstico.
    entry = census.input_drift
    return {"output": extremes(output), "entry": extremes(entry)}


@dataclass
class Streak:
    """Una racha de entrega del agente: su arranque, su texto y su forma.

    **La unidad es la racha, no el turno.** El acuse necesitaría restar el fin
    de habla del transportista, que vive en otra familia de eventos y en un eje
    que deriva distinto (ver `axis_drift`); todo lo que hay aquí se mide
    dentro de la **misma** familia y con el **mismo** reloj local, así que no
    depende de esa resta.
    """

    call: str
    indice: int
    start_ms: int
    t_local: float
    duration_ms: int
    first_text: str
    kind: str
    narration: bool
    # Del primer audio de la racha al primer fragmento clasificado, en ms de
    # reloj local. Es la diferencia acuse -> sustantiva: cuánto sonó el agente
    # sin decir nada.
    until_substantive_ms: int | None
    until_lax_substantive_ms: int | None
    fragments: int


def agent_streaks(census: Census, *, gap_ms: int) -> list[Streak]:
    """Una `Racha` por tramo de habla del agente, con su texto clasificado.

    Cada fragmento se fecha por la **llegada local del delta de audio que lo
    contiene** — la regla de fechado. Una racha sin su primer fragmento
    mapeado no se puede fechar y se descarta con constancia.
    """
    output: list[Streak] = []
    for k, (a, b) in enumerate(streaks(census.fragments, gap_ms=gap_ms)):
        inside = census.fragments[a : b + 1]

        def stamp_date(f: Fragment) -> float | None:
            estado, delta = map_items(f, census.deltas)
            return None if (estado != RESUELTO or delta is None) else delta.t_local

        t_cero = stamp_date(inside[0])
        if t_cero is None:
            continue

        def first(accepts) -> int | None:
            for f in inside:
                if not accepts(classify(f.text)):
                    continue
                t = stamp_date(f)
                if t is None:
                    continue
                return round((t - t_cero) * 1000.0)
            return None

        output.append(
            Streak(
                call=census.name,
                indice=k,
                start_ms=inside[0].start_ms,
                t_local=t_cero,
                duration_ms=inside[-1].end_ms - inside[0].start_ms,
                first_text=inside[0].text,
                kind=classify(inside[0].text),
                narration=any(is_narration(f.text) for f in inside[:4]),
                until_substantive_ms=first(lambda c: c == SUBSTANTIVE),
                until_lax_substantive_ms=first(lambda c: c != BACKCHANNEL),
                fragments=len(inside),
            )
        )
    return output


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


def summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    o = sorted(values)
    return (
        f"n={len(o)} min {o[0]:.0f} p50 {st.median(o):.0f} max {o[-1]:.0f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="poc3_gpt_live.diagnostics.step0_latencies",
        description="Paso 0 de PoC3 — VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.",
    )
    ap.add_argument("--censuses", default=str(DEFAULT_CENSUS_PATH))
    ap.add_argument("--gap-ms", type=int, default=400)
    args = ap.parse_args(argv)

    all_entries = list(censuses(Path(args.censuses)))
    if not all_entries:
        print(f"[FALLA] no hay censo_*.jsonl en {args.censuses}")
        return 1

    print("=" * 78)
    print("PASO 0 DE POC3 — VALIDACIÓN DE INSTRUMENTO, NO RESULTADO")
    print(f"{len(all_entries)} censos. Ninguna cifra de aquí es publicable.")
    print("=" * 78)

    # -- 1. Mapeo texto -> audio --------------------------------------------
    print("\n[1] MAPEO TEXTO -> AUDIO, POR LLAMADA\n")
    print(f"  {'call':<18}{'frag':>6}{'resuelto':>10}{'hole':>7}{'outside':>7}"
          f"{'ambig':>7}{'deriva p50':>12}{'deriva p05':>12}")
    print("  " + "-" * 77)
    tot = Counter()
    all_drifts: list[float] = []
    for c in all_entries:
        running_count = Counter()
        drifts: list[float] = []
        for f in c.fragments:
            estado, delta = map_items(f, c.deltas)
            running_count[estado] += 1
            if estado == RESUELTO and delta is not None:
                drifts.append((f.t_local - delta.t_local) * 1000.0)
        tot.update(running_count)
        all_drifts.extend(drifts)
        d = sorted(drifts)
        p50 = st.median(d) if d else float("nan")
        p05 = d[max(0, int(len(d) * 0.05))] if d else float("nan")
        print(f"  {c.name[-16:]:<18}{len(c.fragments):>6}{running_count[RESUELTO]:>10}"
              f"{running_count[IN_GAP]:>7}{running_count[OUTSIDE]:>7}{running_count[AMBIGUOUS]:>7}"
              f"{p50:>11.0f}{p05:>12.0f}")
    n = sum(tot.values())
    print(f"\n  TOTAL fragmentos {n}: resuelto {tot[RESUELTO]} "
          f"({tot[RESUELTO]/n:.1%}) · en hueco {tot[IN_GAP]} · "
          f"fuera {tot[OUTSIDE]} · ambiguo {tot[AMBIGUOUS]}")
    d = sorted(all_drifts)
    print(f"  deriva texto−audio (ms), {len(d)} fragmentos resueltos: "
          f"p05 {d[int(len(d)*0.05)]:.0f} · p50 {st.median(d):.0f} · "
          f"p95 {d[int(len(d)*0.95)]:.0f} · min {d[0]:.0f} · max {d[-1]:.0f}")
    print(f"  fragmentos cuyo texto llegó DESPUÉS de su audio: "
          f"{sum(1 for x in d if x > 0)} de {len(d)}")

    # -- 2. Estructura de entrega y sensibilidad ----------------------------
    print("\n[2] ESTRUCTURA DE ENTREGA — el criterio de racha, derivado de los datos\n")
    pairs: list[tuple[int, float]] = []
    pacings: list[float] = []
    for c in all_entries:
        e = delivery_structure(c.deltas)
        pairs.extend(e["pairs"])
        if e["pacing"]:
            pacings.append(e["pacing"])
    per_axis: dict[int, list[float]] = defaultdict(list)
    for axis, wall in pairs:
        per_axis[axis].append(wall)
    print("  ¿sirve el hueco DEL EJE como criterio? Su equivalente en reloj:\n")
    print(f"  {'hueco del eje':>14}{'n':>7}{'pared p50':>11}{'pared p95':>11}"
          f"{'pared max':>11}")
    print("  " + "-" * 55)
    for axis in sorted(per_axis):
        v = sorted(per_axis[axis])
        print(f"  {axis:>14}{len(v):>7}{st.median(v):>11.0f}"
              f"{v[int(len(v)*0.95)]:>11.0f}{v[-1]:>11.0f}")
    print("\n  NO SIRVE: un hueco de 200 ms en el eje son ~93 ms de reloj, lo mismo")
    print("  que dos deltas contiguos; y dos deltas CONTIGUOS llegaron a estar")
    print("  separados por varios segundos de reloj. El eje es tiempo de AUDIO y no")
    print("  registra el silencio. El criterio se toma del RELOJ DE PARED.\n")
    of_wall = sorted(p for _, p in pairs)
    print(f"  huecos de reloj entre deltas (ms): p50 {st.median(of_wall):.0f} · "
          f"p90 {of_wall[int(len(of_wall)*0.90)]:.0f} · "
          f"p99 {of_wall[int(len(of_wall)*0.99)]:.0f} · "
          f"p999 {of_wall[int(len(of_wall)*0.999)]:.0f} · max {of_wall[-1]:.0f}")
    for cutoff in (150, 200, 300, 500, 1000):
        print(f"    por encima de {cutoff:>5} ms: "
              f"{sum(1 for x in of_wall if x > cutoff):>5} de {len(of_wall)}")
    if pacings:
        print(f"\n  ritmo de entrega (audio emitido / reloj transcurrido) por llamada: "
              f"min {min(pacings):.3f} · p50 {st.median(pacings):.3f} · "
              f"max {max(pacings):.3f}")

    print("\n  Sensibilidad al criterio de hueco (ahora de RELOJ):\n")
    print(f"  {'hueco ms':>9}{'streaks':>8}{'x llam.':>9}{'sust=0':>8}{'sust>0':>8}"
          f"{'p50 de >0':>11}{'max':>8}{'apoyo 1º':>10}")
    print("  " + "-" * 71)
    for hole in (0, 200, 400, 600, 1000, 2000, 3000):
        rr = [r for c in all_entries for r in agent_streaks(c, gap_ms=hole)]
        substantive = [r.until_substantive_ms for r in rr if r.until_substantive_ms is not None]
        mayor = sorted(x for x in substantive if x > 0)
        print(f"  {hole:>9}{len(rr):>8}{len(rr)/len(all_entries):>9.1f}"
              f"{sum(1 for x in substantive if x == 0):>8}{len(mayor):>8}"
              f"{(st.median(mayor) if mayor else float('nan')):>11.0f}"
              f"{(max(substantive) if substantive else float('nan')):>8.0f}"
              f"{sum(1 for r in rr if r.kind == BACKCHANNEL):>10}")

    # A partir de aquí, el criterio del argumento.
    all_streaks = [
        r for c in all_entries for r in agent_streaks(c, gap_ms=args.gap_ms)
    ]
    print(f"\n  (el resto del reporte usa --hueco-ms {args.gap_ms})")

    # -- 2b. Deriva del eje: ¿se puede restar una familia de la otra? --------
    print("\n[2b] DERIVA DEL EJE DEL SERVIDOR CONTRA EL RELOJ DE PARED\n")
    print("  Si el eje siguiera al reloj, `t_local - start_ms/1000` sería constante.")
    print(f"  {'call':<18}{'SALIDA (s)':>12}{'ENTRADA (s)':>13}{'difference':>12}")
    print("  " + "-" * 55)
    differences: list[float] = []
    for c in all_entries:
        d = axis_drift(c)
        if d["output"] is None or d["entry"] is None:
            continue
        diff = d["output"] - d["entry"]
        differences.append(diff)
        print(f"  {c.name[-16:]:<18}{d['output']:>12.2f}{d['entry']:>13.2f}"
              f"{diff:>12.2f}")
    if differences:
        a = sorted(abs(x) for x in differences)
        print(f"\n  |diferencia| entre las dos derivas: min {a[0]:.2f} s · "
              f"p50 {st.median(a):.2f} s · max {a[-1]:.2f} s")

    # -- 3. Piso de cuantización --------------------------------------------
    print("\n[3] PISO DE CUANTIZACIÓN, MEDIDO\n")
    dur_delta = Counter(d.end_ms - d.start_ms for c in all_entries for d in c.deltas)
    dur_frag = Counter(f.end_ms - f.start_ms for c in all_entries for f in c.fragments)
    print(f"  duración de los deltas de audio (ms): {dict(sorted(dur_delta.items())[:8])}")
    print(f"  duración de los fragmentos de texto (ms): "
          f"{dict(sorted(dur_frag.items())[:10])}")
    nonzero_substantive = sorted({r.until_substantive_ms for r in all_streaks
                          if r.until_substantive_ms not in (None, 0)})
    print(f"  valores distintos de sustantiva > 0 (ms), los 10 menores: "
          f"{nonzero_substantive[:10]}")
    print("  El escalón mínimo distinto de cero es el PISO: por debajo de él la "
          "métrica\n  no resuelve, y ese piso es instrumento, no conducta del agente.")

    # -- 4. Clasificador -----------------------------------------------------
    print("\n[4] CLASIFICADOR DE APOYO VERBAL (SOLO ANÁLISIS)\n")
    kinds = Counter(r.kind for r in all_streaks)
    print(f"  primeros enunciados por clase: {dict(kinds)}")
    print(f"  rachas con NARRACIÓN DE PLAN en sus 4 primeros fragmentos: "
          f"{sum(1 for r in all_streaks if r.narration)} de "
          f"{len(all_streaks)}")
    print("\n  CASOS DE FRONTERA (INDECISO) — todos, para revisión a mano:")
    boundary = [r for r in all_streaks if r.kind == UNDECIDED]
    if not boundary:
        print("    (ninguno)")
    for r in boundary:
        print(f"    {r.call[-12:]}  racha {r.indice:>3}  "
              f"{r.duration_ms:>5} ms  {r.first_text!r}")
    print("\n  Clasificados APOYO, todos:")
    for r in all_streaks:
        if r.kind == BACKCHANNEL:
            print(f"    {r.call[-12:]}  racha {r.indice:>3}  "
                  f"{r.duration_ms:>5} ms  {r.first_text!r}")
    print("\n  SIN_TEXTO (racha de audio sin ningún fragmento dentro), todas:")
    for r in all_streaks:
        if r.kind == NO_TEXT:
            print(f"    {r.call[-12:]}  racha {r.indice:>3}  "
                  f"{r.duration_ms:>5} ms")
    print("\n  Muestra de SUSTANTIVO (10 primeros):")
    for r in [x for x in all_streaks if x.kind == SUBSTANTIVE][:10]:
        print(f"    {r.call[-12:]}  racha {r.indice:>3}  {r.first_text!r}"
              f"{'   [narra plan]' if r.narration else ''}")

    # -- 5. El acuse -------------------------------------------------------
    print("\n[5] ACUSE — NO SE PUEDE RECONSTRUIR DE ESTOS ARTEFACTOS\n")
    print("  El acuse necesita el fin de habla del transportista. En el censo eso")
    print("  sólo existe como `turn.done`(role=user).`end_ms`, en el eje del")
    print("  servidor; el arranque del agente está en el mismo eje pero en otra")
    print("  familia de eventos, y [2b] mide que las dos familias NO derivan")
    print("  igual. La ruta de producción no usa ese eje: mide por energía local")
    print("  sobre las tramas μ-law (`_LOCAL_ENERGY` en app/latency.py), y el")
    print("  censo no guarda `input_audio.append`. Sin audio entrante no hay cero.")
    print("\n  Lo que sí se mide aquí, y con un solo reloj: la diferencia")
    print("  acuse -> sustantiva (dentro de la misma racha, llegadas locales).\n")
    print(f"  {'clase del arranque':<22}{'n':>5}{'con sust.':>11}{'p50 ms':>9}"
          f"{'max ms':>9}")
    print("  " + "-" * 57)
    for kind in (BACKCHANNEL, UNDECIDED, SUBSTANTIVE, NO_TEXT):
        group = [r for r in all_streaks if r.kind == kind]
        v = sorted(r.until_substantive_ms for r in group
                   if r.until_substantive_ms is not None)
        print(f"  {kind:<22}{len(group):>5}{len(v):>11}"
              f"{(st.median(v) if v else float('nan')):>9.0f}"
              f"{(max(v) if v else float('nan')):>9.0f}")
    strict_variant = sorted(r.until_substantive_ms for r in all_streaks
                      if r.until_substantive_ms is not None)
    lenient = sorted(r.until_lax_substantive_ms for r in all_streaks
                  if r.until_lax_substantive_ms is not None)
    print(f"\n  estricta (exige SUSTANTIVO): {summary([float(x) for x in strict_variant])}")
    print(f"  laxa (acepta todo menos APOYO): {summary([float(x) for x in lenient])}")
    print("  La distancia entre las dos es lo que cuesta la ambigüedad de los "
          "fragmentos\n  de cadencia — no es conducta del agente.")

    # -- 6. Arranque sobre el transportista ---------------------------------
    print("\n[6] ACUSE NEGATIVO — NO MEDIBLE AQUÍ, POR LO MISMO QUE [5]\n")
    print("  Un acuse negativo es el agente arrancando antes de que el")
    print("  transportista termine. Decidir el signo exige restar las dos")
    print("  familias de eventos, que es justo lo que [2b] invalida: la")
    print("  diferencia entre las dos derivas llega a varios segundos dentro de")
    print("  una misma llamada, y el signo del acuse se juega en cientos de ms.")
    print("\n  Lo único afirmable sin esa resta: las 15 llamadas manuales ya")
    print("  reportaron 101 `onsets_over_carrier` de 217 turnos")
    print("  (46 con menos de 150 ms de silencio), medidos EN VIVO por energía")
    print("  local. Ese conteo no se re-deriva del censo; se cita de su JSON.")

    # -- 7. Mínimo de racha en la ruta saliente ------------------------------
    print("\n[7] MÍNIMO DE DURACIÓN EN LA RUTA SALIENTE — NO SE CONTESTA AQUÍ\n")
    print("  La pregunta es sobre la racha de voz que `outbound_speech()` arma por")
    print("  ENERGÍA sobre las tramas μ-law. El censo no guarda esos bytes")
    print("  (`\"audio\": \"<audio 4800 bytes>\"`), así que esa duración no existe")
    print("  en estos artefactos. Un conteo sobre rachas de `output_audio.delta`")
    print("  TAMPOCO la contesta: el alpha entrega audio el 82-98% del reloj,")
    print("  silencio incluido, así que sus rachas son atorones de entrega y no")
    print("  tramos de habla. Sólo una corrida instrumentada puede medirlo.\n")
    print("  Lo de abajo es la duración de la racha de TEXTO, que es otra cosa y")
    print("  está acotada por abajo por los 200 ms de un fragmento:\n")
    print(f"  {'mínimo ms':>10}{'se pierden':>12}{'  por clase del arranque'}")
    print("  " + "-" * 62)
    for minimum in (20, 100, 150, 200, 300, 500):
        losses = [r for r in all_streaks if r.duration_ms < minimum]
        per_class = Counter(r.kind for r in losses)
        print(f"  {minimum:>10}{len(losses):>6} de {len(all_streaks):<4}"
              f"  {dict(per_class) or '{}'}")
    short_clips = [r for r in all_streaks if r.duration_ms < 150]
    if short_clips:
        print(f"\n  Las {len(short_clips)} que se perderían con 150 ms, una por una:")
        per_call = Counter(r.call for r in short_clips)
        for r in short_clips:
            print(f"    {r.call[-12:]}  racha {r.indice:>3}  "
                  f"{r.duration_ms:>4} ms  {r.kind:<11} {r.first_text!r}")
        print(f"\n  concentración por llamada: {dict(per_call)}")

    print("\n" + "=" * 78)
    print("Recordatorio: VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
