"""Calibración del umbral de voz de PoC3 — entregables E1 a E6.

VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.
========================================
Lee lo que dejó `energy_probe` en una corrida de calibración y lo cruza con
el censo de la misma llamada. **Ninguna cifra que salga de aquí es reportable**,
ni entra a percentiles, ni se cita en ningún entregable — mismo estatus que los
15 censos de agosto.

## La etiqueta no usa energía, y esa es la única razón por la que esto sirve

Calibrar un detector de energía con una etiqueta sacada de la energía sería
circular. Aquí la etiqueta sale del **texto**: un delta de audio es HABLA si lo
cubre algún `output_transcript.added`, y RELLENO si cae en un tramo sin texto
ninguno. El mapeo texto→delta es el que ya se validó en el paso 0 (99.8%,
3,643 de 3,651 fragmentos resueltos), y el censo se lee con `step0_latencies`
en vez de reescribir el lector.

Los deltas que cruzan el borde de un tramo de texto quedan AMBIGUO y **no se
fuerzan** a ninguna de las dos clases: son justo los arranques y las colas, que
es donde un umbral se juega su error, y meterlos en una clase por conveniencia
inventaría la separación que esto tiene que medir.

## El piso de resolución del eje, que decide cómo se leen los bordes

Los sellos de `output_transcript.added` son múltiplos exactos de **200 ms** (los
de `output_audio.delta` son de 20 ms; medido sobre las 15 llamadas de agosto).
O sea que el borde de un tramo de texto se conoce con ±200 ms mientras que el
delta que se quiere etiquetar dura 100 ms. Por eso `TEXT_GAP_MS` existe y
por eso la clase AMBIGUO no es un detalle: es el ancho real de la duda.

## Lo que este módulo NO hace

No toca `bridge.SPEECH_THRESHOLD_RMS`, ni `observado.RMS_WITH_SPEECH`, ni ninguna ruta
de avance. Lee dos archivos y escribe un reporte a la salida estándar.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from poc3_gpt_live.diagnostics.step0_latencies import Census, Delta, read_census

# El valor heredado que se está poniendo a prueba. Vive en tres sitios
# independientes (`shared/carrier_simulator/observado.py:32`,
# `poc2_livekit/.../humo_sala.py:132`, `poc3_gpt_live/app/bridge.py:329`) y
# **no se unifican**: hacerlo tocaría la ruta de PoC1, que ya entregó campaña.
INHERITED_THRESHOLD_RMS: Final[float] = 200.0

# Piso de ruido de confort del arnés, para tener contra qué comparar en dBFS.
COMFORT_FLOOR_DBFS: Final[float] = -60.0

FINE_WINDOW_S: Final[float] = 0.020
COARSE_WINDOW_S: Final[float] = 0.100

# Dos fragmentos de texto a menos de esto se consideran el mismo tramo hablado.
# 200 ms es **un solo bin** del eje del servidor: unir por debajo de eso sería
# fingir una resolución que los sellos no tienen.
TEXT_GAP_MS: Final[int] = 200

# Banda de guarda alrededor del borde de cada tramo de texto. Un delta que caiga
# dentro de ella es AMBIGUO y no entra a ninguna distribución.
#
# No es una perilla de conveniencia, es el ancho de la ignorancia: el borde del
# texto se conoce con ±200 ms (un bin) y el delta que se etiqueta dura 100 ms,
# así que en los 200 ms pegados a un borde **no se sabe** si hubo voz. Sin la
# guarda, ese habla de borde cae en RELLENO y borra el valle: medido en esta
# corrida, el 80.4% de las ventanas de "relleno" por encima de RMS 200 estaban a
# ≤200 ms de un tramo de texto y el 100% a ≤1,000 ms, mientras que el relleno
# de verdad vive a 2,240 ms (p50) del texto más cercano. El reporte publica la
# sensibilidad a este valor porque es la decisión más discutible del análisis.
EDGE_GUARD_MS: Final[int] = 200

FULL_SCALE: Final[float] = 32768.0

SPEECH: Final[str] = "habla"
RELLENO: Final[str] = "filler"
AMBIGUOUS: Final[str] = "ambiguo"

DOMAINS: Final[tuple[tuple[str, str], ...]] = (
    ("rms_24k", "PCM16 24 kHz (lo que mide hoy el puente)"),
    ("rms_8k", "mu-law 8 kHz decodificado (lo que viaja a Twilio)"),
)

PERCENTILES: Final[tuple[int, ...]] = (1, 5, 25, 50, 75, 95, 99)


def dbfs(rms: float) -> float:
    return 20.0 * math.log10(rms / FULL_SCALE) if rms > 0 else float("-inf")


def _db(rms: float) -> str:
    d = dbfs(rms)
    return "  -inf" if d == float("-inf") else f"{d:6.1f}"


def _pct(values: Sequence[float], p: int) -> float:
    if not values:
        return float("nan")
    return values[min(int(p / 100 * (len(values) - 1)), len(values) - 1)]


@dataclass
class DumpData:
    """Lo que dejó la sonda: un registro por `output_audio.delta`."""

    header: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


def read_dump(ruta: Path) -> DumpData:
    text_lines = ruta.read_text(encoding="utf-8").splitlines()
    return DumpData(
        header=json.loads(text_lines[0]),
        rows=[json.loads(x) for x in text_lines[1:] if x.strip()],
    )


def input_spans(census_path: Path) -> list[tuple[int, int]]:
    """Los rangos de `input_transcript.added`, que `Censo` no guarda.

    Es la transcripción del transportista, fechada por el **mismo** servidor y
    en el **mismo** eje que el texto del agente. Es lo único con lo que se puede
    decidir si un arranque adelantado cayó sobre habla real o sobre el silencio
    que el detector del alpha todavía no había declarado fin de turno.
    """
    output: list[tuple[int, int]] = []
    for linea in census_path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            row = json.loads(linea)
        except ValueError:
            continue
        if row.get("type") != "input_transcript.added":
            continue
        event = row.get("event") or {}
        start = event.get("start_ms")
        if isinstance(start, (int, float)):
            output.append((int(start), int(event.get("end_ms") or start)))
    return sorted(output)


# --------------------------------------------------------------------------
# La etiqueta, que sale del texto y nunca de la energía
# --------------------------------------------------------------------------


def spoken_segments(
    census: Census, *, gap_ms: int = TEXT_GAP_MS
) -> list[tuple[int, int]]:
    """Los rangos del eje del servidor cubiertos por texto del agente."""
    joined: list[list[int]] = []
    for start, end in sorted((f.start_ms, f.end_ms) for f in census.fragments):
        if joined and start - joined[-1][1] <= gap_ms:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(a, b) for a, b in joined]


def assign_label(
    delta: Delta,
    segments: Sequence[tuple[int, int]],
    *,
    guard_ms: int = EDGE_GUARD_MS,
) -> str:
    """AMBIGUO si el delta toca la banda de guarda de cualquier borde de texto.

    Si no la toca: HABLA cuando cae entero dentro de un tramo con texto y
    RELLENO cuando no toca ninguno. El orden importa — la guarda se consulta
    primero, porque en esos 200 ms **no se sabe**, y forzar una clase ahí es
    justo lo que borraría el valle que este análisis tiene que medir.
    """
    for a, b in segments:
        for edge in (a, b):
            if delta.start_ms < edge + guard_ms and edge - guard_ms < delta.end_ms:
                return AMBIGUOUS
    if any(a <= delta.start_ms and delta.end_ms <= b for a, b in segments):
        return SPEECH
    return RELLENO


def windows_per_class(
    dump_data: DumpData,
    census: Census,
    campo: str,
    *,
    guard_ms: int = EDGE_GUARD_MS,
) -> dict[str, list[float]]:
    """El RMS de cada ventana de 20 ms, agrupado por la etiqueta de su delta."""
    segments = spoken_segments(census)
    by_start = {d.start_ms: d for d in census.deltas}
    output: dict[str, list[float]] = {SPEECH: [], RELLENO: [], AMBIGUOUS: []}
    for row in dump_data.rows:
        start = row.get("start_ms")
        if not isinstance(start, (int, float)):
            continue
        delta = by_start.get(int(start))
        if delta is None:
            continue
        output[assign_label(delta, segments, guard_ms=guard_ms)].extend(row.get(campo) or [])
    return output


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------


def e1(
    dump_data: DumpData, census: Census, *, guard_ms: int = EDGE_GUARD_MS
) -> dict[str, dict[str, list[float]]]:
    print("=" * 78)
    print("E1 — LAS DOS DISTRIBUCIONES, ETIQUETADAS POR TEXTO, EN LOS DOS DOMINIOS")
    print("=" * 78)
    segments = spoken_segments(census)
    print(
        f"  tramos con texto del agente: {len(segments)}   "
        f"fragments: {len(census.fragments)}   deltas: {len(census.deltas)}   "
        f"guarda de borde: ±{guard_ms} ms"
    )
    everything: dict[str, dict[str, list[float]]] = {}
    for campo, title in DOMAINS:
        groups = {
            k: sorted(v)
            for k, v in windows_per_class(
                dump_data, census, campo, guard_ms=guard_ms
            ).items()
        }
        everything[campo] = groups
        n_total = sum(len(v) for v in groups.values()) or 1
        print(f"\n  {title}")
        header = "".join(f"{'p' + str(p):>8}" for p in PERCENTILES)
        print(f"    {'kind':<9}{'windows':>9}{'%':>7}{'seg':>7}{header}")
        for kind in (SPEECH, RELLENO, AMBIGUOUS):
            v = groups[kind]
            if not v:
                print(f"    {kind:<9}{0:>9}")
                continue
            print(
                f"    {kind:<9}{len(v):>9}{len(v) / n_total:>7.1%}{len(v) * 0.02:>7.1f}"
                + "".join(f"{_pct(v, p):>8.0f}" for p in PERCENTILES)
            )
            print(
                f"    {'':<9}{'dBFS':>9}{'':>7}{'':>7}"
                + "".join(f"{_db(_pct(v, p)):>8}" for p in PERCENTILES)
            )
        piso = FULL_SCALE * 10 ** (COMFORT_FLOOR_DBFS / 20)
        print(
            f"    piso de ruido de confort del arnés: {COMFORT_FLOOR_DBFS:.0f} dBFS "
            f"(RMS ~ {piso:.0f})"
        )
    return everything


# --------------------------------------------------------------------------
# E2 y E3
# --------------------------------------------------------------------------


def e2(everything: dict[str, dict[str, list[float]]]) -> None:
    print()
    print("=" * 78)
    print("E2 — ¿HAY VALLE ENTRE LAS DOS?")
    print("=" * 78)
    for campo, title in DOMAINS:
        speech, filler = everything[campo][SPEECH], everything[campo][RELLENO]
        if not speech or not filler:
            print(f"\n  {title}: sin datos suficientes")
            continue
        p05_speech = _pct(speech, 5)
        p95_filler = _pct(filler, 95)
        on_top = sum(1 for x in filler if x > p05_speech) / len(filler)
        low = sum(1 for x in speech if x < p95_filler) / len(speech)
        print(f"\n  {title}")
        print(f"    p05 de HABLA    {p05_speech:>9.1f} RMS  ({_db(p05_speech)} dBFS)")
        print(f"    p95 de RELLENO  {p95_filler:>9.1f} RMS  ({_db(p95_filler)} dBFS)")
        if p95_filler < p05_speech:
            width = dbfs(p05_speech) - dbfs(p95_filler)
            print(
                f"    VALLE: sí, de {p95_filler:.1f} a {p05_speech:.1f} RMS "
                f"({width:.1f} dB de ancho)"
            )
        else:
            print("    VALLE: NO — las dos distribuciones se traslapan")
        print(f"    relleno por encima del p05 de habla: {on_top:>7.2%}")
        print(f"    habla por debajo del p95 de relleno: {low:>7.2%}")


def guard_sensitivity(dump_data: DumpData, census: Census) -> None:
    """Cuánto de E2 depende de la banda de guarda, que es la decisión discutible."""
    print()
    print("  SENSIBILIDAD A LA BANDA DE GUARDA (la decisión más discutible de aquí)")
    print(f"    {'guard':>8}{'habla':>8}{'filler':>9}{'ambiguo':>9}"
          f"{'p95 relleno':>13}{'p05 habla':>11}{'¿valle?':>9}")
    for guard in (0, 100, 200, 400, 600):
        groups = windows_per_class(dump_data, census, "rms_24k", guard_ms=guard)
        speech, filler = sorted(groups[SPEECH]), sorted(groups[RELLENO])
        if not speech or not filler:
            continue
        p95_r, p05_h = _pct(filler, 95), _pct(speech, 5)
        print(f"    {guard:>7} ms{len(speech):>8}{len(filler):>9}{len(groups[AMBIGUOUS]):>9}"
              f"{p95_r:>13.0f}{p05_h:>11.0f}{('sí' if p95_r < p05_h else 'NO'):>9}")


def e3(everything: dict[str, dict[str, list[float]]]) -> None:
    print()
    print("=" * 78)
    print("E3 — ¿DISPARA EL 200.0 SOBRE EL RELLENO?")
    print("=" * 78)
    print(
        f"  umbral heredado: RMS {INHERITED_THRESHOLD_RMS} = "
        f"{dbfs(INHERITED_THRESHOLD_RMS):.2f} dBFS\n"
    )
    print(f"  {'dominio':<44}{'relleno > 200':>14}{'habla > 200':>13}")
    for campo, title in DOMAINS:
        speech, filler = everything[campo][SPEECH], everything[campo][RELLENO]
        fr = sum(1 for x in filler if x >= INHERITED_THRESHOLD_RMS) / len(filler) if filler else 0.0
        fh = sum(1 for x in speech if x >= INHERITED_THRESHOLD_RMS) / len(speech) if speech else 0.0
        print(f"  {title[:43]:<44}{fr:>13.2%}{fh:>13.2%}")


# --------------------------------------------------------------------------
# E4 — el costo de la ventana
# --------------------------------------------------------------------------


def e4(dump_data: DumpData, census: Census, *, umbral: float = INHERITED_THRESHOLD_RMS) -> None:
    print()
    print("=" * 78)
    print("E4 — EL COSTO DE LA VENTANA (20 ms contra el delta entero, ~100 ms)")
    print("=" * 78)
    print("  Por cada tramo hablado: cuándo cruza el umbral la primera ventana fina")
    print("  contra cuándo lo cruza el RMS del delta completo, que es como se mide hoy.")
    print()
    print("  Las dos se fechan cuando su decisión ESTÁ DISPONIBLE, no cuando empieza el")
    print("  audio que la produjo: una ventana de 20 ms decide al terminar esos 20 ms y")
    print("  el delta entero decide al terminar el delta. Fechar la gruesa en el inicio")
    print("  del delta le regalaría los ~100 ms que justamente se quieren medir.\n")
    segments = spoken_segments(census)
    by_start = {d.start_ms: d for d in census.deltas}
    sorted_rows = sorted(
        (f for f in dump_data.rows if isinstance(f.get("start_ms"), (int, float))),
        key=lambda f: f["start_ms"],
    )
    for campo, title in DOMAINS:
        lags: list[float] = []
        no_crossing = 0
        for a, b in segments:
            fine: float | None = None
            coarse: float | None = None
            for row in sorted_rows:
                delta = by_start.get(int(row["start_ms"]))
                if delta is None or delta.end_ms <= a or delta.start_ms >= b:
                    continue
                values = row.get(campo) or []
                if not values:
                    continue
                step = (delta.end_ms - delta.start_ms) / len(values)
                if fine is None:
                    for i, x in enumerate(values):
                        if x >= umbral:
                            # La ventana fina decide al CERRAR su ventana.
                            fine = delta.start_ms + (i + 1) * step
                            break
                if coarse is None:
                    coarse_value = math.sqrt(sum(x * x for x in values) / len(values))
                    if coarse_value >= umbral:
                        # La gruesa promedia el delta entero: decide al cerrarlo.
                        coarse = float(delta.end_ms)
                if fine is not None and coarse is not None:
                    break
            if fine is None:
                no_crossing += 1
            elif coarse is not None:
                lags.append(coarse - fine)
        print(f"  {title}")
        print(f"    tramos hablados: {len(segments)}   sin cruce con ventana fina: {no_crossing}")
        if lags:
            lags.sort()
            print(
                f"    retraso de la ventana gruesa: n={len(lags)}  "
                f"min {lags[0]:,.0f}  p50 {st.median(lags):,.0f}  "
                f"max {lags[-1]:,.0f} ms"
            )
            print(
                f"    tramos donde la gruesa llega TARDE: "
                f"{sum(1 for x in lags if x > 0)} de {len(lags)}"
            )
        else:
            print("    (ningún tramo donde crucen las dos)")


# --------------------------------------------------------------------------
# E5
# --------------------------------------------------------------------------


def e5(census: Census, entry: list[tuple[int, int]], wav: Path | None) -> None:
    print()
    print("=" * 78)
    print("E5 — ¿SE ENCIMÓ SOBRE HABLA REAL, O DENTRO DEL HANGOVER DEL DETECTOR?")
    print("=" * 78)
    print("  Separa dos conductas distintas, y solo una es full-duplex de verdad:")
    print("    (a) el agente arrancó mientras el transportista TODAVÍA emitía sonido")
    print("    (b) el agente arrancó después del último sonido, dentro del silencio")
    print("        que el detector del alpha aún no había declarado fin de turno\n")
    print("  Instrumento: `input_transcript.added` — la transcripción del")
    print("  transportista, fechada por el MISMO servidor y en el MISMO eje que el")
    print("  texto del agente. Sin traducción de relojes no hay deriva que corregir.\n")

    if not entry:
        print("  (el censo no trae input_transcript.added: E5 no se contesta así)")
    else:
        segments = spoken_segments(census)
        print(
            f"  {'turno del usuario':>22}{'arranca agente':>16}{'lead':>10}"
            f"{'¿habla del usuario ahí?':>26}"
        )
        cases = 0
        talk_over = 0
        for turn in census.user_turns:
            subsequent = [t[0] for t in segments if t[0] > turn.start_ms]
            if not subsequent:
                continue
            onset = min(subsequent)
            lead = turn.end_ms - onset
            if lead <= 0:
                continue
            cases += 1
            covered = any(a <= onset < b for a, b in entry)
            talk_over += int(covered)
            verdict = "SÍ — habla real" if covered else "no — silencio"
            print(
                f"  {f'{turn.start_ms:,}-{turn.end_ms:,}':>22}{onset:>16,}"
                f"{lead:>9,}{verdict:>26}"
            )
        if cases:
            print(f"\n  arranques por delante del fin de turno: {cases}")
            print(
                f"    sobre habla real del transportista: {talk_over} "
                f"({talk_over / cases:.0%})  -> full-duplex de verdad"
            )
            print(
                f"    dentro del hangover del detector:   {cases - talk_over} "
                f"({(cases - talk_over) / cases:.0%})  -> no es solape audible"
            )

    if wav is not None and wav.is_file():
        _e5_wav(wav)


def _e5_wav(ruta: Path) -> None:
    import numpy as np

    with wave.open(str(ruta)) as file:
        channel_list, sample_rate, n = file.getnchannels(), file.getframerate(), file.getnframes()
        raw_value = np.frombuffer(file.readframes(n), dtype="<i2").reshape(-1, channel_list)
    window = int(sample_rate * FINE_WINDOW_S)
    print(f"\n  VALIDACIÓN GRUESA CONTRA EL WAV  ({n / sample_rate:.1f} s, {channel_list} canales)")
    for canal, name in ((0, "canal 0 — el transportista"), (1, "canal 1 — el agente")):
        cut = raw_value[: (len(raw_value) // window) * window, canal].reshape(-1, window)
        rms = np.sqrt((cut.astype(np.float64) ** 2).mean(axis=1))
        is_active = float((rms >= INHERITED_THRESHOLD_RMS).mean())
        print(
            f"    {name:<28} p50 {float(np.median(rms)):>7.1f} RMS   "
            f"sobre 200: {is_active:>6.1%}   ({is_active * n / sample_rate:>5.1f} s)"
        )
    print("    LÍMITE: los dos canales se fechan al llegar y el cabezal monótono")
    print("    acumula ~1.7-1.9 s de corrimiento sobre 90 s. Sirve para el orden y el")
    print("    contenido; NO para fechar un solape de 200-1,000 ms. Por eso la pregunta")
    print("    de arriba se contesta con el eje del servidor y no con este archivo.")


# --------------------------------------------------------------------------
# E5 sobre el lote de agosto: el solape, separado en sus dos conductas
# --------------------------------------------------------------------------


# Un adelanto de 200 ms es UN SOLO BIN del eje del servidor y es indistinguible
# de cero. Solo cuenta como solape lo que supere el bin — o sea, 400 ms o más,
# porque todos los sellos son múltiplos de 200.
MIN_LEAD_MS: Final[int] = 400

# Cortes para estratificar por duración del turno del transportista: pone a
# prueba si los arranques adelantados caen en los turnos largos y no en los
# cortos.
DURATION_CUTOFFS_MS: Final[tuple[int, ...]] = (1500, 3000, 5000)


@dataclass
class Onset:
    """Un turno del transportista y el arranque de texto del agente que le sigue."""

    call: str
    turn_start_ms: int
    turn_end_ms: int
    onset_ms: int
    lead_ms: int
    turn_duration_ms: int
    talk_over: bool


def onsets_of(census: Census, entry: list[tuple[int, int]]) -> list[Onset]:
    segments = spoken_segments(census)
    output: list[Onset] = []
    for turn in census.user_turns:
        subsequent = [t[0] for t in segments if t[0] > turn.start_ms]
        if not subsequent:
            continue
        onset = min(subsequent)
        output.append(
            Onset(
                call=census.name,
                turn_start_ms=turn.start_ms,
                turn_end_ms=turn.end_ms,
                onset_ms=onset,
                lead_ms=turn.end_ms - onset,
                turn_duration_ms=turn.end_ms - turn.start_ms,
                talk_over=any(a <= onset < b for a, b in entry),
            )
        )
    return output


def _bucket(duration_ms: int) -> str:
    previous = 0
    for cutoff in DURATION_CUTOFFS_MS:
        if duration_ms < cutoff:
            return f"{previous/1000:.1f}-{cutoff/1000:.1f} s"
        previous = cutoff
    return f">= {DURATION_CUTOFFS_MS[-1]/1000:.1f} s"


def e5_lote(folder: Path) -> None:
    """E5 sobre todos los censos de una carpeta, estratificado por duración."""
    print("=" * 78)
    print("E5 EN LOTE — ¿EL SOLAPE ES FULL-DUPLEX O ES EL HANGOVER DEL DETECTOR?")
    print("=" * 78)
    print(f"  Solape = el texto del agente arranca >= {MIN_LEAD_MS} ms antes del")
    print("  end_ms del turno. El piso son 200 ms (un bin del eje), así que un adelanto")
    print("  de 200 ms NO cuenta: es indistinguible de cero.")
    print()

    all_entries: list[Onset] = []
    print(f"  {'call':<14}{'turns':>7}{'overlaps':>9}{'sobre habla':>13}{'hangover':>10}")
    for ruta in sorted(folder.glob("census_*.jsonl")):
        census = read_census(ruta)
        if not census.user_turns or not census.fragments:
            continue
        onset_list = onsets_of(census, input_spans(ruta))
        all_entries.extend(onset_list)
        overlaps = [a for a in onset_list if a.lead_ms >= MIN_LEAD_MS]
        on_top = sum(1 for a in overlaps if a.talk_over)
        print(f"  {ruta.stem[6:18]:<14}{len(onset_list):>7}{len(overlaps):>9}{on_top:>13}"
              f"{len(overlaps) - on_top:>10}")

    overlaps = [a for a in all_entries if a.lead_ms >= MIN_LEAD_MS]
    if not overlaps:
        print()
        print("  (ningún solape por encima del piso)")
        return
    on_top = sum(1 for a in overlaps if a.talk_over)
    print()
    print(f"  AGREGADO: {len(all_entries)} turnos pareados, {len(overlaps)} con solape real")
    print(f"    sobre habla del transportista: {on_top:>4} ({on_top/len(overlaps):>5.1%})"
          "  -> full-duplex de verdad")
    print(f"    dentro del hangover:           {len(overlaps)-on_top:>4} "
          f"({(len(overlaps)-on_top)/len(overlaps):>5.1%})  -> pipeline rápido, no full-duplex")

    print()
    print("  ESTRATIFICADO POR DURACIÓN DEL TURNO DEL TRANSPORTISTA")
    print(f"    {'bucket':>12}{'turns':>8}{'overlaps':>9}{'% solape':>10}"
          f"{'sobre habla':>13}{'% de los solapes':>18}")
    buckets: dict[str, list[Onset]] = {}
    for a in all_entries:
        buckets.setdefault(_bucket(a.turn_duration_ms), []).append(a)
    order = [f"{0/1000:.1f}-{DURATION_CUTOFFS_MS[0]/1000:.1f} s"] + [
        f"{DURATION_CUTOFFS_MS[i]/1000:.1f}-{DURATION_CUTOFFS_MS[i+1]/1000:.1f} s"
        for i in range(len(DURATION_CUTOFFS_MS) - 1)
    ] + [f">= {DURATION_CUTOFFS_MS[-1]/1000:.1f} s"]
    for name in order:
        group = buckets.get(name) or []
        if not group:
            continue
        s_ = [a for a in group if a.lead_ms >= MIN_LEAD_MS]
        sh = sum(1 for a in s_ if a.talk_over)
        print(f"    {name:>12}{len(group):>8}{len(s_):>9}{len(s_)/len(group):>10.0%}"
              f"{sh:>13}{(sh/len(s_) if s_ else 0):>18.0%}")

    v = sorted(a.lead_ms for a in overlaps)
    print()
    print(f"  adelanto de los solapes: p50 {st.median(v):,.0f}  p95 "
          f"{v[int(0.95*(len(v)-1))]:,.0f}  max {v[-1]:,.0f} ms")
    longs = [a.turn_duration_ms for a in all_entries if a.lead_ms >= MIN_LEAD_MS]
    short_items = [a.turn_duration_ms for a in all_entries if a.lead_ms < MIN_LEAD_MS]
    if longs and short_items:
        print(f"  duración del turno — con solape: p50 {st.median(longs):,.0f} ms   "
              f"sin solape: p50 {st.median(short_items):,.0f} ms")


# --------------------------------------------------------------------------
# E6
# --------------------------------------------------------------------------


def e6(everything: dict[str, dict[str, list[float]]]) -> None:
    print()
    print("=" * 78)
    print("E6 — UMBRAL Y VENTANA: SENSIBILIDAD")
    print("=" * 78)
    for campo, title in DOMAINS:
        speech, filler = everything[campo][SPEECH], everything[campo][RELLENO]
        if not speech or not filler:
            continue
        print(f"\n  {title}")
        print(f"    {'umbral':>9}{'dBFS':>9}{'relleno que dispara':>22}{'habla que se pierde':>22}")
        for umbral in (33.0, 50.0, 100.0, 150.0, INHERITED_THRESHOLD_RMS, 300.0, 500.0, 800.0):
            fr = sum(1 for x in filler if x >= umbral) / len(filler)
            fh = sum(1 for x in speech if x < umbral) / len(speech)
            mark = "  <- heredado" if umbral == INHERITED_THRESHOLD_RMS else ""
            print(f"    {umbral:>9.0f}{dbfs(umbral):>9.1f}{fr:>21.2%}{fh:>21.2%}{mark}")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.diagnostics.threshold_calibration",
        description="E1-E6 de la corrida de calibración del detector de voz de PoC3",
    )
    arg_parser.add_argument("--call-id")
    arg_parser.add_argument(
        "--lote",
        action="store_true",
        help="corre solo E5 sobre todos los censos de la carpeta de grabaciones",
    )
    arg_parser.add_argument(
        "--folder", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = arg_parser.parse_args(argv)

    base: Path = args.folder
    if args.lote:
        print(
            "VALIDACIÓN DE INSTRUMENTO, NO RESULTADO — "
            "ninguna cifra de aquí es reportable"
        )
        print()
        e5_lote(base / "recordings")
        return 0
    if not args.call_id:
        arg_parser.error("hace falta --call-id, o --lote")
    census_path = base / "recordings" / f"census_{args.call_id}.jsonl"
    dump_data = read_dump(base / "diagnostics" / "energy" / f"energy_{args.call_id}.jsonl")
    census = read_census(census_path)
    entry = input_spans(census_path)

    print("VALIDACIÓN DE INSTRUMENTO, NO RESULTADO — ninguna cifra de aquí es reportable")
    print(
        f"llamada {args.call_id}   deltas {len(dump_data.rows)}   "
        f"audio crudo volcado: {dump_data.header.get('raw_audio')}\n"
    )
    everything = e1(dump_data, census)
    e2(everything)
    guard_sensitivity(dump_data, census)
    e3(everything)
    e4(dump_data, census)
    e5(census, entry, base / "diagnostics" / "auditoria" / f"auditoria_{args.call_id}.wav")
    e6(everything)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
