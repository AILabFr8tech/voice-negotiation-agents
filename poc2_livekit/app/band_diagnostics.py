"""¿La banda telefónica baja la confianza del detector de fin de turno?

Una sonda previa dejó probabilidades del detector apelmazadas alrededor de su
umbral, con dos explicaciones **confundidas**: la banda telefónica (8 kHz + μ-law
le quitan al audio la parte alta del espectro, donde viven las pistas de final de
frase) y la incoherencia de la propia sonda (el mismo clip repetido contesta "Sí,
soy Carlos. Dígame." a una oferta de precio). Este módulo separa la primera.
Datos en `docs/comparativo_results.md`.

## El diseño

Pareado, mismo clip fuente, dos condiciones, n por brazo configurable:

- **Brazo A (degradado)**: el clip como lo prepara `shared/audio/` — a -18 dBFS,
  8 kHz, ida y vuelta por μ-law, publicado a 8 kHz. El salto a 48 kHz + Opus lo
  da LiveKit, igual que con el trunk SIP.
- **Brazo B (limpio)**: el **mismo** archivo con la **misma** ganancia, sin
  remuestreo ni μ-law, publicado a 48 kHz nativos.

Los dos llevan **el mismo recorte**, al milisegundo: el brazo B se corta con las
fronteras que produjo la degradación del A. Sin eso el corte abrupto del final
—que es una de las sospechas— no quedaría controlado. El ruido de confort corre
continuo en los dos al mismo nivel en dBFS; nada más cambia.

## Por qué los brazos se alternan por LLAMADA y no por turno

Una pista publicada tiene **una** tasa de muestreo y `rtc.AudioSource` no la
cambia en caliente. Alternar dentro de una llamada exigiría republicar la pista
entre turnos (abre un hueco sin audio y contradice el ruido continuo, del que
depende el endpointing) o publicar dos pistas de micrófono a la vez (`RoomIO`
engancha una sola vía `find_micro_track_id`, así que un brazo quedaría mudo sin
avisar).

Así que se alterna por llamada, con **un turno medido por llamada**. Conserva el
propósito de intercalar —que una deriva del servicio no se confunda con la
condición— y trae una ventaja no buscada: cada turno medido cae en la misma
posición de la conversación, contestando el saludo, donde esa línea **sí es
coherente**. El contraste queda libre de la incoherencia en vez de arrastrarla.

## Lo que este diseño NO separa

El brazo limpio cambia también el transporte: al publicar a 48 kHz nativos
LiveKit ya no remuestrea. La comparación aísla **"banda telefónica + códec +
remuestreo del SFU" como bloque**, no el códec solo. Se reporta así.

## Banderas duras

- Los JSON van a `poc2_livekit/diagnostics/banda/` y llevan `exclude_from_pooled`:
  ni filtrándose pueden entrar a una poza de percentiles, al comparativo ni a
  ningún WER.
- El brazo limpio **no es una llamada telefónica** y no se compara con nada de la
  campaña.
- El contrato de métricas no se toca: todo vive en bloques extra.
- Esta ruta limpia es diagnóstico del arnés, **no** un modo de preparación de
  clips: `shared.audio.cli preparar` no la expone y no puede producirla.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from poc2_livekit.app import harness
from poc2_livekit.app.config import (
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_livekit,
    verify_for_voice,
)
from poc2_livekit.app.statistical_tests import describe, fisher_exact, mann_whitney
from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.audio.clips import degrade
from shared.audio.level import load_calibration, normalize_to_target
from shared.audio.wav import leer_wav

logger = logging.getLogger(__name__)

ARM_DEGRADED: Final[str] = "A_degraded"
ARM_CLEAN: Final[str] = "B_clean"

CLEAN_SAMPLE_RATE: Final[int] = 48_000

# Carpeta de diagnóstico, aparte de `grabaciones/`. Ver las banderas duras.
DIAGNOSTICS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "banda"
)

# Cuánto se espera a que el worker escriba su JSON tras desconectarnos.
JSON_WAIT_S: Final[float] = 25.0

# Semilla del sorteo dentro de cada bloque. Se registra en el resultado: la
# asignación es reproducible y no se pudo elegir después de ver los datos.
DEFAULT_SEED: Final[int] = 20260914

# Las 5 probabilidades de la sonda del 14 de septiembre de 2026, con el mismo
# clip degradado pero en una conversación incoherente. Se reportan al lado como
# **observación sugerente** y sin contraste: no son un brazo de este
# experimento. Las diferencias van declaradas en `analizar`.
PROBE_OF_SEPTEMBER_14: Final[tuple[float, ...]] = (
    0.4991,
    0.4709,
    0.2953,
    0.3610,
    0.2982,
)


@dataclass
class Arm:
    """Un brazo del experimento: su clip ya listo y a qué tasa se publica."""

    name: str
    samples: np.ndarray
    sample_rate: int
    descripcion: str
    speech_rms_dbfs: float

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass
class TurnMeasurement:
    """Lo que se registra de cada llamada del experimento."""

    arm: str
    # Ronda del diseño (1..n): identifica el bloque pareado.
    indice: int
    room: str
    # Orden real de ejecución (1..N), incluidos los reintentos. Es lo que
    # permite después buscar una deriva temporal dentro de la propia tanda.
    order: int = 0
    timestamp: str = ""
    # 1 = primer intento de esa casilla; 2, 3… = reposiciones del mismo brazo.
    attempt: int = 1
    discarded: bool = False
    discard_reason: str = ""
    probability: float | None = None
    umbral: float | None = None
    wait_s: float | None = None
    motivo: str = "not_captured"
    end_of_turn_ms: float | None = None
    stt: str = ""
    responded: bool = False
    technical_failures: int = 0

    @property
    def captured(self) -> bool:
        return self.probability is not None

    @property
    def valid(self) -> bool:
        """Entra al análisis. Una descartada nunca entra, ni con probabilidad."""
        return self.captured and not self.discarded


# --------------------------------------------------------------------------
# Descarte y reposición
# --------------------------------------------------------------------------
#
# La regla se fija **antes** de correr y no mira jamás el VALOR de la
# probabilidad: descartar por el valor sería elegir los datos que confirman lo
# que uno espera. Solo se descarta cuando la llamada no produjo la condición que
# se quería medir.

MAX_ATTEMPTS_PER_CELL: Final[int] = 3

# Tope de llamadas extra en toda la tanda. Sin él, una racha de fallos podría
# multiplicar el costo aprobado sin que nadie lo decida.
REDRAW_BUDGET: Final[int] = 6


def evaluate_discard(measurement: TurnMeasurement) -> str:
    """Por qué esta llamada no sirve. Cadena vacía = sirve.

    **No mira el valor de la probabilidad**, solo si existe. Hay una prueba que
    lo verifica: dos mediciones idénticas salvo por el valor tienen que recibir
    el mismo veredicto.
    """
    if not measurement.responded:
        return "el agente no contestó: el turno nunca se cerró"
    if not measurement.stt.strip():
        return "el STT no oyó el clip: la condición no llegó a darse"
    if measurement.technical_failures:
        return (
            f"{measurement.technical_failures} fallo(s) técnico(s) del proveedor "
            "durante la llamada"
        )
    if not measurement.captured:
        return "sin probabilidad del detector: no hay dato para el contraste"
    return ""


def prepare_arms(source_path: Path | str) -> tuple[Arm, Arm]:
    """Los dos clips, del mismo origen y con el mismo recorte al milisegundo.

    El brazo limpio se corta con las fronteras que produjo la degradación del
    otro. Si cada uno recortara por su cuenta, los bordes —y en particular el
    corte del final, que es una de las sospechas— dejarían de estar controlados.
    """
    source, source_sample_rate = leer_wav(source_path)
    if source_sample_rate != CLEAN_SAMPLE_RATE:
        raise harness.HarnessFailure(
            f"el diagnóstico espera una fuente a {CLEAN_SAMPLE_RATE} Hz y "
            f"{Path(source_path).name} está a {source_sample_rate}: el brazo limpio "
            "no puede remuestrear sin dejar de ser limpio"
        )

    calibration = load_calibration()

    # --- Brazo A: la ruta de producción, tal cual ---
    degraded_one = degrade(source, source_sample_rate, calibration=calibration)

    # --- Brazo B: misma ganancia, sin remuestreo ni códec ---
    leveled = normalize_to_target(
        source,
        source_sample_rate,
        target_dbfs=calibration.target_dbfs,
        drop_db=calibration.drop_db,
    )
    start = int(round(degraded_one.start_cut_ms * source_sample_rate / 1000.0))
    clean_samples = leveled.samples[
        start : start + int(round(degraded_one.duration_s * source_sample_rate))
    ]

    from shared.audio.level import rms_of_speech

    return (
        Arm(
            name=ARM_DEGRADED,
            samples=degraded_one.pcm8k,
            sample_rate=TELEPHONY_SAMPLE_RATE,
            descripcion=f"8 kHz + μ-law, publicado a {TELEPHONY_SAMPLE_RATE} Hz",
            speech_rms_dbfs=degraded_one.speech_rms_dbfs,
        ),
        Arm(
            name=ARM_CLEAN,
            samples=np.ascontiguousarray(clean_samples, dtype=np.int16),
            sample_rate=CLEAN_SAMPLE_RATE,
            descripcion=f"sin remuestreo ni códec, publicado a {CLEAN_SAMPLE_RATE} Hz",
            speech_rms_dbfs=round(
                rms_of_speech(clean_samples, source_sample_rate, drop_db=calibration.drop_db), 2
            ),
        ),
    )


async def _wait_for_json(room: str) -> dict[str, Any] | None:
    """El worker escribe su JSON después de que nos desconectamos."""
    ruta = harness.Path(__file__).resolve().parents[1] / "recordings" / f"call_{room}.json"
    limit = time.monotonic() + JSON_WAIT_S
    while time.monotonic() < limit:
        if ruta.is_file():
            await asyncio.sleep(0.4)  # que termine de escribirlo
            try:
                return json.loads(ruta.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        await asyncio.sleep(0.5)
    return None


def _move_to_diagnostics(room: str, destination: Path) -> Path | None:
    """Saca el JSON de `grabaciones/` en cuanto existe. Bandera dura."""
    origin = Path(__file__).resolve().parents[1] / "recordings" / f"call_{room}.json"
    if not origin.is_file():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / origin.name
    origin.replace(final)
    return final


def _read_measurement(data: dict[str, Any], measurement: TurnMeasurement) -> TurnMeasurement:
    """Saca del JSON del worker el turno de régimen —el nuestro— y su motivo.

    Hay exactamente dos muestras por llamada: la apertura (el saludo, que no
    tiene turno de usuario antes) y la nuestra. Se busca la primera que no sea
    apertura; si no hay, se queda en `no_capturado`, que es distinto de
    "cerró por confianza".
    """
    measurement.technical_failures = len(data.get("technical_failures") or [])
    latencies = {m.get("indice"): m for m in data.get("latencies", [])}
    for extra in data.get("livekit_extra", {}).get("per_turn", []):
        if extra.get("is_opening") or extra.get("indice") is None:
            continue
        end = extra.get("end_of_turn") or {}
        lat = latencies.get(extra["indice"], {})
        measurement.probability = end.get("probability")
        measurement.umbral = end.get("umbral")
        measurement.wait_s = end.get("wait_s")
        measurement.motivo = str(end.get("motivo", "not_captured"))
        measurement.end_of_turn_ms = lat.get("end_of_turn_ms")
        break
    return measurement


async def run_one(
    settings: Settings, arm: Arm, indice: int, *, wait_s: float
) -> TurnMeasurement:
    """Una llamada de un turno con el clip del brazo que toque."""
    room = settings.room_name(
        suffix=f"diag-banda-{arm.name}-{indice:02d}-{time.strftime('%H%M%S')}"
    )
    measurement = TurnMeasurement(arm=arm.name, indice=indice, room=room)

    step = harness.ScriptStep(
        turn="T1",
        text="Sí, soy Carlos. Dígame.",
        samples=arm.samples,
        clip_id=f"diag_{arm.name}",
        origin=arm.descripcion,
        of_script=True,
        # El brazo limpio publica a 48 kHz: sin esto, su duración se leería 6
        # veces más larga. No cambia nada de lo ya medido —el fin de voz sale
        # del deadline del reproductor, no de aquí— pero sí el cero real del turno.
        sample_rate=arm.sample_rate,
    )

    observed = await harness.run_session(
        settings,
        room=room,
        script=[step],
        publish_sample_rate=arm.sample_rate,
        wait_after_clip=wait_s,
        # `diagnostico` hace que el agente marque `excluir_de_pooled`.
        identification={"poc": 2, "diagnostics": "banda"},
        silent=True,
    )

    if observed.turns:
        measurement.stt = observed.turns[0].stt
        measurement.responded = observed.turns[0].responded

    data = await _wait_for_json(room)
    if data is not None:
        _read_measurement(data, measurement)
    _move_to_diagnostics(room, DIAGNOSTICS_PATH)
    return measurement


def build_schedule(
    brazos: tuple[Arm, Arm], *, n: int, seed: int
) -> list[tuple[int, Arm]]:
    """Bloques pareados de 2, con el orden **sorteado** dentro de cada bloque.

    A, B, A, B… estricto empareja la condición con la paridad del orden: si algo
    del servicio alternara con ese mismo periodo —o si el segundo turno de cada
    bloque heredara algo del primero— el efecto quedaría confundido con el
    brazo. Sorteando dentro del bloque se conserva el balance exacto (uno de
    cada brazo por bloque) y se rompe esa correlación.

    La semilla se registra en el JSON: la asignación es reproducible y nadie
    puede sospechar que se eligió después de ver los datos.
    """
    import random

    draw = random.Random(seed)
    schedule: list[tuple[int, Arm]] = []
    for round_no in range(1, n + 1):
        block = list(brazos)
        draw.shuffle(block)
        schedule.extend((round_no, arm) for arm in block)
    return schedule


async def run_experiment(
    settings: Settings,
    brazos: tuple[Arm, Arm],
    *,
    n: int,
    wait_s: float,
    seed: int,
) -> list[TurnMeasurement]:
    """Corre el calendario y repone las casillas que se caen, con el mismo brazo."""
    schedule = build_schedule(brazos, n=n, seed=seed)
    measurements: list[TurnMeasurement] = []
    redraws = 0
    order = 0

    for round_no, arm in schedule:
        for attempt in range(1, MAX_ATTEMPTS_PER_CELL + 1):
            order += 1
            measurement = await run_one(settings, arm, round_no, wait_s=wait_s)
            measurement.order = order
            measurement.attempt = attempt
            measurement.timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            motivo = evaluate_discard(measurement)
            measurement.discarded = bool(motivo)
            measurement.discard_reason = motivo
            measurements.append(measurement)

            label = f"  [{order:>2}] {arm.name:<12} ronda {round_no:>2}"
            if attempt > 1:
                label += f" (reposición {attempt - 1})"
            if not motivo:
                print(
                    f"{label}  p={measurement.probability:.4f}  {measurement.motivo}"
                )
                break

            print(f"{label}  DESCARTADA — {motivo}")
            if attempt == MAX_ATTEMPTS_PER_CELL:
                print(f"       casilla perdida tras {attempt} attempts: queda como hueco")
                break
            if redraws >= REDRAW_BUDGET:
                print(
                    f"       presupuesto de reposiciones agotado "
                    f"({REDRAW_BUDGET}): la casilla queda como hueco"
                )
                break
            redraws += 1

    return measurements


# --------------------------------------------------------------------------
# Análisis
# --------------------------------------------------------------------------


def analyze(measurements: list[TurnMeasurement]) -> dict[str, Any]:
    """Las dos distribuciones, Mann-Whitney y Fisher. Con su cobertura."""
    resultado: dict[str, Any] = {"per_arm": {}}

    per_arm: dict[str, list[TurnMeasurement]] = {}
    for measurement in measurements:
        per_arm.setdefault(measurement.arm, []).append(measurement)

    probabilities: dict[str, list[float]] = {}
    for name, group in per_arm.items():
        valid_ones = [m for m in group if m.valid]
        discarded_ones = [m for m in group if m.discarded]
        values = [m.probability for m in valid_ones if m.probability is not None]
        probabilities[name] = values
        expirations = sum(1 for m in valid_ones if m.motivo == "vencimiento")

        discard_count: dict[str, int] = {}
        for measurement in discarded_ones:
            clave = measurement.discard_reason.split(":")[0]
            discard_count[clave] = discard_count.get(clave, 0) + 1

        resultado["per_arm"][name] = {
            "calls": len(group),
            "valid_ones": len(valid_ones),
            "discarded_ones": len(discarded_ones),
            "discard_reasons": discard_count,
            "redraws": sum(1 for m in group if m.attempt > 1),
            # La cobertura va SIEMPRE, y sobre las llamadas que NO se
            # descartaron: cero probabilidades es "no capturado", nunca "sin
            # vencimientos".
            "cobertura": {
                "captured_ones": len(valid_ones),
                "de": len(group) - len(discarded_ones) if len(group) > len(discarded_ones) else 0,
                "total_calls": len(group),
                "porcentaje": (
                    round(100.0 * len(valid_ones) / len(group), 1) if group else 0.0
                ),
            },
            "probabilities": sorted(values),
            "distribution": describe(values),
            "expirations": expirations,
            "confianza": len(valid_ones) - expirations,
            "end_of_turn_ms": describe(
                [m.end_of_turn_ms for m in valid_ones if m.end_of_turn_ms is not None]
            ),
        }

    a = probabilities.get(ARM_DEGRADED, [])
    b = probabilities.get(ARM_CLEAN, [])
    if a and b:
        contrast = mann_whitney(a, b)
        resultado["mann_whitney"] = {
            **asdict(contrast),
            "note": (
                "contraste principal, sobre las probabilidades continuas del "
                "detector"
            ),
        }
    else:
        resultado["mann_whitney"] = {
            "note": "sin probabilidades en algún brazo: no hay contraste que hacer"
        }

    expiry_a = resultado["per_arm"].get(ARM_DEGRADED, {}).get("expirations")
    expiry_b = resultado["per_arm"].get(ARM_CLEAN, {}).get("expirations")
    if expiry_a is not None and expiry_b is not None:
        n_a = resultado["per_arm"][ARM_DEGRADED]["cobertura"]["captured_ones"]
        n_b = resultado["per_arm"][ARM_CLEAN]["cobertura"]["captured_ones"]
        if n_a and n_b:
            contrast = fisher_exact(expiry_a, n_a - expiry_a, expiry_b, n_b - expiry_b)
            resultado["fisher"] = {
                **asdict(contrast),
                "note": (
                    "SECUNDARIO. Con ~10 por brazo la potencia es muy baja: un p "
                    "alto aquí no es evidencia de que no haya diferencia"
                ),
            }

    resultado["inconsistent_probe"] = {
        "probabilities": list(PROBE_OF_SEPTEMBER_14),
        "distribution": describe(list(PROBE_OF_SEPTEMBER_14)),
        "is_suggestive_observation": True,
        "no_p_value": (
            "NO se contrasta y no lleva p: no es un brazo de este experimento "
            "sino una corrida anterior con otro diseño"
        ),
        "declared_differences": [
            "coherencia: ahí el mismo clip contestaba a ofertas de precio, aquí "
            "contesta al saludo, que es la respuesta que corresponde",
            "posición del turno: ahí eran los turnos 1 a 5 de una conversación, "
            "aquí todos son el primer turno",
            "estructura: ahí fue UNA llamada de 5 turnos, aquí son 20 llamadas "
            "de un turno, así que allá los turnos compartían sesión y contexto",
        ],
    }

    resultado["confounder_declared"] = (
        "El brazo limpio también cambia el transporte: publica a 48 kHz nativos "
        "y LiveKit ya no remuestrea de 8 a 48. La comparación aísla 'banda "
        "telefónica + códec + remuestreo del SFU' COMO BLOQUE, no el códec solo."
    )
    return resultado


def print_analysis(analysis: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("Diagnóstico de banda — ¿la degradación baja la confianza del detector?")
    print("=" * 70)

    for name, data in analysis["per_arm"].items():
        coverage = data["cobertura"]
        print(f"\n  {name}")
        print(
            f"    llamadas              {data['calls']} "
            f"({data['valid_ones']} válidas, {data['discarded_ones']} descartadas, "
            f"{data['redraws']} reposiciones)"
        )
        for motivo, how_many in data["discard_reasons"].items():
            print(f"                          {how_many}x {motivo}")
        print(
            f"    cobertura de captura  {coverage['captured_ones']}/"
            f"{coverage['total_calls']} llamadas "
            f"({coverage['porcentaje']:.0f}%)"
        )
        if not coverage["captured_ones"]:
            print("    (sin probabilidades: NO CAPTURADO, que no es 'sin vencimientos')")
            continue

        distribution = data["distribution"]
        print(
            f"    probabilidad          min {distribution['min']:.4f} · "
            f"q1 {distribution['q1']:.4f} · mediana {distribution['median_value']:.4f} · "
            f"q3 {distribution['q3']:.4f} · max {distribution['max']:.4f}"
        )
        print(f"    valores               {[round(v, 4) for v in data['probabilities']]}")
        print(
            f"    cierres               {data['confianza']} por confianza, "
            f"{data['expirations']} por vencimiento"
        )
        eot = data["end_of_turn_ms"]
        if eot.get("n"):
            print(
                f"    end_of_turn_ms       mediana {eot['median_value']:,.1f} "
                f"(min {eot['min']:,.1f} · max {eot['max']:,.1f})"
            )

    contrast = analysis.get("mann_whitney", {})
    print("\n  contraste principal")
    if "p" in contrast:
        print(
            f"    Mann-Whitney          U = {contrast['statistic']:g}, "
            f"p = {contrast['p']:.4f}  ({contrast['method']})"
        )
        print(f"                          {contrast['detail']}")
    else:
        print(f"    {contrast.get('note')}")

    fisher = analysis.get("fisher")
    if fisher:
        print("\n  contraste secundario")
        print(f"    Fisher exacto         p = {fisher['p']:.4f}")
        print(f"                          {fisher['note']}")

    probe = analysis.get("inconsistent_probe")
    if probe:
        distribution = probe["distribution"]
        print("\n  sonda incoherente del 14 de septiembre (OBSERVACIÓN, sin valor p)")
        print(f"    probabilidades        {probe['probabilities']}")
        print(
            f"    distribución          min {distribution['min']:.4f} · "
            f"mediana {distribution['median_value']:.4f} · max {distribution['max']:.4f}"
        )
        print("    no se contrasta porque difiere en tres cosas a la vez:")
        for difference in probe["declared_differences"]:
            print(f"      - {difference}")

    print(f"\n  confounder declarado\n    {analysis['confounder_declared']}")
    print(
        "\n  El contraste A vs B es el único que concluye. Lo de la sonda es\n"
        "  sugerente y nada más."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def estimate_cost(n: int, brazos: tuple[Arm, Arm], wait_s: float) -> float:
    """Cota alta: cada llamada paga TTS y LLM completos."""
    per_call = harness.MAX_AGENT_WAIT_S + wait_s + max(
        b.duration_s for b in brazos
    )
    return sum(harness.estimate_cost(per_call) for _ in range(2 * n))


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.band_diagnostics",
        description=(
            "¿La banda telefónica baja la confianza del detector de fin de turno? "
            "Pareado, mismo clip, dos condiciones, alternadas por llamada."
        ),
    )
    arg_parser.add_argument(
        "--source", required=True, help="WAV de 48 kHz sin degradar (la grabación cruda)"
    )
    arg_parser.add_argument("--n", type=int, default=10, help="turnos por brazo (default 10)")
    arg_parser.add_argument(
        "--wait", type=float, default=20.0, help="tope de espera al agente por turno"
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"semilla del sorteo dentro de cada bloque (default {DEFAULT_SEED}). "
            "Queda registrada en el resultado"
        ),
    )
    arg_parser.add_argument("--accept-cost", action="store_true")
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_voice(settings)

    try:
        print(harness.verify_constants())
        brazos = prepare_arms(args.source)
    except (harness.HarnessFailure, ValueError) as error:
        print(f"[FALLA] {error}")
        return 1

    print("\n  los dos brazos")
    for arm in brazos:
        print(
            f"    {arm.name:<12} {arm.descripcion:<44} "
            f"{arm.duration_s:.3f} s · voz {arm.speech_rms_dbfs:.2f} dBFS"
        )
    schedule = build_schedule(brazos, n=args.n, seed=args.seed)
    order = " ".join("A" if b.name == ARM_DEGRADED else "B" for _, b in schedule)
    print(f"\n  turnos           {args.n} por brazo, {2 * args.n} llamadas")
    print(f"  semilla           {args.seed} (queda en el resultado)")
    print(f"  orden sorteado    {order}")
    print(f"  ruido de confort  {load_calibration().noise_dbfs:.1f} dBFS en los dos")
    print(
        f"  reposición        hasta {MAX_ATTEMPTS_PER_CELL} intentos por "
        f"casilla, tope de {REDRAW_BUDGET} llamadas extra"
    )
    print(f"  JSON a            {DIAGNOSTICS_PATH}")

    cost = estimate_cost(args.n, brazos, args.wait)
    print(f"\n  estimado         ~${cost:.2f} USD (cota alta)")

    if not args.accept_cost:
        print(
            "\n  lo que NO se gasta: Twilio — no se crea participante SIP.\n"
            "\nVuelve a correrlo con --accept-cost para ejecutarlo."
        )
        return 0

    print()
    measurements = asyncio.run(
        run_experiment(
            settings, brazos, n=args.n, wait_s=args.wait, seed=args.seed
        )
    )
    analysis = analyze(measurements)
    print_analysis(analysis)

    DIAGNOSTICS_PATH.mkdir(parents=True, exist_ok=True)
    destination = DIAGNOSTICS_PATH / f"resultado-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(args.source),
                "diseno": {
                    "n_per_arm": args.n,
                    "seed": args.seed,
                    "asignacion": "bloques pareados de 2, orden sorteado dentro del bloque",
                    "max_attempts_per_cell": MAX_ATTEMPTS_PER_CELL,
                    "redraw_budget": REDRAW_BUDGET,
                    "discard_rule": (
                        "fijada antes de correr; NO mira el valor de la "
                        "probabilidad. Se descarta si el agente no contestó, si "
                        "el STT no oyó el clip, si hubo fallo técnico del "
                        "proveedor, o si no llegó probabilidad del detector. La "
                        "casilla se repone con el MISMO brazo"
                    ),
                },
                "brazos": [
                    {
                        "name": b.name,
                        "descripcion": b.descripcion,
                        "duration_s": round(b.duration_s, 3),
                        "speech_rms_dbfs": b.speech_rms_dbfs,
                        "sample_rate": b.sample_rate,
                    }
                    for b in brazos
                ],
                "measurements": [asdict(m) for m in measurements],
                "analysis": analysis,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  resultado en          {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
