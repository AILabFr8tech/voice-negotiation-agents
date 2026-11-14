"""Corpus sintético: valida el cálculo sin gastar una llamada.

Dos usos:

1. **Validar la aritmética.** Los tests construyen corridas con errores
   sembrados a propósito y verifican que el WER y los percentiles dan lo que
   tienen que dar. Ninguna de las 15 llamadas reales se gasta en depurar una
   división.
2. **Documentar el contrato de captura.** Un JSON generado aquí tiene la llave
   `latencias` con la forma exacta que el pipeline telefónico escribe.

Todo es determinista: se siembra un `random.Random` con una semilla fija, así
que dos corridas del mismo corpus dan exactamente los mismos números y una
diferencia en los tests significa que cambió el cálculo, no el azar.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from shared.metrics import reference as ref
from shared.metrics.scenarios import Scenario, load_script
from shared.metrics.latency import TurnSample

DEFAULT_SEED: Final[int] = 20260805

# Latencias plausibles en ms, del orden de lo que se vio en la llamada real.
# No pretenden predecir nada: solo tener una distribución con forma para que los
# percentiles se puedan verificar.
_OPENING_MS: Final[tuple[float, float]] = (1800.0, 2600.0)
_REGIMEN_MS: Final[tuple[float, float]] = (700.0, 1500.0)
_INTERRUPTED_MS: Final[tuple[float, float]] = (250.0, 600.0)

# Sustituciones que imitan errores típicos de un STT en español mexicano.
_CONFUSIONS: Final[dict[str, str]] = {
    "digame": "dime",
    "flete": "fleje",
    "diesel": "diseño",
    "casetas": "cassettes",
    "parejo": "matched_pair",
    "broncas": "brocas",
    "empalma": "en palma",
    "muere": "muere ahi",
}


@dataclass
class SyntheticCall:
    """Una llamada de mentira, con su referencia y lo que el STT "entendió"."""

    data: dict[str, Any]
    reference_text: str
    reference_content: str
    scenario: int
    repetition: int
    poc: int

    @property
    def call_sid(self) -> str:
        return str(self.data["call_sid"])


# --------------------------------------------------------------------------
# Latencia
# --------------------------------------------------------------------------


def synthetic_samples(
    n_turns: int,
    rng: random.Random,
    *,
    interrupted_turns: Sequence[int] = (),
    with_stages: bool = True,
) -> list[TurnSample]:
    """Muestras de latencia de una llamada.

    El turno 0 es apertura y sale caro a propósito (arranque en frío: WebSocket
    del TTS y primera petición al LLM con el prompt completo). Los turnos de
    `interrumpidos` salen rápido, porque medir un barge-in mide otra cosa.

    Con `with_stages=False` se emula PoC3: solo extremo a extremo, sin desglose.
    """
    samples: list[TurnSample] = []
    for indice in range(n_turns):
        interrupted = indice in interrupted_turns
        if interrupted:
            span = _INTERRUPTED_MS
        elif indice == 0:
            span = _OPENING_MS
        else:
            span = _REGIMEN_MS
        total = rng.uniform(*span)

        if not with_stages:
            samples.append(
                TurnSample(
                    indice=indice,
                    full_turn_ms=round(total, 1),
                    interrupted=interrupted,
                )
            )
            continue

        # Se reparte el total entre las etapas para que el desglose sume algo
        # coherente con el extremo a extremo.
        end_of_turn = total * rng.uniform(0.10, 0.20)
        stt = total * rng.uniform(0.10, 0.18)
        llm = total * rng.uniform(0.35, 0.50)
        tts = total - end_of_turn - stt - llm
        samples.append(
            TurnSample(
                indice=indice,
                full_turn_ms=round(total, 1),
                end_of_turn_ms=round(end_of_turn, 1),
                stt_ms=round(stt, 1),
                llm_ttft_ms=round(llm, 1),
                tts_ttfb_ms=round(max(tts, 20.0), 1),
                interrupted=interrupted,
            )
        )
    return samples


# --------------------------------------------------------------------------
# Transcripción
# --------------------------------------------------------------------------


def corrupt(text: str, rng: random.Random, *, sample_rate: float = 0.12) -> str:
    """Simula lo que un STT le hace a una frase.

    Tres cosas que se vieron en la llamada real: se come palabras cortas,
    confunde palabras del dominio, y a veces escribe los montos con dígito en
    lugar de letra (esto último **no** debe contar como error, y verificarlo es
    justo el punto de la normalización).
    """
    words = text.split()
    output: list[str] = []
    for word in words:
        clean = word.lower().strip(".,¿?¡!…")

        if clean in _CONFUSIONS and rng.random() < 0.5:
            output.append(_CONFUSIONS[clean])
            continue
        # Se come una palabra corta de vez en cuando.
        if len(clean) <= 3 and rng.random() < sample_rate:
            continue
        output.append(word)

    resultado = " ".join(output)
    # La mitad de las veces, los montos con letra salen con dígito.
    if rng.random() < 0.5:
        for letter, digit in (
            ("dieciocho mil", "18,000"),
            ("veintiocho mil", "28,000"),
            ("veinticinco mil", "25,000"),
            ("diecisiete", "17"),
        ):
            resultado = resultado.replace(letter, digit)
    return resultado


def _split_up(text: str, rng: random.Random) -> list[str]:
    """Parte una frase como la parte un STT de streaming."""
    words = text.split()
    if len(words) <= 3:
        return [text] if text else []
    fragments: list[str] = []
    i = 0
    while i < len(words):
        size = rng.choice((2, 3, 4, 6))
        fragments.append(" ".join(words[i : i + size]))
        i += size
    return fragments


def _agent_response(indice: int, monto: float | None) -> str:
    """Una línea plausible del agente. Con monto cuando toca proponerlo."""
    if monto is not None:
        return (
            f"Entiendo su punto. Le puedo ofrecer {monto:,.0f} pesos por el viaje "
            "de Monterrey a Laredo."
        )
    templates = (
        "Buenas tardes, le llamo de Freight Technologies. ¿Hablo con Carlos?",
        "Claro que sí. Es una caja seca de 53 pies con carga general.",
        "Tiene sentido lo que dice. Déjeme ver qué puedo hacer por usted.",
        "Le agradezco su tiempo. Quedamos en contacto entonces.",
    )
    return templates[indice % len(templates)]


# --------------------------------------------------------------------------
# Llamada completa
# --------------------------------------------------------------------------


def synthetic_call(
    scenario: Scenario,
    *,
    repetition: int,
    poc: int = 1,
    seed: int = DEFAULT_SEED,
    target_price: float = 18_000.0,
    offers: Sequence[float] = (16_000.0, 17_000.0, 17_500.0),
    resultado: str = "sin_acuerdo",
    motivo: str = "rechazo_definitivo",
    monto_final: float | None = None,
    compensations: Sequence[str] = (),
    deviations: int = 1,
    interrupted_turns: Sequence[int] = (),
    with_stages: bool = True,
) -> SyntheticCall:
    """Construye una llamada completa: JSON de corrida + archivo de referencia.

    `desviaciones` es cuántas líneas del guion se cambian en la referencia, para
    emular que el actor improvisó: así el WER no sale artificialmente en cero y
    se comprueba que el flujo de corrección manual se refleja en el número.

    """
    rng = random.Random(f"{seed}-{scenario.id}-{repetition}")

    spoken_items = [t for t in scenario.turns if not t.is_silence]

    # La referencia: el guion, con algunas líneas cambiadas a mano.
    reference_lines: list[str] = []
    deviated_indices = set(
        rng.sample(range(len(spoken_items)), min(deviations, len(spoken_items)))
        if spoken_items
        else []
    )
    for i, turn in enumerate(spoken_items):
        if i in deviated_indices:
            reference_lines.append(f"{turn.linea} Y otra cosa, ahí le encargo.")
        else:
            reference_lines.append(turn.linea)

    # La transcripción: lo que el STT "entendió" de esa referencia, más lo que
    # dijo el agente, intercalado y con marcas de tiempo crecientes.
    transcript: list[dict[str, Any]] = []
    t = 2.0
    amounts_to_say = list(offers)

    for i, linea in enumerate(reference_lines):
        monto = amounts_to_say[i] if i < len(amounts_to_say) else None
        agent = _agent_response(i, monto)
        transcript.append({"t": round(t, 3), "role": ref.AGENT_ROLE, "text": agent})
        t += rng.uniform(3.0, 6.0)

        for fragment in _split_up(corrupt(linea, rng), rng):
            transcript.append(
                {"t": round(t, 3), "role": ref.CARRIER_ROLE, "text": fragment}
            )
            t += rng.uniform(0.8, 2.2)
        t += rng.uniform(0.5, 1.5)

    n_turns = len(reference_lines)
    samples = synthetic_samples(
        n_turns, rng, interrupted_turns=interrupted_turns, with_stages=with_stages
    )

    call_sid = f"CAsintetico{scenario.id}{repetition}{poc}"

    result_record: dict[str, Any] = {
        "resultado": resultado,
        "timestamp": "2026-08-05T12:00:00+00:00",
        "monto_final": monto_final,
        "condiciones": [],
        "offered_compensations": list(compensations),
        "offer_history": list(offers),
        "counteroffers": max(len(offers) - 1, 0),
        "motivo": None if resultado == "acuerdo" else motivo,
        "reason_category": None if resultado == "acuerdo" else motivo,
        "validated_final_amount": (
            None
            if monto_final is None
            else any(abs(monto_final - o) <= 0.01 for o in offers)
        ),
        "call_id": call_sid,
        "metadata": {},
    }

    data: dict[str, Any] = {
        "call_sid": call_sid,
        "summary": {
            "call_id": call_sid,
            "company": "Freight Technologies",
            "origin": "Monterrey, Nuevo León",
            "destination": "Laredo, Texas",
            "cargo_type": "caja seca de 53 pies con carga general",
            "target_price": target_price,
            "offer_history": list(offers),
            "offered_compensations": list(compensations),
            "resultado": result_record,
        },
        "transcript": transcript,
        # La misma forma que escribe la captura de latencia de cada PoC.
        "latencies": [m.to_dict() for m in samples],
    }

    # El archivo de referencia, ya "corregido": las líneas `yo:` traen la
    # referencia real (con las desviaciones) y las notas manuales están puestas.
    content = _corrected_reference(
        scenario,
        reference_lines,
        transcript,
        poc=poc,
        repetition=repetition,
        call_sid=call_sid,
        naturalness=rng.choice((3, 4, 4, 5)),
        turn_taking=rng.choice((3, 3, 4, 5)),
    )

    return SyntheticCall(
        data=data,
        reference_text=" ".join(reference_lines),
        reference_content=content,
        scenario=scenario.id,
        repetition=repetition,
        poc=poc,
    )


def _corrected_reference(
    scenario: Scenario,
    text_lines: Sequence[str],
    transcript: Sequence[dict[str, Any]],
    *,
    poc: int,
    repetition: int,
    call_sid: str,
    naturalness: int,
    turn_taking: int,
) -> str:
    """El archivo de referencia como quedaría después de corregirlo a mano."""
    base = ref.generate(
        scenario=scenario,
        transcript=transcript,
        poc=poc,
        repetition=repetition,
        call_sid=call_sid,
        script_version="v1",
    )

    spoken_items = [t for t in scenario.turns if not t.is_silence]
    replacements = {
        f"{ref.REFERENCE_PREFIX} {turn.linea}": f"{ref.REFERENCE_PREFIX} {linea}"
        for turn, linea in zip(spoken_items, text_lines)
        if turn.linea != linea
    }
    for old, new in replacements.items():
        base = base.replace(old, new, 1)

    base = base.replace("naturalness: \n", f"naturalness: {naturalness}\n")
    base = base.replace("turn_taking: \n", f"turn_taking: {turn_taking}\n")
    base = base.replace(
        "unwanted_interruptions: \n", "unwanted_interruptions: 1\n"
    )
    base = base.replace("false_end_of_turn: \n", "false_end_of_turn: 0\n")
    return base


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def synthetic_corpus(
    *,
    poc: int = 1,
    repetitions: int = 3,
    seed: int = DEFAULT_SEED,
    script_path: Path | str | None = None,
) -> list[SyntheticCall]:
    """Las 15 llamadas del guion, sintéticas: 5 escenarios × 3 repeticiones.

    Se siembran a propósito los casos que el reporte tiene que saber manejar:
    llamadas de distinta duración, desviaciones respecto del guion para que el
    WER no salga en cero, y turnos interrumpidos que van a su propia cubeta.
    """
    script = load_script(script_path)
    calls: list[SyntheticCall] = []

    for scenario in script.scenarios:
        for repetition in range(1, repetitions + 1):
            extra: dict[str, Any] = {}

            if scenario.id == 1:
                # Cooperativo: cierra.
                extra = {
                    "resultado": "acuerdo",
                    "monto_final": 17_500.0,
                    "offers": (16_000.0, 17_000.0, 17_500.0),
                }
            elif scenario.id == 2:
                # Regateador: cierra a veces, se traba otras.
                if repetition == 2:
                    extra = {
                        "resultado": "acuerdo",
                        "monto_final": 18_000.0,
                        "offers": (16_000.0, 17_500.0, 18_000.0),
                    }
                else:
                    extra = {"motivo": "sin_acuerdo_tras_contraofertas"}
            elif scenario.id == 3:
                # Ancla alta: nunca cierra.
                extra = {"motivo": "rechazo_definitivo"}
            elif scenario.id == 4:
                extra = {
                    "resultado": "acuerdo",
                    "monto_final": 17_000.0,
                    "offers": (16_000.0, 17_000.0),
                    "compensations": ("tiempo_espera_carga",),
                }
            elif scenario.id == 5:
                # Estrés de reconocimiento: cierra en acuerdo, como el guion, con
                # su trayectoria de precios (16,000 → 17,500 → 18,500). Igual que
                # el escenario 4, basta con `resultado`: el motivo queda en None
                # solo, porque un cierre con acuerdo no tiene motivo de cierre.
                # El barge-in se conserva para que la cubeta de interrumpidos no
                # quede vacía en el corpus.
                extra = {
                    "resultado": "acuerdo",
                    "monto_final": 18_500.0,
                    "offers": (16_000.0, 17_500.0, 18_500.0),
                    "interrupted_turns": (1,),
                }

            calls.append(
                synthetic_call(
                    scenario,
                    repetition=repetition,
                    poc=poc,
                    seed=seed,
                    **extra,
                )
            )
    return calls


def write_corpus(
    folder: Path | str,
    calls: Sequence[SyntheticCall],
    *,
    with_references: bool = True,
) -> list[Path]:
    """Deja el corpus en disco con los mismos nombres que usa la telefonía real."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)

    written_list: list[Path] = []
    for call in calls:
        ruta_json = destination / f"call_{call.call_sid}.json"
        ruta_json.write_text(
            json.dumps(call.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written_list.append(ruta_json)

        if with_references:
            ruta_ref = ref.path_to_reference(destination, call.call_sid)
            ruta_ref.write_text(call.reference_content, encoding="utf-8")
            written_list.append(ruta_ref)

    return written_list
