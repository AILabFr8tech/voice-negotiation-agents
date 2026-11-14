"""El guion que el transportista simulado va a actuar: los clips ya en memoria.

Se carga **todo antes de conectarse**, a propósito. Fallar a media llamada por
un clip que falta costaría la corrida entera y dejaría al agente hablando solo,
y leer un archivo con el agente esperando es la clase de pausa que le rompe el
endpointing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, upsample_from_telephony, ulaw_a_pcm16
from shared.audio.codec import pcm16_a_ulaw
from shared.audio.wav import leer_wav
from shared.carrier_simulator.audio import noise_floor_under_clip

# Dónde deja `shared.audio.cli preparar` los artefactos.
RUTA_CLIPS_DEFAULT: Final[Path] = (
    Path(__file__).resolve().parents[1] / "test_scenarios" / "clips"
)


class HarnessFailure(AssertionError):
    """Un supuesto del que depende el arnés no se cumple."""


@dataclass
class ScriptStep:
    """Lo que al arnés le toca hacer en un turno: reproducir un clip o callarse.

    Un turno de silencio **no se graba**: el guion lo pide para ver qué hace el
    agente cuando el transportista no contesta, y un WAV de silencio sería menos
    fiel que dejar corriendo el mismo ruido de confort que ya se publica.
    """

    turn: str
    text: str
    samples: np.ndarray | None = None
    # Los bytes μ-law **tal como están en disco**, cuando el clip vino de un
    # `.ulaw`. No es una copia redundante de `muestras`: es el estímulo que PoC1
    # y PoC3 meten al WebSocket de Media Streams sin tocarlo.
    #
    # Estos bytes son el clip **con el piso de ruido sumado por debajo**
    # (§2.2.9), así que no coinciden con el archivo de disco: ese va en
    # `ulaw_fuente`. Vacío cuando el clip vino de un WAV o se
    # armó en memoria; quien necesite μ-law tiene que **fallar** ahí, no
    # recodificar por su cuenta.
    ulaw: bytes = b""
    # Los bytes del `.ulaw` tal como están en disco, sin el piso. Es contra
    # estos que se verifica el hash del manifiesto: sumar el piso cambia los
    # bytes por definición, así que la evidencia de que las 3 PoC reprodujeron
    # el mismo estímulo se ancla en el archivo fuente, y el piso es una función
    # determinista de él. Vacío = el paso no vino de un `.ulaw`.
    ulaw_source: bytes = b""
    silence_s: float = 0.0
    clip_id: str = ""
    origin: str = ""
    # Tasa a la que están las `muestras`. Casi siempre es la telefónica, pero el
    # camino de respaldo (`--tasa`) y el brazo limpio del diagnóstico de banda
    # las traen ya subidas, y ahí dividir entre 8,000 daría una duración 6 veces
    # más larga — que es de donde sale `t_inicio`, o sea el inicio del clip que
    # se le publica al worker como cero real del turno.
    sample_rate: int = TELEPHONY_SAMPLE_RATE
    # Si `texto` es la línea del guion, el STT se puede comparar contra ella.
    # En la sonda de `--repetir` no lo es, y marcar diferencias ahí sería ruido.
    of_script: bool = False
    # El segundo cero: dónde deja de haber voz DENTRO del clip, con el
    # umbral que lo midió. Sale del manifiesto, que es donde `shared/audio` lo
    # dejó al preparar — no se re-mide aquí, y menos en vivo: sería DSP durante
    # la llamada, y además arriesgaría que el arnés de cada PoC llegara a un
    # número distinto sobre el mismo clip. `None` = el clip no venía del
    # manifiesto (un `--clip` suelto), y entonces solo se publica el cero de
    # archivo. Nunca se inventa.
    end_of_speech_s: float | None = None
    speech_drop_db: float | None = None
    speech_threshold_dbfs: float | None = None
    # El turno del guion con sus ramas, y todas las variantes ya
    # cargadas en memoria por `variantes`. Los campos de arriba traen la rama
    # **por defecto**, así que un turno que no se ramifica —y todo lo que viene
    # después en el arnés— no se entera de que esto existe.
    #
    # Se cargan las variantes ANTES de conectarse, no cuando toque la rama: leer
    # un archivo a media llamada, con el agente esperando, es la clase de pausa
    # que le rompe el endpointing. Un clip que falte tiene que fallar antes de
    # gastar la llamada.
    of_turn: Any = None
    variants: dict[str, "ScriptStep"] = field(default_factory=dict)
    # Solo en el escenario de interrupciones: qué hace el arnés encima de la
    # respuesta del agente en este turno (`interrupciones.EventoDelTurno`).
    # `None` en toda corrida normal, que entonces no cambia en nada.
    event: Any = None

    @property
    def is_silence(self) -> bool:
        """Un turno en que al arnés le toca callarse: no trae audio de ninguna
        clase.

        Se pregunta por las **dos** representaciones a propósito. Un paso puede
        traer solo los bytes μ-law —así lo arma el arnés de PoC1, que no
        necesita el PCM para nada— y leer únicamente `muestras` lo tomaría por
        un silencio del guion: se saltaría el clip, el agente no oiría nada y el
        turno saldría reportado como "el agente no contestó".
        """
        return self.samples is None and not self.ulaw

    @property
    def branches_out(self) -> bool:
        return bool(self.variants)

    @property
    def duration_s(self) -> float:
        if self.is_silence:
            return self.silence_s
        if self.samples is None:
            # Solo μ-law: un byte por muestra, siempre a banda telefónica.
            return len(self.ulaw) / TELEPHONY_SAMPLE_RATE
        return len(self.samples) / self.sample_rate

    def max_duration_s(self) -> float:
        """La rama más larga. Es la que hay que presupuestar al estimar costo."""
        if not self.variants:
            return self.duration_s
        return max(v.duration_s for v in self.variants.values())


def load_clip(ruta: Path | str, *, publish_sample_rate: int) -> tuple[np.ndarray, str]:
    """Lee un artefacto de `shared/audio/` y lo deja listo para publicar.

    Prefiere el `.ulaw` cuando existe: son **exactamente** los bytes que PoC1 y
    PoC3 le meten al WebSocket de Media Streams, así que cargarlo de ahí
    garantiza que las tres PoC reproducen el mismo audio y no tres redondeos del
    mismo audio.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise HarnessFailure(
            f"no existe el clip {ruta}. Prepáralo con "
            "`python -m shared.audio.cli preparar`"
        )

    if ruta.suffix == ".ulaw":
        samples = ulaw_a_pcm16(ruta.read_bytes())
        origin = f"{ruta.name} (μ-law, {TELEPHONY_SAMPLE_RATE} Hz)"
    else:
        samples, sample_rate = leer_wav(ruta)
        if sample_rate != TELEPHONY_SAMPLE_RATE:
            raise HarnessFailure(
                f"{ruta.name} está a {sample_rate} Hz y el arnés espera {TELEPHONY_SAMPLE_RATE}. "
                "Usa un artefacto de `shared/audio/cli preparar`, no la grabación cruda"
            )
        origin = f"{ruta.name} (PCM, {sample_rate} Hz)"

    if publish_sample_rate != TELEPHONY_SAMPLE_RATE:
        # Camino de respaldo: solo si la pila rechazara publicar a 8 kHz. Deja
        # de ser el mismo camino que el trunk SIP y el reporte lo dice.
        samples = upsample_from_telephony(samples, publish_sample_rate)
        origin += f" -> {publish_sample_rate} Hz (respaldo)"

    return np.ascontiguousarray(samples, dtype=np.int16), origin


def load_clip_to_publish(
    ruta: Path | str, *, publish_sample_rate: int
) -> tuple[np.ndarray, bytes, bytes, str]:
    """El clip listo para el cable: con el piso de ruido sumado por debajo.

    Distinto de `load_clip` a propósito. Aquel deja el estímulo **limpio**,
    que es lo que hay que analizar —las pausas del clip son del clip, no del
    ruido que el arnés le pone encima—. Este deja lo que de verdad sale al
    cable, que es clip + piso continuo.

    Devuelve cuatro cosas porque las tres PoC necesitan tres representaciones y
    tienen que ser **el mismo audio**, no tres redondeos:

    - las muestras PCM, que publica PoC2;
    - los bytes μ-law publicados, que meten PoC1 y PoC3 al WebSocket;
    - los bytes μ-law **de disco**, sin tocar, contra los que se verifica el
      hash del manifiesto. Sumar el piso cambia los bytes por definición, así
      que la evidencia de "las 3 PoC reprodujeron el mismo estímulo" se ancla en
      el archivo fuente más la semilla, que es determinista.

    El PCM sale de decodificar el μ-law publicado y no de la suma directa: así
    PoC2 oye exactamente lo mismo que PoC1, con la misma cuantización, igual que
    antes de este cambio.
    """
    ruta = Path(ruta)
    source = ruta.read_bytes()
    cleaned, origin = load_clip(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
    # La semilla sale del contenido del clip: dos corridas publican los mismos
    # bytes y dos clips distintos no comparten realización del ruido, sin tener
    # que pasear un contador por el arnés.
    seed = int.from_bytes(hashlib.sha256(source).digest()[:4], "big")
    published = pcm16_a_ulaw(noise_floor_under_clip(cleaned, seed=seed))
    samples = ulaw_a_pcm16(published)
    origin += " + piso de ruido"

    if publish_sample_rate != TELEPHONY_SAMPLE_RATE:
        # Mismo camino de respaldo que `cargar_clip`: se sube DESPUÉS de sumar
        # el piso, para que el piso viaje remuestreado igual que la voz.
        samples = upsample_from_telephony(samples, publish_sample_rate)
        origin += f" -> {publish_sample_rate} Hz (respaldo)"

    return (
        np.ascontiguousarray(samples, dtype=np.int16),
        published,
        source,
        origin,
    )


def load_scenario(
    numero: int, folder: Path | str, *, publish_sample_rate: int
) -> list[ScriptStep]:
    """Arma el guion de un escenario con los clips ya preparados en disco.

    Carga **todos** los clips antes de conectarse. Fallar a media llamada por un
    clip que falta costaría la corrida entera y dejaría al agente hablando solo.

    Los turnos que se ramifican (escenarios 2 y 4) cargan **todas** sus
    variantes aquí mismo, no cuando toque la rama: leer un archivo a media
    llamada le rompería el endpointing al agente, y un clip que falte tiene que
    fallar antes de gastar la llamada, no en el turno 5.
    """
    from shared.audio.manifest import (
        MANIFEST_NAME,
        load_manifest,
        expected_inventory,
    )
    from shared.metrics.scenarios import load_script

    script = load_script()
    scenarios = {e.id: e for e in script.scenarios}
    if numero not in scenarios:
        raise HarnessFailure(
            f"el guion no tiene un escenario {numero}: hay {sorted(scenarios)}"
        )

    # El fin de voz de cada clip sale del manifiesto y no se re-mide aquí: es
    # una propiedad del clip, y las 3 PoC reproducen el mismo clip. Si cada
    # arnés lo midiera por su cuenta, bastaría una diferencia de versión para
    # que las tres compararan contra ceros distintos.
    manifest_path = Path(folder) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise HarnessFailure(
            f"no hay manifiesto en {manifest_path}. Corre "
            "`python -m shared.audio.cli preparar`: sin él no se sabe dónde "
            "termina la voz dentro de cada clip, y el cero del turno quedaría "
            "fechado en el fin del archivo"
        )
    of_manifest = load_manifest(manifest_path).by_id()

    # El id del clip sale del mismo inventario que usa `shared.audio`, para que
    # el arnés y la preparación no puedan discrepar sobre cómo se llama un clip.
    # Se indexa por (turno, variante) porque un turno ramificado tiene varias.
    of_inventory = {
        (clip.turn, clip.variant): clip
        for clip in expected_inventory(script)
        if clip.scenario == numero
    }

    folder = Path(folder)
    steps: list[ScriptStep] = []
    absent: list[str] = []

    def _load_variant(turn: Any, variant: str) -> ScriptStep | None:
        """Un clip del turno, o `None` anotando qué faltó."""
        expected = of_inventory.get((turn.id, variant))
        if expected is None:  # pragma: no cover - el inventario sale del guion
            absent.append(f"{turn.id} ({variant})")
            return None

        ruta = folder / f"{expected.id}.ulaw"
        if not ruta.exists():
            absent.append(ruta.name)
            return None
        entry = of_manifest.get(expected.id)
        if entry is None:
            absent.append(f"{expected.id} (en disco pero no en el manifiesto)")
            return None

        # Las tres representaciones salen de una sola llamada, y del mismo
        # audio: el clip con el piso de ruido sumado por debajo. Leer el archivo
        # dos veces —PCM para PoC2, μ-law crudo para PoC1 y PoC3— no sirve: el
        # piso hay que sumarlo una sola vez o las PoC oirían realizaciones
        # distintas del ruido.
        samples, published, source, origin = load_clip_to_publish(
            ruta, publish_sample_rate=publish_sample_rate
        )
        return ScriptStep(
            turn=turn.id,
            text=expected.text,
            samples=samples,
            ulaw=published,
            ulaw_source=source,
            clip_id=expected.id,
            origin=origin,
            of_script=True,
            sample_rate=publish_sample_rate,
            end_of_speech_s=entry.end_of_speech_s,
            speech_drop_db=entry.speech_drop_db,
            speech_threshold_dbfs=entry.speech_threshold_dbfs,
            of_turn=turn,
        )

    for turn in scenarios[numero].turns:
        if turn.is_silence:
            steps.append(
                ScriptStep(
                    turn=turn.id,
                    text="(silencio)",
                    silence_s=float(turn.silence_s or 0.0),
                    of_turn=turn,
                )
            )
            continue

        base = _load_variant(turn, "base")
        if base is None:
            continue

        if turn.branches_out:
            # Todas las ramas, incluida la base: así `variantes` es el árbol
            # completo y elegir es un lookup, no una rama especial de código.
            variants: dict[str, ScriptStep] = {}
            for rama in turn.branches:
                name = turn.variant_of(rama.linea)
                if name in variants:
                    continue
                step = base if name == "base" else _load_variant(turn, name)
                if step is not None:
                    variants[name] = step
            base.variants = variants

        steps.append(base)

    if absent:
        raise HarnessFailure(
            f"faltan {len(absent)} clip(s) del escenario {numero}: "
            f"{', '.join(absent[:5])}{' …' if len(absent) > 5 else ''}.\n"
            "        Grábalos y corre `python -m shared.audio.cli preparar`"
        )
    return steps


def negotiation_state(script: list[ScriptStep]) -> Any:
    """El estado del árbol, o `None` si este guion no se ramifica.

    Devolver `None` en un escenario lineal no es una optimización: es lo que
    hace que el JSON de los escenarios 1, 3 y 5 no traiga un bloque de ramas
    vacío, que se leería como "el árbol no funcionó".
    """
    if not any(step.branches_out for step in script):
        return None

    from shared.test_scenarios.decision import NegotiationState

    estado = NegotiationState.from_the_catalog()
    mute_ones = estado.compensations_without_phrases()
    if mute_ones:
        # No se puede seguir: una compensación sin frases no da error, da una
        # rama que nunca se cumple. El escenario 4 saldría diciendo "nunca
        # ofreció nada" sobre un agente que ofreció, y eso es peor que no correr.
        raise HarnessFailure(
            f"{len(mute_ones)} compensación(es) del catálogo no tienen "
            f"`recognized_by`: {', '.join(mute_ones)}.\n"
            "        Sin frases el arnés no las oye y el escenario 4 ramificaría "
            "siempre por\n        'nunca ofreció'. Agrégalas en "
            "shared/negotiation_engine/compensations.yaml"
        )
    return estado
