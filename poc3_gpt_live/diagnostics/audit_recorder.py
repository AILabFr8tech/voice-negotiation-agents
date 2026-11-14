"""Grabación de auditoría de PoC3: las dos direcciones del puente, en 2 canales.

VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.
========================================
Existe para poder **oír** la corrida de calibración del detector de voz y
contrastar de forma gruesa lo que la sonda de energía dice con números. Ninguna
cifra que salga de aquí es reportable — mismo estatus que `energy_probe` y
que los 15 censos de agosto.

## Por qué no se reusa `shared/carrier_simulator/auditoria.py`

Esa grabadora coloca cada trama en su instante de llegada y **asigna**
(`linea[inicio:fin] = muestras`). En PoC1 y PoC2 es correcto, porque las tramas
llegan a ritmo real y no se pisan. En PoC3 se pisan: el alpha entrega deltas de
100 ms con un hueco entre llegadas de **94 ms de mediana** —idéntico en las 15
llamadas de agosto, así que es cadencia del servidor y no jitter—, de modo que
el 55.1% de los deltas pisa la cola del anterior. Con la regla de asignación se
perdería el **5.5% del audio del agente** (122 s de 2,233 s medidos), en
silencio.

Aquí el canal lleva un **cabezal monótono**: un bloque nunca empieza antes de
donde terminó el anterior. No pierde audio, y el estiramiento que produce se
mide y se publica en el JSON hermano (`cabezal`) en vez de estimarse en prosa.

**Ese estiramiento no es pequeño, y no es un artefacto de la grabadora.**
Replayando los tiempos de llegada de los 15 censos de agosto: el cabezal corre
el 73-99% de los bloques, el desplazamiento llega a **3,112 ms** y una ráfaga
llegó a estirar **92%** sobre su propio audio (otra, 345%). La causa es física y
ya estaba anotada en el puente: el alpha entrega más rápido que tiempo real
—15 deltas con 1.86 s de audio en 0.4 s de llegadas— y Twilio acumula cola. O
sea que el desfase del cabezal **mide la cola de reproducción**, y el canal del
agente queda en el eje de lo que el transportista *oyó*, no en el eje de lo que
el puente *recibió*, que es donde vive el acuse.

**Lo que salva el archivo es dónde cae ese desfase.** Entre dos intervenciones
del agente el silencio deja que la llegada alcance al cabezal, así que en el
arranque de cada intervención el desfase vuelve a cero: 71 de 73 arranques en
los 15 censos, con un máximo de 265 ms, contra los 3,112 ms de adentro. El acuse
cuelga justo de ese primer bloque. La tabla `arranques_de_intervencion` publica
el desfase de cada uno **por corrida**, para comprobarlo en vez de confiar en
aquello: si ahí sale distinto de cero, esa marca no se audita con este archivo.

Tampoco se toca la grabadora compartida, que es la ruta de PoC1 y PoC2 y ya
entregó campaña.

## Los dos canales, y en qué dominio

- Canal 0, transportista: el μ-law que llega de Twilio, decodificado.
- Canal 1, agente: el μ-law que **sale** hacia Twilio, decodificado — o sea lo
  que el transportista de verdad oye, y el mismo buffer que la sonda registra
  como `pcm8`.

Los dos a 8 kHz: no hay que remuestrear nada para juntarlos y no se inventa
ancho de banda que la línea nunca tuvo. Quien llama entrega ya el PCM16, igual
que con la sonda, para que este módulo no dependa de `app/`.

## Qué NO valida este archivo

En PoC1 y PoC2 el canal 0 del WAV es exacto porque el reproductor conoce su
deadline. En PoC3 **los dos canales se fechan al llegar**, y el canal del agente
comparte el `t_local` del que cuelga el acuse, así que el WAV hereda el reloj y
**no valida la marca de tiempo de forma independiente**. Valida el contenido en
la marca —si lo que sonó ahí era relleno y no voz, se oye—, que es el modo de
falla que se teme, y nada más.

## LFPDPPP

Mismo trato que `POC3_SONDA_VUELCA_AUDIO`: se enciende a mano para una corrida
donde quien habla es quien la corre, el artefacto **no entra al repo**
(gitignoreado) y se borra cuando la calibración quede congelada.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import write_wav_stereo

logger = logging.getLogger(__name__)

# Variable de entorno que la enciende. Ausente o vacía = apagada, y entonces no
# se ejecuta ni una línea de este módulo dentro del ciclo de audio.
TURN_ON_VARIABLE: Final[str] = "POC3_RECORD_AUDIT"

TELEPHONY_SAMPLE_RATE: Final[int] = 8000
BYTES_PER_SAMPLE: Final[int] = 2

# El canal 0 es siempre el transportista y el 1 siempre el agente, igual que en
# el arnés: quien abra el WAV en seis meses tiene que saber cuál es cuál sin
# abrir el JSON.
CARRIER_CHANNEL: Final[int] = 0
AGENT_CHANNEL: Final[int] = 1

# El prefijo lo comparte con los artefactos del arnés a propósito: significan lo
# mismo y así quedan juntos al ordenar la carpeta. De paso, `.gitignore` ya
# excluye `*.wav` y `auditoria_*.json`.
PREFIX: Final[str] = "auditoria"

RUTA_DEFAULT: Final[Path] = Path(__file__).resolve().parent / "auditoria"

# Silencio de llegadas que separa dos intervenciones del agente. No es una
# perilla de medición —no decide ningún turno ni entra a ninguna latencia—: solo
# agrupa bloques para poder publicar el desfase del cabezal en los arranques,
# que es donde cae el acuse. Va holgado a propósito: partir de más una
# intervención larga solo agrega filas a una tabla de diagnóstico.
LINE_SEPARATION_S: Final[float] = 0.5

# --------------------------------------------------------------------------
# Arranques de VOZ, que no son los de intervención
# --------------------------------------------------------------------------
#
# `arranques_de_intervencion` agrupa por silencio de LLEGADAS y ahí el cabezal
# viene reseteado, así que su desfase es ~0. Pero el acuse de un turno cae en el
# arranque de **voz**, y un turno puede empezar a hablar en medio de una
# intervención que ya venía sonando — sobre `CA221584…` hubo 16 arranques de voz
# y **un solo** arranque de intervención. En esos 15 restantes el cabezal no
# está en cero, y ese desfase es corrección sistemática del acuse, no ruido.
#
# Se publica por arranque para poder restarlo: es el término de encolado del
# presupuesto de error del discriminante (δ), junto con
# `bridge.RETRASO_REMUESTREO_MS`. Sin esta tabla el encolado hay que absorberlo
# en la banda muerta, y absorber un sesgo conocido como si fuera ruido triplica
# δ (150 ms contra 50).
#
# El umbral y la ventana son los mismos con los que el puente decide que hay
# voz, a propósito: si aquí se usara otro criterio, la tabla describiría
# arranques que el acuse no vio.
SPEECH_THRESHOLD_RMS: Final[float] = 200.0
SPEECH_WINDOW_S: Final[float] = 0.020
# Silencio que tiene que haber ANTES para que un cruce cuente como arranque y no
# como la continuación de la misma frase. Es el hangover de racha del arnés
# (`shared/carrier_simulator/observado.py`), para que los arranques que esta
# tabla lista sean los mismos que el arnés agrupa.
SILENCE_BEFORE_ONSET_S: Final[float] = 0.3


def switched_on() -> bool:
    return bool(os.environ.get(TURN_ON_VARIABLE, "").strip())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 2)


@dataclass
class _Canal:
    """Un sentido del audio, con cabezal monótono y su contabilidad.

    En vivo solo hace una resta, una comparación y un `append`: nada de numpy y
    nada de disco dentro del ciclo de audio.
    """

    sample_rate: int
    # (inicio en muestras, PCM16 crudo). Por el cabezal, los bloques nunca se
    # solapan, así que al volcar la asignación no pierde nada.
    blocks: list[tuple[int, bytes]] = field(default_factory=list, repr=False)
    playhead: int = 0
    # Por bloque, en muestras. `desfases` es cuánto se corrió su arranque
    # respecto del instante en que llegó; 0 es "cayó donde le tocaba".
    offsets: list[int] = field(default_factory=list, repr=False)
    durations: list[int] = field(default_factory=list, repr=False)
    arrivals: list[int] = field(default_factory=list, repr=False)

    def annotate(self, moment: float, raw_value: bytes) -> None:
        samples = len(raw_value) // BYTES_PER_SAMPLE
        if samples <= 0:
            return
        arrival = max(0, int(round(moment * self.sample_rate)))
        start = arrival if arrival >= self.playhead else self.playhead
        self.blocks.append((start, raw_value))
        self.offsets.append(start - arrival)
        self.durations.append(samples)
        self.arrivals.append(arrival)
        self.playhead = start + samples

    # -- después de colgar --------------------------------------------------

    def timeline(self, total_samples: int) -> np.ndarray:
        linea = np.zeros(total_samples, dtype=np.int16)
        for start, raw_value in self.blocks:
            pair = len(raw_value) - (len(raw_value) % BYTES_PER_SAMPLE)
            samples = np.frombuffer(raw_value[:pair], dtype="<i2")
            end = min(start + samples.size, total_samples)
            if end > start:
                linea[start:end] = samples[: end - start]
        return linea

    def _bursts(self) -> list[dict[str, Any]]:
        """Tramos corridos consecutivos. Una ráfaga se cierra cuando un bloque
        vuelve a caer en su instante de llegada (desfase 0)."""
        output: list[dict[str, Any]] = []
        actual: list[int] = []
        for i, offset in enumerate(self.offsets):
            if offset > 0:
                actual.append(i)
                continue
            if actual:
                output.append(self._summarize_burst(actual))
                actual = []
        if actual:
            output.append(self._summarize_burst(actual))
        return output

    def _summarize_burst(self, indices: list[int]) -> dict[str, Any]:
        audio = sum(self.durations[i] for i in indices)
        # El desfase del último bloque es lo que el cabezal llegó a adelantarse:
        # la elongación de esa ráfaga.
        elongation = self.offsets[indices[-1]]
        audio_ms = audio / self.sample_rate * 1000.0
        return {
            "blocks": len(indices),
            "audio_ms": round(audio_ms, 1),
            "elongation_ms": round(elongation / self.sample_rate * 1000.0, 1),
            "elongation_pct": round(elongation / audio * 100.0, 2) if audio else 0.0,
        }

    def onsets(self) -> list[dict[str, Any]]:
        """Los bloques que abren una intervención, con su desfase.

        **Es la fila que decide si este archivo sirve para auditar una marca.**
        El acuse cuelga del primer bloque de una intervención, y ahí el cabezal
        viene reseteado por el silencio anterior: en los 15 censos de agosto el
        desfase en el arranque salió 0 en 71 de 73 (máximo 265 ms), mientras que
        dentro de una intervención llegó a 3,112 ms. Publicar la tabla permite
        comprobarlo por corrida en vez de confiar en aquello.
        """
        output: list[dict[str, Any]] = []
        anterior: int | None = None
        for i, arrival in enumerate(self.arrivals):
            separation = (
                (arrival - anterior) / self.sample_rate if anterior is not None else None
            )
            if separation is None or separation > LINE_SEPARATION_S:
                output.append(
                    {
                        "block": i,
                        "t_arrival_s": round(arrival / self.sample_rate, 3),
                        "t_placed_s": round(
                            (arrival + self.offsets[i]) / self.sample_rate, 3
                        ),
                        "offset_ms": round(self.offsets[i] / self.sample_rate * 1000.0, 1),
                    }
                )
            anterior = arrival
        return output

    def speech_onsets(self) -> list[dict[str, Any]]:
        """Los cruces de umbral que abren habla, con el desfase de su bloque.

        **Es la corrección sistemática del acuse, no diagnóstico de adorno.**
        El acuse de un turno se fecha cuando el delta llega al puente; el
        transportista lo oye `desfase_ms` después, porque el cabezal tuvo que
        correr ese bloque para no pisar el anterior. Restarlo es lo que permite
        bajar la banda muerta del discriminante de 150 a 50 ms.

        Se calcula al volcar y no en vivo: recorre la línea de tiempo ya armada,
        que es trabajo de después de colgar. Dentro del ciclo de audio este
        módulo sigue haciendo una resta, una comparación y un `append`.

        Un arranque es la primera ventana que cruza el umbral después de al
        menos `SILENCE_BEFORE_ONSET_S` sin cruzarlo. El bloque que lo
        contiene es el que aporta el desfase; si el arranque cae fuera de todo
        bloque —no debería, pero el relleno de silencio existe— se publica con
        `bloque: null` y sin desfase, en vez de imputarle el de un vecino.
        """
        if not self.blocks:
            return []
        total = self.playhead
        linea = self.timeline(total).astype(np.float64)
        window = max(1, int(self.sample_rate * SPEECH_WINDOW_S))
        n = linea.size // window
        if n <= 0:
            return []
        rms = np.sqrt((linea[: n * window].reshape(n, window) ** 2).mean(axis=1))
        with_voice = rms >= SPEECH_THRESHOLD_RMS
        back = max(1, int(round(SILENCE_BEFORE_ONSET_S / SPEECH_WINDOW_S)))

        # (inicio, fin) de cada bloque, para localizar en cuál cae el arranque.
        segments = [
            (start, start + (len(raw_value) // BYTES_PER_SAMPLE))
            for start, raw_value in self.blocks
        ]
        output: list[dict[str, Any]] = []
        for i in range(n):
            if not with_voice[i]:
                continue
            since = max(0, i - back)
            if i and with_voice[since:i].any():
                continue
            sample = i * window
            indice = next(
                (j for j, (a, b) in enumerate(segments) if a <= sample < b), None
            )
            output.append(
                {
                    "t_seg": round(sample / self.sample_rate, 3),
                    "block": indice,
                    "offset_ms": (
                        round(self.offsets[indice] / self.sample_rate * 1000.0, 1)
                        if indice is not None
                        else None
                    ),
                    "rms": round(float(rms[i]), 1),
                }
            )
        return output

    def accounting(self) -> dict[str, Any]:
        """Lo que el cabezal le hizo a este canal. Medido, no estimado."""
        in_ms = [d / self.sample_rate * 1000.0 for d in self.offsets]
        runs_done = [d for d in self.offsets if d > 0]
        bursts = self._bursts()
        # Por milisegundos y no por porcentaje: una ráfaga de un solo bloque con
        # 40 ms de audio corrido 143 ms da 345%, y eso no describe nada. El
        # porcentaje se publica al lado, dentro de la misma ráfaga.
        worst = max(bursts, key=lambda r: r["elongation_ms"], default=None)
        onsets = self.onsets()
        offset_at_onsets = [a["offset_ms"] for a in onsets]
        of_voice = self.speech_onsets()
        with_offset = [a["offset_ms"] for a in of_voice if a["offset_ms"] is not None]
        return {
            "speech_onsets": {
                "n": len(of_voice),
                "no_block": sum(1 for a in of_voice if a["block"] is None),
                "at_zero": sum(1 for d in with_offset if d == 0),
                "offset_ms": {
                    "p50": _percentile(with_offset, 50),
                    "p95": _percentile(with_offset, 95),
                    "maximum": round(max(with_offset), 2) if with_offset else 0.0,
                },
                "what_it_is_for": (
                    "el acuse se fecha cuando el delta LLEGA al puente; el "
                    "transportista lo oye `desfase_ms` después, porque el "
                    "cabezal corrió ese bloque. Restarlo —junto con "
                    "`bridge.RESAMPLING_LAG_MS`— es la corrección "
                    "sistemática que deja la banda muerta del discriminante en "
                    "50 ms en vez de 150. NO es lo mismo que "
                    "`arranques_de_intervencion`: ésos agrupan por silencio de "
                    "llegadas y salen en cero por construcción; éstos son los "
                    "cruces de umbral, y un turno puede arrancar a hablar en "
                    "medio de una intervención que ya venía sonando"
                ),
                "criterion": (
                    f"RMS >= {SPEECH_THRESHOLD_RMS:.0f} sobre ventanas de "
                    f"{SPEECH_WINDOW_S * 1000:.0f} ms, con al menos "
                    f"{SILENCE_BEFORE_ONSET_S * 1000:.0f} ms sin cruce "
                    "antes. Mismos valores con los que el puente decide que hay "
                    "voz y mismo hangover con el que el arnés agrupa rachas"
                ),
                "tabla": of_voice,
            },
            "line_onsets": {
                "n": len(onsets),
                "at_zero": sum(1 for d in offset_at_onsets if d == 0),
                "max_offset_ms": (
                    max(offset_at_onsets) if offset_at_onsets else 0.0
                ),
                "tabla": onsets,
            },
            "what_it_measures": (
                "cuánto tuvo que correr el cabezal cada bloque respecto de su "
                "instante de llegada, porque el anterior todavía no terminaba"
            ),
            "blocks": len(self.blocks),
            "blocks_run": len(runs_done),
            "blocks_run_pct": (
                round(len(runs_done) / len(self.blocks) * 100.0, 1)
                if self.blocks
                else 0.0
            ),
            "offset_ms": {
                "p50": _percentile(in_ms, 50),
                "p95": _percentile(in_ms, 95),
                "maximum": round(max(in_ms), 2) if in_ms else 0.0,
            },
            # El "total" que pide el contrato: la suma de lo que se adelantó el
            # cabezal en cada ráfaga. No es la duración del archivo ni un error
            # acumulado de punta a punta — entre ráfagas el cabezal se resetea
            # solo, porque el silencio deja que la llegada lo alcance.
            "accumulated_offset_ms": round(
                sum(r["elongation_ms"] for r in bursts), 1
            ),
            "bursts": len(bursts),
            "worst_burst": worst,
            "worst_elongation_pct": worst["elongation_pct"] if worst else 0.0,
        }


@dataclass
class CallRecorder:
    """Las dos direcciones en memoria; el archivo se arma al colgar.

    Volcar al final y no escribir en vivo es el mismo criterio que la grabadora
    del arnés: meter el sistema de archivos dentro del ciclo que se está
    midiendo sería medir el disco. Una llamada de 40 s ocupa ~1.3 MB.
    """

    destination: Path
    sample_rate: int = TELEPHONY_SAMPLE_RATE
    start: float = field(default_factory=time.time)
    carrier: _Canal = field(init=False, repr=False)
    agent: _Canal = field(init=False, repr=False)
    _ultimo_end_ms: float | None = field(default=None, init=False, repr=False)
    # Silencio que el alpha omitió entre dos rangos y que el puente no repone.
    # Se cuenta, no se dibuja: ver `anotar_agente`.
    skipped_gaps: int = field(default=0, init=False)
    skipped_silence_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.carrier = _Canal(sample_rate=self.sample_rate)
        self.agent = _Canal(sample_rate=self.sample_rate)

    # -- camino en vivo ------------------------------------------------------

    def annotate_carrier(self, raw_value: bytes, *, t: float | None = None) -> None:
        """Una trama entrante de Twilio, ya decodificada a PCM16 de 8 kHz."""
        # `is None` y no `or`: un `t` de 0.0 es un instante válido —el primer
        # bloque de la llamada— y `or` lo mandaría a `time.time()`.
        self.carrier.annotate(
            (time.time() if t is None else t) - self.start, raw_value
        )

    def annotate_agent(
        self,
        raw_value: bytes,
        *,
        t: float | None = None,
        start_ms: Any = None,
        end_ms: Any = None,
    ) -> None:
        """Un `output_audio.delta`, ya convertido al μ-law que sale y decodificado.

        `t` es el instante en que el delta llegó al puente — el mismo que usa el
        colector para marcar la voz saliente. Se recibe en vez de tomarse aquí
        para que el WAV no quede ~24 ms tarde: eso es lo que cuesta el remuestreo
        entre esa marca y este punto.

        `start_ms` y `end_ms` son el rango del alpha y **no** colocan nada: solo
        se cuenta el silencio que el alpha omitió entre dos rangos. No se repone,
        porque el puente tampoco lo repone — manda los bytes del delta tal cual,
        así que el transportista oye los dos tramos pegados. Reponerlo aquí
        pintaría en el archivo una pausa que nadie oyó. Que el eje del alpha
        tampoco sirva para colocar es cosa aparte: avanza más despacio que el
        reloj de pared (97.6 s de eje en 128.8 s de llamada, medido en la 1.2).
        """
        if isinstance(start_ms, (int, float)) and isinstance(
            self._ultimo_end_ms, (int, float)
        ):
            hole = max(0.0, float(start_ms) - self._ultimo_end_ms)
            if hole > 0.0:
                self.skipped_gaps += 1
                self.skipped_silence_ms += hole
        if isinstance(end_ms, (int, float)):
            self._ultimo_end_ms = float(end_ms)
        self.agent.annotate((time.time() if t is None else t) - self.start, raw_value)

    # -- después de colgar ---------------------------------------------------

    def shut(self, *, call_id: str = "") -> Path | None:
        """Escribe el WAV y su JSON hermano. `None` si no hubo audio."""
        if not self.carrier.blocks and not self.agent.blocks:
            return None

        samples = max(self.carrier.playhead, self.agent.playhead)
        left_side = self.carrier.timeline(samples)
        derecho = self.agent.timeline(samples)

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        write_wav_stereo(self.destination, left_side, derecho, self.sample_rate)

        descriptive = {
            "what_it_is": (
                "grabación de auditoría de una corrida de CALIBRACIÓN de PoC3, "
                "para oír lo que la sonda de energía dice con números"
            ),
            "ADVERTENCIA": (
                "VALIDACIÓN DE INSTRUMENTO, NO RESULTADO. Ninguna cifra de este "
                "archivo es reportable ni entra a ningún percentil"
            ),
            "NOT_SUITABLE_FOR_JUDGING_BY_EAR_RESPONSE_TIMES": (
                "los dos canales de este archivo NO están alineados entre sí: el "
                "canal del agente queda ~1.8 s ANTES de donde ocurrió, así que "
                "todo hueco entre el transportista y la respuesta suena ~1.8 s "
                "más corto de lo que fue. Es consecuencia del cabezal monótono, "
                "no un defecto que se pueda corregir en el archivo. Para oír el "
                "ritmo de la conversación se usa el WAV del ARNÉS "
                "(diagnostics/harness/auditoria_*.wav), cuyo canal 0 sale del "
                "deadline del reproductor. Medido y declarado en "
                "docs/hallazgos_20260922_poc3.md"
            ),
            "is_campaign_evidence": False,
            "does_not_fit_in": [
                "latencies",
                "percentiles",
                "WER / CER",
                "turn-taking",
                "cualquier juicio de oído sobre el hueco de respuesta",
            ],
            "call_id": call_id,
            "file": self.destination.name,
            "sample_rate_hz": self.sample_rate,
            "duration_s": round(samples / self.sample_rate, 3),
            "channel_list": {
                str(CARRIER_CHANNEL): (
                    "el μ-law que llegó de Twilio, decodificado: lo que el "
                    "puente le mandó al alpha"
                ),
                str(AGENT_CHANNEL): (
                    "el μ-law que salió hacia Twilio, decodificado: lo que el "
                    "transportista oyó, y el mismo buffer que mide la sonda"
                ),
            },
            "colocacion": (
                "cabezal monótono: un bloque nunca empieza antes de donde "
                "terminó el anterior. No pierde audio. Dentro de una "
                "intervención el cabezal se adelanta a la llegada —eso mide la "
                "cola de reproducción de Twilio, porque el alpha entrega más "
                "rápido que tiempo real— y en el arranque de cada intervención "
                "vuelve a cero. Cuánto, en `cabezal`"
            ),
            "what_it_is_and_is_not_for": (
                "el canal del agente queda en el eje de lo que el transportista "
                "OYÓ; el acuse vive en el eje de lo que el puente RECIBIÓ. Los "
                "dos coinciden en el arranque de cada intervención, que es donde "
                "cae el acuse: revisar `arranques_de_intervencion` antes de "
                "auditar una marca, y no comparar posiciones de media "
                "intervención"
            ),
            "playhead": {
                str(CARRIER_CHANNEL): self.carrier.accounting(),
                str(AGENT_CHANNEL): self.agent.accounting(),
            },
            "silence_skipped_by_the_alpha": {
                "what_it_is": (
                    "huecos entre dos rangos de `output_audio`: silencio que el "
                    "alpha quitó del stream. NO se repone en el archivo porque "
                    "el puente tampoco lo repone — manda los bytes del delta tal "
                    "cual y el transportista oye los dos tramos pegados"
                ),
                "gaps": self.skipped_gaps,
                "total_ms": round(self.skipped_silence_ms, 1),
            },
            "alignment": (
                "En PoC1 y PoC2 el canal 0 del WAV es exacto porque el "
                "reproductor conoce su deadline; en PoC3 los dos canales se "
                "fechan al llegar y el canal del agente comparte el t_local del "
                "que cuelga el acuse, así que el WAV hereda el reloj y no valida "
                "la marca de tiempo de forma independiente. Valida el contenido "
                "en la marca, que es el modo de falla que se teme, y nada más. "
                "El acuse de PoC3 no tendrá validación temporal independiente en "
                "ninguna corrida. Asimetría declarada."
            ),
            "lfpdppp": (
                "corrida de calibración actuada por quien la corre. El artefacto "
                "no entra al repo y se borra al congelar la calibración"
            ),
        }
        side = self.destination.with_suffix(".json")
        side.write_text(
            json.dumps(descriptive, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "auditoría de PoC3 en %s (%.1f s, %d/%d bloques)",
            self.destination,
            samples / self.sample_rate,
            len(self.carrier.blocks),
            len(self.agent.blocks),
        )
        return self.destination


def begin(call_id: str, *, folder: Path | None = None) -> CallRecorder | None:
    """La grabadora, o `None` si no está encendida por entorno."""
    if not switched_on():
        return None
    destination = (folder or RUTA_DEFAULT) / f"{PREFIX}_{call_id or 'no_sid'}.wav"
    return CallRecorder(destination=destination)
