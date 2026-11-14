"""Sonda de energía del audio SALIENTE — calibración del detector de voz.

VALIDACIÓN DE INSTRUMENTO, NO RESULTADO.
========================================
Lo que escribe esta sonda sirve para calibrar el umbral con el que PoC3 fecha
la primera voz del agente. **Ninguna cifra que salga de aquí es reportable**, ni
entra a percentiles, ni se cita en ningún entregable — mismo estatus que los 15
censos de agosto.

## Para qué sirve

El acuse es la única métrica de latencia que PoC3 comparte con PoC1 y PoC2, y
cuelga de `bridge.SPEECH_THRESHOLD_RMS = 200.0`. RMS 200 sobre fondo de escala son
−44.3 dBFS; el piso de ruido de confort del arnés son −60 dBFS (RMS ≈ 33). El
modelo entrega audio casi todo el tiempo **con el silencio adentro**: si su
relleno cayera por encima de −44.3 dBFS, el umbral dispararía en toda trama y el
acuse saldría ~100 ms o negativo en **todos** los turnos — lo que se leería como
firma de full-duplex siendo artefacto de umbral. Esta sonda mide esa energía
para calibrar el umbral (resultado en el anexo de
`docs/comparativo_results.md`).

No se puede contestar con los artefactos de la llamada: `bridge.no_audio`
sustituye la carga por `<audio N bytes>` antes de que el censo la escriba.

## Qué mide, y las tres cosas que la hacen servir

1. **Sub-ventanas de 20 ms, no el delta entero.** Una ventana de ~100 ms diluye
   los arranques breves —los apoyos verbales, que son la firma de esta
   arquitectura— y pone el piso de resolución del acuse en 100 ms. PoC1 mide en
   20 ms, y `bridge.has_speech` también.
2. **Los dos dominios.** (a) PCM16 24 kHz, como se mide hoy; (b) el μ-law 8 kHz
   que de verdad viaja a Twilio, decodificado a PCM16 — el mismo dominio que
   PoC1. El umbral heredado no significa lo mismo en banda ancha que en banda
   telefónica, y esta sonda es lo que permite medir cuánto se mueve.
3. **Nada de etiquetar aquí.** La etiqueta SPEECH/RELLENO sale del **texto**
   (`output_transcript.added`), no de la energía, y se arma **después** cruzando
   este volcado con el censo de la misma llamada. Etiquetar en vivo con la misma
   señal que se quiere calibrar sería circular.

## Lo que NO hace

- **No cambia ninguna decisión.** No toca `has_speech`, ni el colector, ni el
  turn-taking. Es un `append` a una lista y un archivo al colgar.
- **No se enciende sola.** Requiere `POC3_SONDA_DE_ENERGIA=1` en el entorno. Con
  la variable ausente, `abrir()` devuelve `None` y el puente no hace nada.

## El tiempo por trama es IMPUTADO, y hay que leerlo como tal

Las 5 tramas de un delta llegan **en el mismo mensaje del WebSocket**: no hay un
instante de llegada por trama que observar. Se registra el instante del delta
(`t_local_del_delta`) y el índice de la trama; el tiempo por trama se reconstruye
como `t_local_del_delta + indice * 0.02`, que es imputación y no medición. Para
lo que esta corrida contesta —la distribución de energía de habla contra
relleno— da igual; para fechar un acuse al milisegundo, no.

## La alineación entre los dos dominios es aproximada

`PuenteDeAudio.a_twilio` es un remuestreador **con estado**: el μ-law que sale al
procesar un delta no corresponde exactamente al mismo tramo de audio que el
PCM24 de ese delta, porque el filtro arrastra cola. Por eso las tramas de cada
dominio se numeran por separado y el JSONL publica los dos conteos. Comparar las
dos distribuciones agregadas es válido; emparejar la trama `k` de un dominio con
la `k` del otro, no.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# Variable de entorno que la enciende. Ausente o vacía = apagada.
TURN_ON_VARIABLE: Final[str] = "POC3_ENERGY_PROBE"

# Segunda variable, para volcar además el audio crudo en base64. Apagada por
# default: `bridge.sin_audio` existe como guardarraíl de LFPDPPP y esta sonda no
# lo deroga por su cuenta. Se enciende a mano para una corrida de calibración
# donde quien habla es quien la corre, y así la corrida queda re-analizable sin
# volver a gastar.
RAW_AUDIO_VARIABLE: Final[str] = "POC3_PROBE_DUMPS_AUDIO"

# La ventana de análisis. 20 ms es la trama de Media Streams y la ventana con la
# que PoC1 mide su propia energía (`Observado.anotar_audio` sobre tramas de
# Twilio), así que es lo que hace comparables las dos distribuciones.
WINDOW_S: Final[float] = 0.020

LIVE_SAMPLE_RATE: Final[int] = 24000
TELEPHONY_SAMPLE_RATE: Final[int] = 8000
BYTES_PER_SAMPLE: Final[int] = 2

SAMPLES_PER_WINDOW_24K: Final[int] = int(LIVE_SAMPLE_RATE * WINDOW_S)  # 480
SAMPLES_PER_WINDOW_8K: Final[int] = int(TELEPHONY_SAMPLE_RATE * WINDOW_S)  # 160

# Fondo de escala de PCM16, para expresar el RMS en dBFS.
FULL_SCALE: Final[float] = 32768.0

RUTA_DEFAULT: Final[Path] = Path(__file__).resolve().parent / "energy"


def switched_on() -> bool:
    return bool(os.environ.get(TURN_ON_VARIABLE, "").strip())


def dumps_audio() -> bool:
    return bool(os.environ.get(RAW_AUDIO_VARIABLE, "").strip())


def dbfs(rms: float) -> float | None:
    """RMS de PCM16 a dBFS. `None` para el silencio digital exacto.

    `None` y no `-inf`: un `-inf` en un JSON no es serializable y un número
    grande y negativo inventado se confundiría con una medición.
    """
    if rms <= 0.0:
        return None
    return round(20.0 * math.log10(rms / FULL_SCALE), 2)


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt((samples.astype(np.float64) ** 2).mean()))


def energy_per_window(pcm: bytes, samples_per_window: int) -> list[float]:
    """El RMS de cada ventana completa. La cola incompleta se descarta.

    Se descarta en vez de medirse a medias porque un resto de 3 muestras da un
    RMS que no es comparable con el de una ventana entera, y esta corrida existe
    justamente para comparar distribuciones.
    """
    if len(pcm) < BYTES_PER_SAMPLE:
        return []
    pair = len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)
    all_items = np.frombuffer(pcm[:pair], dtype="<i2")
    n = len(all_items) // samples_per_window
    return [
        _rms(all_items[i * samples_per_window : (i + 1) * samples_per_window])
        for i in range(n)
    ]


class EnergyProbe:
    """Acumula la energía del audio saliente y la escribe al colgar.

    En vivo solo hace aritmética sobre un array y un `append` a una lista —
    mismo criterio que la grabadora de auditoría del arnés: nada de I/O dentro
    del ciclo de 20 ms.
    """

    def __init__(self, destination: Path, *, with_audio: bool = False) -> None:
        self.destination = destination
        self.with_audio = with_audio
        self._rows: list[dict[str, Any]] = []
        self._deltas = 0

    def annotate(
        self,
        *,
        start_ms: Any,
        end_ms: Any,
        pcm24: bytes,
        ulaw: bytes,
        pcm8: bytes,
    ) -> None:
        """Un `output_audio.delta` ya convertido. `pcm8` es el μ-law decodificado.

        Se recibe `pcm8` ya decodificado y no el μ-law crudo para que la sonda no
        tenga que importar el códec del puente: quien llama ya lo tiene a mano y
        así este módulo no depende de `app/`.
        """
        self._deltas += 1
        ancha = energy_per_window(pcm24, SAMPLES_PER_WINDOW_24K)
        narrow = energy_per_window(pcm8, SAMPLES_PER_WINDOW_8K)
        row: dict[str, Any] = {
            "n": self._deltas,
            "t_local_of_delta": round(time.time(), 6),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "bytes_pcm24": len(pcm24),
            "bytes_ulaw": len(ulaw),
            # Los dos conteos, por lo del remuestreador con estado: si difieren
            # más de una ventana, emparejar por índice sería inventar.
            "windows_24k": len(ancha),
            "windows_8k": len(narrow),
            "rms_24k": [round(x, 2) for x in ancha],
            "rms_8k": [round(x, 2) for x in narrow],
        }
        if self.with_audio:
            import base64

            row["pcm24_b64"] = base64.b64encode(pcm24).decode("ascii")
            row["ulaw_b64"] = base64.b64encode(ulaw).decode("ascii")
        self._rows.append(row)

    def shut(self) -> Path | None:
        """Escribe el JSONL. Devuelve la ruta, o `None` si no hubo nada."""
        if not self._rows:
            return None
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "what_it_is": (
                "energía RMS del audio SALIENTE de PoC3, por ventana de 20 ms y "
                "en los dos dominios, para calibrar el umbral de voz"
            ),
            "ADVERTENCIA": (
                "VALIDACIÓN DE INSTRUMENTO, NO RESULTADO. Ninguna cifra de este "
                "archivo es reportable ni entra a ningún percentil"
            ),
            "window_s": WINDOW_S,
            "sample_rate_24k": LIVE_SAMPLE_RATE,
            "sample_rate_8k": TELEPHONY_SAMPLE_RATE,
            "inherited_threshold_rms": 200.0,
            "inherited_threshold_dbfs": dbfs(200.0),
            "time_per_frame": (
                "IMPUTADO: las tramas de un delta llegan en el mismo mensaje. "
                "Se reconstruye como t_local_del_delta + indice * window_s"
            ),
            "alignment_across_domains": (
                "APROXIMADA: `a_twilio` remuestrea con estado, así que la trama k "
                "de un dominio no es el mismo tramo de audio que la k del otro. "
                "Comparar distribuciones sí; emparejar por índice no"
            ),
            "label": (
                "SPEECH/RELLENO NO se decide aquí. Sale del texto "
                "(`output_transcript.added` del censo de la misma llamada), "
                "cruzado después. Etiquetar con la energía sería circular"
            ),
            "raw_audio": self.with_audio,
            "deltas": len(self._rows),
        }
        with self.destination.open("w", encoding="utf-8") as file:
            file.write(json.dumps(header, ensure_ascii=False) + "\n")
            for row in self._rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("sonda de energía en %s (%d deltas)", self.destination, len(self._rows))
        return self.destination


def begin(call_id: str, *, folder: Path | None = None) -> EnergyProbe | None:
    """La sonda, o `None` si no está encendida por entorno.

    Devolver `None` y no un objeto inerte es deliberado: quien llama escribe
    `if self._sonda is not None`, y así en una corrida normal no se ejecuta ni
    una línea de este módulo dentro del ciclo de audio.
    """
    if not switched_on():
        return None
    destination = (folder or RUTA_DEFAULT) / f"energy_{call_id or 'no_sid'}.jsonl"
    return EnergyProbe(destination, with_audio=dumps_audio())
