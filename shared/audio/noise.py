"""Ruido de confort y medidas de nivel.

## Por qué el arnés no manda silencio

Entre clip y clip el arnés tiene que seguir mandando audio, y **no puede mandar
ceros**. Dos capas distintas rompen con el silencio digital:

- En PoC2, con DTX de Opus activo, las tramas de ceros hacen que el encoder deje
  de mandar paquetes; el `AudioStream` del agente no recibe nada durante la
  pausa y el VAD de Silero nunca acumula los 0.55 s de silencio *observado* que
  pide su `min_silence_duration`. El fin de turno sale tarde o no sale. El
  arnés apaga DTX **y** manda ruido: el `dtx=False` lo desactiva en el
  publicador y el ruido evita que el encoder clasifique la trama como silencio
  de todos modos.
- En PoC1 y PoC3 el transporte es μ-law sobre WebSocket y no hay DTX, pero el
  silencio absoluto tampoco existe en una llamada real: un teléfono siempre
  entrega piso de ruido. Mandar ceros le daría al VAD una frontera más limpia
  de la que tendría con un transportista de verdad.

## Por qué es determinista

El ruido entra en el audio que se reproduce, así que entra en el hash del clip.
Con una semilla fija, dos corridas del arnés mandan **los mismos bytes** y la
reproducibilidad es verificable y no una intención. Sin semilla, cada corrida
sería irrepetible por diseño.

El nivel por default (-60 dBFS) es el de `calibracion_nivel.json`, medido sobre
una grabación real de la línea de Twilio. Queda por debajo de cualquier umbral de
voz y por encima de "esto es silencio, apaga el encoder" (validación en el anexo
de `docs/comparativo_results.md`).

Es la única constante de este paquete que se duplica: el valor vive aquí **y**
en el JSON de calibración, porque `nivel.py` ya importa de este módulo y hacerlo
al revés cerraría el ciclo. Una prueba verifica que los dos sigan de acuerdo.

**No es una constante de la comparativa**: no entra en ningún número reportado.
Es una perilla del arnés y vive aquí, no en el contrato de métricas.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

# Nivel del ruido de confort, en dBFS RMS. Ver el docstring del módulo.
COMFORT_NOISE_DBFS: Final[float] = -60.0

# Fondo de escala de PCM16. Se nombra porque aparece en las tres conversiones.
_FULL_SCALE: Final[float] = 32768.0


def comfort_noise(
    samples: int,
    *,
    dbfs: float = COMFORT_NOISE_DBFS,
    seed: int = 0,
) -> np.ndarray:
    """Ruido uniforme de `muestras` a `dbfs` RMS, reproducible por `semilla`.

    Uniforme y no gaussiano a propósito: acotado, así que no hay picos raros que
    un VAD pueda leer como un golpe de voz.
    """
    if samples <= 0:
        return np.zeros(0, dtype=np.int16)
    # RMS de una uniforme en [-pico, pico] es pico/sqrt(3).
    target_rms = _FULL_SCALE * (10.0 ** (dbfs / 20.0))
    pico = target_rms * math.sqrt(3.0)
    generator = np.random.default_rng(seed)
    values = generator.uniform(-pico, pico, size=samples)
    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


def dbfs_rms(samples: np.ndarray) -> float:
    """Nivel RMS en dBFS. `-inf` si el buffer es silencio absoluto."""
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return float("-inf")
    rms = math.sqrt(float(np.mean(data * data)))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms / _FULL_SCALE)


def dbfs_pico(samples: np.ndarray) -> float:
    """Nivel de pico en dBFS. Sirve para cazar clips recortados (cerca de 0)."""
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return float("-inf")
    pico = float(np.max(np.abs(data)))
    if pico <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(pico / _FULL_SCALE)
