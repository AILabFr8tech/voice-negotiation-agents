"""Utilidades de audio del arnés de clips. **Idénticas en las 3 PoC.**

Mismo criterio que `shared/metrics/`: si cada PoC degradara sus clips a su
manera, el WER mediría el preprocesamiento y no el STT. Por eso la preparación
del audio vive aquí y no dentro de una PoC, y por eso cada clip lleva un hash
que prueba que las tres reprodujeron **los mismos bytes**.

## El flujo

    1. Grabas las 49 líneas del guion  -> WAV 16 bits, una por archivo
    2. `cli preparar`                  -> calibra nivel, degrada y escribe
                                          artefactos + manifiesto
    3. `cli verificar`                 -> re-hashea contra manifiesto, guion y nivel
    4. El arnés de cada PoC reproduce los artefactos ya degradados

El paso 2 corre **una sola vez**: el arnés nunca degrada en vivo, porque hacer
DSP durante la llamada le metería su propio retraso a lo que se está midiendo.

## El nivel y la degradación son parte de la medición

Cada clip se normaliza al nivel que entrega la línea telefónica real antes de
degradarlo (`nivel.py`). Sin eso, un clip grabado más cerca del micrófono
cruzaría el umbral del VAD antes y lo reconocería mejor el STT: el arnés estaría
midiendo su propia ganancia. **El nivel de referencia de hoy es provisional** —
ver `calibracion_nivel.json`.

Y se degrada a 8 kHz + μ-law porque es lo que pasó por las 15 llamadas de cada
PoC. Un clip de laptop a 48 kHz tiene un espectro que el transportista nunca pudo
haber mandado, y un STT lo reconoce mucho mejor: medir WER con audio de estudio
daría un número bonito y sin relación con lo que se comparó.

Lo que cambia entre PoC es solo **cómo se entrega** ese mismo audio:

| PoC | Transporte | Qué recibe |
|---|---|---|
| 1 y 3 | WebSocket de Media Streams | los bytes μ-law, idénticos a los de Twilio |
| 2 | pista de audio de la sala | el PCM de 8 kHz, que LiveKit sube a 48 kHz y codifica en Opus |

Esa última fila es una asimetría **de la arquitectura, no del arnés**: en una
llamada real por el trunk SIP, el mismo salto de 8 kHz a Opus lo hace LiveKit
igual. Va declarada en `docs/METRICS_CONTRACT.md` con el resto.

**Los clips no se versionan y el manifiesto sí** (`*.wav` está ignorado: pesa y
puede traer datos personales — LFPDPPP). El manifiesto es la única evidencia
commiteable de qué audio se reprodujo, y si se pierden los clips dice exactamente
qué se perdió.

Dependencias: solo `numpy` y la biblioteca estándar. `soxr` se usa **si está**, y
solo al preparar — ver `remuestreo.py`.
"""

from shared.audio.clips import (
    DegradedClip,
    EmptyClip,
    degrade,
    write_artifacts,
    prepare_clip,
)
from shared.audio.codec import (
    round_trip_ulaw,
    pcm16_a_ulaw,
    ulaw_a_pcm16,
)
from shared.audio.manifest import (
    ExpectedClip,
    ManifestEntry,
    Manifest,
    Problem,
    load_manifest,
    expected_inventory,
    verify,
)
from shared.audio.level import (
    DROP_FOR_LEVEL_DB,
    TOLERANCE_DB,
    Calibration,
    EndOfSpeech,
    load_calibration,
    end_of_speech,
    normalize_to_target,
    rms_of_speech,
)
from shared.audio.resampling import (
    TELEPHONY_SAMPLE_RATE,
    downsample_to_telephony,
    engine_description,
    upsample_from_telephony,
)
from shared.audio.noise import (
    COMFORT_NOISE_DBFS,
    dbfs_pico,
    dbfs_rms,
    comfort_noise,
)
from shared.audio.wav import (
    AudioNotSupported,
    write_wav,
    write_wav_stereo,
    leer_wav,
    read_wav_stereo,
)

__all__ = [
    "AudioNotSupported",
    "Calibration",
    "DegradedClip",
    "ExpectedClip",
    "EmptyClip",
    "DROP_FOR_LEVEL_DB",
    "COMFORT_NOISE_DBFS",
    "ManifestEntry",
    "EndOfSpeech",
    "Manifest",
    "Problem",
    "TELEPHONY_SAMPLE_RATE",
    "TOLERANCE_DB",
    "downsample_to_telephony",
    "load_calibration",
    "load_manifest",
    "dbfs_pico",
    "dbfs_rms",
    "degrade",
    "engine_description",
    "write_artifacts",
    "write_wav",
    "write_wav_stereo",
    "end_of_speech",
    "round_trip_ulaw",
    "expected_inventory",
    "leer_wav",
    "read_wav_stereo",
    "normalize_to_target",
    "pcm16_a_ulaw",
    "prepare_clip",
    "rms_of_speech",
    "comfort_noise",
    "upsample_from_telephony",
    "ulaw_a_pcm16",
    "verify",
]
