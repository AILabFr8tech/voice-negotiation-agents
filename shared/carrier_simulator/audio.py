"""The frames the simulated carrier publishes: comfort noise and clip.

None of this knows where the audio goes out. Each PoC provides the transport (the
room's `AudioSource` in PoC2, the Media Streams WebSocket in PoC1 and
PoC3); here we only build the bytes and measure their energy.
"""

from __future__ import annotations

import math
import random
from array import array
from typing import Final

import numpy as np

from shared.audio.level import load_calibration
from shared.audio.noise import comfort_noise

# Duration of the published frame. 20 ms is what Media Streams sends and
# what `AudioSource` expects, so the 3 PoCs publish at the same pacing.
MS_PER_FRAME: Final[int] = 20


# Peak amplitude of the comfort noise, in int16 counts. The target level comes
# from `calibracion_nivel.json` (-60 dBFS), measured on a real recording of the
# line: it is the order of a telephone's noise floor and sits far below any
# speech threshold. With a uniform distribution on [-peak, peak] the RMS is
# peak/sqrt(3), and the amplitude follows from that.
#
# It is **independent of the sample rate**, so two arms that publish at different
# rates publish exactly the same noise level.
def _peak_for(dbfs: float) -> int:
    return max(1, round(32768.0 * (10.0 ** (dbfs / 20.0)) * math.sqrt(3.0)))


COMFORT_NOISE_PEAK: Final[int] = _peak_for(load_calibration().noise_dbfs)


def comfort_noise_frame(samples: int) -> bytes:
    """Mono int16 PCM of uniform noise, far below any speech.

    Zeros are deliberately **not** sent: with DTX active the encoder would stop
    sending packets, the agent would receive no frames during the pause, and its
    VAD would never accumulate the *observed* silence its `min_silence_duration`
    requires. Turning DTX off on publication fixes the publisher side; the noise
    keeps the encoder from classifying it as silence anyway.
    """
    pico = COMFORT_NOISE_PEAK
    return array("h", (random.randint(-pico, pico) for _ in range(samples))).tobytes()


def noise_floor_under_clip(samples: np.ndarray, *, seed: int) -> np.ndarray:
    """Adds the comfort noise **underneath** the clip, instead of alternating them.

    Alternating two sources (noise between clips and the raw clip during the clip)
    leaves two artificial noise-floor steps at the two edges of each
    line, exactly the two instants the end-of-turn detector has to
    find. A telephone line does not do that: the floor is always there and the
    speech is added on top. See `docs/METRICS_CONTRACT.md` §2.2.9.

    The caller sets the seed, derived from the clip, so that two runs
    publish exactly the same bytes: the harness is reproducible by design
    and the noise is part of the audio that gets played.
    """
    # No guard for the empty clip: `comfort_noise(0)` already returns an empty
    # array and the sum of two empty arrays is empty. A guard here would be a branch
    # that no test can tell apart from its absence.
    piso = comfort_noise(int(samples.size), seed=seed)
    summed = samples.astype(np.int32) + piso.astype(np.int32)
    return np.clip(summed, -32768, 32767).astype(np.int16)


def rms_pcm16(raw_value: bytes) -> float:
    """RMS of an int16 PCM buffer."""
    samples = array("h")
    samples.frombytes(raw_value[: len(raw_value) - len(raw_value) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(m * m for m in samples) / len(samples))


def split_into_frames(samples: np.ndarray, per_frame: int) -> list[bytes]:
    """Splits the clip into equal-sized frames, padding the last one.

    The last frame is filled with comfort noise rather than zeros, for the
    same reason as the padding between clips.
    """
    frames: list[bytes] = []
    for start in range(0, len(samples), per_frame):
        piece = samples[start : start + per_frame]
        if len(piece) < per_frame:
            filler = np.frombuffer(
                comfort_noise_frame(per_frame - len(piece)), dtype=np.int16
            )
            piece = np.concatenate([piece, filler])
        frames.append(np.ascontiguousarray(piece, dtype=np.int16).tobytes())
    return frames
