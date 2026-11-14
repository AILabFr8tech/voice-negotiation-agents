"""Synthetic clips with the same manifest as the real ones, for testing without audio.

The campaign's `.ulaw` files are a person's voice and are not in the repo. To
test the validation without them, for each entry of the real manifest a clip of
the same duration is generated with a speech-like signal (harmonics with a
syllabic envelope), normalized to −18 dBFS of speech, plus a manifest with its
own hashes and its own end of speech, measured with the same function as the
real ones. **None of this enters any result**: it only lets the code run on a
machine without the clips.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, pcm16_a_ulaw, ulaw_a_pcm16
from shared.audio.manifest import MANIFEST_NAME
from shared.audio.level import end_of_speech, rms_of_speech

RAIZ = Path(__file__).resolve().parents[1]
REAL_MANIFEST = RAIZ / "shared" / "test_scenarios" / "clips" / MANIFEST_NAME


def speech_signal(duration_s: float, cola_seg: float, chance: np.random.Generator) -> np.ndarray:
    n = int(round(duration_s * TELEPHONY_SAMPLE_RATE))
    # Random intonation and syllables, so that no clip resembles a shifted copy
    # of itself or another clip: the PoC3 validation recognizes clips by their
    # waveform, and a periodic signal would fool it.
    step = chance.normal(0.0, 1.0, n // 400 + 2)
    f0 = 110.0 + 25.0 * np.tanh(np.interp(np.arange(n) / 400, np.arange(step.size), np.cumsum(step)) / 4)
    phase = 2 * np.pi * np.cumsum(f0) / TELEPHONY_SAMPLE_RATE
    sig = sum((0.7**k) * np.sin((k + 1) * phase) for k in range(8))
    sig = sig + 0.3 * chance.normal(0.0, 1.0, n)
    durations = chance.uniform(0.12, 0.30, int(duration_s / 0.12) + 2)
    edges = np.cumsum(durations) * TELEPHONY_SAMPLE_RATE
    syllable = np.searchsorted(edges, np.arange(n))
    heights = chance.uniform(0.3, 1.0, syllable.max() + 1)
    envelope = np.convolve(heights[syllable], np.ones(240) / 240, mode="same")
    edge = int(0.03 * TELEPHONY_SAMPLE_RATE)
    envelope[:edge] *= np.linspace(0, 1, edge)
    speech_until = n - int(cola_seg * TELEPHONY_SAMPLE_RATE)
    envelope[speech_until:] *= 0.01
    pcm = sig * envelope
    pcm = pcm / np.max(np.abs(pcm)) * 20000
    return pcm.astype(np.int16)


def generate(folder: Path, *, seed: int = 7) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    real = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    chance = np.random.default_rng(seed)
    clips = []
    for entry in real["clips"]:
        pcm = speech_signal(float(entry["duration_s"]), float(entry["cola_seg"]), chance)
        gain = 10 ** ((-18.0 - rms_of_speech(pcm, TELEPHONY_SAMPLE_RATE)) / 20)
        pcm = np.clip(pcm * gain, -32768, 32767).astype(np.int16)
        ulaw = pcm16_a_ulaw(pcm)
        decoded = ulaw_a_pcm16(ulaw)
        eos = end_of_speech(decoded, TELEPHONY_SAMPLE_RATE)
        (folder / f"{entry['id']}.ulaw").write_bytes(ulaw)
        new_item = dict(entry)
        new_item.update({
            "sha256_ulaw": hashlib.sha256(ulaw).hexdigest(),
            "samples": len(ulaw),
            "duration_s": round(len(ulaw) / TELEPHONY_SAMPLE_RATE, 6),
            "end_of_speech_s": eos.end_s,
            "cola_seg": eos.cola_seg,
            "speech_threshold_dbfs": eos.umbral_dbfs,
            "warning_list": ["SINTÉTICO: solo para probar la validación"],
        })
        clips.append(new_item)
    manifest = dict(real)
    manifest["clips"] = clips
    ruta = folder / MANIFEST_NAME
    ruta.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


if __name__ == "__main__":
    import sys

    print(generate(Path(sys.argv[1])))
