"""Recognize which clip arrived and timestamp its end of speech, on the fake agent's side.

Shared by the PoC3 (`fake_model_poc3`) and PoC2 (`fake_agent_livekit`) fake
agents. In PoC1 the fake agent recognizes the clip by the bytes of its first
frame, because the WebSocket delivers exactly what the harness published. In PoC3
the audio arrives resampled to 24 kHz by the bridge, and in PoC2 it also goes
through Opus in the LiveKit room: the bytes no longer work as a fingerprint.

Here recognition is by waveform. When speech appears in the incoming audio, a
window around it is taken and correlated against the onset of each line the
scenario can play, resampled to the arrival rate. The peak gives which clip it is
and at which sample it started; from there, with the manifest's
`end_of_speech_s`, comes the end-of-speech sample. **The real end of speech is the
instant at which the chunk containing that sample arrived**: that is when a real
agent would have it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE
from shared.audio.manifest import MANIFEST_NAME, load_manifest
from shared.carrier_simulator.script import load_clip_to_publish

# Recognition window: a little before the speech onset, the template and a
# margin after it. The trigger is the start of the chunk where speech appeared,
# which may begin before the clip; the margin makes the whole clip fit in the
# window.
BEFORE_ONSET_S: Final[float] = 0.30
TEMPLATE_LENGTH_S: Final[float] = 0.60
MARGIN_AFTER_TEMPLATE_S: Final[float] = 0.20
# RMS in int16 from which a chunk counts as speech. It is the bridge's and the
# harness's threshold: the carrier is at −18 dBFS and the background noise at −60.
RMS_OF_SPEECH: Final[float] = 200.0


def resample_audio(pcm: np.ndarray, since: int, until: int) -> np.ndarray:
    """With the same library and quality as the PoC3 bridge."""
    if since == until:
        return pcm.astype(np.int16)
    import soxr

    output = soxr.resample(pcm.astype(np.float32), since, until, quality="HQ")
    return np.clip(np.round(output), -32768, 32767).astype(np.int16)


@dataclass(frozen=True)
class Template:
    clip_id: str
    text: str
    # The onset of the published clip, at the arrival rate, as floating point.
    start: np.ndarray
    # End-of-speech sample, counted from the start of the clip.
    end_of_speech: int
    # Length of the published clip, in samples.
    largo: int


def templates(folder: Path, clip_ids: list[str], sample_rate: int) -> list[Template]:
    """The candidate lines, as the harness publishes them, at rate `sample_rate`."""
    from shared.audio import ulaw_a_pcm16

    manifest = {e.id: e for e in load_manifest(folder / MANIFEST_NAME).entries}
    output = []
    for clip_id in clip_ids:
        entry = manifest.get(clip_id)
        ruta = folder / f"{clip_id}.ulaw"
        if entry is None or entry.end_of_speech_s is None or not ruta.is_file():
            continue
        _, published, _, _ = load_clip_to_publish(ruta, publish_sample_rate=TELEPHONY_SAMPLE_RATE)
        pcm = resample_audio(ulaw_a_pcm16(published), TELEPHONY_SAMPLE_RATE, sample_rate)
        output.append(Template(
            clip_id=clip_id,
            text=entry.text,
            start=pcm[: int(TEMPLATE_LENGTH_S * sample_rate)].astype(np.float64),
            end_of_speech=min(len(pcm), round(float(entry.end_of_speech_s) * sample_rate)),
            largo=len(pcm),
        ))
    if not output:
        raise SystemExit(f"no candidate clips in {folder}")
    return output


def normalized_correlation(sig: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized correlation of `template` along `sig`, via FFT."""
    n, m = len(sig), len(template)
    tam = 1 << (n + m - 1).bit_length()
    raw = np.fft.irfft(
        np.fft.rfft(sig, tam) * np.conj(np.fft.rfft(template, tam)), tam
    )[: n - m + 1]
    cumulative = np.concatenate([[0.0], np.cumsum(sig**2)])
    energy = cumulative[m:] - cumulative[: n - m + 1]
    norma = np.sqrt(np.maximum(energy, 1e-9)) * np.sqrt(np.sum(template**2))
    return raw / norma


@dataclass(frozen=True)
class Candidate:
    template: Template
    displacement: int
    correlation: float


def recognize(sig: np.ndarray, candidate_ones: list[Template]) -> list[Candidate]:
    """Each candidate with its best peak, from highest to lowest correlation."""
    output = []
    for p in candidate_ones:
        if len(sig) < len(p.start):
            continue
        corr = normalized_correlation(sig, p.start)
        k = int(np.argmax(corr))
        output.append(Candidate(p, k, float(corr[k])))
    return sorted(output, key=lambda c: c.correlation, reverse=True)


@dataclass(frozen=True)
class EndOfSpeech:
    """A recognized clip and the instant its end of speech reached the agent."""

    template: Template
    correlation: float
    second_one: float
    t_first_sample: float
    t_end_of_speech: float
    # Sample of the received audio where the clip started; with it any other
    # edge of the clip is timestamped via `Recognizer.arrival_of`.
    start_sample: int = -1


@dataclass(frozen=True)
class NotRecognized:
    correlation: float
    second_one: float
    best: str
    sample: int


class Recognizer:
    """Listens to the carrier's audio and reports when each clip's speech ended.

    `min_correlation` and `min_advantage` are what is required of the best peak,
    in absolute terms and over the second best. Through the PoC3 bridge the clip
    arrives almost intact and the peak comes out ≈1; through Opus it comes out
    lower, and the advantage over the second candidate is what prevents confusing
    two lines.
    """

    def __init__(
        self,
        candidate_ones: list[Template],
        sample_rate: int,
        *,
        min_correlation: float,
        min_advantage: float = 0.10,
    ) -> None:
        self.candidate_ones = candidate_ones
        self.sample_rate = sample_rate
        self.min_correlation = min_correlation
        self.min_advantage = min_advantage
        self._audio: list[np.ndarray] = []
        self._received = 0
        self._end_arrivals: list[int] = []
        self._arrivals_t: list[float] = []
        self._trigger: int | None = None
        self._pending: tuple[Candidate, float, int] | None = None
        self._ignore_until = 0

    def _window(self, since: int, until: int) -> np.ndarray:
        everything = np.concatenate(self._audio) if len(self._audio) > 1 else self._audio[0]
        self._audio = [everything]
        return everything[max(0, since):until].astype(np.float64)

    def arrival_of(self, sample: int) -> float:
        """Instant at which the chunk containing `sample` arrived."""
        return self._arrivals_t[bisect.bisect_right(self._end_arrivals, sample)]

    def receive(self, piece: np.ndarray, arrival: float) -> EndOfSpeech | NotRecognized | None:
        antes = self._received
        self._audio.append(np.asarray(piece, dtype=np.int16).copy())
        self._received += len(piece)
        self._end_arrivals.append(self._received)
        self._arrivals_t.append(arrival)

        if self._pending is not None:
            candidate, second_one, start = self._pending
            end = start + candidate.template.end_of_speech
            if self._received > end:
                self._pending = None
                return EndOfSpeech(
                    candidate.template, candidate.correlation, second_one,
                    self.arrival_of(start), self.arrival_of(end), start,
                )
            return None

        if self._trigger is None:
            if len(piece) and antes >= self._ignore_until:
                rms = float(np.sqrt(np.mean(np.asarray(piece, dtype=np.float64) ** 2)))
                if rms >= RMS_OF_SPEECH:
                    self._trigger = antes
            return None

        until = self._trigger + int(
            (TEMPLATE_LENGTH_S + MARGIN_AFTER_TEMPLATE_S) * self.sample_rate
        )
        if self._received < until:
            return None
        since = self._trigger - int(BEFORE_ONSET_S * self.sample_rate)
        self._trigger = None
        candidates = recognize(self._window(since, until), self.candidate_ones)
        best = candidates[0]
        second_one = candidates[1].correlation if len(candidates) > 1 else 0.0
        start = max(0, since) + best.displacement
        if (
            best.correlation < self.min_correlation
            or best.correlation - second_one < self.min_advantage
        ):
            self._ignore_until = self._received + self.sample_rate
            return NotRecognized(round(best.correlation, 4), round(second_one, 4),
                                best.template.clip_id, start)
        self._ignore_until = start + best.template.largo
        end = start + best.template.end_of_speech
        if self._received > end:
            return EndOfSpeech(best.template, best.correlation, second_one,
                            self.arrival_of(start), self.arrival_of(end), start)
        self._pending = (best, second_one, start)
        return None
