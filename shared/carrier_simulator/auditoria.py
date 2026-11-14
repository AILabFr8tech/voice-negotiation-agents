"""Audit recording: listen to a whole run, in two separate channels.

It exists so that one can **verify by ear what the tests verify with
numbers**. It is diagnostics and nothing more: it does not go into `latencias`, nor WER,
nor any percentile, nor the campaign pool. The file name and the JSON
next to it say so, because a loose `.wav` in a folder six months from now does not
defend itself.

## Why two channels and not a mix

Mixing them would erase exactly the moment that matters. What one wants to hear is
whether the agent started over the clip and who spoke first; in a mix, two simultaneous
voices sound like one voice with noise. Kept separate, one turns the balance to one
side and listens to the other.

## Where the audio comes from, and where it does NOT

From what the harness already handles in both directions: channel 0 is the same
frames that the `Player` pushes (clips and comfort noise, without distinction) and
channel 1 is the same frames that arrive from the agent and feed the energy
measurement. **Egress is not enabled and `record` is not touched**: the comparison's
environment is frozen, and recording through LiveKit would change what is being
measured, besides adding one more service to the bill.

## Why it is dumped at the end and not written live

Writing to disk inside the loop that sets the pacing would put file-system time
right on the path whose delay is being measured. `annotate_*` only appends a
tuple to a list, which is O(1) and does not touch the disk; the file is built at
the end. The cost is memory: a 3-minute run is around 12 MB across the
two channels, which fits easily.

## Alignment

Both directions are timestamped against the same zero (`Observed.start`) and each
frame is placed at the instant it belongs to, padding whatever is missing with silence.
Channel 0 comes out exact because the player knows its own deadline; channel 1
is timestamped **when the frame arrives**, which is as close as one can get without a
clock on the other side. To hear who spoke first that is more than enough; to measure
milliseconds it is useless, and that is what the turn-zero marks are for.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from shared.audio import TELEPHONY_SAMPLE_RATE, write_wav_stereo, upsample_from_telephony

# Channel 0 is always the harness and 1 always the agent. Fixed: whoever opens the
# WAV in six months must be able to tell which is which without opening the JSON.
HARNESS_CHANNEL: Final[int] = 0
AGENT_CHANNEL: Final[int] = 1

# Marker in the file name. Having "auditoria" read before the room name is
# deliberate: sorting the folder by name, all of these end up together and
# apart from the campaign artifacts.
AUDIT_PREFIX: Final[str] = "auditoria"


class UnauditableSampleRate(ValueError):
    """The publication sample rate cannot be brought to that of the audit file."""


def output_sample_rate(harness_sample_rate: int, agent_sample_rate: int) -> int:
    """The WAV's sample rate. Raises if the two sides cannot be combined.

    It is validated **before** connecting to anything, not when dumping: finding out
    that the file cannot be written after having spent a call would be the
    worst possible moment to learn it.

    It is written at the agent's rate because it is the higher of the pair and because
    its voice is the one that will be judged by ear. The harness channel is upsampled
    from telephone band, which does not invent bandwidth: it still sounds like a
    phone, which is how it really sounds.
    """
    if harness_sample_rate == agent_sample_rate:
        return agent_sample_rate
    if harness_sample_rate == TELEPHONY_SAMPLE_RATE and agent_sample_rate % TELEPHONY_SAMPLE_RATE == 0:
        return agent_sample_rate
    raise UnauditableSampleRate(
        f"no se puede armar el WAV de auditoría con el arnés a {harness_sample_rate} Hz y "
        f"el agente a {agent_sample_rate} Hz. La grabación de auditoría soporta "
        f"--sample-rate {TELEPHONY_SAMPLE_RATE} (la del trunk SIP, que es el default) o "
        f"--sample-rate {agent_sample_rate}. La corrida sí puede correr a otra tasa: quita "
        "--record-audit"
    )


def _to_timeline(
    frames: list[tuple[float, bytes]], *, sample_rate: int, duration_s: float
) -> np.ndarray:
    """Places each frame at its instant and pads the rest with silence.

    Padding instead of concatenating is what keeps the file honest: if the
    agent's audio dropped out for ten seconds, concatenating would move everything
    after it ten seconds earlier and the WAV would show the agent
    answering before the clip had ended.
    """
    total = max(1, int(math.ceil(duration_s * sample_rate)))
    linea = np.zeros(total, dtype=np.int16)
    for moment, raw_value in frames:
        if len(raw_value) < 2:
            continue
        samples = np.frombuffer(raw_value[: len(raw_value) - len(raw_value) % 2], dtype="<i2")
        start = int(round(moment * sample_rate))
        if start < 0:
            samples = samples[-start:]
            start = 0
        end = min(start + samples.size, total)
        if end > start:
            linea[start:end] = samples[: end - start]
    return linea


def _carry_to(linea: np.ndarray, *, since: int, until: int, samples: int) -> np.ndarray:
    """Resamples if needed and leaves the channel with exactly `samples`."""
    if since != until:
        linea = upsample_from_telephony(linea, until)
    if linea.size < samples:
        linea = np.concatenate(
            [linea, np.zeros(samples - linea.size, dtype=np.int16)]
        )
    return np.ascontiguousarray(linea[:samples], dtype=np.int16)


@dataclass
class AuditRecorder:
    """Accumulates both directions in memory and dumps them at the end.

    It is passed to the `Player` and the `Observed`; both call it only if it is not
    `None`, so with the flag off it does not exist and there is nothing to pay.
    """

    harness_sample_rate: int
    agent_sample_rate: int
    start: float = field(default_factory=time.time)

    _harness: list[tuple[float, bytes]] = field(default_factory=list, repr=False)
    _agent: list[tuple[float, bytes]] = field(default_factory=list, repr=False)

    # --- live path: only this runs during the call -------------------------

    def annotate_harness(self, moment: float, raw_value: bytes) -> None:
        """`moment` is the start of the frame, relative to the run's zero."""
        self._harness.append((moment, raw_value))

    def annotate_agent(self, moment: float, raw_value: bytes) -> None:
        self._agent.append((moment, raw_value))

    # --- after hanging up ---------------------------------------------------

    @property
    def harness_frames(self) -> int:
        return len(self._harness)

    @property
    def agent_frames(self) -> int:
        return len(self._agent)

    def _duration_s(self) -> float:
        end = 0.0
        for frames, sample_rate in ((self._harness, self.harness_sample_rate), (self._agent, self.agent_sample_rate)):
            for moment, raw_value in frames:
                end = max(end, moment + (len(raw_value) // 2) / sample_rate)
        return end

    def dump(self, ruta_wav: Path | str, *, room: str = "") -> dict[str, Any]:
        """Builds the two-channel WAV and returns the descriptor for the JSON.

        It returns the dictionary instead of writing it so that the caller decides
        where it goes and can add its own fields: the recorder knows nothing about rooms
        or scenarios.
        """
        ruta_wav = Path(ruta_wav)
        output = output_sample_rate(self.harness_sample_rate, self.agent_sample_rate)
        duration = self._duration_s()
        samples = max(1, int(math.ceil(duration * output)))

        harness_channel = _carry_to(
            _to_timeline(self._harness, sample_rate=self.harness_sample_rate, duration_s=duration),
            since=self.harness_sample_rate,
            until=output,
            samples=samples,
        )
        agent_channel = _carry_to(
            _to_timeline(self._agent, sample_rate=self.agent_sample_rate, duration_s=duration),
            since=self.agent_sample_rate,
            until=output,
            samples=samples,
        )
        write_wav_stereo(ruta_wav, harness_channel, agent_channel, output)

        return {
            "what_it_is": (
                "grabación de auditoría de una corrida del arnés, para escuchar "
                "lo que las pruebas verifican con números"
            ),
            "is_campaign_evidence": False,
            "does_not_fit_in": [
                "latencies",
                "percentiles",
                "WER / CER",
                "la poza de las 15 llamadas",
            ],
            "room": room,
            "file": ruta_wav.name,
            "sample_rate_hz": output,
            "duration_s": round(duration, 3),
            "channel_list": {
                str(HARNESS_CHANNEL): (
                    "lo que el arnés publicó: los clips del guion y el ruido de "
                    "confort entre ellos, sin distinción — es la pista tal como "
                    "salió"
                ),
                str(AGENT_CHANNEL): "el audio del agente, tal como llegó",
            },
            "original_sample_rate_hz": {
                str(HARNESS_CHANNEL): self.harness_sample_rate,
                str(AGENT_CHANNEL): self.agent_sample_rate,
            },
            "frames": {
                str(HARNESS_CHANNEL): self.harness_frames,
                str(AGENT_CHANNEL): self.agent_frames,
            },
            "where_it_comes_from": (
                "del audio que el arnés ya maneja en las dos direcciones. NO es "
                "Egress de LiveKit: `record` sigue apagado y no se activó ninguna "
                "grabación del servicio"
            ),
            "alignment": (
                "las dos direcciones se fechan contra el mismo cero de la corrida. "
                "El canal 0 sale exacto (el reproductor conoce su deadline); el "
                "canal 1 se fecha cuando la trama llega, con el jitter del "
                "transporte dentro. Sirve para oír quién habló primero, NO para "
                "medir milisegundos: para eso están las marcas del cero del turno"
            ),
            "if_channel_0_sounds_like_a_phone": (
                "es correcto: se publica en banda telefónica de "
                f"{self.harness_sample_rate} Hz a propósito, y subirlo a {output} Hz para "
                "este archivo no le devuelve el ancho de banda que nunca tuvo"
            ),
        }
