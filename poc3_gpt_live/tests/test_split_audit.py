"""Split the audit WAV into two mono files, without ffmpeg or sox.

What matters here is that **the channels do not get crossed**: a file labeled
"agent" that actually carries the carrier would lead whoever listens to it to
the opposite conclusion, and on 8 kHz audio of two different voices that cannot
be reliably detected by ear.
"""

from __future__ import annotations

import json
import wave

import numpy as np

from poc3_gpt_live.diagnostics.split_audit import (
    BRIDGE_KEY,
    is_from_bridge,
    main,
    split,
)


def _stereo(destination, left_side: np.ndarray, derecho: np.ndarray, sample_rate=8_000):
    interleaved = np.empty(len(left_side) * 2, dtype="<i2")
    interleaved[0::2] = left_side
    interleaved[1::2] = derecho
    with wave.open(str(destination), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(interleaved.tobytes())
    return destination


def _leer(ruta) -> np.ndarray:
    with wave.open(str(ruta), "rb") as f:
        assert f.getnchannels() == 1
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")


class TestTheChannelsDoNotCross:
    def test_each_mono_carries_its_own_sample_by_sample(self, tmp_path) -> None:
        left = (np.arange(1_000) % 100).astype("<i2")
        right = (np.arange(1_000) % 7 - 3).astype("<i2")

        c0, c1 = split(_stereo(tmp_path / "a.wav", left, right))

        assert np.array_equal(_leer(c0), left)
        assert np.array_equal(_leer(c1), right)

    def test_the_name_says_who_each_one_is(self, tmp_path) -> None:
        """The order is that of the two recorders: 0 carrier, 1 agent."""
        zeros = np.zeros(10, dtype="<i2")

        c0, c1 = split(_stereo(tmp_path / "a.wav", zeros, zeros))

        assert c0.name.endswith("_channel0_carrier.wav")
        assert c1.name.endswith("_channel1_agent.wav")

    def test_preserves_the_sample_rate_and_the_duration(self, tmp_path) -> None:
        samples = np.zeros(4_000, dtype="<i2")

        for output in split(_stereo(tmp_path / "a.wav", samples, samples)):
            with wave.open(str(output), "rb") as f:
                assert f.getframerate() == 8_000
                assert f.getnframes() == 4_000


class TestAWavThatIsNotTwoChannel:
    def test_is_rejected_instead_of_producing_garbage(self, tmp_path) -> None:
        import pytest

        destination = tmp_path / "mono.wav"
        with wave.open(str(destination), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8_000)
            f.writeframes(np.zeros(100, dtype="<i2").tobytes())

        with pytest.raises(SystemExit, match="no 2"):
            split(destination)


class TestTheBridgeWavWarning:
    """Its two channels are misaligned by ~1.8 s and it cannot be noticed by ear.

    The agent's speech onsets land almost two seconds earlier than in the
    harness WAV, while the channel-0 clips match between the two files. Whoever
    splits this WAV and listens to it will conclude that the agent answers
    almost immediately, and the warning is the only thing that stops them.
    """

    @staticmethod
    def _with_json(tmp_path, content: dict):
        ruta = _stereo(
            tmp_path / "auditoria_x.wav",
            np.zeros(10, dtype="<i2"),
            np.zeros(10, dtype="<i2"),
        )
        ruta.with_suffix(".json").write_text(
            json.dumps(content, ensure_ascii=False), encoding="utf-8"
        )
        return ruta

    def test_the_bridge_one_is_recognized_by_its_json(self, tmp_path) -> None:
        ruta = self._with_json(tmp_path, {BRIDGE_KEY: {"0": {}, "1": {}}})

        assert is_from_bridge(ruta) is True

    def test_the_harness_one_does_not_warn(self, tmp_path) -> None:
        """Its channel 0 comes from the player's deadline: that one is usable by ear."""
        ruta = self._with_json(tmp_path, {"room": "x", "alignment": "..."})

        assert is_from_bridge(ruta) is False

    def test_without_json_it_does_not_guess(self, tmp_path) -> None:
        ruta = _stereo(
            tmp_path / "loose.wav",
            np.zeros(10, dtype="<i2"),
            np.zeros(10, dtype="<i2"),
        )

        assert is_from_bridge(ruta) is False

    def test_a_corrupt_json_does_not_take_down_the_split(self, tmp_path) -> None:
        ruta = _stereo(
            tmp_path / "roto.wav",
            np.zeros(10, dtype="<i2"),
            np.zeros(10, dtype="<i2"),
        )
        ruta.with_suffix(".json").write_text("{esto no es json", encoding="utf-8")

        assert is_from_bridge(ruta) is False
        assert len(split(ruta)) == 2

    def test_the_cli_prints_it_and_names_the_harness_one(self, tmp_path, capsys) -> None:
        ruta = self._with_json(tmp_path, {BRIDGE_KEY: {}})

        assert main([str(ruta)]) == 0
        output = capsys.readouterr().out

        assert "NO ESTÁN ALINEADOS" in output
        assert "1.8 s" in output
        assert "harness" in output

    def test_the_cli_does_not_print_it_for_the_harness_one(self, tmp_path, capsys) -> None:
        ruta = self._with_json(tmp_path, {"room": "x"})

        assert main([str(ruta)]) == 0
        assert "NO ESTÁN ALINEADOS" not in capsys.readouterr().out
