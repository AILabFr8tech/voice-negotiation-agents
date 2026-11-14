"""La grabación de auditoría: que separe los canales, que alinee, y que apagada no exista.

La prueba que más importa es la última clase: con la bandera apagada, todo lo
que el arnés mide tiene que salir idéntico. Una grabación que mueva las latencias
invalida la corrida entera, y entonces el archivo no sirve ni para oírlo.
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.audio import write_wav_stereo
from shared.carrier_simulator.auditoria import (
    AGENT_CHANNEL,
    HARNESS_CHANNEL,
    AuditRecorder,
    UnauditableSampleRate,
    output_sample_rate,
)
from shared.carrier_simulator.player import Player

AGENT_SAMPLE_RATE = 24_000
HARNESS_SAMPLE_RATE = 8_000

# Por debajo de esto un canal está callado, y **no se exige cero exacto**.
#
# El canal 0 pasa por el remuestreo de banda telefónica a la tasa del agente, y
# un remuestreador es un filtro: donde el tono se corta de golpe, su respuesta
# repica un poco más allá del corte, y además deja un piso de redondeo. Con
# `soxr` (HQ), un tono de 6000 (-14.7 dBFS) que se apaga en seco deja 744 cuentas
# justo en la transición, **12** cien muestras después (-68.7 dBFS) y **1** de
# ahí en adelante (-90 dBFS), hasta el final del archivo.
#
# Por eso no se exige `== 0`: solo se cumpliría con el respaldo de numpy, cuyo
# FIR es de ~49 taps y muere antes; con `soxr` la prueba fallaría por 12 cuentas
# de un audio que nadie puede oír. Lo que la prueba quiere decir es "aquí no hay
# voz del arnés", y eso se mide contra un umbral audible.
#
# 60 cuentas (-54.7 dBFS) es cinco veces el repique y queda por debajo del piso
# de ruido de confort (-60 dBFS son ~33 cuentas RMS), así que sigue siendo un
# límite estricto: cualquier fuga real de audio de un canal al otro lo pasa por
# mucho — la señal de prueba está en 6000, cien veces más arriba.
CHANNEL_SILENCE = 60


def _tone(samples: int, valor: int = 8000) -> bytes:
    return np.full(samples, valor, dtype=np.int16).tobytes()


def _read_channels(ruta: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Los dos canales por separado. `leer_wav` promedia el estéreo y aquí eso
    borraría justo lo que se quiere comprobar."""
    with wave.open(str(ruta), "rb") as file:
        assert file.getnchannels() == 2
        assert file.getsampwidth() == 2
        sample_rate = file.getframerate()
        raw_value = file.readframes(file.getnframes())
    pairs = np.frombuffer(raw_value, dtype="<i2").reshape(-1, 2)
    return pairs[:, HARNESS_CHANNEL], pairs[:, AGENT_CHANNEL], sample_rate


class _TestPlayer(Player):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frames: list[bytes] = []

    async def publish(self, raw_value: bytes) -> None:
        self.frames.append(raw_value)


class TestOutputRate:
    def test_telephone_band_upsamples_to_the_agents(self) -> None:
        assert output_sample_rate(HARNESS_SAMPLE_RATE, AGENT_SAMPLE_RATE) == AGENT_SAMPLE_RATE

    def test_equal_rates_stay(self) -> None:
        assert output_sample_rate(AGENT_SAMPLE_RATE, AGENT_SAMPLE_RATE) == AGENT_SAMPLE_RATE

    def test_a_rate_that_cannot_be_pooled_is_rejected(self) -> None:
        with pytest.raises(UnauditableSampleRate) as error:
            output_sample_rate(44_100, AGENT_SAMPLE_RATE)
        # El mensaje tiene que decir qué hacer: quien lo lee está a punto de
        # gastar una llamada.
        assert "--record-audit" in str(error.value)


class TestTwoSeparateChannels:
    def test_each_side_goes_on_its_channel_and_they_do_not_mix(self, tmp_path: Path) -> None:
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        # El arnés habla en el segundo 0; el agente, en el 1.
        recorder.annotate_harness(0.0, _tone(HARNESS_SAMPLE_RATE, 6000))
        recorder.annotate_agent(1.0, _tone(AGENT_SAMPLE_RATE, 9000))

        ruta = tmp_path / "auditoria_room.wav"
        recorder.dump(ruta, room="room")
        harness, agent, sample_rate = _read_channels(ruta)

        assert sample_rate == AGENT_SAMPLE_RATE
        assert harness.size == agent.size

        # Primer segundo: solo el arnés.
        assert np.abs(harness[: AGENT_SAMPLE_RATE // 2]).max() > 1000
        assert np.abs(agent[: AGENT_SAMPLE_RATE // 2]).max() == 0
        # Segundo segundo: solo el agente. El canal del arnés se mide contra
        # `SILENCIO_DE_CANAL` y no contra cero porque pasó por el remuestreo;
        # el del agente sí va exacto, que no lo remuestrea nadie.
        assert np.abs(agent[AGENT_SAMPLE_RATE + 100 : AGENT_SAMPLE_RATE * 2 - 100]).max() > 1000
        assert np.abs(harness[AGENT_SAMPLE_RATE + 100 :]).max() < CHANNEL_SILENCE

    def test_the_talk_over_is_audible_because_both_sound_at_once(
        self, tmp_path: Path
    ) -> None:
        """Lo que el archivo existe para mostrar: quién habló encima de quién."""
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        recorder.annotate_harness(0.0, _tone(HARNESS_SAMPLE_RATE, 6000))
        recorder.annotate_agent(0.5, _tone(AGENT_SAMPLE_RATE // 2, 9000))

        ruta = tmp_path / "a.wav"
        recorder.dump(ruta)
        harness, agent, _ = _read_channels(ruta)

        overlapped = slice(int(0.6 * AGENT_SAMPLE_RATE), int(0.9 * AGENT_SAMPLE_RATE))
        assert np.abs(harness[overlapped]).max() > 1000
        assert np.abs(agent[overlapped]).max() > 1000


class TestAlignment:
    def test_a_gap_is_filled_with_silence_and_what_follows_does_not_shift(
        self, tmp_path: Path
    ) -> None:
        """Concatenar en vez de rellenar mostraría al agente contestando antes
        de tiempo, que es exactamente lo que este archivo sirve para juzgar."""
        recorder = AuditRecorder(
            harness_sample_rate=AGENT_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        recorder.annotate_agent(0.0, _tone(AGENT_SAMPLE_RATE // 10))
        # Diez segundos de nada, y después otra trama.
        recorder.annotate_agent(10.0, _tone(AGENT_SAMPLE_RATE // 10))

        ruta = tmp_path / "a.wav"
        recorder.dump(ruta)
        _, agent, _ = _read_channels(ruta)

        assert np.abs(agent[: AGENT_SAMPLE_RATE // 10]).max() > 1000
        assert np.abs(agent[AGENT_SAMPLE_RATE : AGENT_SAMPLE_RATE * 9]).max() == 0
        assert np.abs(agent[AGENT_SAMPLE_RATE * 10 :]).max() > 1000

    def test_the_channels_measure_the_same_even_if_one_ends_earlier(
        self, tmp_path: Path
    ) -> None:
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        recorder.annotate_harness(0.0, _tone(HARNESS_SAMPLE_RATE // 10))
        recorder.annotate_agent(0.0, _tone(AGENT_SAMPLE_RATE * 3))

        ruta = tmp_path / "a.wav"
        recorder.dump(ruta)
        harness, agent, _ = _read_channels(ruta)
        assert harness.size == agent.size

    def test_a_stereo_wav_with_mismatched_channels_is_rejected(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="miden distinto"):
            write_wav_stereo(
                tmp_path / "a.wav",
                np.zeros(10, dtype=np.int16),
                np.zeros(11, dtype=np.int16),
                AGENT_SAMPLE_RATE,
            )


class TestDescriptive:
    def test_says_it_is_not_campaign_evidence(self, tmp_path: Path) -> None:
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        recorder.annotate_harness(0.0, _tone(HARNESS_SAMPLE_RATE))
        descriptive = recorder.dump(tmp_path / "a.wav", room="sala-x")

        assert descriptive["is_campaign_evidence"] is False
        assert "latencies" in descriptive["does_not_fit_in"]
        assert "percentiles" in descriptive["does_not_fit_in"]
        # Y que el audio no salió de LiveKit, que es la restricción dura.
        assert "Egress" in descriptive["where_it_comes_from"]
        # Serializable: se escribe como JSON al lado del WAV.
        json.dumps(descriptive, ensure_ascii=False)

    def test_counts_the_frames_of_each_side(self, tmp_path: Path) -> None:
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        for i in range(5):
            recorder.annotate_harness(i * 0.02, _tone(160))
        for i in range(3):
            recorder.annotate_agent(i * 0.02, _tone(480))

        descriptive = recorder.dump(tmp_path / "a.wav")
        assert descriptive["frames"][str(HARNESS_CHANNEL)] == 5
        assert descriptive["frames"][str(AGENT_CHANNEL)] == 3


class TestOffChangesNothing:
    """La restricción dura: sin la bandera, todo sale idéntico a como salía."""

    @staticmethod
    def _launch(recorder):
        observed = cs.Observed(silent=True)
        player = _TestPlayer(
            sample_rate=HARNESS_SAMPLE_RATE, observed=observed, recorder=recorder
        )
        clip = np.full(1600, 5000, dtype=np.int16)  # 0.2 s = 10 tramas

        async def _run_loop() -> float:
            halt = asyncio.Event()
            tarea = asyncio.create_task(player.launch(halt))
            await asyncio.sleep(0.05)
            end = await player.play(clip)
            halt.set()
            await tarea
            return end

        end = asyncio.run(_run_loop())
        return player.frames, observed, end, player.pacing

    def test_the_published_frames_are_byte_identical(self) -> None:
        sin, _, _, _ = self._launch(None)
        having, _, _, _ = self._launch(
            AuditRecorder(harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE)
        )
        # El ruido de confort es aleatorio, así que se comparan las del clip:
        # esas sí tienen que salir byte por byte iguales.
        of_the_clip_without = [t for t in sin if t == np.full(160, 5000, dtype=np.int16).tobytes()]
        of_the_clip_with = [t for t in having if t == np.full(160, 5000, dtype=np.int16).tobytes()]
        assert of_the_clip_without == of_the_clip_with
        assert len(of_the_clip_without) == 10

    def test_the_zero_marks_do_not_move(self) -> None:
        """Encender la grabadora no puede correr el cero del turno.

        **Se compara el calendario nominal, no la duración medida**, y la razón
        no es de tolerancia sino de instrumento. El cero sale del instante de
        envío **medido**, así que la duración de dos corridas difiere
        en lo que difiera el planificador; con 10 tramas eso es ruido del mismo
        orden que lo que se busca. Peor todavía: como el ritmo va contra deadline
        **absoluto**, un costo por trama menor que los 20 ms de holgura se
        absorbe entero y **no aparece nunca** en la duración. Medir por ahí no
        funciona.

        El calendario nominal sí es aritmética pura: si la grabadora empujara una
        trama, la descartara o cambiara el conteo, el calendario cambiaría y esto
        falla de forma determinista. La duración medida se sigue mirando, pero
        solo como cota gruesa de que el bucle no se atoró.
        """
        _, obs_without, end_without, pacing_without = self._launch(None)
        _, obs_with, end_with, pacing_with = self._launch(
            AuditRecorder(harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE)
        )
        assert obs_without.clip_frames == obs_with.clip_frames == 10

        # El calendario nominal: cada trama a exactamente 20 ms de la anterior,
        # sin deriva acumulada, con y sin grabadora.
        #
        # Se comparan los **espaciados** y no las dos listas enteras porque su
        # largo no dice nada: el bucle de ruido de confort sigue corriendo hasta
        # que la prueba pide parar, así que dos corridas terminan con una trama
        # de diferencia según dónde caiga ese `parar`. Lo del clip ya está
        # cubierto arriba (`tramas_del_clip == 10`).
        #
        # La tolerancia de 2 µs no es holgura de reloj —esto es aritmética— sino
        # el redondeo a 1 µs que hace `_instante_de`.
        for pacing in (pacing_without, pacing_with):
            onset = pacing.nominal_values[0]
            for i, nominal in enumerate(pacing.nominal_values):
                assert nominal - onset == pytest.approx(i * 0.020, abs=2e-6)

        # Cota gruesa: 10 tramas de 20 ms. Holgada a propósito —el planificador
        # de Windows se lleva decenas de ms bajo carga— porque lo que esta
        # prueba verifica de verdad es la igualdad de arriba.
        for duration in (
            end_without - (obs_without.t_clip_start or 0),
            end_with - (obs_with.t_clip_start or 0),
        ):
            assert duration == pytest.approx(0.2, abs=0.100)

    def test_logging_audio_measures_the_same_with_and_without_recorder(self) -> None:
        raw_value = _tone(480, 9000)
        sin = cs.Observed(silent=True)
        having = cs.Observed(
            silent=True,
            recorder=AuditRecorder(
                harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
            ),
        )
        for observed in (sin, having):
            observed.annotate_audio(raw_value)

        assert sin.audio_frames == having.audio_frames == 1
        assert sin.rms == having.rms
        assert (sin.t_agent_first_speech is None) == (
            having.t_agent_first_speech is None
        )

    def test_without_a_recorder_not_a_single_frame_is_saved(self) -> None:
        """El costo de la bandera apagada es una comparación contra `None`."""
        observed = cs.Observed(silent=True)
        assert observed.recorder is None
        player = _TestPlayer(sample_rate=HARNESS_SAMPLE_RATE, observed=observed)
        assert player.recorder is None

    def test_with_recorder_both_directions_are_saved(self) -> None:
        recorder = AuditRecorder(
            harness_sample_rate=HARNESS_SAMPLE_RATE, agent_sample_rate=AGENT_SAMPLE_RATE
        )
        _, observed, _, _ = self._launch(recorder)
        observed.recorder = recorder
        observed.annotate_audio(_tone(480))

        assert recorder.harness_frames >= 10
        assert recorder.agent_frames == 1
