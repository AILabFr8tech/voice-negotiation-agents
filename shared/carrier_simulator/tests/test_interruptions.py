"""The interruption scenario: preparation, plan, measurement and a whole run.

The whole run uses an acted agent that hears what the harness publishes: it answers
with a long sentence and, if an interruption clip plays over it, it goes quiet a
fixed time later and replies. The harness knows nothing about that; what is tested is
that it measures it.
"""

from __future__ import annotations

import asyncio
import json
import wave

import numpy as np
import pytest

import shared.carrier_simulator as cs
from shared.audio import interruptions as prep
from shared.carrier_simulator import interruptions as it
from shared.carrier_simulator.observed import SpeechStreak
from shared.carrier_simulator.player import Player, play_script

VOICE = np.full(480, 8000, dtype=np.int16).tobytes()  # 20 ms at 24 kHz, with speech


def _recording(ruta, seconds: float, seed: int) -> None:
    """A 48 kHz "voice": noise with a syllabic envelope and silence at both ends."""
    sample_rate = 48_000
    chance = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    syllables = np.repeat(chance.uniform(0.3, 1.0, n // 4800 + 1), 4800)[:n]
    voice = chance.normal(0, 1, n) * syllables * 6000
    pcm = np.concatenate([np.zeros(sample_rate // 5), voice, np.zeros(sample_rate // 5)]).astype(np.int16)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


@pytest.fixture()
def prepared_clips(tmp_path):
    raw_values = tmp_path / "raw_values"
    raw_values.mkdir()
    for k, c in enumerate(prep.load_catalog()):
        _recording(raw_values / f"{c.id}.wav", 1.5 if c.tipo == "interrupcion" else 0.4, k)
    destination = tmp_path / "interruption_clips"
    entries, warning_list = prep.prepare(raw_values, destination)
    return destination, entries


def test_prepare_writes_the_manifest_with_speech_start_and_end(prepared_clips):
    destination, entries = prepared_clips
    assert len(entries) == len(prep.load_catalog())
    for e in entries:
        assert 0.0 <= e.speech_onset_s < e.end_of_speech_s <= e.duration_s + 1e-6
        assert (destination / f"{e.id}.ulaw").is_file()
    assert prep.verify(destination) == []
    manifest = json.loads((destination / prep.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {c["tipo"] for c in manifest["clips"]} == {"interrupcion", "apoyo"}


def test_the_plan_puts_two_events_of_each_class_and_rotates_the_clips():
    catalog = [{"id": c.id, "tipo": c.tipo} for c in prep.load_catalog()]
    turns = [f"T{k}" for k in range(2, 8)]
    used_items: dict[str, int] = {}
    for scenario in range(1, 6):
        for repetition in range(1, 4):
            plan = it.event_plan(scenario, repetition, turns, catalog)
            kinds = [t for t, _ in plan.values()]
            assert sorted(kinds) == sorted(it.PATRON)
            # Same scenario and repetition, same plan: that is what makes the 3 PoCs equal.
            assert plan == it.event_plan(scenario, repetition, turns, catalog)
            for _, clip in plan.values():
                if clip:
                    used_items[clip] = used_items.get(clip, 0) + 1
    # Every clip comes up, and none is used twice as often as another of its class.
    assert set(used_items) == {c["id"] for c in catalog}
    for tipo in ("int_", "apo_"):
        tallies = [n for k, n in used_items.items() if k.startswith(tipo)]
        assert max(tallies) <= 2 * min(tallies)


def test_measure_the_stop_and_replicate_event():
    streaks = [SpeechStreak(start=1.0, end=3.2), SpeechStreak(start=3.5, end=3.9), SpeechStreak(start=6.0, end=8.0)]
    m = it.measure_event(streaks, reference=3.0, end_of_speech=4.5, tipo="interrupcion")
    # The 0.3 s pause between 3.2 and 3.5 is not a stop (< 0.7 s); the one from 3.9 to 6.0 is.
    assert m["stopped"] is True
    assert m["stop_time_ms"] == pytest.approx(900.0)
    assert m["reply_latency_ms"] == pytest.approx(1500.0)


def test_the_backchannel_measures_the_pause_and_when_it_resumes():
    # Brief pause after the "ajá" (its speech ends at 2.5 s) and it keeps talking.
    streaks = [SpeechStreak(start=1.0, end=2.4), SpeechStreak(start=3.3, end=6.0)]
    m = it.measure_event(streaks, reference=2.0, end_of_speech=2.5, tipo="apoyo", until=8.0)
    assert m["stopped"] is True and m["endless_pause"] is False
    assert m["pause_duration_ms"] == pytest.approx(900.0)
    assert m["speaks_again_ms"] == pytest.approx(800.0)
    assert "reply_latency_ms" not in m


def test_if_it_does_not_resume_the_pause_is_a_lower_bound():
    streaks = [SpeechStreak(start=1.0, end=2.4)]
    m = it.measure_event(streaks, reference=2.0, end_of_speech=2.0, tipo="control", until=4.5)
    assert m["endless_pause"] is True and "speaks_again_ms" not in m
    assert m["pause_duration_ms"] == pytest.approx(2100.0)


def test_in_the_interruption_speaks_again_is_the_replicate():
    streaks = [SpeechStreak(start=1.0, end=3.2), SpeechStreak(start=6.0, end=8.0)]
    m = it.measure_event(streaks, reference=3.0, end_of_speech=4.5, tipo="interrupcion")
    assert m["speaks_again_ms"] == m["reply_latency_ms"] == pytest.approx(1500.0)


def test_measure_the_event_without_stop_within_the_window():
    streaks = [SpeechStreak(start=1.0, end=9.0)]
    m = it.measure_event(streaks, reference=2.0, end_of_speech=2.5, tipo="apoyo")
    assert m["stopped"] is False
    assert m["stop_time_ms"] == pytest.approx(7000.0)


class _Player(Player):
    """Publishes to a list and tells the acted agent which clip is playing."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.playing: str | None = None
        self.start_of: dict[str, float] = {}

    async def publish(self, raw_value: bytes) -> None:
        return None

    async def play_step(self, step):
        self.playing = step.clip_id
        self.start_of[step.clip_id] = self.observed.now()
        try:
            return await super().play_step(step)
        finally:
            self.playing = None


async def _agent(observed, player, *, stop_s: float, replica_seg: float):
    """Answers each line for 4 s; if an interruption plays, it goes quiet and replies."""
    seen_end: float | None = None
    while True:
        # Waits for a script line to finish.
        while player.playing is None or player.playing.startswith("int_"):
            await asyncio.sleep(0.01)
        clip = player.playing
        while player.playing == clip:
            await asyncio.sleep(0.01)
        # What the STT transcribed of the line, before answering.
        observed.annotate_text(observed.own_identity, f"línea {clip}", final=True)
        await asyncio.sleep(0.5)
        end = asyncio.get_running_loop().time() + 4.0
        interrupted_at = None
        while asyncio.get_running_loop().time() < end:
            if player.playing and player.playing.startswith("int_") and interrupted_at is None:
                interrupted_at = asyncio.get_running_loop().time()
                # And what it transcribed of the clip that played over it. One segment
                # later: written in the same millisecond the clip starts, it
                # could land on the line's side, and the test failed sometimes.
                await asyncio.sleep(0.02)
                observed.annotate_text(observed.own_identity, "espéreme", final=True)
            if interrupted_at is not None and asyncio.get_running_loop().time() >= interrupted_at + stop_s:
                break
            observed.annotate_audio(VOICE)
            await asyncio.sleep(0.02)
        if interrupted_at is not None:
            while player.playing is not None:
                await asyncio.sleep(0.01)
            await asyncio.sleep(replica_seg)
            for _ in range(50):  # 1 s of reply
                observed.annotate_audio(VOICE)
                await asyncio.sleep(0.02)
        seen_end = observed.now()
        del seen_end


def test_a_run_with_interruption_measures_the_stop(prepared_clips):
    destination, _ = prepared_clips
    script = [
        cs.ScriptStep(turn=f"T{k}", text=f"línea {k}", samples=np.full(4000, 3000, np.int16),
                        clip_id=f"e1_t{k}", sample_rate=8000, end_of_speech_s=0.5)
        for k in (1, 2, 3)
    ]
    plan = it.attach_events(script, scenario=1, repetition=1, folder=destination)
    # With three turns only T2 is eligible, and the plan attaches an event to it.
    assert set(plan) == {"T2"} and script[1].event is not None
    # For the test, a 1 s interruption with speech from the start is forced.
    entry = json.loads((destination / prep.MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(e for e in entry["clips"] if e["tipo"] == "interrupcion")
    clip_int = entry["id"]
    event = it.TurnEvent(
        tipo="interrupcion", clip_id=clip_int, text=entry["text"],
        step=cs.ScriptStep(turn="T2", text=entry["text"],
                             samples=np.full(8000, 3000, np.int16), clip_id=clip_int, sample_rate=8000),
        speech_onset_s=0.0, end_of_speech_s=1.0,
    )
    script[1].event = event
    script[0].event = script[2].event = None

    observed = cs.Observed(silent=True)
    player = _Player(sample_rate=8000, observed=observed)

    async def _everything():
        halt = asyncio.Event()
        pumping = asyncio.create_task(player.launch(halt))
        actor = asyncio.create_task(_agent(observed, player, stop_s=0.4, replica_seg=1.0))
        await play_script(script, player=player, observed=observed, wait_for_agent_s=15.0)
        actor.cancel()
        halt.set()
        await asyncio.gather(pumping, actor, return_exceptions=True)

    asyncio.run(_everything())
    t2 = observed.turns[1]
    assert t2.event is not None and t2.event["fired"] is True
    assert t2.event["tipo"] == "interrupcion"
    assert t2.event["stopped"] is True
    # It goes quiet 0.4 s after the interruption starts playing (±60 ms of
    # polling and framing).
    assert t2.event["stop_time_ms"] == pytest.approx(400.0, abs=80.0)
    # It replies 1 s after the interruption's speech ends.
    assert t2.event["reply_latency_ms"] == pytest.approx(1000.0, abs=120.0)
    # What the STT heard, split into the script line and the event's clip.
    assert t2.event["line_stt"] == "línea e1_t2"
    assert isinstance(t2.event["agent_before_event"], str)
    assert isinstance(t2.event["agent_after_event"], str)
    assert t2.event["event_stt"] == "espéreme"
    # Turns without an event proceed as usual.
    assert observed.turns[0].event is None and observed.turns[2].event is None
    assert observed.turns[2].advance_reason == "contestó y terminó"
