"""PoC3's harness, end to end and without spending a cent.

The lesson of `TestBridgeConstruction` and `TestTheFullTextPathAgainstTheHarness`:
the pieces can all be green separately and still not fit together. Here they
run together —the Media Streams client, the player, the script from `shared/`,
the **production** bridge with its text echo, its anchored clock and its
recorder— against the fake alpha in `conftest.py`.

What this protects is what cannot be tested piece by piece: that the text the
bridge emits reaches the harness and is reassembled, that the audio makes the
full round trip and is heard as voice on this side, and that the run leaves the
three artifacts with the same key.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from poc3_gpt_live.app import harness, media_streams as ms
from poc3_gpt_live.app.marks import marks_de, zeros_path
from poc3_gpt_live.tests.conftest import ASSEMBLED_RESPONSE
from shared.carrier_simulator import RUTA_CLIPS_DEFAULT

# The quote measures scenario 1's clips, and the `.ulaw` files do not travel in
# the repo (they are a person's voice): without them these tests have nothing to measure.
with_script_clips = pytest.mark.skipif(
    not (RUTA_CLIPS_DEFAULT / "e1_t1.ulaw").is_file(),
    reason="sin los clips del guion en esta máquina",
)


class TestTheAudioMakesTheFullLoop:
    def test_the_clips_went_out_over_the_wire(self, run) -> None:
        observed, *_ = run
        assert observed.published_frames > 0
        assert observed.frames_without_destination == 0

    def test_the_agent_voice_was_heard_on_this_side(self, run) -> None:
        """Full round trip: the alpha's PCM24 -> μ-law -> the harness measures it
        as voice. If the resampling broke, this comes out as 0."""
        observed, *_ = run
        assert observed.audio_frames > 0
        assert observed.speech_streaks, "el arnés no midió una sola racha de voz"
        assert observed.t_agent_first_speech is not None

    def test_the_script_advanced_to_the_end(self, run) -> None:
        observed, *_ = run
        assert [t.step.turn for t in observed.turns] == ["T1", "T2"]


class TestTheTextArrivesAndIsReassembled:
    def test_the_run_is_not_deaf(self, run) -> None:
        observed, *_ = run
        assert observed.deaf_to_text is False
        assert observed.text_block()["eligible"] is True

    def test_the_agent_is_reassembled_without_gluing_words(self, run) -> None:
        """The alpha's separator goes at the start of the fragment: concatenating
        is correct and re-inserting spaces would be the mistake."""
        observed, *_ = run
        text = observed.assemble_agent(observed.agent_fragments())
        assert ASSEMBLED_RESPONSE in text
        assert "Claro,con" not in text

    def test_each_turn_keeps_what_the_agent_answered_it(
        self, run
    ) -> None:
        observed, *_ = run
        replies = [t.response for t in observed.turns]
        assert all(r.strip() for r in replies), replies


class TestTheThreeArtifactsShareAKey:
    def test_the_zeros_were_written_with_the_call_sid(self, run) -> None:
        _, call_sid, folder, _ = run
        ruta = zeros_path(call_sid, folder=folder)

        assert ruta.is_file()
        assert [m.turn for m in marks_de(ruta)] == ["T1", "T2"]

    def test_the_bridge_left_its_json_and_its_census_with_the_same_key(
        self, run
    ) -> None:
        _, call_sid, _, recordings = run

        assert (recordings / f"call_{call_sid}.json").is_file()
        assert (recordings / f"census_{call_sid}.jsonl").is_file()

    def test_the_bridge_json_declares_the_echo_on_and_fit(
        self, run
    ) -> None:
        _, call_sid, _, recordings = run
        content = json.loads(
            (recordings / f"call_{call_sid}.json").read_text(encoding="utf-8")
        )
        echo = content["text_echo"]

        assert echo["enabled"] is True
        assert echo["emitted_fragments"] > 0
        assert echo["eligible"] is True

    def test_both_sides_count_the_same_fragments(self, run) -> None:
        """The bridge says how many it emitted and the harness how many it received.
        If they do not match, what failed is the transport and not the model."""
        observed, call_sid, _, recordings = run
        content = json.loads(
            (recordings / f"call_{call_sid}.json").read_text(encoding="utf-8")
        )

        emitted = content["text_echo"]["emitted_fragments"]
        assert len(observed.fragments) == emitted

    def test_the_bridge_clock_came_out_interpretable(self, run) -> None:
        """The whole of step 3, exercised: anchor, R measured at opening and at
        hang-up, and the drift below its threshold."""
        _, call_sid, _, recordings = run
        content = json.loads(
            (recordings / f"call_{call_sid}.json").read_text(encoding="utf-8")
        )
        clock = content["clock"]

        assert clock["interpretable"] is True
        assert clock["effective_resolution"]["on_open_ms"] > 0
        assert clock["effective_resolution"]["on_hang_up_ms"] is not None


class TestTheRunLog:
    def test_the_left_queue_leaves_a_trace_of_each_streak(self, run) -> None:
        """With the queue silent it cannot be audited: the block is always included."""
        observed, *_ = run
        block = observed.left_tail_block()

        assert block["left_tail_s"] == ms.TEXT_LEFT_TAIL_S
        assert block["evaluated_streaks"] > 0
        assert "times_decided" in block

    def test_the_branches_are_published_even_if_the_script_is_linear(self, run) -> None:
        observed, *_ = run
        branches = observed.branches_block()

        assert branches["deaf_to_text"] is False
        assert branches["branched_turns"] == len(observed.decisions)


class TestWhatCanBeAskedWithoutStartingAnything:
    def test_the_agent_hang_up_cuts_the_script(self) -> None:
        observed = ms.ObservedFromPoC3(silent=True)
        closed = asyncio.Event()

        harness.annotate_hang_up(observed, closed)("el agente cerró")

        assert closed.is_set()
        assert observed.was_closed_by_agent is True
        assert observed.close_reason == "el agente cerró"

    def test_warning_twice_does_not_rewrite_the_reason(self) -> None:
        """The client may report both a clean close and an exception."""
        observed = ms.ObservedFromPoC3(silent=True)
        closed = asyncio.Event()
        hung_up = harness.annotate_hang_up(observed, closed)

        hung_up("el agente cerró")
        hung_up("se cayó el socket")

        assert observed.close_reason == "el agente cerró"

    def test_the_cost_comes_from_the_table_and_is_computed_per_second(self) -> None:
        """$0.05/min billed per second, plus the delegate at its mean."""
        assert harness.VOICE_COST_PER_MINUTE == 0.05
        # 150.8 s is PoC3's measured mean duration in `docs/costos_poc3.md`.
        assert harness.estimate_cost(150.8) == pytest.approx(0.1282, abs=0.0001)
        # And it is not rounded up to the minute: 150.8 s does not cost the same as 180.
        assert harness.estimate_cost(150.8) < harness.estimate_cost(180.0)

    def test_the_cli_accepts_what_the_batch_sends_it(self) -> None:
        """`shared/lote/executor.call_command` builds exactly this."""
        args = harness.build_analyzer().parse_args(
            ["--scenario", "2", "--repetition", "3", "--accept-cost"]
        )

        assert (args.scenario, args.repetition, args.accept_cost) == (2, 3, True)

    @with_script_clips
    def test_without_accept_cost_nothing_runs(self, capsys, monkeypatch) -> None:
        """The quote is the $0 path, and it is the one the batch uses first."""
        # Quoting does not touch the network, but it loads the settings: with the
        # fake key the test does not depend on the `.env` of whoever runs it.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
        code_value = harness.main(["--scenario", "1"])
        output = capsys.readouterr().out

        assert code_value == 0
        assert "Vuelve a correrlo con --accept-cost" in output
        assert "docs/costos_poc3.md" in output

    def test_asking_for_scenario_and_clip_at_once_is_an_error(self, capsys) -> None:
        assert harness.main(["--scenario", "1", "--clip", "x.ulaw"]) == 1
        assert "and only one" in capsys.readouterr().out


class TestTheBatchInput:
    """The contract with `shared/lote/`: four flags and a small JSON.

    The batch quotes by running the harness **without** `--accept-cost` and
    reading the summary; with that it builds its cost table before touching
    anything. If these flags stopped existing, PoC3's campaign would fail call
    after call and the batch would report it as "the harness finished without
    leaving a summary".
    """

    def test_the_cli_accepts_the_exact_command_the_batch_builds(self, tmp_path) -> None:
        from shared.lote.executor import (
            BatchItem,
            call_command,
            default_descriptors,
        )

        command = call_command(
            default_descriptors(tmp_path)[3],
            BatchItem(indice=1, poc=3, scenario=4, repetition=2),
            summary_path=tmp_path / "r.json",
            accept_cost=True,
        )
        # The interpreter and the `-m module` are stripped from the command: what
        # the CLI sees are the arguments.
        args = harness.build_analyzer().parse_args(command[3:])

        assert (args.scenario, args.repetition) == (4, 2)
        assert args.accept_cost is True
        assert args.summary == str(tmp_path / "r.json")

    @with_script_clips
    def test_the_quote_leaves_the_estimate_and_spends_nothing(self, tmp_path, monkeypatch) -> None:
        destination = tmp_path / "summary.json"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")

        code_value = harness.main(["--scenario", "1", "--summary", str(destination)])
        summary = json.loads(destination.read_text(encoding="utf-8"))

        assert code_value == 0
        assert summary["estimate_only"] is True
        assert summary["poc"] == 3
        assert summary["estimated_usd"] == pytest.approx(
            harness.estimate_cost(summary["seconds_cap"]), abs=0.001
        )

    @with_script_clips
    def test_the_campaign_estimate_fits_within_the_cap(self, tmp_path, monkeypatch) -> None:
        """86 calls at the upper bound against the table's $25 cap.

        The harness's upper bound is more pessimistic than the measured duration
        (it adds up the maximum waits of every turn), so if it fits with that,
        it fits.
        """
        destination = tmp_path / "summary.json"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
        harness.main(["--scenario", "1", "--summary", str(destination)])
        estimated = json.loads(destination.read_text(encoding="utf-8"))["estimated_usd"]

        assert 86 * estimated < 25.00

    def test_a_failed_check_counts_as_a_failed_criterion(self) -> None:
        """If it did not add up, the batch would accept the call whose clock was
        not interpretable — which is exactly the one that must not enter the cross-check."""
        revision = {
            "reviews": [
                {"que": "sample_reconciliation", "ok": True},
                {"que": "interpretable_clock", "ok": False},
                {"que": "text_echo", "ok": False},
            ]
        }

        assert harness.unverified(revision) == 2
        assert harness.unverified({"reviews": []}) == 0
        assert harness.unverified({}) == 0
