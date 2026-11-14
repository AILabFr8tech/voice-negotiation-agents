"""The WER reference derived from the clips, and its two exclusion rules.

What these tests have to uphold, ordered by how much it would hurt if it failed
silently:

1. **The dead tail is distinguished from a real deletion.** A clip that played
   after the conversation had already ended is not the same as a clip that
   played and the STT did not understand. The first is excluded, the second
   counts. Today there is no case of the second kind in the campaign; the day
   one appears, it has to count and not disappear.
2. **Pairing intersects by `clip_id`, not by turn name.** With divergent
   branches the two PoCs reach the same `T5` through different clips, and
   pairing by name would compare the WER over two texts that are not the same.
3. **An excluded turn takes its STT group with it.** The hypothesis is
   distributed by position; if a turn drops out in the middle and its group
   stays, everything that comes after is shifted against the wrong reference.

See `docs/METRICS_CONTRACT.md` §2.10.
"""

from __future__ import annotations

import pytest

from shared.metrics import reference as ref
from shared.metrics.clips_reference import (
    DEAD_TAIL_REASON,
    OUT_OF_PAIR_REASON,
    ClipRun,
    PlayedTurn,
    match_pairs,
    classify_tail,
    common_clips,
    generate,
    read_run,
    trim_to_pair,
    log_turns,
    verify_against_manifest,
)


def _t(turn: str, clip: str, linea: str, *, stt: str = "", agent: str = "") -> PlayedTurn:
    return PlayedTurn(
        turn=turn, clip_id=clip, variant="base", linea=linea, stt=stt, agent=agent
    )


def _run(poc: int, turns: list[PlayedTurn], *, rep: str = "1") -> ClipRun:
    alive_ones, excluded_clips = classify_tail(turns)
    return ClipRun(
        clave=f"poc{poc}", poc=poc, scenario=2, repetition=rep,
        turns=alive_ones, excluded_clips=excluded_clips,
    )


def _speech(text: str, t: float, role: str) -> dict:
    return {"t": t, "role": role, "text": text}


# --------------------------------------------------------------------------
# Rule 1: the dead tail
# --------------------------------------------------------------------------


class TestDeadTail:
    def test_tail_turn_without_stt_and_without_agent_is_excluded(self) -> None:
        """The measured case: 45 turns, all 45 from PoC1, all 45 at the end.

        The clip played (it is visible in the WAV) but 17 s after the agent
        closed and with 13 s of nothing but comfort noise in between.
        """
        turns = [
            _t("T1", "c1", "hola", stt="hola", agent="qué tal"),
            _t("T2", "c2", "sí, así quedamos"),
        ]
        alive_ones, excluded_clips = classify_tail(turns)
        assert [t.turn for t in alive_ones] == ["T1"]
        assert [e.turn.turn for e in excluded_clips] == ["T2"]
        assert excluded_clips[0].motivo == DEAD_TAIL_REASON
        assert excluded_clips[0].indice == 1

    def test_without_stt_but_with_the_agent_speaking_it_is_a_real_deletion_and_COUNTS(self) -> None:
        """The distinction that needs to exist even though it is not used today.

        The STT had its chance (the agent answered, so the conversation was
        still alive) and understood nothing. That is exactly what the WER must
        measure. Excluding it would hide the worst possible STT error: losing a
        whole turn.
        """
        turns = [
            _t("T1", "c1", "hola", stt="hola", agent="qué tal"),
            _t("T2", "c2", "dieciocho mil quinientos", agent="de acuerdo"),
        ]
        alive_ones, excluded_clips = classify_tail(turns)
        assert [t.turn for t in alive_ones] == ["T1", "T2"]
        assert excluded_clips == []

    def test_with_stt_but_no_agent_response_ALSO_counts(self) -> None:
        """The mirror case of the previous one, and it is also in the campaign:
        5 turns, all 5 from PoC2.

        The clip played, the STT transcribed it, and the agent no longer
        answered. The transcript exists and carries real errors (one of those 5
        lost "pero de ahí no me muevo" entirely), so discarding it would mean
        erasing deletions, and from a single PoC at that. The cutoff condition
        is a conjunction precisely so that this case survives.
        """
        turns = [
            _t("T1", "c1", "hola", stt="hola", agent="qué tal"),
            _t("T2", "c2", "le bajo a veinticinco mil, pero de ahí no me muevo",
               stt="le bajo a veinticinco mil"),
        ]
        alive_ones, excluded_clips = classify_tail(turns)
        assert [t.turn for t in alive_ones] == ["T1", "T2"]
        assert excluded_clips == []

    def test_a_mute_turn_in_the_MIDDLE_is_not_tail_and_stays(self) -> None:
        """Trimming only happens from the end. In the middle, a mute turn is a
        failure and has to show up as such, not disappear from the denominator."""
        turns = [
            _t("T1", "c1", "hola", stt="hola", agent="qué tal"),
            _t("T2", "c2", "silent"),
            _t("T3", "c3", "adiós", stt="adiós", agent="hasta luego"),
        ]
        alive_ones, excluded_clips = classify_tail(turns)
        assert [t.turn for t in alive_ones] == ["T1", "T2", "T3"]
        assert excluded_clips == []

    def test_trims_several_tail_turns_not_just_the_last(self) -> None:
        turns = [
            _t("T1", "c1", "hola", stt="hola", agent="qué tal"),
            _t("T2", "c2", "uno"),
            _t("T3", "c3", "dos"),
        ]
        alive_ones, excluded_clips = classify_tail(turns)
        assert [t.turn for t in alive_ones] == ["T1"]
        assert [e.indice for e in excluded_clips] == [1, 2]

    def test_an_entirely_mute_run_does_not_blow_up(self) -> None:
        alive_ones, excluded_clips = classify_tail([_t("T1", "c1", "hola")])
        assert alive_ones == []
        assert len(excluded_clips) == 1

    def test_without_turns(self) -> None:
        assert classify_tail([]) == ([], [])


# --------------------------------------------------------------------------
# Rule 2: pairing
# --------------------------------------------------------------------------


class TestMatching:
    def test_intersects_by_clip_not_by_turn_name(self) -> None:
        """The divergent-branches case: 6 pairs in the campaign.

        Both PoCs reach `T5`, but one through `alt1` and the other through
        `alt2`. They are different phrases. Pairing by turn name would measure
        PoC1's WER against the text that played in PoC2.
        """
        a = _run(1, [
            _t("T4", "e2_t4", "cuatro", stt="cuatro", agent="ok"),
            _t("T5", "e2_t5_alt1", "por aquí", stt="por aquí", agent="ok"),
            _t("T6", "e2_t6", "seis", stt="seis", agent="ok"),
        ])
        b = _run(2, [
            _t("T4", "e2_t4", "cuatro", stt="cuatro", agent="ok"),
            _t("T5", "e2_t5_alt2", "por allá", stt="por allá", agent="ok"),
            _t("T6", "e2_t6", "seis", stt="seis", agent="ok"),
        ])
        commons = common_clips([a, b])
        assert commons == frozenset({"e2_t4", "e2_t6"})
        assert "e2_t5_alt1" not in commons and "e2_t5_alt2" not in commons

    def test_both_runs_end_up_with_the_SAME_reference_text(self) -> None:
        """The whole point of pairing, checked on the final text."""
        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2_alt1", "dos por aquí", stt="dos", agent="ok"),
            _t("T3", "c3", "tres", stt="tres", agent="ok"),
        ])
        b = _run(2, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2_alt2", "dos por allá", stt="dos", agent="ok"),
        ])
        commons = common_clips([a, b])
        ra = trim_to_pair(a, commons)
        rb = trim_to_pair(b, commons)
        assert [t.linea for t in ra.turns] == [t.linea for t in rb.turns] == ["uno"]

    def test_trim_does_not_touch_the_original_run(self) -> None:
        """The unpaired count is still needed: it is the coverage table."""
        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2", "dos", stt="dos", agent="ok"),
        ])
        antes = list(a.turns)
        trim_to_pair(a, frozenset({"c1"}))
        assert a.turns == antes

    def test_the_trim_preserves_the_original_index_of_the_dropped_turn(self) -> None:
        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2", "dos", stt="dos", agent="ok"),
            _t("T3", "c3", "tres", stt="tres", agent="ok"),
        ])
        r = trim_to_pair(a, frozenset({"c1", "c3"}))
        dropped = [e for e in r.excluded_clips if e.motivo == OUT_OF_PAIR_REASON]
        assert [(e.indice, e.turn.turn) for e in dropped] == [(1, "T2")]

    def test_common_clips_without_runs(self) -> None:
        assert common_clips([]) == frozenset()

    def test_pairing_groups_by_scenario_and_repetition(self) -> None:
        a = _run(1, [_t("T1", "c1", "uno", stt="uno", agent="ok")], rep="3")
        b = _run(2, [_t("T1", "c1", "uno", stt="uno", agent="ok")], rep="3")
        c = _run(1, [_t("T1", "c1", "uno", stt="uno", agent="ok")], rep="4")
        pairs = match_pairs([a, b, c])
        assert sorted(pairs) == [(2, "3"), (2, "4")]
        assert len(pairs[(2, "3")]) == 2 and len(pairs[(2, "4")]) == 1


# --------------------------------------------------------------------------
# File generation
# --------------------------------------------------------------------------


class TestGenerate:
    def _transcript(self, *turns: str) -> list[dict]:
        rows: list[dict] = []
        t = 0.0
        for text in turns:
            rows.append(_speech(text, t, ref.CARRIER_ROLE))
            t += 1.0
            rows.append(_speech("ajá", t, ref.AGENT_ROLE))
            t += 1.0
        return rows

    def test_excluded_turn_IN_THE_MIDDLE_takes_its_stt_group(self) -> None:
        """Otherwise everything that comes after is compared against the
        neighboring phrase.

        It is the most expensive failure in this module, because it does not
        blow up: it produces a believable, wrong WER.
        """
        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2", "dos", stt="dos", agent="ok"),
            _t("T3", "c3", "tres", stt="tres", agent="ok"),
        ])
        r = trim_to_pair(a, frozenset({"c1", "c3"}))
        text = generate(r, transcript=self._transcript("uno", "dos", "tres"))
        file = ref.parse(text)
        assert file.reference_text == "uno tres"
        # The key point: "tres" ended up against "tres", not against "dos".
        assert file.hypothesis_text == "uno tres"

    def test_excluded_tail_turn_contributes_neither_reference_nor_hypothesis(self) -> None:
        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2", "así quedamos"),
        ])
        text = generate(a, transcript=self._transcript("uno"))
        file = ref.parse(text)
        assert file.reference_text == "uno"
        assert "así quedamos" not in file.reference_text
        assert DEAD_TAIL_REASON in text

    def test_the_leftover_stt_groups_stay(self) -> None:
        """Fragmentation: the STT produced more groups than there were turns.
        They are real output and count toward the channel WER, even though they
        do not match a turn."""
        a = _run(1, [_t("T1", "c1", "uno", stt="uno", agent="ok")])
        text = generate(a, transcript=self._transcript("uno", "de más"))
        file = ref.parse(text)
        assert file.reference_text == "uno"
        assert "de más" in file.hypothesis_text

    def test_the_yo_lines_come_from_the_clip_and_not_from_the_stt(self) -> None:
        """What defines the method: the reference is what played, not what was
        understood. If it came from the STT, the WER would be 0 by construction."""
        a = _run(1, [_t("T1", "c1", "dieciocho mil", stt="18 mil", agent="ok")])
        file = ref.parse(generate(a, transcript=self._transcript("18 mil")))
        assert file.reference_text == "dieciocho mil"
        assert file.hypothesis_text == "18 mil"

    def test_declares_that_its_hypothesis_is_the_files(self) -> None:
        a = _run(1, [_t("T1", "c1", "uno", stt="uno", agent="ok")])
        file = ref.parse(generate(a, transcript=self._transcript("uno")))
        assert file.metadata["hypothesis"] == ref.FILE_HYPOTHESIS

    def test_the_text_of_the_excluded_turn_does_not_return_as_INSERTION(self) -> None:
        """The declaration above, checked end to end on the WER.

        `llamadas._wer_del_canal` takes the hypothesis from the JSON by default
        (the `stt:` lines of a hand-corrected file might have been edited). If a
        derived file did not declare its provenance, the text of the excluded
        turns would come back in through the JSON and be counted as
        insertions.
        """
        from shared.metrics.calls import compute

        a = _run(1, [
            _t("T1", "c1", "uno", stt="uno", agent="ok"),
            _t("T2", "c2", "dos", stt="dos", agent="ok"),
        ])
        r = trim_to_pair(a, frozenset({"c1"}))
        transcript = self._transcript("uno", "dos")
        file = ref.parse(generate(r, transcript=transcript))

        data = {
            "call_sid": "CA1",
            "summary": {"resultado": {}},
            # The JSON does carry "dos": it is real STT output, but from a turn
            # that is not part of this comparison.
            "transcript": transcript,
        }
        canal = (compute(data, reference_file=file).stt or {})["full_channel"]
        assert canal["wer"]["insertions"] == 0, "el turno excluido volvió por el JSON"
        assert canal["wer"]["sample_rate"] == 0.0

    def test_leaves_the_manual_notes_empty(self) -> None:
        """They are the only thing that is still manual; making them up would be
        worse than leaving them pending."""
        a = _run(1, [_t("T1", "c1", "uno", stt="uno", agent="ok")])
        file = ref.parse(generate(a, transcript=self._transcript("uno")))
        assert file.notes.naturalness is None
        assert file.notes.turn_taking is None

    def test_writes_the_metadata_the_report_needs(self) -> None:
        a = ClipRun(
            clave="k", poc=2, scenario=5, repetition="7", call_sid="CA9",
            turns=[_t("T1", "c1", "uno", stt="uno", agent="ok")],
        )
        file = ref.parse(generate(a, transcript=self._transcript("uno")))
        assert file.metadata["poc"] == "2"
        assert file.metadata["scenario"] == "5"
        assert file.metadata["repetition"] == "7"
        assert file.metadata["call_sid"] == "CA9"

    def test_two_exclusions_with_the_same_index_do_not_pass_silently(self) -> None:
        from shared.metrics.clips_reference import ExcludedTurn

        t1 = _t("T1", "c1", "uno")
        mala = ClipRun(
            clave="k", poc=1, scenario=1, repetition="1", turns=[],
            excluded_clips=[
                ExcludedTurn(turn=t1, indice=0, motivo="a"),
                ExcludedTurn(turn=t1, indice=0, motivo="b"),
            ],
        )
        with pytest.raises(ValueError, match="mismo índice"):
            generate(mala)


# --------------------------------------------------------------------------
# Reading and cross-verification
# --------------------------------------------------------------------------


class TestReading:
    def test_reads_the_log_and_trims_the_tail_at_once(self) -> None:
        bit = {
            "scenario": 3,
            "turns": [
                {"turn": "T1", "clip_id": "c1", "variant": "base",
                 "played_line": "uno", "stt": "uno", "agent": "ok"},
                {"turn": "T2", "clip_id": "c2", "variant": "alt1",
                 "played_line": "dos", "stt": "", "agent": ""},
            ],
        }
        c = read_run(bit, clave="k", poc=1, repetition="2", call_sid="CA1")
        assert c.scenario == 3 and c.pair == (3, "2")
        assert [t.turn for t in c.turns] == ["T1"]
        assert [e.motivo for e in c.excluded_clips] == [DEAD_TAIL_REASON]

    def test_log_turns_tolerates_missing_fields(self) -> None:
        turns = log_turns({"turns": [{"turn": "T1"}]})
        assert turns[0].linea == "" and turns[0].stt == ""

    def test_the_cross_verification_against_the_manifest_passes_when_they_match(self) -> None:
        """Two independent records of the same fact: what the harness logged
        when playing and what the clip generator logged when synthesizing."""
        a = _run(1, [_t("T1", "c1", "dieciocho mil", stt="x", agent="ok")])
        assert verify_against_manifest(a, {"c1": "dieciocho mil"}) == []

    def test_the_cross_verification_detects_a_wrong_reference(self) -> None:
        a = _run(1, [_t("T1", "c1", "dieciocho mil", stt="x", agent="ok")])
        problems = verify_against_manifest(a, {"c1": "diecinueve mil"})
        assert len(problems) == 1 and "T1" in problems[0]

    def test_the_cross_verification_detects_an_unknown_clip(self) -> None:
        a = _run(1, [_t("T1", "cX", "uno", stt="x", agent="ok")])
        problems = verify_against_manifest(a, {"c1": "uno"})
        assert len(problems) == 1 and "no está en el manifiesto" in problems[0]
