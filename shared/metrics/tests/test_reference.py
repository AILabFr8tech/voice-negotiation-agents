"""Tests for the reference file: it is generated pre-seeded and read back corrected.

`TestParseoDeEncabezados` covers a real regression: since `"## T1"` also starts
with `#`, discarding comments before recognizing headers collapsed the 8 turns
into one and the notes section was never opened. The full-channel WER came out
right by chance, but the manual scores were lost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.metrics import reference as ref
from shared.metrics.scenarios import load_script


@pytest.fixture(scope="module")
def script():
    return load_script()


@pytest.fixture
def transcript() -> list[dict[str, object]]:
    """Two carrier turns separated by one agent turn."""
    return [
        {"t": 1.0, "role": ref.AGENT_ROLE, "text": "¿Hablo con Carlos?"},
        {"t": 3.0, "role": ref.CARRIER_ROLE, "text": "Sí, claro"},
        {"t": 4.0, "role": ref.CARRIER_ROLE, "text": "que sí."},
        {"t": 8.0, "role": ref.AGENT_ROLE, "text": "Le ofrezco 16,000."},
        {"t": 12.0, "role": ref.CARRIER_ROLE, "text": "está muy bajo"},
    ]


class TestTurnGrouping:
    def test_cuts_where_the_agent_speaks(self, transcript) -> None:
        """The agent's turn is the real signal of a turn change."""
        groups = ref.group_carrier_turns(transcript)
        assert groups == ["Sí, claro | que sí.", "está muy bajo"]

    def test_without_agent_turns_it_cuts_by_gap(self) -> None:
        """Fallback for calls recorded before the agent annotator existed."""
        transcript = [
            {"t": 1.0, "role": ref.CARRIER_ROLE, "text": "uno"},
            {"t": 2.0, "role": ref.CARRIER_ROLE, "text": "dos"},
            {"t": 30.0, "role": ref.CARRIER_ROLE, "text": "tres"},
        ]
        groups = ref.group_carrier_turns(transcript, hole=4.0)
        assert groups == ["uno | dos", "tres"]

    def test_ignores_empty_text(self) -> None:
        transcript = [
            {"t": 1.0, "role": ref.CARRIER_ROLE, "text": "  "},
            {"t": 2.0, "role": ref.CARRIER_ROLE, "text": "algo"},
        ]
        assert ref.group_carrier_turns(transcript) == ["algo"]

    def test_empty_transcript(self) -> None:
        assert ref.group_carrier_turns([]) == []


class TestGeneration:
    def test_seeds_the_script_lines(self, script, transcript) -> None:
        scenario = script.scenario(1)
        content = ref.generate(
            scenario=scenario, transcript=transcript, poc=1, repetition=1
        )
        # The first line of scenario 1 has to come already written.
        assert f"{ref.REFERENCE_PREFIX} Sí, soy Carlos. Dígame." in content

    def test_includes_the_stt_fragments(self, script, transcript) -> None:
        content = ref.generate(
            scenario=script.scenario(1), transcript=transcript
        )
        assert "Sí, claro | que sí." in content

    def test_brings_the_alternatives_as_a_comment(self, script) -> None:
        """The script branches are offered without getting in the parser's way."""
        content = ref.generate(scenario=script.scenario(2), transcript=[])
        assert "#   alt:" in content

    def test_marks_the_silence_turns(self, tmp_path: Path) -> None:
        """The campaign script no longer has silences; the generator still marks them.

        This used to be tested with scenario 5, back when it was "channel
        failure". With its own script, the coverage is no longer tied to the
        shape of the campaign script.
        """
        destination = tmp_path / "con_silence.yaml"
        destination.write_text(
            "version: prueba\nscenarios:\n"
            "  - id: 9\n    name: Silencio\n    turns:\n"
            "      - id: T1\n        linea: Sí, soy Carlos.\n"
            '      - id: T2\n        linea: ""\n        silence_s: 8\n',
            encoding="utf-8",
        )
        scenario = load_script(destination).scenario(9)
        content = ref.generate(scenario=scenario, transcript=[])
        assert "silence" in content
        assert "alucinación" in content

    def test_seeds_the_corrected_amount_of_scenario_five(self, script) -> None:
        """The mid-sentence self-correction has to reach the reference intact.

        It is the hard case for the WER: if seeding swallowed one of the two
        amounts, `error_montos` would measure against an incomplete reference.
        """
        content = ref.generate(scenario=script.scenario(5), transcript=[])
        assert "dieciocho mil… no, dieciocho mil quinientos" in content

    def test_without_scenario_it_leaves_the_stt_turns(self, transcript) -> None:
        content = ref.generate(scenario=None, transcript=transcript)
        assert "está muy bajo" in content
        assert "scenario: \n" in content or "scenario: " in content

    def test_warns_about_bias(self, script) -> None:
        """Copying the `stt:` into the `yo:` yields a false WER of 0: it must be warned about."""
        content = ref.generate(scenario=script.scenario(1), transcript=[])
        assert "no las copies" in content.lower() or "bias" in content.lower()

    def test_leftover_fragments_are_not_lost(self, script) -> None:
        """More STT groups than script turns: they go to a separate section."""
        many = []
        t = 0.0
        for i in range(12):
            many.append({"t": t, "role": ref.AGENT_ROLE, "text": f"agente {i}"})
            t += 1.0
            many.append(
                {"t": t, "role": ref.CARRIER_ROLE, "text": f"transportista {i}"}
            )
            t += 1.0
        content = ref.generate(scenario=script.scenario(1), transcript=many)
        assert ref.LEFTOVERS_HEADER in content
        assert "transportista 11" in content


class TestHeaderParsing:
    """Regression: `##` also starts with `#`."""

    def test_recognizes_all_the_turns(self, script, transcript) -> None:
        content = ref.generate(
            scenario=script.scenario(1), transcript=transcript
        )
        file = ref.parse(content)
        # Scenario 1 has 8 spoken turns, all with a seeded line.
        assert len(file.turns) == 8
        assert [t.id for t in file.turns] == [f"T{i}" for i in range(1, 9)]

    def test_reads_the_manual_notes(self) -> None:
        content = (
            "## T1\n"
            "yo: hola\n"
            f"{ref.NOTES_HEADER}\n"
            "#   comentario que se ignora\n"
            "naturalness: 4\n"
            "turn_taking: 3\n"
            "unwanted_interruptions: 2\n"
            "false_end_of_turn: 0\n"
        )
        file = ref.parse(content)
        assert file.notes.naturalness == 4
        assert file.notes.turn_taking == 3
        assert file.notes.unwanted_interruptions == 2
        assert file.notes.false_end_of_turn == 0
        assert file.notes.completed is True

    def test_empty_notes_stay_none(self) -> None:
        content = f"{ref.NOTES_HEADER}\nnaturalness: \nturn_taking: \n"
        file = ref.parse(content)
        assert file.notes.naturalness is None
        assert file.notes.completed is False


class TestParsing:
    def test_only_the_yo_lines_feed_the_calculation(self) -> None:
        content = (
            "## T1\n"
            "yo: esto sí cuenta\n"
            "stt: esto no cuenta\n"
            "# comentario tampoco\n"
        )
        file = ref.parse(content)
        assert file.reference_text == "esto sí cuenta"
        assert file.hypothesis_text == "esto no cuenta"

    def test_several_me_lines_are_concatenated(self) -> None:
        """This is how an improvisation is added without reordering the file."""
        content = "## T1\nyo: primera parte\nyo: y lo que improvisé\n"
        file = ref.parse(content)
        assert file.reference_text == "primera parte y lo que improvisé"

    def test_reads_metadata(self) -> None:
        content = "poc: 1\nscenario: 3\nrepetition: 2\ncall_sid: CA123\n"
        file = ref.parse(content)
        assert file.integer("poc") == 1
        assert file.integer("scenario") == 3
        assert file.metadata["call_sid"] == "CA123"

    def test_free_notes_do_not_break_the_parsing(self) -> None:
        content = (
            "## T1\n"
            "yo: la línea\n"
            f"{ref.FREE_HEADER}\n"
            "El agente tardó mucho en el primer turno.\n"
            "yo: esto de aquí abajo ya no es una referencia\n"
        )
        file = ref.parse(content)
        assert file.reference_text == "la línea"
        assert "tardó mucho" in file.notes.free

    def test_fragment_separator_is_undone(self) -> None:
        file = ref.parse("## T1\nstt: uno | dos | tres\n")
        assert file.hypothesis_text == "uno dos tres"


class TestRoundTrip:
    def test_generate_and_read(self, tmp_path: Path, script, transcript) -> None:
        content = ref.generate(
            scenario=script.scenario(2),
            transcript=transcript,
            poc=1,
            repetition=3,
            call_sid="CAtest",
            script_version="v1",
        )
        destination = ref.path_to_reference(tmp_path, "CAtest")
        ref.write(content, destination)

        file = ref.leer(destination)
        assert file.integer("poc") == 1
        assert file.integer("scenario") == 2
        assert file.integer("repetition") == 3
        assert file.metadata["call_sid"] == "CAtest"
        # The lines of script 2 come pre-seeded.
        assert "dieciocho mil" in file.reference_text


class TestDoNotLoseManualWork:
    def test_does_not_overwrite_by_default(self, tmp_path: Path) -> None:
        """Regenerating by accident would erase the 3 min of manual correction."""
        destination = tmp_path / "reference_CA1.txt"
        ref.write("original", destination)
        with pytest.raises(FileExistsError):
            ref.write("new", destination)
        assert destination.read_text(encoding="utf-8") == "original"

    def test_overwrites_if_requested(self, tmp_path: Path) -> None:
        destination = tmp_path / "reference_CA1.txt"
        ref.write("original", destination)
        ref.write("new", destination, overwrite=True)
        assert destination.read_text(encoding="utf-8") == "new"

    def test_search_finds_the_file(self, tmp_path: Path) -> None:
        assert ref.search(tmp_path, "CA1") is None
        ref.write("algo", ref.path_to_reference(tmp_path, "CA1"))
        assert ref.search(tmp_path, "CA1") is not None


class TestValidation:
    def test_without_correcting_it_warns(self) -> None:
        file = ref.parse("## T1\nyo: \nstt: algo que dijo el stt\n")
        warning_list = ref.validate(file)
        assert any("no se corrigió" in a for a in warning_list)

    def test_detects_copying_from_the_stt(self) -> None:
        """The only workflow error that gives a nice-looking, false number."""
        content = "## T1\nyo: texto idéntico\nstt: texto idéntico\n"
        warning_list = ref.validate(ref.parse(content))
        assert any("copiado el STT" in a for a in warning_list)

    def test_warns_about_missing_notes(self) -> None:
        warning_list = ref.validate(ref.parse("## T1\nyo: algo\n"))
        assert any("puntuaciones manuales" in a for a in warning_list)

    def test_warns_about_out_of_range_score(self) -> None:
        content = (
            "## T1\nyo: algo\n"
            f"{ref.NOTES_HEADER}\nnaturalness: 9\nturn_taking: 3\n"
        )
        warning_list = ref.validate(ref.parse(content))
        assert any("fuera del rango" in a for a in warning_list)

    def test_complete_file_does_not_warn(self) -> None:
        content = (
            "## T1\nyo: lo que dije\nstt: lo que entendio\n"
            f"{ref.NOTES_HEADER}\nnaturalness: 4\nturn_taking: 4\n"
        )
        assert ref.validate(ref.parse(content)) == []
