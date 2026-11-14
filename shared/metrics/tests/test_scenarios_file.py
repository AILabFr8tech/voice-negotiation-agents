"""Tests for the script as data.

`TestSinDeriva` is the one that justifies having the script twice (prose + YAML):
without this test, changing a line in the .md and forgetting it in the .yaml
would be discovered in the middle of the 15 calls, when the reference file comes
seeded with a text that nobody says anymore.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.metrics.scenarios import SCENARIOS_PATH, load_script

PROSE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "negotiation_engine"
    / "tests"
    / "script_scenarios.md"
)


@pytest.fixture(scope="module")
def script():
    return load_script()


class TestCarga:
    def test_the_file_exists(self) -> None:
        assert SCENARIOS_PATH.is_file()

    def test_five_scenarios(self, script) -> None:
        assert [e.id for e in script.scenarios] == [1, 2, 3, 4, 5]

    def test_names(self, script) -> None:
        assert script.scenario(1).name == "Cooperativo"
        assert script.scenario(3).name == "Agresivo / ancla alta"

    def test_trip_config(self, script) -> None:
        assert script.trip["target_price"] == 18_000
        assert script.trip["max_price"] == 21_000
        assert script.trip["carrier_name"] == "Carlos"

    def test_all_scenarios_have_turns(self, script) -> None:
        for scenario in script.scenarios:
            assert scenario.turns, f"escenario {scenario.id} sin turnos"

    def test_nonexistent_scenario_raises(self, script) -> None:
        with pytest.raises(KeyError):
            script.scenario(99)

    def test_alternate_path(self, tmp_path: Path) -> None:
        destination = tmp_path / "otro.yaml"
        destination.write_text(
            "version: prueba\nscenarios:\n"
            "  - id: 7\n    name: Inventado\n    turns:\n"
            "      - id: T1\n        linea: hola\n",
            encoding="utf-8",
        )
        otro = load_script(destination)
        assert otro.version == "prueba"
        assert otro.scenario(7).turns[0].linea == "hola"

    def test_invalid_file(self, tmp_path: Path) -> None:
        destination = tmp_path / "empty.yaml"
        destination.write_text("version: x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="scenarios"):
            load_script(destination)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_script(tmp_path / "does_not_exist.yaml")


class TestTurns:
    def test_eight_turns_in_the_negotiation_scenarios(self, script) -> None:
        """~8 turns per call is what gives density of latency data."""
        for scenario_key in (1, 2, 3, 4):
            assert len(script.scenario(scenario_key).turns) == 8

    def test_no_scenario_has_silences_or_hang_up(self, script) -> None:
        """The campaign's 5 scenarios all close with speech.

        Scenario 5 used to be "channel failure" and had two silence turns and a
        hang-up. It was replaced by the recognition-stress one, and this test
        stays as a guard: failure handling is app correctness, not a comparison
        metric, and it must not slip back into the script by accident.
        """
        for scenario in script.scenarios:
            assert not [t for t in scenario.turns if t.is_silence], scenario.label
            assert not [t for t in scenario.turns if t.hangs_up], scenario.label
            assert not [
                t for t in scenario.turns if t.silence_s is not None
            ], scenario.label

    def test_branch_alternatives(self, script) -> None:
        turn = next(t for t in script.scenario(2).turns if t.id == "T5")
        assert len(turn.alternates) == 2

    def test_reference_text_includes_the_spoken_lines(self, script) -> None:
        text = script.scenario(5).reference_text()
        assert "Carlos" in text
        # All 8 turns of the scenario are spoken: none is silence.
        assert len(text.splitlines()) == 8

    def test_reference_text_omits_silences(self, tmp_path: Path) -> None:
        """The loader still supports silence turns, even though the script does not use them.

        It is tested with its own script: piggybacking on scenario 5 is what tied
        this coverage to the old shape of the script.
        """
        destination = tmp_path / "con_silence.yaml"
        destination.write_text(
            "version: prueba\nscenarios:\n"
            "  - id: 9\n    name: Silencio\n    turns:\n"
            "      - id: T1\n        linea: Sí, soy Carlos.\n"
            '      - id: T2\n        linea: ""\n        silence_s: 8\n',
            encoding="utf-8",
        )
        text = load_script(destination).scenario(9).reference_text()
        assert text.splitlines() == ["Sí, soy Carlos."]

    def test_the_script_amounts_can_be_extracted(self, script) -> None:
        """The script says the amounts on purpose both in words and in digits."""
        from shared.metrics.text import extract_amounts

        text = script.scenario(2).reference_text()
        amounts = extract_amounts(text)
        assert 18_000.0 in amounts

        text3 = script.scenario(3).reference_text()
        # Scenario 3 anchors well above the 21k ceiling.
        assert any(m > 21_000 for m in extract_amounts(text3))


class TestNoDrift:
    """The YAML lines have to keep existing in the prose script."""

    def test_the_prose_script_exists(self) -> None:
        assert PROSE_SCRIPT_PATH.is_file()

    def test_each_line_is_in_the_prose(self, script) -> None:
        # Whitespace is collapsed on both sides: the .md has lines wrapped at
        # the column width.
        prose = " ".join(PROSE_SCRIPT_PATH.read_text(encoding="utf-8").split())

        absent: list[str] = []
        for scenario in script.scenarios:
            for turn in scenario.turns:
                candidate_ones = [turn.linea, *turn.alternates]
                for linea in candidate_ones:
                    if not linea.strip():
                        continue
                    if " ".join(linea.split()) not in prose:
                        absent.append(f"{scenario.id}/{turn.id}: {linea}")

        assert not absent, (
            "estas líneas del YAML ya no están en guion_escenarios.md "
            "(deriva entre las dos copias):\n  " + "\n  ".join(absent)
        )

    def test_the_trip_config_matches(self, script) -> None:
        prose = PROSE_SCRIPT_PATH.read_text(encoding="utf-8")
        assert "18,000 MXN" in prose
        assert "21,000 MXN" in prose
        assert script.trip["target_price"] == 18_000
        assert script.trip["max_price"] == 21_000
