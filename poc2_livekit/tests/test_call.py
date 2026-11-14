"""Dialer rules: which runs are rejected before spending on the call.

No test touches the network or the `.env`: `verify_campaign_run` is a pure
function over the arguments and the script on disk.
"""

from __future__ import annotations

import pytest

from poc2_livekit.app.call import (
    REJECTED_OUTPUT,
    verify_campaign_run,
)
from shared.metrics.scenarios import load_script


class TestCampaignRun:
    @pytest.mark.parametrize(
        "scenario", [e.id for e in load_script().scenarios]
    )
    def test_the_five_scenarios_of_the_script_are_valid(self, scenario: int):
        assert verify_campaign_run(scenario) == ""

    def test_a_scenario_outside_the_script_is_rejected(self):
        """A typo in the number would produce a run labeled as campaign that
        matches no script, and it would be discovered when consolidating."""
        error = verify_campaign_run(9)
        assert "no existe el escenario 9" in error

    def test_scenario_zero_is_rejected(self):
        assert verify_campaign_run(0)


class TestDebugRun:
    def test_without_scenario_nothing_is_validated(self):
        """Without `--scenario` the run does not go into the consolidated results: nothing to validate."""
        assert verify_campaign_run(None) == ""


def test_the_rejection_code_is_distinguished_from_the_config_one():
    """So that the 15 runs can be scripted and the two failures told apart."""
    assert REJECTED_OUTPUT == 2
