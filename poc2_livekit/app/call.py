"""Dials PoC2's outbound call. **The only module that spends on telephony.**

Two steps against the LiveKit API, in this order:

1. `CreateAgentDispatch` — puts **our** agent (by `agent_name`) into a new
   room, with the run label in the `metadata`.
2. `CreateSIPParticipant` — dials over the outbound trunk and puts the carrier
   into that same room.

The order matters: if it dialed first, the phone could ring and be answered
before the agent is inside, and the carrier would hear silence.

On the shared account (CLAUDE.md): this module **never** creates, modifies or
deletes trunks, dispatch rules or inbound rules. It only references the trunk
by id and creates ephemeral resources —a room, a dispatch, a SIP participant—
that die with the call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any, Final

from poc2_livekit.app.config import (
    CARRIER_IDENTITY,
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_livekit,
    verify_for_telephony,
)
from shared.metrics.scenarios import load_script

logger = logging.getLogger(__name__)

# Exit code when a run is rejected by policy, not because of incomplete config.
# It differs from 1 so that the 15 runs can be scripted and "something is
# missing in the .env" can be told apart from "this run is not allowed".
REJECTED_OUTPUT: Final[int] = 2


def verify_campaign_run(scenario: int | None) -> str:
    """Rules that only apply to scripted calls. Returns the error, or "".

    A campaign run is one that carries `--scenario`: its JSON goes into the
    consolidated results and into the technical decision.

    The scenario must exist in `shared/test_scenarios/`: a typo in the number
    would produce a run labeled as campaign that matches no script, and that
    is discovered when consolidating, too late.

    Nothing about audio is validated **here**, because PoC2 does not record.
    The WER is computed against the acted script: the `stt:` lines of the JSON
    are the hypothesis and the `yo:` lines the reference, which the person
    corrects by hand against what they said. It is the same method PoC1 was
    measured with.
    """
    if scenario is None:
        return ""

    try:
        load_script().scenario(scenario)
    except KeyError as error:
        # `str()` of a KeyError drags along the repr quotes: take the message
        # exactly as the script loader wrote it.
        return str(error.args[0]) if error.args else f"invalid scenario {scenario}"

    return ""

# Rates verified in `docs/costos_poc2.md`. They are repeated here so that the
# cost warning does not depend on someone opening the document.
TRUNK_COST_PER_MINUTE: Final[float] = 0.012
STT_COST_PER_MINUTE: Final[float] = 0.0077
TTS_COST_PER_CALL: Final[float] = 0.095  # midpoint of the ElevenLabs range
LLM_COST_PER_CALL: Final[float] = 0.008
# LiveKit Cloud (participant minutes + SIP) and turn detector inference are
# NOT yet verified against the account. See docs/costos_poc2.md.
ESTIMATED_LIVEKIT_COST_PER_MINUTE: Final[float] = 0.005

TYPICAL_DURATION_MIN: Final[float] = 4.0
DEFAULT_LIMIT_S: Final[int] = 600


def estimate_cost(minutes: float) -> float:
    """Approximate cost of a call lasting `minutes`. See docs/costos_poc2.md."""
    per_minute = (
        TRUNK_COST_PER_MINUTE
        + STT_COST_PER_MINUTE
        + ESTIMATED_LIVEKIT_COST_PER_MINUTE
    )
    return minutes * per_minute + TTS_COST_PER_CALL + LLM_COST_PER_CALL


def _print_cost_warning(settings: Settings, destination: str, limit_s: int) -> None:
    estimated = estimate_cost(TYPICAL_DURATION_MIN)
    tope = estimate_cost(limit_s / 60.0)
    print("Real outbound call over the LiveKit SIP trunk.\n")
    print(f"  destination      {destination}")
    print(f"  trunk            {settings.sip_outbound_trunk_id}")
    print(f"  project          {settings.livekit_url}")
    print(f"  agent            {settings.agent_name}")
    print(f"\n  estimate         ~${estimated:.2f} USD ({TYPICAL_DURATION_MIN:.0f} min)")
    print(f"  hard cap         ~${tope:.2f} USD ({limit_s} s)")
    print(
        "\n  LiveKit Cloud minutes and turn detector inference are an\n"
        "  UNVERIFIED ESTIMATE: confirm them with this first call\n"
        "  (docs/costos_poc2.md).\n"
    )
    print("Run it again with --accept-cost to dial.")


async def set_mark(
    settings: Settings,
    *,
    destination: str,
    room: str,
    identification: dict[str, Any],
    limit_s: int,
) -> int:
    """Dispatches the agent and dials. Returns 0 if the call was answered."""
    from livekit import api

    async with api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    ) as lk:
        # --- 1. The agent goes in first ---
        dispatch_result = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=settings.agent_name,
                room=room,
                metadata=json.dumps(identification, ensure_ascii=False),
            )
        )
        print(f"  agent dispatched: {getattr(dispatch_result, 'id', '?')} -> room {room}")

        # --- 2. Dial ---
        print(f"  dialing {destination} …")
        try:
            participant = await lk.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=settings.sip_outbound_trunk_id,
                    sip_call_to=destination,
                    room_name=room,
                    # The same identity the agent expects (config.py): it is
                    # how it knows whom to watch to find out the call was answered.
                    participant_identity=CARRIER_IDENTITY,
                    participant_name="Transportista",
                    # Blocks until answered or failed: without this the script
                    # would finish before knowing whether the call went through.
                    wait_until_answered=True,
                )
            )
        except api.TwirpError as error:
            print(f"\n  [failure] the call did not go through: {error}")
            return 1

        print(f"  answered: {getattr(participant, 'identity', '?')}")
        print(
            f"\n  The agent is now negotiating. Hang up when it finishes, or wait for the\n"
            f"  {limit_s} s cap. The run is saved in poc2_livekit/recordings/."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc2_livekit.app.call",
        description=(
            "Dials PoC2's outbound call. Requires the worker to be up in "
            "another terminal (`python -m poc2_livekit.app.agent`)."
        ),
    )
    arg_parser.add_argument("--a", dest="destination", default="", help="destination number")
    arg_parser.add_argument("--scenario", type=int, help="script scenario (1-5)")
    arg_parser.add_argument("--repetition", type=int, help="repetition (1-3)")
    arg_parser.add_argument("--poc", type=int, default=2, help="PoC number")
    arg_parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT_S, help="hard cap in seconds"
    )
    arg_parser.add_argument(
        "--accept-cost", action="store_true", help="required to dial"
    )
    args = arg_parser.parse_args(argv)

    force_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Before anything else, and before touching the .env or the network: if this
    # is a campaign run, it must follow the campaign rules.
    if error := verify_campaign_run(args.scenario):
        print(f"[rejected] {error}")
        return REJECTED_OUTPUT

    settings = load_settings()
    verify_for_livekit(settings)
    verify_for_telephony(settings)

    destination = args.destination.strip() or settings.destination_number

    if not args.accept_cost:
        _print_cost_warning(settings, destination, args.limit)
        return 0

    identification: dict[str, Any] = {"poc": args.poc}
    if args.scenario is not None:
        identification["scenario"] = args.scenario
    if args.repetition is not None:
        identification["repetition"] = args.repetition
    if args.scenario is None or args.repetition is None:
        print(
            "[warning] without --scenario/--repetition the run is left unlabeled "
            "and has to be identified by hand afterwards\n"
        )

    # The suffix keeps two runs of the same scenario from landing in the same room.
    room = settings.room_name(
        scenario=args.scenario,
        repetition=args.repetition,
        suffix=time.strftime("%Y%m%d-%H%M%S"),
    )

    return asyncio.run(
        set_mark(
            settings,
            destination=destination,
            room=room,
            identification=identification,
            limit_s=args.limit,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
