"""Dials PoC1's outbound call. It is the only module that spends on telephony.

Guardrails from CLAUDE.md, implemented here:

  - `--accept-cost` is mandatory. Without the flag, it prints the estimate
    (table in `docs/costos_poc1.md`) and exits without dialing.
  - Before dialing it checks `/health` on the public tunnel: if the server or
    ngrok is not up, the call would be answered with a Twilio error and the
    minute would be billed anyway.
  - Hard duration cap (`time_limit`), so that a bug cannot leave a call open
    burning minutes.
  - **It does not touch the number's or the trunk's configuration.** The TwiML
    `url` travels with this outbound call; the webhook on the dispatcher
    agent's number stays intact. This module never calls any `.update()`.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Final

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from poc1_pipecat.app.config import (
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc1_pipecat.app.telephony import HEALTH_PATH, RUTA_TWIML

logger = logging.getLogger(__name__)

# 10 minutes
# call left hanging by a bug.
DEFAULT_LIMIT_SECONDS: Final[int] = 600

# Account rates; source and date in docs/costos_poc1.md.
VOICE_COST_PER_MINUTE: Final[float] = 0.016  # outbound-api to MX mobile
STREAMS_COST_PER_MINUTE: Final[float] = 0.0044
RECORDING_COST_PER_MINUTE: Final[float] = 0.0025
STT_COST_PER_MINUTE: Final[float] = 0.0077
# What is not per minute: TTS per character and LLM tokens, ~$0.13 for a
# 4-minute negotiation.
AI_COST_PER_CALL: Final[float] = 0.13


def estimate_cost(minutes: float, *, is_recording: bool) -> float:
    """Estimate for a call of `minutes`. Twilio rounds up to the next minute."""
    billed_minutes = max(1.0, float(int(minutes)) + (1.0 if minutes % 1 else 0.0))
    per_minute = (
        VOICE_COST_PER_MINUTE + STREAMS_COST_PER_MINUTE + STT_COST_PER_MINUTE
    )
    if is_recording:
        per_minute += RECORDING_COST_PER_MINUTE
    return billed_minutes * per_minute + AI_COST_PER_CALL


def verify_tunnel(settings: Settings, *, timeout: float = 10.0) -> dict[str, object]:
    """Confirm that the server responds on the public domain.

    Raises RuntimeError if it does not: dialing without the tunnel up pays for
    the minute and negotiates nothing.
    """
    url = settings.url_webhook(HEALTH_PATH)
    # ngrok's free plan interposes a warning page (ERR_NGROK_6024) when the
    # request looks like it comes from a browser. Today urllib does not trigger
    # it, but the header disables it explicitly: without it, a change in that
    # heuristic would return HTML and this check would block the call for no
    # reason. Twilio's TwiML POST never goes through that page.
    request = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"the server is not responding at {url} ({error}).\n"
            "Start both pieces before dialing:\n"
            "  1) ngrok http 8000 --domain=" + settings.public_host() + "\n"
            "  2) uv run --project poc1_pipecat python -m poc1_pipecat.app.telephony"
        ) from error


def run_twiml_url(
    settings: Settings, *, poc: int | None, scenario: int | None, repetition: int | None
) -> str:
    """The TwiML URL with the run label in the query string.

    It is the first link in the stamping chain: Twilio keeps the query string
    when requesting the TwiML, the endpoint turns it into `<Parameter>`s of the
    `<Stream>`, and that way scenario/repetition end up inside the call JSON.
    Without this, one would have to reconstruct by hand which call was which
    from the Call SID — with 45 calls, that is a guaranteed source of error.
    """
    base = settings.url_webhook(RUTA_TWIML)
    label = {
        "poc": poc,
        "scenario": scenario,
        "repetition": repetition,
    }
    parameters = {k: str(v) for k, v in label.items() if v is not None}
    if not parameters:
        return base
    return f"{base}?{urllib.parse.urlencode(parameters)}"


def set_mark(
    settings: Settings,
    *,
    destination: str,
    record: bool,
    limit_seconds: int,
    url_twiml: str | None = None,
) -> str:
    """Dial the outbound call and return the Call SID.

    `recording_channels="dual"` separates the agent's audio from the carrier's
    audio into two channels. That is what makes the recording usable for
    computing WER and scoring turn-taking: in a single mixed channel you cannot
    tell who talked over whom.
    """
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    call = client.calls.create(
        to=destination,
        from_=settings.twilio_phone_number,
        url=url_twiml or settings.url_webhook(RUTA_TWIML),
        method="POST",
        record=record,
        recording_channels="dual" if record else "mono",
        time_limit=limit_seconds,
    )
    return call.sid


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc1_pipecat.app.call",
        description=(
            "Dials PoC1's outbound call. Requires the telephony server "
            "and the tunnel to be up."
        ),
    )
    arg_parser.add_argument(
        "--accept-cost",
        action="store_true",
        help="confirms the telephony spend (see docs/costos_poc1.md)",
    )
    arg_parser.add_argument(
        "--a",
        dest="destination",
        default=None,
        help="destination number in E.164 (default: TWILIO_DESTINATION_NUMBER from the .env)",
    )
    arg_parser.add_argument(
        "--no-record",
        action="store_true",
        help="turns off Twilio's native recording (records by default)",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT_SECONDS,
        help=f"hard duration cap in seconds (default {DEFAULT_LIMIT_SECONDS})",
    )
    # Run label. It is stored inside the call JSON, so nobody has to remember
    # later which Call SID was which scenario.
    arg_parser.add_argument(
        "--scenario", type=int, choices=range(1, 6), help="script scenario (1-5)"
    )
    arg_parser.add_argument(
        "--repetition", type=int, choices=range(1, 4), help="repetition (1-3)"
    )
    arg_parser.add_argument(
        "--poc", type=int, default=1, choices=range(1, 4), help="PoC number (1-3)"
    )
    args = arg_parser.parse_args(argv)
    force_utf8_output()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = load_settings()
    verify_for_telephony(settings)
    verify_for_voice(settings)

    destination = args.destination or settings.twilio_destination_number
    if not destination:
        print(
            "No destination number: pass it with --a +52... or set "
            "TWILIO_DESTINATION_NUMBER in the .env"
        )
        return 1
    if not destination.startswith("+"):
        print(f"The destination {destination!r} must be in E.164 format, with + and country code.")
        return 1

    record = not args.no_record
    minutes = args.limit / 60
    typical_estimate = estimate_cost(4, is_recording=record)
    estimate_cap = estimate_cost(minutes, is_recording=record)

    url_twiml = run_twiml_url(
        settings, poc=args.poc, scenario=args.scenario, repetition=args.repetition
    )

    print("PoC1 outbound call")
    print(f"  from              {settings.twilio_phone_number}")
    print(f"  to                {destination}")
    print(f"  TwiML             {url_twiml}")
    if args.scenario is not None:
        print(
            f"  run               PoC {args.poc}, scenario {args.scenario}"
            f", repetition {args.repetition if args.repetition is not None else '?'}"
        )
    else:
        print(
            "  run               UNLABELED — pass --scenario and --repetition "
            "or you will have to label it by hand"
        )
    print(f"  recording         {'yes, 2 channels' if record else 'no'}")
    print(f"  duration cap      {args.limit}s")
    print(f"  typical cost      ~${typical_estimate:.2f} USD (4-min call)")
    print(f"  maximum cost      ~${estimate_cap:.2f} USD (if it hits the cap)")

    if not args.accept_cost:
        print(
            "\nNot dialed: --accept-cost is missing.\n"
            "Rate breakdown in docs/costos_poc1.md."
        )
        return 0

    try:
        health = verify_tunnel(settings)
    except RuntimeError as error:
        print(f"\n{error}")
        return 1
    print(f"\n  server up         {health.get('websocket')}")

    try:
        call_sid = set_mark(
            settings,
            destination=destination,
            record=record,
            limit_seconds=args.limit,
            url_twiml=url_twiml,
        )
    except TwilioRestException as error:
        print(f"\nTwilio rejected the call ({error.status} {error.code}): {error.msg}")
        return 1

    print(f"\n  dialing...        call_sid={call_sid}")
    print(
        "\nAnswer the phone. The agent speaks first.\n"
        "The transcript and the outcome appear in the server terminal on "
        "hang-up, and are saved to poc1_pipecat/recordings/call_<call_sid>.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
