"""Dials PoC3's outbound call. It is the only module that spends money on telephony.

Guardrails from CLAUDE.md, implemented here:

  - `--accept-cost` is mandatory. Without the flag, it prints the estimate
    (table in `docs/costos_poc3.md`) and exits without dialing.
  - Before dialing it checks `/salud` on the public tunnel: if the server or
    ngrok is not up, the call would be answered with a Twilio error and the
    minute would be paid anyway.
  - Hard duration cap (`time_limit`), so that a bug cannot leave a call
    open burning minutes.
  - **It does not touch the number or trunk configuration.** The TwiML `url`
    travels in this outbound call; the webhook of the dispatcher agent's number
    stays intact. This module does not call any `.update()`.

## PoC3 does not record

`record=False`, with no flag to turn it on. The metrics contract freezes this
for the 3 PoCs: the WER reference is the **acted script**, not a recording,
and that method is what makes WER comparable across architectures. On top of that
there is an extra reason here: the alpha terms forbid exposing generated audio
outside the organization, and a recording in Twilio's cloud is exactly
that. (PoC1 did leave 2-channel recordings because its dialer recorded by
default; they were not used for any metric and remain pending under LFPDPPP.)
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

from poc3_gpt_live.app.config import (
    Settings,
    load_settings,
    force_utf8_output,
    verify_for_telephony,
    verify_for_voice,
)
from poc3_gpt_live.app.telephony import HEALTH_PATH, RUTA_TWIML
from shared.metrics.scenarios import load_script

logger = logging.getLogger(__name__)

# 10 minutes: hard cap so that a bug cannot leave a call hanging open.
DEFAULT_LIMIT_SECONDS: Final[int] = 600

# Exit code when a run is rejected by policy, not because of incomplete
# config. It differs from 1 so the 15 runs can be scripted. Same
# criterion as PoC2.
REJECTED_OUTPUT: Final[int] = 2

# Twilio rates, the same as PoC1 because the telephony layer is the same
# (source and date in docs/costos_poc1.md).
VOICE_COST_PER_MINUTE: Final[float] = 0.016  # outbound-api to MX mobile
STREAMS_COST_PER_MINUTE: Final[float] = 0.0044

# The `gpt-live` model is NOT billed during the alpha (`docs_alpha/`). What is
# billed is the delegation to the text model, which is where tool calling
# runs. An estimate, not a verified rate: see docs/costos_poc3.md.
DELEGATION_COST_PER_CALL: Final[float] = 0.02


def estimate_cost(minutes: float) -> float:
    """Estimate for a call of `minutos` minutes. Twilio rounds up to the minute."""
    billed_minutes = max(1.0, float(int(minutes)) + (1.0 if minutes % 1 else 0.0))
    per_minute = VOICE_COST_PER_MINUTE + STREAMS_COST_PER_MINUTE
    return billed_minutes * per_minute + DELEGATION_COST_PER_CALL


def verify_campaign_run(scenario: int | None) -> str:
    """Rules that apply only to scripted calls. Returns the error, or "".

    A campaign run is one that carries `--scenario`: its JSON goes into the
    consolidated dataset and the technical decision. The scenario has to exist in
    `shared/test_scenarios/`, because a typo in the number would produce a run
    labeled as campaign that matches no script — and that is
    discovered at consolidation time, too late. Same check as PoC2.
    """
    if scenario is None:
        return ""
    try:
        load_script().scenario(scenario)
    except KeyError as error:
        return str(error.args[0]) if error.args else f"invalid scenario {scenario}"
    return ""


def verify_tunnel(settings: Settings, *, timeout: float = 10.0) -> dict[str, object]:
    """Confirm that the server responds on the public domain.

    Raises RuntimeError if it does not: dialing without the tunnel up pays for the minute and
    negotiates nothing.
    """
    url = settings.url_webhook(HEALTH_PATH)
    # ngrok's free plan inserts a warning page when the request looks like it
    # comes from a browser. The header disables it
    # explicitly; Twilio's TwiML POST never goes through there.
    request = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"the server is not responding at {url} ({error}).\n"
            "Start both pieces before dialing:\n"
            "  1) ngrok http 8000 --domain=" + settings.public_host() + "\n"
            "  2) uv run --project poc3_gpt_live python -m poc3_gpt_live.app.telefonia"
        ) from error


def run_twiml_url(
    settings: Settings, *, poc: int | None, scenario: int | None, repetition: int | None
) -> str:
    """The TwiML URL with the run label in the query string.

    It is the first link in the stamping chain: Twilio keeps the query string when requesting
    the TwiML, the endpoint turns it into `<Parameter>` elements of the `<Stream>`, and that is how
    `escenario`/`repeticion` end up inside the call JSON.
    """
    base = settings.url_webhook(RUTA_TWIML)
    label = {"poc": poc, "scenario": scenario, "repetition": repetition}
    parameters = {k: str(v) for k, v in label.items() if v is not None}
    if not parameters:
        return base
    return f"{base}?{urllib.parse.urlencode(parameters)}"


def set_mark(
    settings: Settings,
    *,
    destination: str,
    limit_seconds: int,
    url_twiml: str | None = None,
) -> str:
    """Dial the outbound call and return the Call SID.

    `record=False` with no option to turn it on: see the module docstring.
    """
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    call = client.calls.create(
        to=destination,
        from_=settings.twilio_phone_number,
        url=url_twiml or settings.url_webhook(RUTA_TWIML),
        method="POST",
        record=False,
        time_limit=limit_seconds,
    )
    return call.sid


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.app.call",
        description=(
            "Dials PoC3's outbound call. Requires the telephony "
            "server and the tunnel to be up."
        ),
    )
    arg_parser.add_argument(
        "--accept-cost",
        action="store_true",
        help="confirm the telephony spend (see docs/costos_poc3.md)",
    )
    arg_parser.add_argument(
        "--a",
        dest="destination",
        default=None,
        help="destination number in E.164 (default: TWILIO_NUMERO_DESTINO from .env)",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT_SECONDS,
        help=f"hard duration cap in seconds (default {DEFAULT_LIMIT_SECONDS})",
    )
    arg_parser.add_argument(
        "--scenario", type=int, choices=range(1, 6), help="script scenario (1-5)"
    )
    arg_parser.add_argument(
        "--repetition", type=int, choices=range(1, 4), help="repetition (1-3)"
    )
    arg_parser.add_argument(
        "--poc", type=int, default=3, choices=range(1, 4), help="PoC number (1-3)"
    )
    args = arg_parser.parse_args(argv)
    force_utf8_output()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Before touching the .env or the network: if this is a campaign run, it has to
    # follow the campaign rules.
    if error := verify_campaign_run(args.scenario):
        print(f"[rejected] {error}")
        return REJECTED_OUTPUT

    settings = load_settings()
    verify_for_telephony(settings)
    verify_for_voice(settings)

    destination = args.destination or settings.twilio_destination_number
    if not destination:
        print(
            "No destination number: pass it with --a +52... or set "
            "TWILIO_NUMERO_DESTINO in the .env"
        )
        return 1
    if not destination.startswith("+"):
        print(f"The destination {destination!r} must be in E.164 format, with + and country code.")
        return 1

    typical_estimate = estimate_cost(4)
    estimate_cap = estimate_cost(args.limit / 60)

    url_twiml = run_twiml_url(
        settings, poc=args.poc, scenario=args.scenario, repetition=args.repetition
    )

    print("PoC3 outbound call — GPT-Live alpha")
    print(f"  from              {settings.twilio_phone_number}")
    print(f"  to                {destination}")
    print(f"  TwiML             {url_twiml}")
    print(f"  model             {settings.live_model} (alpha, NOT billed)")
    print(f"  delegation        {settings.delegation_model} (billed)")
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
    print("  recording         no (WER is measured against the acted script)")
    print(f"  duration cap      {args.limit}s")
    print(f"  typical cost      ~${typical_estimate:.2f} USD (4-min call)")
    print(f"  maximum cost      ~${estimate_cap:.2f} USD (if it hits the cap)")

    if not args.accept_cost:
        print(
            "\nNot dialed: --accept-cost is missing.\n"
            "Rate breakdown in docs/costos_poc3.md."
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
            limit_seconds=args.limit,
            url_twiml=url_twiml,
        )
    except TwilioRestException as error:
        print(f"\nTwilio rejected the call ({error.status} {error.code}): {error.msg}")
        return 1

    print(f"\n  dialing...        call_sid={call_sid}")
    print(
        "\nAnswer the phone. The agent speaks first.\n"
        "The transcript and the result are printed in the server terminal on "
        "hang-up, and saved to poc3_gpt_live/recordings/llamada_<call_sid>.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
