# negotiation_engine — layer 4: negotiation logic

**Pure** logic: no telephony, no STT/TTS, no LLM, no network, no
persistence. Each PoC wraps these functions in its framework's tool-calling
schemas; the negotiating behavior stays identical across the four and the
comparison measures only the voice layer.

## Modules

| File | Responsibility |
|---|---|
| [pricing.py](pricing.py) | `proponer_precio`, `validate_decreasing_concession`, `last_increment` |
| [compensations.py](compensations.py) | Loading and validating the catalog; rendering it as prose for the prompt |
| [compensations.yaml](compensations.yaml) | **PLACEHOLDER**, pending confirmation with the business |
| [outcomes.py](outcomes.py) | `registrar_acuerdo`, `finalizar_sin_acuerdo`, `NegotiationResult` |

## The non-negotiable rule

`max_price` **only** exists inside `pricing.py`. It is not in the system
prompt, not in the LLM context, not in the result record.

`proponer_precio` returns exactly `"aceptado"` or `"rechazado"`, nothing
else. Not the ceiling, not the distance to the ceiling, not the rejection
reason: that return value travels into the model's context, and an "exceeds
the maximum by 200" would reveal at a stroke that a limit exists and where it
is. The reasons (`RejectionReason`) go to the server's `logging`, which is
where they are read for the metrics.

For the same reason, the messages of the configuration exceptions do not
include amounts: an exception that bubbles up through the tool-calling layer
may end up serialized to the model.

```python
from shared.negotiation_engine import proponer_precio

proponer_precio(monto=21_000, target_price=20_000, max_price=24_000,
                offer_history=[18_000])
# -> "aceptado"
```

`target_price` is **not** a hard barrier: the prompt allows exceeding it
after exhausting the non-monetary compensations. It is only logged, for the
margin-vs-target-price metric. The only hard ceiling is `max_price`, and it
is inclusive (paying exactly the maximum is accepted; exceeding it by one
centavo is not).

## Decreasing concessions

Each increase must be **strictly smaller** than the previous one: less is
conceded each time, so as not to signal that there is room left. Details that
matter:

- The increment is measured against the **highest offer already spoken**, not
  against the last one. If it were measured against the last one, going down
  and back up would reset the concession margin and an increasing raise would
  slip through.
- Repeating or going below what has already been offered **passes**
  validation. Step 6 of the prompt requires repeating the final amount aloud
  before closing; if that were rejected, the agent could not confirm the
  agreement.
- The first offer and the first increase always pass: there is no previous
  increment to compare against. The financial ceiling bounds them anyway.

**Known v0 simplification:** the engine is a set of pure functions and the
negotiation state (history, compensations already offered) is carried by the
caller. That means the ceiling is passed as an argument on every call. If
that proves fragile when integrating with Pipecat, the next increment can add
a session object that holds `max_price` and does not expose it in its repr.

## Non-monetary compensations

`compensations.yaml` is the single source of truth and feeds two things at
once, so they cannot get out of sync:

- `render_compensations()` → prose for `{COMPENSACIONES_DISPONIBLES}` in the
  system prompt (it is synthesized as audio: no bullets or ids).
- `compensation_ids()` / `validate_compensation(tipo)` → the only valid values
  of `tipo` in `ofrecer_compensacion_no_monetaria(tipo)`. Guards against the
  model inventing a compensation Spots does not offer.

The three records in the YAML are **the illustrative examples from CLAUDE.md,
not the real catalog**. Pending confirmation with the business.

## Closing the negotiation

`registrar_acuerdo` and `finalizar_sin_acuerdo` **persist nothing yet**: they
build a `NegotiationResult` (an immutable dataclass, serializable to JSON with
`.to_dict()`) and write it to the log. What is fixed here is the shape of the
data, so that the 4 PoCs produce comparable results. Real persistence comes
with `shared/metrics/`.

`registrar_acuerdo` receives `offer_history` to check that the agreed amount
came from an offer `proponer_precio` already accepted. If it matches none, it
sets `validated_final_amount=False` and emits a warning, but **raises no
exception**: blowing up there would drop a call already closed with the
carrier. The case is caught in the metrics review, where compliance with the
financial limit is a binary that must come out at 100%.

## Pending in this module

- [ ] Confirm the real compensation catalog with the business
- [ ] Tool-calling JSON schemas per framework (each PoC defines them)
- [ ] System prompt interpolation with `string.Template` (not a blind
      `str.format()`); lives in `shared/system_prompt/`, consumes
      `render_compensations()`
- [ ] `carrier_simulator` to exercise the engine end to end without calls
