"""Data for the trip being negotiated: a single fixed test trip, the same in all 5 scenarios.

In production this is inherited from Spots' negotiation configuration (target
price and maximum price of the spot route that did not close over WhatsApp).
For the PoC it is a constant.

`max_price` lives in this structure but **does not go out to the prompt**: the
only exit path to the LLM is `prompt_data()`, which does not include it. Any
code that needs to render the prompt must use that method and never
`dataclasses.asdict()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

Amount = float


def format_amount_for_voice(monto: Amount) -> str:
    """Format an amount so the LLM reads it aloud naturally."""
    return f"{monto:,.0f} pesos mexicanos"


@dataclass(frozen=True)
class TripData:
    """A trip to negotiate, with its inherited financial configuration."""

    company: str
    origin: str
    destination: str
    cargo_type: str
    target_price: Amount
    # NEVER to the prompt or the LLM context. It only travels to `proponer_precio`.
    max_price: Amount
    carrier_name: str = "no se conoce"

    def __post_init__(self) -> None:
        for name in ("target_price", "max_price"):
            valor = getattr(self, name)
            if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(valor) or valor <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_price < self.target_price:
            raise ValueError(
                "invalid configuration: the maximum price cannot be "
                "below the target price"
            )

    def prompt_data(self) -> dict[str, str]:
        """The 6 system prompt variables that come from the trip.

        The seventh, `COMPENSACIONES_DISPONIBLES`, is supplied by the catalog in
        `shared/negotiation_engine/compensations.yaml`.

        `max_price` is not here on purpose and must not be added.
        """
        return {
            "EMPRESA": self.company,
            "ORIGEN": self.origin,
            "DESTINO": self.destination,
            "TIPO_CARGA": self.cargo_type,
            "PRECIO_OBJETIVO": format_amount_for_voice(self.target_price),
            "NOMBRE_TRANSPORTISTA": self.carrier_name,
        }


# The PoC's test trip. A typical Freight Technologies cross-border route; the
# amounts are illustrative, not real market rates (calibrating them is not
# part of this increment).
#
# **It is the same trip in all 3 PoCs, and it must stay that way**: the script
# uses the same 18k/21k configuration in all 5 scenarios on purpose, and if one
# PoC negotiated a different trip the WER vocabulary would no longer be
# comparable.
TEST_TRIP: Final[TripData] = TripData(
    company="Freight Technologies",
    origin="Monterrey, Nuevo León",
    destination="Laredo, Texas",
    cargo_type="caja seca de 53 pies con carga general",
    target_price=18_000.0,
    max_price=21_000.0,
    carrier_name="Carlos",
)
