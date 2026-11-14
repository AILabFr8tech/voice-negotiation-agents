"""Carga e interpolación del system prompt v0, compartido por las 3 PoC.

`shared/system_prompt/v0.md` es la fuente de verdad y no se duplica ni se
edita desde aquí. Este módulo lo lee, sustituye las 7 variables y verifica que
el resultado no filtre el precio máximo.

**Por qué no `string.Template`:** el prompt usa delimitadores `{VAR}`, no
`$VAR`. Hacer que `Template` entienda `{VAR}` exige reescribir su `pattern`
completo, y ese pattern hace que cualquier llave literal suelta caiga en el
grupo `invalid` y levante excepción — justo el problema que CLAUDE.md quiere
evitar cuando una revisión futura del prompt agregue un ejemplo con JSON.
CLAUDE.md admite las dos vías ("reemplazo explícito por clave o
`string.Template`"); aquí va la explícita, con un regex que solo puede
reconocer los 7 nombres conocidos. Cualquier otra llave del archivo es
invisible para el sustituidor, por construcción.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from shared.negotiation_engine import render_compensations
from shared.negotiation_engine.trip import TripData

# El módulo vive al lado de la plantilla que renderiza, así que no necesita la
# raíz del repo ni la configuración de ninguna PoC para encontrarla.
RUTA_SYSTEM_PROMPT: Final[Path] = Path(__file__).resolve().parent / "v0.md"

# Los únicos placeholders reales del prompt (CLAUDE.md).
PROMPT_VARIABLES: Final[tuple[str, ...]] = (
    "EMPRESA",
    "ORIGEN",
    "DESTINO",
    "TIPO_CARGA",
    "PRECIO_OBJETIVO",
    "COMPENSACIONES_DISPONIBLES",
    "NOMBRE_TRANSPORTISTA",
)

_VARIABLES_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{(" + "|".join(PROMPT_VARIABLES) + r")\}"
)


class InvalidPrompt(RuntimeError):
    """El prompt en disco y el código que lo interpola se desincronizaron."""


class MaxPriceLeak(RuntimeError):
    """El prompt renderizado contiene el precio máximo. Nunca debe pasar."""


def load_template(ruta: Path | str | None = None) -> str:
    """Lee `v0.md` tal cual, sin sustituir nada."""
    destination = Path(ruta) if ruta is not None else RUTA_SYSTEM_PROMPT
    return destination.read_text(encoding="utf-8")


def variables_in_template(template: str) -> set[str]:
    """Nombres de las variables conocidas que aparecen en la plantilla."""
    return set(_VARIABLES_PATTERN.findall(template))


def amount_representations(monto: float) -> set[str]:
    """Formas en que un monto podría aparecer escrito en el prompt."""
    with_commas = f"{monto:,.0f}"
    return {
        f"{monto:.0f}",
        with_commas,
        with_commas.replace(",", "."),
        with_commas.replace(",", " "),
        f"{monto:,.2f}",
        f"{monto:.2f}",
    }


def verify_absence_of_max_price(prompt: str, trip: TripData) -> None:
    """Falla si el precio máximo quedó escrito en el prompt renderizado.

    Es el guardarraíl estructural de la regla no negociable de CLAUDE.md:
    `max_price` nunca vive en el system prompt ni en el contexto del LLM.
    Se corre en cada render, no solo en los tests.

    Si el máximo coincide con el objetivo, la verificación se salta: el
    objetivo sí va en el prompt a propósito, y no hay forma de distinguirlos.

    El mensaje de la excepción no incluye el monto: una excepción que sube por
    la capa de tool-calling puede terminar serializada hacia el LLM.
    """
    if trip.max_price == trip.target_price:
        return
    if any(shape in prompt for shape in amount_representations(trip.max_price)):
        raise MaxPriceLeak(
            "el system prompt renderizado contiene el precio máximo: revisa "
            "qué valor se está inyectando en las variables del prompt"
        )


def render_system_prompt(
    trip: TripData,
    *,
    ruta_prompt: Path | str | None = None,
    compensations_path: Path | str | None = None,
) -> str:
    """Devuelve el system prompt v0 con las 7 variables ya sustituidas.

    Levanta `PromptInvalido` si la plantilla y `PROMPT_VARIABLES` dejan de
    coincidir en cualquier dirección, y `FugaDePrecioMaximo` si el resultado
    contiene el precio máximo.
    """
    template = load_template(ruta_prompt)

    values = dict(trip.prompt_data())
    values["COMPENSACIONES_DISPONIBLES"] = render_compensations(
        compensations_path
    )

    expected_variables = set(PROMPT_VARIABLES)
    if set(values) != expected_variables:
        missing_items = expected_variables - set(values)
        surplus = set(values) - expected_variables
        raise InvalidPrompt(
            f"los valores a inyectar no cubren las variables del prompt "
            f"(faltan: {sorted(missing_items)}, sobran: {sorted(surplus)})"
        )

    in_template = variables_in_template(template)
    if in_template != expected_variables:
        missing_items = expected_variables - in_template
        surplus = in_template - expected_variables
        raise InvalidPrompt(
            f"la plantilla del prompt no tiene las 7 variables esperadas "
            f"(faltan en el archivo: {sorted(missing_items)}, "
            f"desconocidas en el código: {sorted(surplus)})"
        )

    # Un valor que traiga un placeholder adentro reabriría la puerta a
    # inyección desde los datos del viaje.
    for clave, valor in values.items():
        if _VARIABLES_PATTERN.search(valor):
            raise InvalidPrompt(
                f"el valor de {clave} contiene un placeholder del prompt"
            )

    prompt = _VARIABLES_PATTERN.sub(lambda m: values[m.group(1)], template)

    if variables_in_template(prompt):
        raise InvalidPrompt(
            "quedaron placeholders sin sustituir en el prompt renderizado"
        )

    verify_absence_of_max_price(prompt, trip)
    return prompt
