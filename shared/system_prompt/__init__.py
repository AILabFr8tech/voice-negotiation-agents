"""System prompt v0 y su interpolación, compartidos por las 3 PoC.

`v0.md` es la fuente de verdad y está congelado (CLAUDE.md): no se duplica por
PoC ni se edita a media campaña de llamadas. `prompt.py` lo lee, sustituye las
7 variables y verifica en cada render que el precio máximo no se haya filtrado.

Ese último punto es el que justifica que el código viva aquí y no en cada PoC:
`verify_absence_of_max_price` es el guardarraíl estructural de una regla no
negociable, y una copia por PoC son tres versiones que pueden divergir sin que
nadie se entere.
"""

from shared.system_prompt.prompt import (
    RUTA_SYSTEM_PROMPT,
    PROMPT_VARIABLES,
    MaxPriceLeak,
    InvalidPrompt,
    load_template,
    render_system_prompt,
    amount_representations,
    variables_in_template,
    verify_absence_of_max_price,
)

__all__ = [
    "RUTA_SYSTEM_PROMPT",
    "PROMPT_VARIABLES",
    "MaxPriceLeak",
    "InvalidPrompt",
    "load_template",
    "render_system_prompt",
    "amount_representations",
    "variables_in_template",
    "verify_absence_of_max_price",
]
