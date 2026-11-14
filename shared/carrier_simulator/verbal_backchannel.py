"""Backchannel classifier for Mexican Spanish.

A backchannel is what one says to signal "I'm still listening" without taking the
turn: *ajá*, *claro*, *de acuerdo*. For the harness the distinction is not about style
but about synchronization: if an agent's "ajá" counts as its answer, the harness
releases the next clip against an agent that is still thinking, and from then on
the conversation is shifted by one turn.

## Why the rule is "the whole utterance", not "contains one of these words"

Because the same words open substantive answers. *"Claro, le explico la
ruta"* starts with `claro` and is an answer; *"Claro."* is a backchannel. What
separates the two is that in the backchannel **there is nothing else**. That is why
*every* word of the utterance must be in the list and the utterance must be short:
the condition breaks as soon as a single content word appears.

## What this module does NOT decide

It does not decide on its own whether the harness advances. The text arrives late and
sometimes not at all (a short fragment may produce no transcript whatsoever, and then
there is nothing here to classify). Whoever advances the turn cross-checks this with
the duration of the speech streak; see `wait_for_agent_turn`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Words that on their own do not take the turn. They are stored already folded
# (no accents, lowercase), which is how `_normalize` leaves them.
#
# Includes the particles of the two- and three-word phrases (*de acuerdo*,
# *muy bien*, *por supuesto*, *claro que sí*): on their own they say nothing, and the
# word cap prevents a real sentence from passing by stringing several together.
BACKCHANNELS: Final[frozenset[str]] = frozenset(
    {
        # non-lexical assent
        "aja", "ajam", "aha", "aha", "mm", "mmm", "mhm", "mjm", "ujum", "ah",
        "ahh", "oh", "eh", "este",
        # affirmation
        "si", "sip", "simon", "claro", "correcto", "exact", "exactamente",
        "efectivamente", "seguro",
        # acknowledgement of receipt
        "ok", "okey", "okay", "vale", "entiendo", "entendido", "comprendo",
        "ready", "perfecto", "excelente", "bien", "bueno", "ya",
        # regional
        "orale", "andale", "sale",
        # phrase particles
        "de", "acuerdo", "muy", "por", "supuesto", "que", "esta", "tiene",
        "sentido",
        # standalone courtesy
        "gracias",
    }
)

# Word cap. A real backchannel never goes beyond a short phrase; from the
# fourth word on there is content even if every word is in the list.
MAX_WORDS: Final[int] = 3

_NON_WORD: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize(text: str) -> list[str]:
    """Lowercase, no accents, no punctuation. Returns the words.

    Accents are folded because the STT is not consistent with them and *ajá* /
    *aja* have to land in the same place. It is the same normalization that
    `shared/metrics/text.py` uses for WER, for the same reason.
    """
    folded_form = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in folded_form if unicodedata.category(c) != "Mn")
    return _NON_WORD.sub(" ", without_accents).split()


def is_backchannel(text: str) -> bool:
    """Whether the whole utterance is a backchannel rather than an answer.

    An empty text returns `False`: "there is no text" is not the same as "the text
    is a backchannel", and conflating them would let a transcription failure
    masquerade as a backchannel.
    """
    words = _normalize(text)
    if not words or len(words) > MAX_WORDS:
        return False
    return all(word in BACKCHANNELS for word in words)
