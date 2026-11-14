"""Spanish text normalization for comparing transcripts.

Raw WER penalizes differences that are not recognition errors. If the carrier
says "dieciocho mil" and the STT writes "18,000", recognition was **perfect**
but a naive WER counts it as two deletions and one insertion. The other way
around too: an STT with `numerals` on writes digits and another with
`numerals` off writes words, and the Deepgram vs. ElevenLabs Scribe bake-off
would compare output format instead of listening quality.

This module brings both texts to a canonical form before comparing them:

    "Yo ando pensando en dieciocho mil."  ->  ['yo','ando','pensando','en','18000']
    "yo ando pensando en $18,000"         ->  ['yo','ando','pensando','en','18000']

The golden rule: **the same normalization is applied to the reference and to
the hypothesis**. Any aggressive transformation (folding accents, turning "un"
into 1) is harmless to the WER because it affects both sides equally; what
does matter is that it is deterministic and does not depend on the STT
provider.

No external dependency is used: `shared/` is consumed by the 3 PoCs, each with
its own `uv` environment, so only the standard library goes in here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

# --------------------------------------------------------------------------
# Spanish numeric vocabulary
# --------------------------------------------------------------------------

# Written already without accents: normalization folds accents before looking
# up here, so "dieciséis" arrives as "dieciseis".
_UNITS: Final[dict[str, int]] = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiun": 21,
    "veintiuno": 21,
    "veintiuna": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}

_TENS: Final[dict[str, int]] = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}

_HUNDREDS: Final[dict[str, int]] = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "doscientas": 200,
    "trescientos": 300,
    "trescientas": 300,
    "cuatrocientos": 400,
    "cuatrocientas": 400,
    "quinientos": 500,
    "quinientas": 500,
    "seiscientos": 600,
    "seiscientas": 600,
    "setecientos": 700,
    "setecientas": 700,
    "ochocientos": 800,
    "ochocientas": 800,
    "novecientos": 900,
    "novecientas": 900,
}

# "mil" multiplies the group that precedes it; "millon" likewise. With no
# preceding group they are worth 1000 and 1000000 ("mil pesos" = 1000).
_MULTIPLIERS: Final[dict[str, int]] = {
    "mil": 1_000,
    "miles": 1_000,
    "millon": 1_000_000,
    "millones": 1_000_000,
}

_NUMERIC_WORDS: Final[frozenset[str]] = frozenset(
    {*_UNITS, *_TENS, *_HUNDREDS, *_MULTIPLIERS}
)

# Currency symbols dropped before tokenizing. The currency name ("mxn",
# "pesos") is kept as a word: it is real speech and must count.
_CURRENCY_SYMBOLS: Final[str] = "$\u20ac\u00a3\u00a2"

# Thousands separator attached to digits: "18,000" and "18.000" -> "18000". It
# only applies when exactly three digits follow and no more digits come after,
# which is the only way a thousands separator appears. That way "1.5" (a real
# decimal) stays intact. The plain space is **not** in the class: "18 000" is
# ambiguous with two consecutive numbers and no bake-off STT produces it; the
# no-break and thin spaces, which some formatters use, are included.
_THOUSANDS_SEPARATOR: Final[re.Pattern[str]] = re.compile(
    "(?<=\\d)[.,\u00a0\u202f](?=\\d{3}(?!\\d))"
)

# Anything that is not a letter (including ñ; accented vowels have already been
# folded), digit, dot or space becomes a space.
_NO_TOKEN: Final[re.Pattern[str]] = re.compile("[^0-9a-zA-Z\u00f1\u00d1.\\s]+")

# A dot that does not separate two digits is punctuation, not a decimal.
_LOOSE_DOT: Final[re.Pattern[str]] = re.compile(r"(?<!\d)\.|\.(?!\d)")

_SPACES: Final[re.Pattern[str]] = re.compile(r"\s+")

# Filler words of spontaneous speech. **Off by default**: in the real call the
# carrier said "Este," and that is speech the STT has to recognize; deleting
# them would dress up the WER. They are kept available because to read the
# "pure" recognition number it is sometimes useful to isolate them.
FILLERS: Final[frozenset[str]] = frozenset(
    {"eh", "ehh", "em", "este", "mmm", "mm", "ah", "ay"}
)

# Below this amount, a number is not considered a freight price: "dos horas de
# espera" or "un día de flexibilidad" are legitimate domain numbers that are
# not amounts. The threshold can be raised/lowered via config.
AMOUNT_THRESHOLD_DEFAULT: Final[float] = 1_000.0


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization knobs. The same config for the 3 PoCs.

    The defaults are the comparison's contract: changing them between PoCs
    would invalidate the numbers. They exist as parameters so the report can
    document what was folded and what was not, and for the tests.
    """

    fold_accents: bool = True
    lowercase: bool = True
    canonicalize_numbers: bool = True
    remove_fillers: bool = False
    amount_threshold: float = AMOUNT_THRESHOLD_DEFAULT

    def descripcion(self) -> str:
        """One line for the report: which normalization produced these numbers."""
        return ", ".join(
            [
                f"acentos={'plegados' if self.fold_accents else 'literales'}",
                f"caja={'baja' if self.lowercase else 'literal'}",
                f"numeros={'canonicos' if self.canonicalize_numbers else 'literales'}",
                f"muletillas={'outside' if self.remove_fillers else 'inside'}",
                f"amount_threshold={self.amount_threshold:,.0f}",
            ]
        )


CONFIG_DEFAULT: Final[NormalizationConfig] = NormalizationConfig()


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def fold_accents(text: str) -> str:
    """Strip diacritics while keeping ñ.

    An STT that writes "dieciseis" and another that writes "dieciséis" heard the
    same thing. The ñ is kept: "año" and "ano" are different words and confusing
    them would be a real recognition error.
    """
    protected = text.replace("\u00f1", "\x00").replace("\u00d1", "\x01")
    decomposed = unicodedata.normalize("NFD", protected)
    no_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    recomposed = unicodedata.normalize("NFC", no_marks)
    return recomposed.replace("\x00", "\u00f1").replace("\x01", "\u00d1")


def _is_digit(token: str) -> bool:
    """True if the token starts with a digit, i.e. it is a number written that way."""
    return bool(token) and token[0].isdigit()


def _leer_numero(tokens: list[str], i: int) -> tuple[float, int] | None:
    """Read a numeric expression starting at `tokens[i]`.

    Returns `(value, how many tokens it consumed)`, or None if no number starts
    at that position. It gathers the whole group, which is what makes
    "dieciocho mil" collapse into a single token just like "18000" — without
    that, the two sides of the WER would have a different number of tokens and
    comparing amounts in words against digits would be impossible:

        ['dieciocho','mil']          -> (18000.0, 2)
        ['18000']                    -> (18000.0, 1)
        ['18','mil']                 -> (18000.0, 2)
        ['veintiocho','mil']         -> (28000.0, 2)
        ['ciento','cincuenta','mil'] -> (150000.0, 3)
        ['treinta','y','cinco']      -> (35.0, 3)
    """
    n = len(tokens)
    total = 0.0
    group = 0.0
    consumed = 0
    had_something = False
    # A standalone decimal ("1.5") does not combine with multipliers or with "y".
    pure_decimal = False
    # The last multiplier applied and where the group after it started: a
    # second "mil" after a "mil" starts another number ("dieciocho mil,
    # dieciocho mil" are two amounts of 18,000, not one of 36,000).
    last_multiplier: float | None = None
    group_start = i

    j = i
    while j < n:
        token = tokens[j]

        if _is_digit(token):
            # Two consecutive numbers are two numbers, not one: if something
            # was already read, another expression starts here. ("18 000" was
            # already joined earlier if it was a thousands separator.)
            if had_something:
                break
            try:
                valor = float(token)
            except ValueError:
                break
            group += valor
            pure_decimal = "." in token
            had_something = True
            j += 1
            consumed = j - i
            continue

        if token in _MULTIPLIERS:
            if pure_decimal:
                break
            multiplier = _MULTIPLIERS[token]
            if last_multiplier is not None and multiplier >= last_multiplier:
                # End the previous number where this group started; the group is
                # read again as the start of the next number.
                consumed = group_start - i
                group = 0.0
                break
            # "mil" with no preceding group is 1000; with a group, it multiplies it.
            total += (group if group else 1.0) * multiplier
            group = 0.0
            last_multiplier = multiplier
            had_something = True
            j += 1
            consumed = j - i
            group_start = j
            continue

        if token in _HUNDREDS:
            if pure_decimal or group >= 100:
                break
            group += _HUNDREDS[token]
            had_something = True
            j += 1
            consumed = j - i
            continue

        if token in _TENS:
            # The group already has a ten or a teen: "treinta cuarenta" is not a number.
            if pure_decimal or group % 100 >= 10:
                break
            group += _TENS[token]
            had_something = True
            j += 1
            consumed = j - i
            continue

        if token in _UNITS:
            if pure_decimal:
                break
            valor = _UNITS[token]
            rest = group % 100
            if rest == 0:
                # Empty group or a round hundred: any unit fits
                # ("cinco", "ciento veintiocho").
                pass
            elif rest >= 30 and rest % 10 == 0 and valor < 10:
                # Round ten + unit: "treinta y cinco".
                pass
            else:
                # Teen or veinti- already complete, or a unit already there: no combining.
                break
            group += valor
            had_something = True
            j += 1
            consumed = j - i
            continue

        if token == "y" and had_something and not pure_decimal:
            # The "y" is swallowed only if it really joins the number: "treinta
            # y cinco", "mil y quinientos". Otherwise it is a conjunction and
            # stays as a word (it is not added to `consumed`).
            following = tokens[j + 1] if j + 1 < n else None
            if following is not None and (
                following in _UNITS
                or following in _HUNDREDS
                or _is_digit(following)
            ):
                j += 1
                continue
            break

        break

    if not had_something or consumed == 0:
        return None
    return total + group, consumed


def _format_number(valor: float) -> str:
    """Number to canonical token: integer without decimal tail, decimal if needed.

    `:g` is not used: that format applies 6 significant digits and would turn
    18000.0001 into "18000", losing the decimal part precisely in the large
    amounts, which are the ones that matter here.
    """
    if valor == int(valor):
        return str(int(valor))
    return f"{valor:.4f}".rstrip("0").rstrip(".")


def tokenize_text(text: str, config: NormalizationConfig = CONFIG_DEFAULT) -> list[str]:
    """Raw text -> list of canonical tokens.

    Order matters: thousands separators are resolved **before** dropping
    punctuation, otherwise "18,000" would split into "18" and "000".
    """
    if not text:
        return []

    work = text
    if config.lowercase:
        work = work.lower()
    if config.fold_accents:
        work = fold_accents(work)

    for symbol in _CURRENCY_SYMBOLS:
        work = work.replace(symbol, " ")

    work = _THOUSANDS_SEPARATOR.sub("", work)
    work = _NO_TOKEN.sub(" ", work)
    work = _LOOSE_DOT.sub(" ", work)
    work = _SPACES.sub(" ", work).strip()

    if not work:
        return []

    raw_values = work.split(" ")

    if config.remove_fillers:
        raw_values = [t for t in raw_values if t not in FILLERS]
        if not raw_values:
            return []

    if not config.canonicalize_numbers:
        return raw_values

    output: list[str] = []
    i = 0
    while i < len(raw_values):
        read_result = _leer_numero(raw_values, i)
        if read_result is None:
            output.append(raw_values[i])
            i += 1
            continue
        valor, consumed = read_result
        output.append(_format_number(valor))
        i += consumed
    return output


def normalize(text: str, config: NormalizationConfig = CONFIG_DEFAULT) -> str:
    """Canonical text as a string. It is the basis of the CER."""
    return " ".join(tokenize_text(text, config))


def extract_amounts(
    text: str, config: NormalizationConfig = CONFIG_DEFAULT
) -> list[float]:
    """The text's amounts, in order of appearance.

    A number counts as an amount if it reaches the config's `amount_threshold`.
    The threshold is what keeps "dos horas de espera" or "un día de
    flexibilidad" —real domain numbers, but not prices— out of the amount
    extraction metric.

    They are returned with repetition and in order: "dieciocho mil… dieciocho
    mil" are two mentions, and that is information (the carrier insisted).
    """
    amounts: list[float] = []
    for token in tokenize_text(text, config):
        try:
            valor = float(token)
        except ValueError:
            continue
        if valor >= config.amount_threshold:
            amounts.append(valor)
    return amounts
