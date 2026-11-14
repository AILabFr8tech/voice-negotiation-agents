"""WER, CER and amount extraction error. Automatic computation, zero hand counting.

All three metrics come from the same operation: aligning two sequences by edit
distance and counting substitutions, deletions and insertions.

    WER     -> sequences of normalized words
    CER     -> sequences of characters of the normalized text
    amounts -> sequences of extracted amounts

Reusing the same aligner for the amounts is not laziness: it gives for free the
distinction between "the STT heard a different number" (substitution), "it
swallowed the number" (deletion) and "it made up a number" (insertion), which is
exactly the breakdown needed to decide the bake-off. A low WER with misheard
amounts is useless: getting an amount wrong is worse than getting any other word
wrong.

**Aggregation across calls:** per-call WERs are never averaged. The WER of a set
is (total errors / total reference words); the mean of percentages would give
an 8-word call the same weight as a 200-word one. `combine()` sums the counts,
which is the correct way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable, Sequence

from shared.metrics.text import (
    CONFIG_DEFAULT,
    NormalizationConfig,
    extract_amounts,
    normalize,
    tokenize_text,
)

# Tolerance when comparing two amounts. The same one the negotiation engine uses.
_AMOUNT_TOLERANCE: Final[float] = 0.01


@dataclass(frozen=True)
class Alignment:
    """Raw counts of an edit-distance alignment."""

    hits: int = 0
    substitutions: int = 0
    deleted: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deleted + self.insertions

    @property
    def longitud_reference(self) -> int:
        """Reference elements: hits + substitutions + deletions."""
        return self.hits + self.substitutions + self.deleted

    @property
    def sample_rate(self) -> float:
        """Errors / reference length.

        It can exceed 1.0 if the system made up more than there was to
        recognize — it is not clipped, because clipping it hides a serious
        failure. With an empty reference it returns 0.0 if there was no
        hypothesis either, and 1.0 if there were insertions (one cannot divide
        by zero but the error is real).
        """
        if self.longitud_reference == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.longitud_reference

    def __add__(self, other: Alignment) -> Alignment:
        return Alignment(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deleted=self.deleted + other.deleted,
            insertions=self.insertions + other.insertions,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate": round(self.sample_rate, 6),
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deleted": self.deleted,
            "insertions": self.insertions,
            "errors": self.errors,
            "longitud_reference": self.longitud_reference,
        }


def align(reference: Sequence[object], hypothesis: Sequence[object]) -> Alignment:
    """Edit distance with S/D/I breakdown.

    Dynamic programming over two rows, carrying the counts inside the cell
    instead of reconstructing the path at the end: memory stays O(len(hyp))
    instead of O(len(ref)·len(hyp)), which matters when the reference is a
    whole call.

    Ties are broken in the order substitution > deletion > insertion. The total
    number of errors is the same under any criterion; what changes is how it is
    split among the three categories, so one is fixed and documented so that
    the breakdown is comparable across PoCs.
    """
    n_ref = len(reference)
    n_hip = len(hypothesis)

    # row[j] = best alignment of reference[:i] against hypothesis[:j]
    row: list[tuple[int, Alignment]] = [
        (j, Alignment(insertions=j)) for j in range(n_hip + 1)
    ]

    for i in range(1, n_ref + 1):
        anterior = row
        actual: list[tuple[int, Alignment]] = [
            (i, Alignment(deleted=i)),
        ]
        for j in range(1, n_hip + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost_diag, ali_diag = anterior[j - 1]
                candidates = [
                    (
                        cost_diag,
                        Alignment(
                            hits=ali_diag.hits + 1,
                            substitutions=ali_diag.substitutions,
                            deleted=ali_diag.deleted,
                            insertions=ali_diag.insertions,
                        ),
                    )
                ]
            else:
                cost_sub, ali_sub = anterior[j - 1]
                candidates = [
                    (
                        cost_sub + 1,
                        Alignment(
                            hits=ali_sub.hits,
                            substitutions=ali_sub.substitutions + 1,
                            deleted=ali_sub.deleted,
                            insertions=ali_sub.insertions,
                        ),
                    )
                ]

            cost_del, ali_del = anterior[j]
            candidates.append(
                (
                    cost_del + 1,
                    Alignment(
                        hits=ali_del.hits,
                        substitutions=ali_del.substitutions,
                        deleted=ali_del.deleted + 1,
                        insertions=ali_del.insertions,
                    ),
                )
            )

            cost_ins, ali_ins = actual[j - 1]
            candidates.append(
                (
                    cost_ins + 1,
                    Alignment(
                        hits=ali_ins.hits,
                        substitutions=ali_ins.substitutions,
                        deleted=ali_ins.deleted,
                        insertions=ali_ins.insertions + 1,
                    ),
                )
            )

            # min() keyed only on the cost keeps the first minimum, which is the
            # preference order declared above.
            actual.append(min(candidates, key=lambda c: c[0]))
        row = actual

    return row[n_hip][1]


# --------------------------------------------------------------------------
# WER / CER
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextResult:
    """WER and CER of a (reference, hypothesis) pair, already normalized."""

    wer: Alignment
    cer: Alignment
    words_reference: int = 0
    hypothesis_words: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "wer": self.wer.to_dict(),
            "cer": self.cer.to_dict(),
            "words_reference": self.words_reference,
            "hypothesis_words": self.hypothesis_words,
        }


def compute_text(
    reference: str,
    hypothesis: str,
    config: NormalizationConfig = CONFIG_DEFAULT,
) -> TextResult:
    """WER and CER of a pair of texts, with Spanish normalization applied.

    The CER is computed over the **already normalized text**, so it inherits
    number canonicalization: "dieciocho mil" and "18,000" are identical at the
    character level too, and no huge CER appears from a formatting difference
    the WER already declared equivalent.
    """
    ref_tokens = tokenize_text(reference, config)
    hyp_tokens = tokenize_text(hypothesis, config)

    ref_text = " ".join(ref_tokens)
    hyp_text = " ".join(hyp_tokens)

    return TextResult(
        wer=align(ref_tokens, hyp_tokens),
        cer=align(ref_text, hyp_text),
        words_reference=len(ref_tokens),
        hypothesis_words=len(hyp_tokens),
    )


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------


class _ComparableAmount:
    """Amount that compares with tolerance, so it can be aligned with `align()`.

    A bare float would compare by exact equality and 18000.0 vs 18000.000001
    would count as a substitution.
    """

    __slots__ = ("valor",)

    def __init__(self, valor: float) -> None:
        self.valor = float(valor)

    def __eq__(self, otro: object) -> bool:
        if not isinstance(otro, _ComparableAmount):
            return NotImplemented
        return abs(self.valor - otro.valor) <= _AMOUNT_TOLERANCE

    def __hash__(self) -> int:  # pragma: no cover - not used in sets
        return hash(round(self.valor, 2))

    def __repr__(self) -> str:
        return f"{self.valor:,.0f}"


@dataclass(frozen=True)
class AmountsResult:
    """How well the amounts were recovered from speech."""

    alignment: Alignment
    expected_items: tuple[float, ...] = ()
    obtained: tuple[float, ...] = ()

    @property
    def error_rate(self) -> float:
        return self.alignment.sample_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "error_rate": round(self.error_rate, 6),
            "expected_items": list(self.expected_items),
            "obtained": list(self.obtained),
            "correctos": self.alignment.hits,
            "badly_heard": self.alignment.substitutions,
            "absent": self.alignment.deleted,
            "espurios": self.alignment.insertions,
        }


def compute_amounts(
    reference: str,
    hypothesis: str,
    config: NormalizationConfig = CONFIG_DEFAULT,
) -> AmountsResult:
    """Amount extraction error rate.

    It is the metric that rules the bake-off: an STT that transcribes
    everything right except the numbers is useless for negotiating. The
    breakdown distinguishes `badly_heard` (it wrote a different number),
    `absent` (it did not transcribe it) and `espurios` (it made up a number
    nobody said).
    """
    expected_items = extract_amounts(reference, config)
    obtained = extract_amounts(hypothesis, config)
    alignment = align(
        [_ComparableAmount(m) for m in expected_items],
        [_ComparableAmount(m) for m in obtained],
    )
    return AmountsResult(
        alignment=alignment,
        expected_items=tuple(expected_items),
        obtained=tuple(obtained),
    )


# --------------------------------------------------------------------------
# Aggregation across calls
# --------------------------------------------------------------------------


@dataclass
class AccumulatorSTT:
    """Sum the counts of several calls to give the PoC's number.

    Counts are summed, rates are not averaged: see the module docstring.
    """

    wer: Alignment = field(default_factory=Alignment)
    cer: Alignment = field(default_factory=Alignment)
    amounts: Alignment = field(default_factory=Alignment)
    calls: int = 0

    def add(self, text: TextResult, amounts: AmountsResult) -> None:
        self.wer = self.wer + text.wer
        self.cer = self.cer + text.cer
        self.amounts = self.amounts + amounts.alignment
        self.calls += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "wer": self.wer.to_dict(),
            "cer": self.cer.to_dict(),
            "amounts": self.amounts.to_dict(),
        }


def combine(alignments: Iterable[Alignment]) -> Alignment:
    """Sum alignments. Empty returns the zero alignment."""
    total = Alignment()
    for una in alignments:
        total = total + una
    return total
