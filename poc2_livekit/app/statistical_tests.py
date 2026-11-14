"""Mann-Whitney and Fisher's exact test, in pure Python.

`scipy` would provide them ready-made, but adding it to PoC2 would change the
`uv.lock` of the environment its 15 calls ran in — the same reason
`shared/audio/resampling.py` did not add `soxr`. They are two textbook tests
and the code fits on one screen; the dependency is not worth the
reproducibility risk.

Both are computed **exactly**, not by normal approximation: with n=10 per arm
the approximation is poor precisely where it matters, in the tails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

# Beyond this, the exact distribution of U stops being cheap and we switch to
# the normal approximation. 20 per arm is ~10^5 states: instantaneous.
MAX_FOR_EXACT: Final[int] = 20

# Cap on the assignments to enumerate in the exact permutation. 10 against 10
# is 184,756: one second. Above this we switch to the normal approximation.
MAX_COMBINATIONS: Final[int] = 500_000


@dataclass(frozen=True)
class ContrastResult:
    """The result of a test, with what is needed to be able to cite it."""

    statistic: float
    p: float
    method: str
    n_a: int
    n_b: int
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.method}: statistic {self.statistic:g}, p = {self.p:.4f}"


def _spans(values: Sequence[float]) -> list[float]:
    """Ranks 1..n, averaging ties."""
    ordered_items = sorted(range(len(values)), key=lambda i: values[i])
    spans = [0.0] * len(values)
    i = 0
    while i < len(ordered_items):
        j = i
        while j + 1 < len(ordered_items) and values[ordered_items[j + 1]] == values[ordered_items[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            spans[ordered_items[k]] = average
        i = j + 1
    return spans


def _counts_of_u(n_a: int, n_b: int) -> list[int]:
    """How many of the C(n_a+n_b, n_a) assignments yield each value of U.

    It is the exact null distribution: under the null hypothesis every way of
    distributing the labels among the observations is equally likely.
    """
    # `tabla[a][b][u]`: ways of interleaving `a` observations from arm A with
    # `b` from arm B accumulating `u`. Placing one from B after `a` from A adds
    # exactly `a` to the statistic, which is the definition of U.
    tabla = [[[0] * (n_a * n_b + 1) for _ in range(n_b + 1)] for _ in range(n_a + 1)]
    tabla[0][0][0] = 1
    for a in range(n_a + 1):
        for b in range(n_b + 1):
            for u in range(n_a * n_b + 1):
                valor = tabla[a][b][u]
                if not valor:
                    continue
                if a < n_a:
                    tabla[a + 1][b][u] += valor
                if b < n_b and u + a <= n_a * n_b:
                    tabla[a][b + 1][u + a] += valor
    return tabla[n_a][n_b]


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> ContrastResult:
    """Two-sample independent test, two-tailed.

    It does not assume normality —the detector's probabilities are not
    normal— and it tests central tendency without depending on the mean.
    """
    n_a, n_b = len(a), len(b)
    if not n_a or not n_b:
        raise ValueError("Mann-Whitney needs observations in both arms")

    together = list(a) + list(b)
    spans = _spans(together)
    add_to = sum(spans[:n_a])
    u_a = add_to - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    has_ties = len(set(together)) != len(together)
    combos = math.comb(n_a + n_b, n_a)
    exact = max(n_a, n_b) <= MAX_FOR_EXACT and combos <= MAX_COMBINATIONS

    if exact and not has_ties:
        counts = _counts_of_u(n_a, n_b)
        total = sum(counts)
        # P(U <= u) by symmetry covers both tails.
        cumulative = sum(counts[: int(u) + 1])
        p = min(1.0, 2.0 * cumulative / total)
        method = "Mann-Whitney exact"
        detail = f"null distribution over {total:,} assignments"
    elif exact:
        # With ties the null of U is no longer the textbook one, but the
        # permutation is still exact: every way of distributing the observed
        # ranks —averaged ties included— between the two arms is enumerated.
        # Falling back to the normal approximation over a single tie would
        # waste an exact p-value that costs one second.
        p = _exact_permutation(spans, n_a)
        method = "Mann-Whitney exact (permutation)"
        detail = (
            f"full permutation over {combos:,} assignments, "
            "with tied ranks averaged"
        )
    else:
        mean = n_a * n_b / 2.0
        # Tie correction on the variance.
        n = n_a + n_b
        ties = {}
        for valor in together:
            ties[valor] = ties.get(valor, 0) + 1
        correction = sum(t**3 - t for t in ties.values())
        variance = (n_a * n_b / 12.0) * ((n + 1) - correction / (n * (n - 1)))
        if variance <= 0:
            return ContrastResult(u, 1.0, "Mann-Whitney (degenerate)", n_a, n_b)
        z = (abs(u - mean) - 0.5) / math.sqrt(variance)
        p = min(1.0, 2.0 * (1.0 - _normal_cumulative(z)))
        method = "Mann-Whitney normal"
        detail = "normal approximation with tie and continuity correction"

    return ContrastResult(u, p, method, n_a, n_b, detail)


def _exact_permutation(spans: Sequence[float], n_a: int) -> float:
    """Two-tailed p by enumerating every distribution of the ranks.

    Valid with ties, where the tabulated null distribution of U no longer
    applies: what gets permuted are the **observed** ranks, so the averaged
    ties travel along and the p-value remains exact.
    """
    from itertools import combinations

    total_ranks = sum(spans)
    n = len(spans)
    observed_value = sum(spans[:n_a])
    # Under the null, an arm's rank sum is centered here.
    center = n_a * total_ranks / n
    deviation = abs(observed_value - center)

    cases = 0
    extremes = 0
    for group in combinations(spans, n_a):
        cases += 1
        if abs(sum(group) - center) >= deviation - 1e-9:
            extremes += 1
    return min(1.0, extremes / cases)


def _normal_cumulative(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fisher_exact(a: int, b: int, c: int, d: int) -> ContrastResult:
    """Two-tailed Fisher test on the 2x2 table `[[a, b], [c, d]]`.

    Two-tailed by the probability criterion: it sums every table with the same
    marginals whose probability does not exceed that of the observed one.
    """
    rows = (a + b, c + d)
    columns = (a + c, b + d)
    total = a + b + c + d
    if total == 0:
        raise ValueError("the table is empty")

    def probability(x: int) -> float:
        return (
            math.comb(rows[0], x)
            * math.comb(rows[1], columns[0] - x)
            / math.comb(total, columns[0])
        )

    minimum = max(0, columns[0] - rows[1])
    maximum = min(rows[0], columns[0])
    observed_value = probability(a)
    # The relative margin keeps a floating-point error from leaving out a table
    # whose value is exactly the same as the observed one.
    p = sum(
        probability(x)
        for x in range(minimum, maximum + 1)
        if probability(x) <= observed_value * (1.0 + 1e-9)
    )
    return ContrastResult(
        statistic=float(a),
        p=min(1.0, p),
        method="Fisher exact",
        n_a=rows[0],
        n_b=rows[1],
        detail="two-tailed by the probability criterion",
    )


def describe(values: Sequence[float]) -> dict[str, float | int]:
    """Full distribution, not just the median.

    With n=10 the median alone hides exactly what matters: whether the values
    spread toward the extremes or stay bunched around the threshold.
    """
    if not values:
        return {"n": 0}
    ordered_items = sorted(values)
    n = len(ordered_items)

    def percentile(fraction: float) -> float:
        # Linear interpolation; with n=10 the quartiles fall between observations.
        position = fraction * (n - 1)
        low = math.floor(position)
        alto = math.ceil(position)
        if low == alto:
            return ordered_items[int(position)]
        peso = position - low
        return ordered_items[low] * (1 - peso) + ordered_items[alto] * peso

    return {
        "n": n,
        "min": ordered_items[0],
        "q1": percentile(0.25),
        "median_value": percentile(0.50),
        "q3": percentile(0.75),
        "max": ordered_items[-1],
        "mean": sum(ordered_items) / n,
    }
