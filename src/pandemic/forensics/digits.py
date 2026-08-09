"""Digit-distribution tests for human interference in count data.

Two independent signals, sensitive to different failure modes.

First digit (Benford's law). Counts growing multiplicatively across several
orders of magnitude should have leading digit ``d`` with probability
``log10(1 + 1/d)``. Invented numbers come out too uniform. Daily COVID counts span
1 to 10^6 within a single country, which is the regime where Benford applies.

Terminal digit. The last digit of a real count above ~100 carries no information
and should be uniform on 0-9. Excess 0s and 5s mean the number was rounded,
estimated or typed rather than counted. This is stronger evidence than Benford,
because the null needs no assumption about the underlying process beyond the
counts being large.

Either way, non-conformity flags a series for investigation and is not proof of
fabrication. A series can fail Benford for innocent reasons, such as a hard
reporting cap or a short run stuck in one order of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# P(first digit = d) under Benford, d = 1..9
BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))

# Nigrini's MAD thresholds for first-digit conformity (Nigrini 2012,
# "Benford's Law: Applications for Forensic Accounting", Table 5.1).
NIGRINI_BOUNDS = ((0.006, "close"), (0.012, "acceptable"), (0.015, "marginal"))


@dataclass(frozen=True)
class DigitTest:
    """Result of one digit-distribution test."""

    n: int
    statistic: float
    p_value: float
    mad: float
    verdict: str
    observed: np.ndarray
    expected: np.ndarray


def _classify_mad(mad: float) -> str:
    for bound, label in NIGRINI_BOUNDS:
        if mad < bound:
            return label
    return "nonconformity"


def first_digit_test(values: np.ndarray, min_value: int = 10) -> DigitTest:
    """Chi-square goodness-of-fit of leading digits against Benford's law.

    Values below ``min_value`` are dropped: a series of 0s, 1s and 2s carries no
    Benford signal and would swamp the test with structural zeros.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= min_value)]
    if v.size < 50:
        return DigitTest(int(v.size), np.nan, np.nan, np.nan, "insufficient-data",
                         np.zeros(9), BENFORD_P * max(v.size, 1))

    lead = (v / np.power(10.0, np.floor(np.log10(v)))).astype(int)
    observed = np.bincount(lead, minlength=10)[1:10].astype(float)
    expected = BENFORD_P * observed.sum()

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=8))
    mad = float(np.abs(observed / observed.sum() - BENFORD_P).mean())

    return DigitTest(int(v.size), chi2, p, mad, _classify_mad(mad), observed, expected)


def terminal_digit_test(values: np.ndarray, min_value: int = 100) -> DigitTest:
    """Chi-square test that final digits are uniform on 0-9.

    Restricted to values >= ``min_value`` so the last digit is genuinely
    uninformative. ``mad`` here is mean absolute deviation from 0.1 per digit;
    it doubles as an interpretable "heaping severity" score.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= min_value)]
    if v.size < 50:
        return DigitTest(int(v.size), np.nan, np.nan, np.nan, "insufficient-data",
                         np.zeros(10), np.full(10, 0.1))

    last = np.mod(v.astype(np.int64), 10)
    observed = np.bincount(last, minlength=10).astype(float)
    expected = np.full(10, observed.sum() / 10.0)

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=9))
    mad = float(np.abs(observed / observed.sum() - 0.1).mean())

    # Round-number heaping is specifically an excess of 0 and 5.
    share_05 = float((observed[0] + observed[5]) / observed.sum())
    verdict = "heaping" if (p < 0.01 and share_05 > 0.25) else (
        "non-uniform" if p < 0.01 else "uniform"
    )
    return DigitTest(int(v.size), chi2, p, mad, verdict, observed, expected)


def round_number_excess(values: np.ndarray, min_value: int = 100) -> float:
    """Excess share of values ending in 0 or 5, over the uniform expectation 0.2.

    Zero means no heaping; 0.8 means every reported number is a round one.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= min_value)]
    if v.size < 50:
        return np.nan
    last = np.mod(v.astype(np.int64), 10)
    share = float(np.isin(last, (0, 5)).mean())
    return max(0.0, (share - 0.2) / 0.8)
