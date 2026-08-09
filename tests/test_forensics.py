"""Tests for the reporting-forensics detectors.

Each detector is given a synthetic series containing exactly the pathology it
hunts for, and a clean series that must not trigger it. False positives matter
as much as detections here: a data-quality flag that fires on well-behaved
series is worse than no flag, because it teaches its reader to ignore it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandemic.forensics.digits import (
    BENFORD_P,
    first_digit_test,
    round_number_excess,
    terminal_digit_test,
)
from pandemic.forensics.flags import (
    backlog_dumps,
    frozen_runs,
    negative_revisions,
    weekday_profile,
)
from pandemic.forensics.naive import decompose_flags, naive_zscore_flags, rolling_zscore_flags


def _series(values, start="2020-03-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, float), index=idx)


def _wave(n=400, peak=200, height=5000.0, width=45.0) -> np.ndarray:
    """A smooth single-wave epidemic: realistic, and free of reporting artefacts."""
    t = np.arange(n)
    return height * np.exp(-((t - peak) ** 2) / (2 * width**2)) + 50.0


def _multiwave(n=900) -> np.ndarray:
    """A multi-wave epidemic shaped like a real national series.

    A single broad wave is a poor test bed for the global z-score: spread over
    the whole series it lifts the mean and standard deviation together, so the
    peak never reaches three SD. Real epidemics are a long quiet baseline
    punctuated by short, tall waves, which is exactly the shape that makes a
    global z-score fire on the peak.
    """
    t = np.arange(n)
    waves = [(200, 800.0, 40.0), (450, 8000.0, 25.0), (700, 5000.0, 20.0)]
    out = np.full(n, 100.0)
    for centre, height, width in waves:
        out = out + height * np.exp(-((t - centre) ** 2) / (2 * width**2))
    return out


# ------------------------------------------------------------------ digit tests


def test_benford_accepts_benford_distributed_data():
    rng = np.random.default_rng(0)
    # Sampling exponents uniformly gives a log-uniform variable, which obeys Benford.
    values = 10 ** rng.uniform(1, 5, 4000)
    result = first_digit_test(values)
    assert result.verdict in {"close", "acceptable"}
    assert result.p_value > 0.01


def test_benford_rejects_uniform_leading_digits():
    rng = np.random.default_rng(1)
    values = rng.integers(100, 999, 3000).astype(float)  # uniform leading digit
    result = first_digit_test(values)
    assert result.verdict == "nonconformity"
    assert result.p_value < 1e-6


def test_benford_reports_insufficient_data_rather_than_guessing():
    assert first_digit_test(np.array([12.0, 34.0, 56.0])).verdict == "insufficient-data"


def test_benford_probabilities_are_a_distribution():
    assert BENFORD_P.sum() == pytest.approx(1.0)
    assert BENFORD_P[0] == pytest.approx(np.log10(2.0))


def test_terminal_digit_accepts_uniform_final_digits():
    rng = np.random.default_rng(2)
    values = rng.integers(100, 100_000, 3000).astype(float)
    assert terminal_digit_test(values).verdict == "uniform"


def test_terminal_digit_detects_round_number_heaping():
    rng = np.random.default_rng(3)
    values = (rng.integers(10, 5000, 2000) * 5).astype(float)  # everything ends 0 or 5
    result = terminal_digit_test(values)
    assert result.verdict == "heaping"
    assert result.p_value < 1e-6
    assert round_number_excess(values) > 0.9


def test_round_number_excess_is_zero_for_clean_data():
    rng = np.random.default_rng(4)
    values = rng.integers(100, 100_000, 3000).astype(float)
    assert round_number_excess(values) < 0.15


# ------------------------------------------------------------ structural flags


def test_negative_revisions_finds_impossible_days():
    s = _series([10, 20, -5, 30, -100, 40])
    neg = negative_revisions(s)
    assert len(neg) == 2
    assert set(neg["magnitude"]) == {5.0, 100.0}


def test_no_negative_revisions_in_a_clean_series():
    assert len(negative_revisions(_series(_wave()))) == 0


def test_frozen_runs_detects_a_reporting_gap():
    values = _wave(n=400)
    values[200:210] = 0.0  # ten days of silence at the peak
    runs = frozen_runs(_series(values))
    gaps = runs[runs["kind"] == "reporting-gap"]
    assert len(gaps) == 1
    assert gaps["length"].iloc[0] == 10


def test_frozen_runs_detects_constant_fill():
    values = _wave(n=400)
    values[150:158] = 1234.0  # a constant, non-zero value: interpolated
    runs = frozen_runs(_series(values))
    fills = runs[runs["kind"] == "constant-fill"]
    assert len(fills) == 1
    assert fills["length"].iloc[0] == 8


def test_frozen_runs_ignores_quiet_periods_outside_the_epidemic():
    """Zeros before an epidemic starts are not a reporting failure."""
    values = np.concatenate([np.zeros(120), _wave(n=280)])
    runs = frozen_runs(_series(values))
    assert len(runs[runs["kind"] == "reporting-gap"]) == 0


def test_backlog_dump_is_detected_with_mass_conservation():
    values = _wave(n=400)
    # Suppress reporting for a week, then release the accumulated backlog.
    withheld = values[220:227].sum()
    values[220:227] = values[220:227] * 0.1
    values[227] += withheld * 0.9
    dumps = backlog_dumps(_series(values))
    assert dumps["is_dump"].any()
    assert dumps.loc[dumps["is_dump"], "conservation"].max() > 0.25


def test_genuine_outbreak_is_not_flagged_as_a_dump():
    """Explosive growth has elevated neighbours; a dump has depressed ones."""
    values = _wave(n=400)
    dumps = backlog_dumps(_series(values))
    assert not dumps["is_dump"].any() if len(dumps) else True


def test_weekday_profile_detects_weekly_batch_reporting():
    """A country reporting once a week must show an extreme amplitude."""
    n = 400
    weekly = np.zeros(n)
    trend = _wave(n)
    for t in range(n):
        if t % 7 == 2:  # everything lands on one weekday
            weekly[t] = trend[max(t - 6, 0): t + 1].sum()
    prof = weekday_profile(_series(weekly), n_permutations=200, seed=0)
    assert prof["amplitude"] > 3.0
    assert prof["p_value"] < 0.01


def test_weekday_profile_is_flat_for_continuous_reporting():
    rng = np.random.default_rng(5)
    values = _wave(n=400) * rng.uniform(0.95, 1.05, 400)
    prof = weekday_profile(_series(values), n_permutations=200, seed=0)
    assert prof["amplitude"] < 0.25
    assert prof["p_value"] > 0.05


def test_weekday_profile_reports_nan_on_a_short_series():
    prof = weekday_profile(_series(np.full(20, 100.0)), n_permutations=50)
    assert np.isnan(prof["amplitude"])


# -------------------------------------------------- naive detector benchmark


def test_naive_zscore_flags_the_wave_peak_not_anomalies():
    """The central claim of Pillar 1, as an assertion.

    On a clean epidemic with no reporting artefacts at all, the textbook detector
    still fires -- and every flag is attributed to the trend.
    """
    s = _series(_multiwave())
    flags = naive_zscore_flags(s, threshold=3.0)
    assert flags.sum() > 0, "the naive detector fires on an artefact-free series"

    decomposed = decompose_flags(s, flags)
    assert (decomposed["mechanism"] == "trend").all()
    assert not decomposed["is_true_anomaly"].any()
    # And every flag sits near a wave peak, which is the point: the detector
    # rediscovers the epidemic rather than finding anything wrong with the data.
    assert decomposed["trend_pctile"].min() > 0.9


def test_decomposition_attributes_a_real_dump_correctly():
    values = _wave(n=400)
    withheld = values[218:225].sum()
    values[218:225] *= 0.1
    values[225] += withheld * 0.9
    s = _series(values)

    decomposed = decompose_flags(s, pd.Series(True, index=s.index))
    dumps = decomposed[decomposed["mechanism"] == "dump"]
    assert len(dumps) >= 1
    assert dumps["is_true_anomaly"].all()


def test_rolling_zscore_is_defined_and_bounded():
    s = _series(_wave(n=300))
    flags = rolling_zscore_flags(s, window=14, threshold=3.0)
    assert flags.dtype == bool
    assert len(flags) == len(s)
    assert not flags.iloc[:14].any(), "no flags before the window is full"
