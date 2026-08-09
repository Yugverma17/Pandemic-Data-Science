"""Tests for the forecasting stack.

The tests that matter most here are not the ones checking that functions run.
They are:

* ``test_rt_estimate_is_causal`` -- the rolling-origin backtest precomputes R_t
  once over each country's full series and then indexes into it. That is only
  legitimate if R_t at index i depends on nothing after i. The whole backtest is
  invalid if this fails, so it is asserted rather than assumed.
* ``test_wis_matches_hand_calculation`` -- WIS is the headline metric; an error
  in it silently reorders every model.
* ``test_rt_recovers_known_r`` -- the estimator is checked against data
  simulated from the model it assumes, where the right answer is known exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandemic.forecast.conformal import ConformalCalibrator, calibrate
from pandemic.forecast.metrics import (
    coverage,
    interval_score,
    mase_scale,
    relative_skill,
    weighted_interval_score,
)
from pandemic.forecast.models import LogLinearDrift, Persistence, SeriesCache, trailing_average
from pandemic.forecast.rt import (
    discretise_serial_interval,
    estimate_rt,
    project_renewal,
    total_infectiousness,
)

# ------------------------------------------------------------------ serial interval


def test_serial_interval_is_a_normalised_pmf():
    w = discretise_serial_interval()
    assert w[0] == 0.0, "same-day transmission would make Lambda_t depend on I_t"
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(("mean", "sd"), [(4.7, 2.9), (5.5, 3.5), (3.0, 1.5)])
def test_serial_interval_recovers_its_moments(mean, sd):
    """The naive CDF-difference discretisation is half a day too slow; this one is not."""
    w = discretise_serial_interval(mean=mean, sd=sd, max_days=60)
    days = np.arange(w.size)
    got_mean = float(np.sum(w * days))
    got_sd = float(np.sqrt(np.sum(w * (days - got_mean) ** 2)))
    assert got_mean == pytest.approx(mean, abs=0.1)
    assert got_sd == pytest.approx(sd, abs=0.15)


def test_serial_interval_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        discretise_serial_interval(mean=0.5, sd=2.0)


# ---------------------------------------------------------------------------- R_t


def _simulate_constant_r(r: float, n: int = 160, seed_size: float = 1e6) -> np.ndarray:
    """Deterministic renewal epidemic with a known, constant reproduction number."""
    w = discretise_serial_interval()
    inc = np.zeros(n)
    inc[:5] = seed_size
    for t in range(5, n):
        lam = sum(inc[t - s] * w[s] for s in range(1, min(w.size, t + 1)))
        inc[t] = r * lam
    return inc


@pytest.mark.parametrize("true_r", [0.8, 1.0, 1.5, 2.5])
def test_rt_recovers_known_r(true_r):
    inc = _simulate_constant_r(true_r)
    est = estimate_rt(inc, tau=7)
    recovered = float(np.nanmean(est.mean[40:70]))
    assert recovered == pytest.approx(true_r, rel=0.02)


def test_rt_estimate_is_causal():
    """R_t at index i must not change when future data is appended or altered.

    This is the identity the backtest's precomputed cache relies on.
    """
    rng = np.random.default_rng(0)
    inc = np.abs(rng.normal(500, 120, 300)).cumsum() % 4000 + 50

    full = estimate_rt(inc, tau=7).mean
    for cut in (120, 200, 260):
        truncated = estimate_rt(inc[:cut], tau=7).mean
        np.testing.assert_allclose(truncated, full[:cut], rtol=1e-9, atol=1e-9)

    # Changing only the future must leave the past untouched. assert_allclose
    # rather than np.allclose: the leading entries are NaN by design (R_t is not
    # identified before infectious pressure exists), and np.allclose treats NaN
    # as unequal to itself.
    perturbed = inc.copy()
    perturbed[200:] *= 7.0
    np.testing.assert_allclose(estimate_rt(perturbed, tau=7).mean[:200],
                               full[:200], rtol=1e-9, atol=1e-9)


def test_rt_is_nan_before_infectious_pressure_exists():
    inc = np.zeros(60)
    inc[40:] = 100.0
    est = estimate_rt(inc, tau=7)
    assert np.all(np.isnan(est.mean[:40])), "R_t is not identified with no prior cases"


def test_total_infectiousness_excludes_the_current_day():
    w = discretise_serial_interval()
    inc = np.zeros(20)
    inc[10] = 1000.0
    lam = total_infectiousness(inc, w)
    assert lam[10] == 0.0
    assert lam[11] == pytest.approx(1000.0 * w[1])


def test_renewal_projection_reproduces_the_generating_process():
    w = discretise_serial_interval()
    inc = _simulate_constant_r(1.3, n=140, seed_size=10.0)
    projected = project_renewal(inc[:110], 1.3, 25, w=w, damping=1.0)
    np.testing.assert_allclose(projected, inc[110:135], rtol=1e-8)


def test_renewal_damping_pulls_growth_toward_one():
    w = discretise_serial_interval()
    inc = _simulate_constant_r(1.6, n=100, seed_size=100.0)
    undamped = project_renewal(inc, 1.6, 30, w=w, damping=1.0)
    damped = project_renewal(inc, 1.6, 30, w=w, damping=0.9)
    assert damped[-1] < undamped[-1]

    shrinking = project_renewal(inc, 0.5, 30, w=w, damping=0.9)
    shrinking_hard = project_renewal(inc, 0.5, 30, w=w, damping=1.0)
    assert shrinking[-1] > shrinking_hard[-1], "damping must lift a decline toward 1 too"


# --------------------------------------------------------------------- scoring


def test_wis_matches_hand_calculation():
    levels = np.array([0.25, 0.5, 0.75])
    preds = np.array([[10.0, 12.0, 14.0]])

    # Inside the interval: IS = width = 4, no penalty. WIS = (0.5*0 + 0.25*4)/1.5
    got = weighted_interval_score(np.array([12.0]), levels, preds)[0]
    assert got == pytest.approx((0.5 * 0.0 + 0.25 * 4.0) / 1.5)

    # Above the upper bound: IS = 4 + (2/0.5)*(20-14) = 28
    got = weighted_interval_score(np.array([20.0]), levels, preds)[0]
    assert got == pytest.approx((0.5 * 8.0 + 0.25 * 28.0) / 1.5)


def test_wis_is_zero_for_a_perfect_point_forecast():
    levels = np.array([0.25, 0.5, 0.75])
    preds = np.array([[7.0, 7.0, 7.0]])
    assert weighted_interval_score(np.array([7.0]), levels, preds)[0] == pytest.approx(0.0)


def test_wis_rewards_sharpness_when_both_forecasts_are_correct():
    levels = np.array([0.25, 0.5, 0.75])
    sharp = np.array([[11.0, 12.0, 13.0]])
    vague = np.array([[2.0, 12.0, 22.0]])
    y = np.array([12.0])
    assert (weighted_interval_score(y, levels, sharp)[0]
            < weighted_interval_score(y, levels, vague)[0])


def test_wis_decomposition_sums_to_the_total():
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    rng = np.random.default_rng(1)
    preds = np.sort(rng.normal(50, 10, size=(40, 5)), axis=1)
    y = rng.normal(50, 15, size=40)
    parts = weighted_interval_score(y, levels, preds, decompose=True)
    total = parts["sharpness"] + parts["underprediction"] + parts["overprediction"] \
        + parts["median_ae"]
    np.testing.assert_allclose(parts["wis"], total, rtol=1e-9)


def test_wis_repairs_crossing_quantiles():
    levels = np.array([0.25, 0.5, 0.75])
    crossed = np.array([[14.0, 12.0, 10.0]])  # decreasing: invalid
    ordered = np.array([[10.0, 12.0, 14.0]])
    y = np.array([13.0])
    assert (weighted_interval_score(y, levels, crossed)[0]
            == pytest.approx(weighted_interval_score(y, levels, ordered)[0]))


def test_wis_requires_a_median():
    with pytest.raises(ValueError, match="median"):
        weighted_interval_score(np.array([1.0]), np.array([0.25, 0.75]),
                                np.array([[1.0, 2.0]]))


def test_interval_score_penalty_scales_with_confidence():
    """Missing a 95% interval must hurt more than missing a 50% one."""
    y, lo, hi = np.array([20.0]), np.array([10.0]), np.array([14.0])
    assert interval_score(y, lo, hi, 0.05)[0] > interval_score(y, lo, hi, 0.5)[0]


def test_coverage_and_relative_skill():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(y, np.array([0.0] * 4), np.array([2.5] * 4)) == pytest.approx(0.5)
    assert relative_skill(np.array([2.0, 8.0]), np.array([4.0, 4.0])) == pytest.approx(1.0)
    assert relative_skill(np.array([1.0, 1.0]), np.array([4.0, 4.0])) == pytest.approx(0.25)


def test_mase_scale_is_the_seasonal_naive_error():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    assert mase_scale(x, season=7) == pytest.approx(1.0)
    assert np.isnan(mase_scale(np.ones(5), season=7))


# ---------------------------------------------------------------------- models


def _cache(values: np.ndarray, population: float = 1e7) -> SeriesCache:
    dates = pd.date_range("2020-03-01", periods=values.size, freq="D")
    return SeriesCache("Testland", dates.to_numpy(), values, population)


def test_trailing_average_requires_a_full_window():
    avg = trailing_average(np.arange(10, dtype=float))
    assert np.isnan(avg[:6]).all()
    assert avg[6] == pytest.approx(3.0)


def test_persistence_returns_the_current_level():
    c = _cache(np.full(80, 100.0))
    assert Persistence().predict(c, 70, 14) == pytest.approx(100.0)


def test_drift_extrapolates_exponential_growth():
    """A series doubling weekly should be projected to double again."""
    days = np.arange(90)
    values = 100.0 * 2 ** (days / 7.0)
    c = _cache(values)
    pred = LogLinearDrift(window=14).predict(c, 80, 7)
    assert pred == pytest.approx(c.avg[80] * 2, rel=0.05)


def test_damped_drift_is_more_conservative_than_undamped():
    days = np.arange(90)
    c = _cache(100.0 * 2 ** (days / 7.0))
    plain = LogLinearDrift(window=14).predict(c, 80, 14)
    damped = LogLinearDrift(window=14, damping=0.9).predict(c, 80, 14)
    assert damped < plain


def test_series_cache_running_peak_is_monotone():
    rng = np.random.default_rng(3)
    c = _cache(np.abs(rng.normal(400, 200, 200)))
    assert np.all(np.diff(c.runmax) >= -1e-9)


# ------------------------------------------------------------------- conformal


def test_calibrate_produces_ordered_non_negative_quantiles():
    residuals = np.array([-0.4, -0.1, 0.0, 0.15, 0.6])
    levels = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    q = calibrate(100.0, residuals, levels)
    assert np.all(np.diff(q) >= 0)
    assert np.all(q >= 0)


def test_conformal_intervals_attain_nominal_coverage_when_exchangeable():
    """Split conformal's guarantee holds under exchangeability; check it empirically."""
    rng = np.random.default_rng(7)
    cal = ConformalCalibrator((0.05, 0.5, 0.95), max_history=400)

    truth, lower, upper = [], [], []
    for i in range(600):
        actual = float(np.expm1(np.log1p(1000.0) + rng.normal(0, 0.3)))
        q, _ = cal.quantiles("m", "e", 7, 1000.0)
        if i > 150 and np.isfinite(q).all():
            truth.append(actual)
            lower.append(q[0])
            upper.append(q[2])
        cal.add_residual("m", "e", 7, actual, 1000.0)

    covered = np.mean((np.array(truth) >= np.array(lower))
                      & (np.array(truth) <= np.array(upper)))
    assert covered == pytest.approx(0.90, abs=0.05)


def test_calibrator_falls_back_to_pooled_residuals():
    cal = ConformalCalibrator((0.1, 0.5, 0.9))
    for i in range(30):
        cal.add_residual("m", "other", 7, 100.0 + i, 100.0)
    _, source = cal.residuals_for("m", "unseen-country", 7)
    assert source == "pooled"
