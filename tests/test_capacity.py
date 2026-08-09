"""Tests for the cases-to-ICU capacity model.

The properties asserted here are the ones the model's usefulness depends on:
the kernels are proper distributions with the intended moments, the convolution
is strictly causal, and census *lags* admissions rather than tracking them. That
last one is the entire operational claim of Pillar 4 -- if occupancy moved with
cases, the model would provide no warning time and would not be worth building.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandemic.capacity.convolve import (
    admissions_to_icu_census,
    cases_to_admissions,
    cases_to_icu_census,
    gamma_kernel,
    los_survival,
    peak_lag_days,
)
from pandemic.capacity.risk import assess_region, exceedance_curve, lead_time, simulate_census
from pandemic.capacity.validate import validate_country
from pandemic.config import EPI

# ------------------------------------------------------------------- kernels


def test_gamma_kernel_is_a_normalised_pmf():
    k = gamma_kernel(7.0, 4.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(k >= 0)


@pytest.mark.parametrize(("mean", "sd"), [(7.0, 4.0), (12.0, 8.0), (4.0, 2.0)])
def test_gamma_kernel_recovers_its_mean(mean, sd):
    k = gamma_kernel(mean, sd, max_days=120)
    got = float(np.sum(k * np.arange(k.size)))
    assert got == pytest.approx(mean, abs=0.15)


def test_gamma_kernel_rejects_bad_parameters():
    with pytest.raises(ValueError):
        gamma_kernel(0.0, 4.0)


def test_los_survival_is_monotone_and_integrates_to_the_mean():
    """The area under a survival curve is the mean of the distribution."""
    s = los_survival(12.0, 8.0, max_days=300)
    assert np.all(np.diff(s) <= 1e-12), "survival must be non-increasing"
    assert s[0] == pytest.approx(1.0, abs=0.02)
    assert s.sum() == pytest.approx(12.0, rel=0.05)


# --------------------------------------------------------------- convolution


def test_convolution_is_causal():
    """Occupancy today cannot depend on cases reported tomorrow."""
    cases = np.zeros(60)
    cases[30] = 1000.0
    census = cases_to_icu_census(cases)
    assert np.all(census[:30] == 0.0)
    assert census[31:].sum() > 0


def test_appending_future_cases_does_not_change_the_past():
    rng = np.random.default_rng(0)
    cases = np.abs(rng.normal(500, 150, 200))
    short = cases_to_icu_census(cases[:120])
    long = cases_to_icu_census(cases)
    np.testing.assert_allclose(short, long[:120], rtol=1e-10, atol=1e-10)


def test_admissions_scale_linearly_with_the_hospitalisation_rate():
    cases = np.full(120, 1000.0)
    a1 = cases_to_admissions(cases, ihr=0.02)
    a2 = cases_to_admissions(cases, ihr=0.04)
    np.testing.assert_allclose(2.0 * a1, a2, rtol=1e-10)


def test_steady_admissions_give_census_equal_to_admissions_times_stay():
    """At equilibrium, census = admission rate x mean stay (Little's law).

    The relevant mean is the *discrete* one the model actually uses. Summing
    ``P(X > s)`` over integers gives ``E[ceil(X)]``, which sits about half a day
    above the continuous mean, so the exact target is the kernel's own sum
    rather than the nominal 12 days.
    """
    admissions = np.full(400, 100.0)
    survival = los_survival(12.0, 8.0)
    census = admissions_to_icu_census(admissions, icu_share=1.0,
                                      los_mean=12.0, los_sd=8.0)

    assert census[-1] == pytest.approx(100.0 * survival.sum(), rel=1e-6)
    # And that discrete mean must sit within half a day of the intended one.
    assert survival.sum() == pytest.approx(12.5, abs=0.3)


def test_census_peaks_after_cases_peak():
    """The operational claim: watching cases buys warning time."""
    t = np.arange(400)
    cases = 10_000 * np.exp(-((t - 200) ** 2) / (2 * 30.0**2))
    census = cases_to_icu_census(cases)
    lag = peak_lag_days(cases, census)
    assert lag > 7, "the model must produce a usable lead time"
    assert lag < 30


def test_peak_lag_is_zero_for_empty_input():
    assert peak_lag_days(np.zeros(10), np.zeros(10)) == 0


def test_longer_stays_delay_and_raise_the_peak():
    t = np.arange(400)
    cases = 10_000 * np.exp(-((t - 200) ** 2) / (2 * 30.0**2))
    short = cases_to_icu_census(cases, los_mean=6.0)
    long = cases_to_icu_census(cases, los_mean=18.0)
    assert long.max() > short.max()
    assert int(np.argmax(long)) > int(np.argmax(short))


# ------------------------------------------------------------------- risk


def test_simulate_census_returns_the_expected_shape_and_is_ordered():
    rng = np.random.default_rng(1)
    cases = np.abs(rng.normal(2000, 400, 180))
    paths = simulate_census(cases, horizon=21, n_draws=40, seed=0)
    assert paths.shape == (40, 180 + 21)
    assert np.all(paths >= 0)
    q = np.percentile(paths, [10, 50, 90], axis=0)
    assert np.all(q[0] <= q[1] + 1e-9) and np.all(q[1] <= q[2] + 1e-9)


def test_simulation_uncertainty_widens_with_horizon():
    rng = np.random.default_rng(2)
    cases = np.abs(rng.normal(2000, 300, 180))
    paths = simulate_census(cases, horizon=28, n_draws=60, seed=0)
    spread = np.percentile(paths, 90, axis=0) - np.percentile(paths, 10, axis=0)
    assert spread[-1] > spread[180], "the fan must widen into the projection"


def test_exceedance_and_lead_time():
    paths = np.tile(np.arange(0.0, 30.0), (10, 1))
    curve = exceedance_curve(paths, threshold=10.0, start_index=0)
    assert curve[0] == 0.0
    assert curve[-1] == 1.0
    assert lead_time(curve, 0.5) == pytest.approx(11.0)


def test_lead_time_is_nan_when_no_breach_occurs():
    assert np.isnan(lead_time(np.zeros(21), 0.5))


def test_assess_region_declines_on_a_series_with_no_epidemic():
    assert assess_region(np.zeros(200), 1e7) == {}


def test_assess_region_suppresses_the_ratio_without_a_prior_peak():
    """A region with no previous wave gets no utilisation figure, not a huge one."""
    cases = np.zeros(200)
    cases[190:] = 3.0  # a tiny, brand-new outbreak in a large population
    result = assess_region(cases, population=5e7, n_draws=40)
    if result:
        assert not result["benchmark_usable"]
        assert np.isnan(result["utilisation_median"])


# -------------------------------------------------------------- validation


def test_validate_country_recovers_a_planted_multiplier():
    """Feed the validator observed data generated by the model itself."""
    rng = np.random.default_rng(3)
    t = np.arange(400)
    cases = 5_000 * np.exp(-((t - 200) ** 2) / (2 * 40.0**2)) + 100
    truth = cases_to_icu_census(cases)

    g = pd.DataFrame({
        "entity": "Testland",
        "date": pd.date_range("2020-03-01", periods=400, freq="D"),
        "new_cases": cases,
        "icu_patients": 3.0 * truth * rng.uniform(0.97, 1.03, 400),
    })

    result = validate_country(g)
    assert result is not None
    assert result["correlation"] > 0.99
    assert result["level_multiplier"] == pytest.approx(3.0, rel=0.05)
    assert abs(result["peak_timing_error_days"]) <= 2


def test_validate_country_returns_none_without_enough_observations():
    g = pd.DataFrame({
        "entity": "Sparse",
        "date": pd.date_range("2020-03-01", periods=60, freq="D"),
        "new_cases": np.full(60, 100.0),
        "icu_patients": np.full(60, 5.0),
    })
    assert validate_country(g) is None


def test_validation_window_restricts_evaluation_only():
    """Restricting the window must not restart the model from an empty ICU."""
    t = np.arange(400)
    cases = 5_000 * np.exp(-((t - 150) ** 2) / (2 * 40.0**2)) + 100
    truth = cases_to_icu_census(cases)
    g = pd.DataFrame({
        "entity": "Testland",
        "date": pd.date_range("2020-03-01", periods=400, freq="D"),
        "new_cases": cases,
        "icu_patients": 2.0 * truth,
    })
    late = validate_country(g, start="2020-08-01", min_days=100)
    assert late is not None
    assert late["correlation"] > 0.99
    assert late["level_multiplier"] == pytest.approx(2.0, rel=0.02)


# ------------------------------------------------------------------ config


def test_published_parameter_ranges_bracket_their_central_estimates():
    assert EPI.ihr_low < EPI.ihr_mean < EPI.ihr_high
    assert EPI.icu_share_low < EPI.icu_share_mean < EPI.icu_share_high
    assert EPI.icu_los_mean > 0 and EPI.icu_los_sd > 0
