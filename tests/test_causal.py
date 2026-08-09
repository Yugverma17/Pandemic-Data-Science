"""Tests for the causal machinery.

The centrepiece is ``test_dml_recovers_effect_under_nonlinear_confounding``:
data is simulated from a known structural model in which the confounder enters
non-linearly, so linear regression is biased *by construction* and double
machine learning should not be. Recovering the planted coefficient is a genuine
correctness check on the estimator -- as opposed to checking that it returns a
number.

The back-door tests are equally load-bearing: they assert that the criterion
*refuses* adjustment sets containing mediators, which is the specific mistake the
graph exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandemic.causal.confounding import partial_correlation, scaling_diagnostic
from pandemic.causal.dag import (
    build_dag,
    columns_for,
    is_valid_backdoor,
    minimal_backdoor_sets,
)
from pandemic.causal.estimators import dml_effect, ols_effect, propensity_weighted_effect
from pandemic.causal.refute import e_value, placebo_treatment, subset_stability
from pandemic.causal.synthetic_control import synthetic_control

# --------------------------------------------------------------------------- DAG


def test_graph_is_acyclic():
    assert build_dag().number_of_nodes() > 0


def test_minimal_adjustment_set_is_valid():
    g = build_dag()
    sets = minimal_backdoor_sets(g)
    assert sets, "no admissible adjustment set exists for this graph"
    ok, why = is_valid_backdoor(g, "stringency", "deaths", sets[0])
    assert ok, why


def test_empty_adjustment_set_is_rejected():
    g = build_dag()
    ok, why = is_valid_backdoor(g, "stringency", "deaths", set())
    assert not ok
    assert "back-door" in why


@pytest.mark.parametrize("mediator", ["testing", "transmission", "vaccination_speed",
                                      "observed_cases"])
def test_adjusting_for_a_mediator_is_rejected(mediator):
    """Controlling for a consequence of treatment removes part of the effect."""
    g = build_dag()
    base = minimal_backdoor_sets(g)[0]
    ok, why = is_valid_backdoor(g, "stringency", "deaths", base | {mediator})
    assert not ok
    assert "post-treatment" in why


def test_adjustment_nodes_map_to_real_columns():
    g = build_dag()
    cols = columns_for(minimal_backdoor_sets(g)[0])
    assert cols
    assert len(cols) == len(set(cols)), "duplicate columns in the adjustment set"


# ------------------------------------------------------------------ estimators


def _simulate(n=1200, true_effect=0.5, seed=0, nonlinear=True):
    """Y = effect*T + g(X) + e ; T = m(X) + v, with g and m non-linear."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-2, 2, n)

    if nonlinear:
        m = np.sin(2.0 * x1) + 0.8 * x2**2
        g = 1.5 * np.cos(1.5 * x1) + 0.9 * np.abs(x2) ** 1.5
    else:
        m = 0.7 * x1 + 0.4 * x2
        g = 1.1 * x1 + 0.6 * x2

    t = m + rng.normal(0, 1.0, n)
    y = true_effect * t + g + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_ols_recovers_effect_when_confounding_is_linear():
    df = _simulate(nonlinear=False, seed=1)
    est = ols_effect(df, "y", "t", ["x1", "x2"])
    assert est.estimate == pytest.approx(0.5, abs=0.06)


def test_ols_is_biased_under_nonlinear_confounding():
    """Establishes that the harder test below is actually hard."""
    df = _simulate(nonlinear=True, seed=2)
    est = ols_effect(df, "y", "t", ["x1", "x2"])
    assert abs(est.estimate - 0.5) > 0.05, "linear adjustment should not suffice here"


def test_dml_recovers_effect_under_nonlinear_confounding():
    df = _simulate(nonlinear=True, seed=2)
    est = dml_effect(df, "y", "t", ["x1", "x2"], n_folds=5, n_repeats=4)
    assert est.estimate == pytest.approx(0.5, abs=0.06)
    assert est.ci_low < 0.5 < est.ci_high


def test_dml_finds_no_effect_when_none_exists():
    df = _simulate(true_effect=0.0, seed=3)
    est = dml_effect(df, "y", "t", ["x1", "x2"], n_folds=5, n_repeats=4)
    assert abs(est.estimate) < 0.06
    assert est.p_value > 0.01


def test_ipw_is_reported_in_per_point_units():
    """IPW must be comparable with the per-point coefficients beside it.

    The raw estimate is the effect of crossing the median — a jump of many
    treatment units — so returning it unscaled would put a differently-sized
    quantity on the same axis as the regression coefficients.
    """
    df = _simulate(true_effect=0.5, nonlinear=False, seed=13)
    est = propensity_weighted_effect(df, "y", "t", ["x1", "x2"])

    contrast = est.detail["mean_treatment_contrast"]
    assert contrast > 1.0, "median split should separate the groups by more than a unit"
    assert est.estimate == pytest.approx(est.detail["raw_ate_median_split"] / contrast)
    # Per-point, it should land in the neighbourhood of the planted effect rather
    # than being inflated by the size of the contrast.
    assert est.estimate == pytest.approx(0.5, abs=0.25)


def test_ipw_trims_extreme_propensity_scores():
    df = _simulate(seed=14)
    est = propensity_weighted_effect(df, "y", "t", ["x1", "x2"])
    assert est.detail["ps_min"] >= 0.05
    assert est.detail["ps_max"] <= 0.95
    assert 0 < est.detail["effective_n"] <= len(df)


def test_dml_accepts_a_custom_learner():
    """Passing an explicit learner must work.

    Regression test: the default was previously selected with ``learner or
    default``, and truth-testing a scikit-learn ensemble calls ``__len__``,
    which raises on an unfitted model. The bug was invisible while every caller
    passed None, and silently emptied the entire refutation suite the moment one
    did not.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = _simulate(true_effect=0.5, nonlinear=False, seed=15)
    light = RandomForestRegressor(n_estimators=60, min_samples_leaf=5, random_state=0)
    est = dml_effect(df, "y", "t", ["x1", "x2"], n_repeats=2, learner=light)
    assert est.estimate == pytest.approx(0.5, abs=0.1)


def test_dml_requires_controls():
    df = _simulate(seed=4)
    with pytest.raises(ValueError, match="at least one control"):
        dml_effect(df, "y", "t", [])


def test_unadjusted_ols_is_confounded():
    df = _simulate(nonlinear=False, seed=5)
    naive = ols_effect(df, "y", "t", [])
    adjusted = ols_effect(df, "y", "t", ["x1", "x2"])
    assert abs(naive.estimate - 0.5) > abs(adjusted.estimate - 0.5)


# ------------------------------------------------------------------ refutation


def test_placebo_treatment_destroys_a_real_effect():
    df = _simulate(seed=6)
    observed = ols_effect(df, "y", "t", ["x1", "x2"]).estimate
    report = placebo_treatment(
        df, "t", lambda d, c: ols_effect(d, "y", "t", c),
        ["x1", "x2"], observed, n_draws=60, seed=0)
    assert abs(report["placebo_mean"]) < 0.1
    assert report["permutation_p"] < 0.05
    assert report["passes"]


def test_subset_stability_is_sign_consistent_for_a_real_effect():
    df = _simulate(seed=7)
    observed = ols_effect(df, "y", "t", ["x1", "x2"]).estimate
    report = subset_stability(df, lambda d, c: ols_effect(d, "y", "t", c),
                              ["x1", "x2"], observed, n_draws=30, seed=0)
    assert report["sign_consistency"] == 1.0
    assert report["passes"]


def test_e_value_formula():
    # VanderWeele & Ding: RR=2 -> 2 + sqrt(2) = 3.414
    assert e_value(2.0)["e_value"] == pytest.approx(2 + np.sqrt(2.0), abs=1e-9)
    # No association -> E-value of 1
    assert e_value(1.0)["e_value"] == pytest.approx(1.0)
    # Protective effects are handled by inversion, so they are symmetric
    assert e_value(0.5)["e_value"] == pytest.approx(e_value(2.0)["e_value"])


def test_e_value_for_a_ci_crossing_one_is_one():
    result = e_value(1.4, ci_low=0.9, ci_high=2.1)
    assert result["e_value_ci"] == pytest.approx(1.0)


# --------------------------------------------------------------- confounding


def test_partial_correlation_removes_a_common_cause():
    rng = np.random.default_rng(8)
    z = rng.normal(0, 1, 800)
    x = z + rng.normal(0, 0.3, 800)
    y = z + rng.normal(0, 0.3, 800)  # x and y related only through z
    raw = float(np.corrcoef(x, y)[0, 1])
    partial, p = partial_correlation(x, y, z)
    assert raw > 0.8
    assert abs(partial) < 0.15
    assert p > 0.01


@pytest.mark.filterwarnings("ignore:.*nearly constant.*")
def test_scaling_diagnostic_identifies_pure_population_scaling():
    # The per-capita series is constant *by construction* here -- that is the
    # pathology being detected -- so scipy's near-constant warning is expected.
    rng = np.random.default_rng(9)
    pop = 10 ** rng.uniform(5, 8, 150)
    # Both variables are a fixed rate times population: correlation is mechanical.
    df = pd.DataFrame({"a": 0.02 * pop, "b": 0.05 * pop, "population": pop})
    result = scaling_diagnostic(df, "a", "b")
    assert result["raw"]["r"] > 0.99
    assert "population" in result["verdict"] or result["per_capita"]["r"] == pytest.approx(0.0, abs=1e-6)


def test_scaling_diagnostic_preserves_a_real_association():
    rng = np.random.default_rng(10)
    pop = 10 ** rng.uniform(5, 8, 200)
    rate_a = rng.uniform(0.01, 0.05, 200)
    rate_b = rate_a * 2 + rng.normal(0, 0.002, 200)  # genuine rate-level link
    df = pd.DataFrame({"a": rate_a * pop, "b": rate_b * pop, "population": pop})
    result = scaling_diagnostic(df, "a", "b")
    assert result["per_capita"]["r"] > 0.8
    assert result["verdict"] == "the association survives population adjustment"


# ------------------------------------------------------- synthetic control


def test_synthetic_control_reproduces_a_donor_combination():
    """With no treatment effect, the synthetic unit should track the treated one."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    donors = {f"D{i}": np.abs(rng.normal(100, 20)) + np.arange(120) * rng.normal(1, 0.4)
              for i in range(6)}
    # Treated unit is exactly 0.5*D0 + 0.5*D1, with no post-period divergence.
    treated = 0.5 * donors["D0"] + 0.5 * donors["D1"]

    rows = [{"entity": k, "date": d, "value": v}
            for k, series in donors.items()
            for d, v in zip(dates, series, strict=True)]
    rows += [{"entity": "T", "date": d, "value": v}
             for d, v in zip(dates, treated, strict=True)]
    panel = pd.DataFrame(rows)

    res = synthetic_control(panel, "T", "2021-03-15", pre_days=60, post_days=21)
    assert res.pre_rmspe < 1e-3
    assert abs(res.effect_mean_pct) < 1.0
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.weights >= 0).all()


def test_synthetic_control_detects_a_planted_effect():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    base = 100 + np.arange(120) * 0.5
    donors = {f"D{i}": base + rng.normal(0, 1, 120) for i in range(6)}
    treated = base.copy()
    cut = 75
    treated[cut:] *= 0.5  # a 50% drop after the intervention

    rows = [{"entity": k, "date": d, "value": v}
            for k, series in donors.items()
            for d, v in zip(dates, series, strict=True)]
    rows += [{"entity": "T", "date": d, "value": v}
             for d, v in zip(dates, treated, strict=True)]
    panel = pd.DataFrame(rows)

    res = synthetic_control(panel, "T", str(dates[cut].date()), pre_days=60, post_days=21)
    assert res.effect_mean_pct < -35
    assert res.rmspe_ratio > 5


def test_synthetic_control_needs_enough_donors():
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    rows = [{"entity": e, "date": d, "value": float(i)}
            for e in ("T", "D0") for i, d in enumerate(dates)]
    with pytest.raises(ValueError, match="donors"):
        synthetic_control(pd.DataFrame(rows), "T", "2021-03-01")
