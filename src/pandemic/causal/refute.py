"""Refutation and sensitivity checks.

Each routine has a predicted outcome if the estimate is real, so a failure is
informative rather than decorative:

``placebo_treatment``     shuffle the treatment. A real effect should vanish. If a
                          randomised treatment reproduces it, we are picking up
                          structure in the covariates, not the exposure.
``random_common_cause``   add a pure-noise covariate. The estimate should not move.
``subset_stability``      re-estimate on random subsamples. Wide spread means a
                          few countries are carrying the result.
``leave_one_group_out``   drop each continent in turn. A result that only exists
                          with Europe in it is a European result.
``e_value``               how strong would an unmeasured confounder need to be to
                          explain the estimate away?

The placebo distribution also gives a permutation p-value, which avoids leaning on
asymptotic normality that a 150-country sample only roughly supports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pandemic.causal.estimators import EffectEstimate
from pandemic.config import get_logger, rng

log = get_logger(__name__)

Estimator = Callable[[pd.DataFrame, list[str]], EffectEstimate]


def placebo_treatment(df: pd.DataFrame, treatment: str, estimator: Estimator,
                      controls: list[str], observed: float,
                      n_draws: int = 200, seed: int = 0) -> dict:
    """Re-estimate with the treatment randomly permuted across countries."""
    r = rng(seed)
    placebo = []
    for _ in range(n_draws):
        d = df.copy()
        d[treatment] = r.permutation(d[treatment].to_numpy())
        try:
            placebo.append(estimator(d, controls).estimate)
        except Exception:  # noqa: BLE001 - a degenerate draw should not abort the suite
            continue

    arr = np.asarray(placebo)
    if arr.size == 0:
        return {"n": 0}
    # Two-sided permutation p-value with the +1 correction.
    p = float((np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1))
    return {
        "n": int(arr.size),
        "placebo_mean": float(arr.mean()),
        "placebo_sd": float(arr.std(ddof=1)),
        "placebo_abs_p95": float(np.percentile(np.abs(arr), 95)),
        "observed": float(observed),
        "permutation_p": p,
        "passes": bool(abs(arr.mean()) < abs(observed) / 3 and p < 0.10),
    }


def random_common_cause(df: pd.DataFrame, estimator: Estimator, controls: list[str],
                        observed: float, n_draws: int = 30, seed: int = 0) -> dict:
    """Add an irrelevant random covariate; the estimate should be unchanged."""
    r = rng(seed)
    out = []
    for i in range(n_draws):
        d = df.copy()
        col = f"_noise_{i}"
        d[col] = r.normal(size=len(d))
        try:
            out.append(estimator(d, [*controls, col]).estimate)
        except Exception:  # noqa: BLE001
            continue

    arr = np.asarray(out)
    if arr.size == 0:
        return {"n": 0}
    shift = float(np.mean(arr) - observed)
    denom = abs(observed) if observed else 1.0
    return {
        "n": int(arr.size),
        "mean_estimate": float(arr.mean()),
        "observed": float(observed),
        "mean_shift": shift,
        "relative_shift": float(shift / denom),
        "passes": bool(abs(shift / denom) < 0.10),
    }


def subset_stability(df: pd.DataFrame, estimator: Estimator, controls: list[str],
                     observed: float, fraction: float = 0.8,
                     n_draws: int = 60, seed: int = 0) -> dict:
    """Re-estimate on random subsamples to expose leverage from a few countries."""
    r = rng(seed)
    n_keep = max(int(round(fraction * len(df))), 20)
    out = []
    for _ in range(n_draws):
        idx = r.choice(len(df), size=n_keep, replace=False)
        try:
            out.append(estimator(df.iloc[idx].reset_index(drop=True), controls).estimate)
        except Exception:  # noqa: BLE001
            continue

    arr = np.asarray(out)
    if arr.size == 0:
        return {"n": 0}
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "n": int(arr.size),
        "fraction": fraction,
        "mean_estimate": float(arr.mean()),
        "p2.5": float(lo), "p97.5": float(hi),
        "observed": float(observed),
        "sign_consistency": float(np.mean(np.sign(arr) == np.sign(observed))),
        "passes": bool(np.mean(np.sign(arr) == np.sign(observed)) > 0.90),
    }


def leave_one_group_out(df: pd.DataFrame, estimator: Estimator, controls: list[str],
                        group_col: str, observed: float, min_remaining: int = 40) -> dict:
    """Drop each group in turn; report the range of estimates that survive."""
    results = {}
    for group in sorted(df[group_col].dropna().unique()):
        remaining = df[df[group_col] != group]
        if len(remaining) < min_remaining:
            continue
        try:
            results[str(group)] = float(estimator(remaining.reset_index(drop=True),
                                                  controls).estimate)
        except Exception:  # noqa: BLE001
            continue

    if not results:
        return {"n": 0}
    vals = np.asarray(list(results.values()))
    return {
        "n": len(results),
        "by_group": results,
        "min": float(vals.min()), "max": float(vals.max()),
        "observed": float(observed),
        "sign_consistency": float(np.mean(np.sign(vals) == np.sign(observed))),
        "passes": bool(np.all(np.sign(vals) == np.sign(observed))),
    }


def e_value(risk_ratio: float, ci_low: float | None = None,
            ci_high: float | None = None) -> dict:
    """E-value for unmeasured confounding (VanderWeele & Ding 2017, *Ann Intern Med*).

    The E-value is the minimum strength of association -- on the risk-ratio scale
    -- that an unmeasured confounder would need with *both* the treatment and the
    outcome, above and beyond the measured covariates, to fully explain away the
    observed association.

    Reading it: an E-value of 1.3 means a modest unmeasured confounder suffices,
    so the finding is fragile. An E-value of 4 means the confounder would have to
    be stronger than almost anything already measured, which is a much harder
    story to tell.
    """
    def _ev(rr: float) -> float:
        if not np.isfinite(rr) or rr <= 0:
            return np.nan
        if rr < 1:
            rr = 1.0 / rr
        return float(rr + np.sqrt(rr * (rr - 1.0)))

    out = {"risk_ratio": float(risk_ratio), "e_value": _ev(risk_ratio)}

    if ci_low is not None and ci_high is not None:
        if ci_low <= 1.0 <= ci_high:
            # The interval already includes "no effect"; nothing to explain away.
            out["e_value_ci"] = 1.0
        else:
            limit = ci_low if ci_low > 1.0 else ci_high
            out["e_value_ci"] = _ev(limit)
        out["ci"] = [float(ci_low), float(ci_high)]
    return out


def run_suite(df: pd.DataFrame, treatment: str, estimator: Estimator,
              controls: list[str], estimate: EffectEstimate,
              group_col: str = "continent", *, contrast: float = 10.0,
              n_placebo: int = 200, seed: int = 0) -> dict:
    """Run every refutation and return one report.

    ``contrast`` is the treatment change the risk ratio is expressed over -- here
    a 10-point move on the 0-100 stringency scale, roughly the gap between a
    country that closed schools and one that did not.
    """
    obs = estimate.estimate
    log.info("running refutation suite (%d placebo draws)", n_placebo)

    rr = float(np.exp(obs * contrast))
    rr_lo = float(np.exp(estimate.ci_low * contrast))
    rr_hi = float(np.exp(estimate.ci_high * contrast))

    report = {
        "estimate": estimate.as_row(),
        "contrast_points": contrast,
        "risk_ratio_per_contrast": rr,
        "placebo_treatment": placebo_treatment(df, treatment, estimator, controls,
                                               obs, n_draws=n_placebo, seed=seed),
        "random_common_cause": random_common_cause(df, estimator, controls, obs, seed=seed),
        "subset_stability": subset_stability(df, estimator, controls, obs, seed=seed),
        "leave_one_group_out": leave_one_group_out(df, estimator, controls,
                                                   group_col, obs),
        "e_value": e_value(rr, min(rr_lo, rr_hi), max(rr_lo, rr_hi)),
    }
    checks = [v.get("passes") for v in report.values()
              if isinstance(v, dict) and "passes" in v]
    report["n_checks_passed"] = int(sum(bool(c) for c in checks))
    report["n_checks"] = len(checks)

    # Each routine catches its own exceptions so that one degenerate draw cannot
    # abort the suite. The cost is that a *systematic* failure -- a broken
    # estimator, a bad argument -- returns an empty report that reads exactly
    # like a clean run. Anything that produced no estimates at all is therefore
    # surfaced loudly rather than being reported as "0 of 0 checks passed".
    empty = [name for name, v in report.items()
             if isinstance(v, dict) and v.get("n") == 0]
    if empty:
        log.error("refutations produced no estimates: %s -- the estimator is "
                  "failing on every draw, not passing", ", ".join(empty))
    report["failed_to_run"] = empty
    return report
