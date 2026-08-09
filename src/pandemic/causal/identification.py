"""Tests for whether the design identifies anything.

The refutation checks in :mod:`pandemic.causal.refute` ask whether an estimate is
stable. They say nothing about whether it is causal. An estimate driven entirely
by reverse causality is perfectly stable, passes every placebo, and gets a large
E-value. Stability and identification are different properties, and treating them
as the same is how a confounded number ends up with a confident standard error.

The two tests here go at identification directly.

Response channel. Governments tighten restrictions because cases are rising. If
stringency is well predicted by case growth just before the policy window, then
treatment is assigned by a cause of the outcome and adjusting for fixed country
traits cannot fix it.

Effect before cause. Deaths within 21 days of the window opening cannot have been
prevented by the policy, since infection to death runs about three weeks and a
policy needs one or two more to change infections. So an estimated "effect" on
those early deaths reads out the confounding directly. If it is as large as the
effect on one-year deaths, the headline estimate is measuring the same thing:
which regions were already in trouble.

A design that fails these is not fixed by a better estimator. It is the wrong
design, and the useful output is a bound plus a statement of what would work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.causal.estimators import EffectEstimate, ols_effect


def response_channel(df: pd.DataFrame, treatment: str, controls: list[str],
                     growth_col: str = "onset_growth_rate",
                     concurrent_col: str = "log_cases_per_million_window") -> dict:
    """How much of the treatment is explained by the outbreak around it?

    Two probes, because they have very different power:

    ``lagged``      treatment against case growth measured *before* the window
                    opened. Weak by construction -- at the 100th case almost every
                    country still has a tiny, noisy series, so a null here is
                    uninformative rather than reassuring.
    ``concurrent``  treatment against the outbreak size *during* the window. This
                    is what governments were actually watching. It cannot separate
                    cause from effect on its own -- restrictions also change case
                    counts -- but a strong association establishes that treatment
                    and outcome are determined simultaneously, which is exactly
                    the condition under which the back-door adjustment fails.
    """
    lagged = ols_effect(df, outcome=treatment, treatment=growth_col,
                        controls=[c for c in controls if c != growth_col],
                        method="stringency ~ pre-window case growth")

    concurrent = None
    if concurrent_col in df.columns:
        sub = df.dropna(subset=[concurrent_col])
        if len(sub) > len(controls) + 10:
            concurrent = ols_effect(
                sub, outcome=treatment, treatment=concurrent_col,
                controls=[c for c in controls if c != concurrent_col],
                method="stringency ~ outbreak size during the window")

    simultaneous = bool(concurrent is not None and concurrent.p_value < 0.05
                        and concurrent.estimate > 0)
    lagged_sig = bool(lagged.p_value < 0.05 and lagged.estimate > 0)

    if simultaneous:
        interp = ("treatment tracks the concurrent outbreak: policy and deaths are "
                  "determined simultaneously, so adjusting for time-invariant "
                  "covariates cannot identify the effect")
    elif lagged_sig:
        interp = "treatment responds to the outbreak observed before the window"
    else:
        interp = "no response channel detected by either probe"

    return {
        "lagged": lagged.as_row(),
        "concurrent": concurrent.as_row() if concurrent else None,
        "raw_correlation_lagged": float(df[[treatment, growth_col]].corr().iloc[0, 1]),
        "raw_correlation_concurrent": (
            float(df[[treatment, concurrent_col]].corr().iloc[0, 1])
            if concurrent_col in df.columns else None),
        "reverse_causality_detected": bool(simultaneous or lagged_sig),
        "interpretation": interp,
    }


def effect_before_cause(df: pd.DataFrame, treatment: str, controls: list[str],
                        *, main_outcome: str = "log_deaths_per_million_1y",
                        placebo_outcome: str = "log_deaths_per_million_first21d",
                        estimator=None) -> dict:
    """Compare the estimated effect on a period the policy could not have affected.

    Returns both estimates and their ratio. A ratio near or above 1 means the
    headline estimate carries no more causal content than a quantity that is
    definitionally unaffected by treatment.
    """
    estimator = estimator or (lambda d, y: ols_effect(d, y, treatment, controls))

    sample = df.dropna(subset=[main_outcome, placebo_outcome, treatment, *controls])
    main: EffectEstimate = estimator(sample, main_outcome)
    placebo: EffectEstimate = estimator(sample, placebo_outcome)

    ratio = (placebo.estimate / main.estimate) if main.estimate else np.nan
    contaminated = bool(placebo.p_value < 0.05
                        and np.sign(placebo.estimate) == np.sign(main.estimate)
                        and abs(ratio) > 0.3)

    return {
        "n": int(len(sample)),
        "main": main.as_row(),
        "placebo_pre_period": placebo.as_row(),
        "ratio_placebo_to_main": float(ratio) if np.isfinite(ratio) else None,
        "identification_compromised": contaminated,
        "interpretation": (
            "the estimate on the pre-causal window is of comparable size and the "
            "same sign, so the headline association is confounding rather than effect"
            if contaminated else
            "no detectable association on the pre-causal window, which is what a "
            "credible causal estimate requires"
        ),
    }


def assess(df: pd.DataFrame, treatment: str, controls: list[str],
           estimate: EffectEstimate, **kwargs) -> dict:
    """Run both identification tests and issue a verdict on the design."""
    channel = response_channel(df, treatment, controls)
    timing = effect_before_cause(df, treatment, controls, **kwargs)

    reverse = channel["reverse_causality_detected"]
    identified = not (reverse or timing["identification_compromised"])

    if identified:
        verdict = (
            "Neither identification probe fires. That is necessary but not "
            "sufficient: both tests have limited power in this design -- the "
            "pre-window series is short and noisy, and the 21-day falsification "
            "window contains few deaths for most countries. A null here should be "
            "read as 'no contradiction found', not as licence to call the estimate "
            "causal."
        )
    else:
        verdict = (
            "The design does NOT identify a causal effect. Treatment is assigned in "
            "response to the outbreak, and the association reproduces on a window "
            "the policy could not have affected. The reported coefficient is a "
            "measure of which countries were already in trouble when they acted -- "
            "reporting it as the effect of policy would invert the direction of the "
            "real relationship. Identification requires variation in policy that is "
            "unrelated to local epidemic severity: staggered adoption with matched "
            "pre-trends, a discontinuity, or an instrument."
        )

    return {
        "response_channel": channel,
        "effect_before_cause": timing,
        "naive_estimate": estimate.as_row(),
        "identified": identified,
        "verdict": verdict,
        "power_caveat": (
            "Both probes are one-sided: they can reveal confounding but cannot "
            "demonstrate its absence. The sign of the estimate is the stronger "
            "evidence here -- a positive coefficient means stricter responses "
            "accompany higher mortality, which is implausible as a causal effect "
            "and expected under simultaneity."
        ),
    }
