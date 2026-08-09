"""Pillar 3 -- causal analysis of why regional outcomes diverged."""

from pandemic.causal.confounding import (
    build_absolute_frame,
    partial_correlation,
    scaling_diagnostic,
)
from pandemic.causal.dag import (
    build_dag,
    columns_for,
    describe,
    is_valid_backdoor,
    minimal_backdoor_sets,
)
from pandemic.causal.dataset import (
    CONFOUNDERS,
    analysis_sample,
    build_country_design,
)
from pandemic.causal.estimators import (
    EffectEstimate,
    dml_effect,
    ols_effect,
    propensity_weighted_effect,
)
from pandemic.causal.refute import e_value, run_suite
from pandemic.causal.synthetic_control import (
    placebo_in_time,
    placebo_inference,
    synthetic_control,
)

__all__ = [
    "CONFOUNDERS",
    "EffectEstimate",
    "analysis_sample",
    "build_absolute_frame",
    "build_country_design",
    "build_dag",
    "columns_for",
    "describe",
    "dml_effect",
    "e_value",
    "is_valid_backdoor",
    "minimal_backdoor_sets",
    "ols_effect",
    "partial_correlation",
    "placebo_in_time",
    "placebo_inference",
    "propensity_weighted_effect",
    "run_suite",
    "scaling_diagnostic",
    "synthetic_control",
]
