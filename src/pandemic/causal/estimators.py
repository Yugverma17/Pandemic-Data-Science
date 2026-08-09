"""Effect estimators: OLS and double machine learning with cross-fitting.

A linear model forces every confounder to enter additively and linearly. COVID
mortality risk rises roughly exponentially with age, so a linear age term leaves
residual confounding that ends up in the treatment coefficient. Double machine
learning (Chernozhukov et al., 2018, Econometrics Journal 21(1):C1-C68) lets
flexible learners soak up the confounders while keeping a valid confidence
interval for the parameter of interest.

Two pieces make that work:

Neyman orthogonality. The estimating equation is residual-on-residual, so the
score is insensitive to first-order error in the nuisance functions. Feeding a
machine-learned E[Y|X] into an ordinary regression instead would inherit the
learner's regularisation bias.

Cross-fitting. Each observation's nuisance predictions come from a model fitted
on the folds excluding it. Without this the learner's overfitting correlates with
the residuals and the estimate stays biased even at large n.

One fold assignment can shift the estimate, so everything runs over several
random splits. The median is reported, with the across-split spread folded into
the standard error (Chernozhukov et al., section 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from pandemic.config import SEED


@dataclass
class EffectEstimate:
    """A point estimate with everything needed to judge it."""

    method: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    detail: dict = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_row(self) -> dict:
        return {
            "method": self.method, "estimate": self.estimate, "std_error": self.std_error,
            "ci_low": self.ci_low, "ci_high": self.ci_high, "p_value": self.p_value,
            "n": self.n,
        }


def _design(df: pd.DataFrame, treatment: str, controls: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = df[treatment].to_numpy(float)
    x = df[controls].to_numpy(float) if controls else np.empty((len(df), 0))
    return t, x


def ols_effect(df: pd.DataFrame, outcome: str, treatment: str,
               controls: list[str] | None = None,
               method: str | None = None) -> EffectEstimate:
    """Least squares with HC3 heteroskedasticity-robust standard errors.

    HC3 rather than the classical errors because country-level data are
    strongly heteroskedastic and the sample is small, which is exactly the
    regime where HC3 outperforms HC0/HC1 (Long & Ervin 2000).
    """
    controls = controls or []
    y = df[outcome].to_numpy(float)
    t, x = _design(df, treatment, controls)
    design = sm.add_constant(np.column_stack([t, x]) if x.size else t.reshape(-1, 1))

    fit = sm.OLS(y, design).fit(cov_type="HC3")
    idx = 1  # column 0 is the intercept
    ci = fit.conf_int(alpha=0.05)[idx]
    return EffectEstimate(
        method=method or ("OLS, unadjusted" if not controls else "OLS, adjusted"),
        estimate=float(fit.params[idx]),
        std_error=float(fit.bse[idx]),
        ci_low=float(ci[0]), ci_high=float(ci[1]),
        p_value=float(fit.pvalues[idx]),
        n=int(len(df)),
        detail={"r_squared": float(fit.rsquared), "n_controls": len(controls)},
    )


def _default_learner(seed: int) -> RandomForestRegressor:
    """A deliberately conservative forest.

    Shallow-ish trees with a leaf-size floor: with ~150 countries the nuisance
    models are the part most likely to overfit, and cross-fitting protects the
    estimate's validity but not its precision.
    """
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=5, max_features=0.6,
        # n_jobs=1 deliberately: with ~150 rows the forests are tiny, thread
        # dispatch costs more than it saves, and nesting joblib inside the
        # repeat loop emits a warning per fit.
        random_state=seed, n_jobs=1,
    )


def dml_effect(df: pd.DataFrame, outcome: str, treatment: str,
               controls: list[str], *, n_folds: int = 5, n_repeats: int = 20,
               learner=None, seed: int = SEED,
               method: str = "Double ML (partialling out)") -> EffectEstimate:
    """Partialling-out DML for the partially linear model ``Y = theta*T + g(X) + e``."""
    y = df[outcome].to_numpy(float)
    t, x = _design(df, treatment, controls)
    n = len(df)
    if x.shape[1] == 0:
        raise ValueError("DML needs at least one control; use ols_effect otherwise")

    thetas, variances, fit_quality = [], [], []

    for rep in range(n_repeats):
        rng_seed = seed + rep
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
        y_hat = np.empty(n)
        t_hat = np.empty(n)

        # `learner is None`, never `learner or ...`: truth-testing a scikit-learn
        # ensemble calls __len__, which reads self.estimators_ and raises on an
        # unfitted model. The `or` form works only while the argument is None.
        base = _default_learner(rng_seed) if learner is None else learner

        for train, test in folds.split(x):
            m_y = clone(base)
            m_t = clone(base)
            m_y.fit(x[train], y[train])
            m_t.fit(x[train], t[train])
            y_hat[test] = m_y.predict(x[test])
            t_hat[test] = m_t.predict(x[test])

        y_res = y - y_hat
        t_res = t - t_hat
        denom = float(np.mean(t_res**2))
        if denom <= 0:
            continue

        theta = float(np.mean(t_res * y_res) / denom)
        # Asymptotic variance of the orthogonal score.
        psi = (y_res - theta * t_res) * t_res
        var = float(np.mean(psi**2) / (denom**2) / n)

        thetas.append(theta)
        variances.append(var)
        fit_quality.append({
            "outcome_r2": 1.0 - float(np.var(y_res) / np.var(y)),
            "treatment_r2": 1.0 - float(np.var(t_res) / np.var(t)),
        })

    if not thetas:
        raise ValueError("DML failed: no usable split (treatment fully explained by controls)")

    thetas_arr = np.asarray(thetas)
    theta = float(np.median(thetas_arr))
    # Combine within-split variance and across-split spread, so a result that is
    # an artefact of one lucky fold assignment cannot look precise.
    var = float(np.median(np.asarray(variances) + (thetas_arr - theta) ** 2))
    se = float(np.sqrt(max(var, 0.0)))

    z = theta / se if se > 0 else np.inf
    p = float(2 * stats.norm.sf(abs(z)))
    return EffectEstimate(
        method=method, estimate=theta, std_error=se,
        ci_low=theta - 1.96 * se, ci_high=theta + 1.96 * se,
        p_value=p, n=n,
        detail={
            "n_folds": n_folds,
            "n_repeats": len(thetas),
            "theta_spread_iqr": float(np.subtract(*np.percentile(thetas_arr, [75, 25]))),
            "outcome_r2": float(np.mean([f["outcome_r2"] for f in fit_quality])),
            "treatment_r2": float(np.mean([f["treatment_r2"] for f in fit_quality])),
            "n_controls": len(controls),
        },
    )


def propensity_weighted_effect(df: pd.DataFrame, outcome: str, treatment: str,
                               controls: list[str], *, seed: int = SEED,
                               n_folds: int = 5) -> EffectEstimate:
    """Binarised treatment with stabilised inverse-probability weights.

    A second identification strategy on the same question. Continuous stringency
    is dichotomised at the median into "strict early response" versus not, a
    generalised propensity score is cross-fitted, and the ATE is formed from
    stabilised weights trimmed at [0.05, 0.95] to prevent one country with an
    extreme score from carrying the estimate.

    Agreement with DML is evidence the answer is not an artefact of one
    estimator's assumptions; disagreement would be a warning worth reporting.
    """
    from sklearn.ensemble import RandomForestClassifier

    y = df[outcome].to_numpy(float)
    t_cont, x = _design(df, treatment, controls)
    cutoff = float(np.median(t_cont))
    t = (t_cont > cutoff).astype(int)
    n = len(df)

    ps = np.empty(n)
    for train, test in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                     max_features=0.6, random_state=seed, n_jobs=1)
        clf.fit(x[train], t[train])
        ps[test] = clf.predict_proba(x[test])[:, 1]

    ps = np.clip(ps, 0.05, 0.95)
    p_treated = float(t.mean())
    w = np.where(t == 1, p_treated / ps, (1 - p_treated) / (1 - ps))

    treated = t == 1
    mu1 = float(np.sum(w[treated] * y[treated]) / np.sum(w[treated]))
    mu0 = float(np.sum(w[~treated] * y[~treated]) / np.sum(w[~treated]))
    ate = mu1 - mu0

    # Influence-function standard error for the weighted difference in means.
    infl = w * (t * (y - mu1) / p_treated - (1 - t) * (y - mu0) / (1 - p_treated))
    se = float(np.std(infl, ddof=1) / np.sqrt(n))

    # Rescale to *per stringency point* before returning. The raw ATE answers a
    # different question from the other estimators -- the effect of crossing the
    # median, a jump of roughly 20 points -- and reporting it beside per-point
    # coefficients on one axis silently compares incompatible units. The contrast
    # is the actual mean gap in treatment between the two groups.
    contrast = float(t_cont[treated].mean() - t_cont[~treated].mean())
    if not np.isfinite(contrast) or abs(contrast) < 1e-9:
        contrast = 1.0
    ate_pp, se_pp = ate / contrast, se / contrast

    z = ate_pp / se_pp if se_pp > 0 else np.inf
    return EffectEstimate(
        method="IPW (median split, per point)",
        estimate=ate_pp, std_error=se_pp,
        ci_low=ate_pp - 1.96 * se_pp, ci_high=ate_pp + 1.96 * se_pp,
        p_value=float(2 * stats.norm.sf(abs(z))), n=n,
        detail={
            "cutoff": cutoff,
            "raw_ate_median_split": ate,
            "mean_treatment_contrast": contrast,
            "n_treated": int(t.sum()),
            "ps_min": float(ps.min()), "ps_max": float(ps.max()),
            # Effective sample size: how much of the data the weights actually use.
            "effective_n": float(w.sum() ** 2 / np.sum(w**2)),
        },
    )
