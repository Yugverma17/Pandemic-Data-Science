"""Synthetic control for a single treated unit.

Abadie & Gardeazabal (2003) and Abadie, Diamond & Hainmueller (2010). When one
region adopts a policy and no single other region is a good comparison, build one:
a convex combination of untreated regions whose pre-intervention trajectory tracks
the treated region. The post-intervention gap between the real unit and its
synthetic twin estimates the effect.

Two constraints do the work. Weights are non-negative and sum to one, so the
estimate can only ever be a weighted average of things that actually happened, with
no extrapolation past the donor pool. And weights are chosen on pre-period fit
alone, keeping the post period out of sample.

Inference is by placebo permutation rather than a standard error. The same
procedure runs pretending each donor was treated, and the treated unit's post/pre
RMSPE ratio is ranked in that distribution. With one treated unit there is no
sampling variation to appeal to, so the rank is the test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pandemic.config import get_logger

log = get_logger(__name__)


@dataclass
class SyntheticControlResult:
    treated: str
    weights: pd.Series
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    observed: pd.Series
    synthetic: pd.Series
    gap: pd.Series
    intervention: pd.Timestamp
    effect_total: float
    effect_mean_pct: float


def _fit_weights(y_treated: np.ndarray, y_donors: np.ndarray) -> np.ndarray:
    """Simplex-constrained least squares: min ||y - Dw||, w >= 0, sum(w) = 1."""
    n_donors = y_donors.shape[1]
    w0 = np.full(n_donors, 1.0 / n_donors)

    def loss(w):
        return float(np.mean((y_treated - y_donors @ w) ** 2))

    def grad(w):
        resid = y_donors @ w - y_treated
        return 2.0 * (y_donors.T @ resid) / y_treated.size

    result = minimize(
        loss, w0, jac=grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    w = np.clip(result.x, 0, None)
    total = w.sum()
    return w / total if total > 0 else w0


def synthetic_control(panel: pd.DataFrame, treated: str, intervention: str,
                      *, value_col: str = "value", unit_col: str = "entity",
                      date_col: str = "date", donors: list[str] | None = None,
                      pre_days: int = 60, post_days: int = 21) -> SyntheticControlResult:
    """Fit a synthetic control for ``treated`` around ``intervention``."""
    cut = pd.Timestamp(intervention)
    pre_start = cut - pd.Timedelta(days=pre_days)
    post_end = cut + pd.Timedelta(days=post_days)

    wide = (panel[(panel[date_col] >= pre_start) & (panel[date_col] <= post_end)]
            .pivot_table(index=date_col, columns=unit_col, values=value_col)
            .sort_index())

    if treated not in wide.columns:
        raise ValueError(f"treated unit {treated!r} not present in the panel")

    pool = [c for c in wide.columns if c != treated]
    if donors is not None:
        pool = [c for c in pool if c in donors]
    # A donor with gaps cannot be used: interpolating it would invent the very
    # counterfactual the method is supposed to estimate.
    wide = wide[[treated, *pool]].dropna(axis=1, how="any")
    pool = [c for c in wide.columns if c != treated]
    if len(pool) < 3:
        raise ValueError(f"only {len(pool)} complete donors available; need at least 3")

    # Weights are fitted on the pre-period alone; the post-period is never seen
    # by the optimiser, which is what keeps the counterfactual out of sample.
    pre = wide[wide.index < cut]

    w = _fit_weights(pre[treated].to_numpy(float), pre[pool].to_numpy(float))
    weights = pd.Series(w, index=pool).sort_values(ascending=False)

    synthetic = pd.Series(wide[pool].to_numpy(float) @ w, index=wide.index)
    observed = wide[treated]
    gap = observed - synthetic

    pre_rmspe = float(np.sqrt(np.mean(gap[gap.index < cut] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[gap.index >= cut] ** 2)))

    post_obs = float(observed[observed.index >= cut].sum())
    post_syn = float(synthetic[synthetic.index >= cut].sum())

    return SyntheticControlResult(
        treated=treated,
        weights=weights[weights > 1e-4],
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        rmspe_ratio=post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf,
        observed=observed, synthetic=synthetic, gap=gap,
        intervention=cut,
        effect_total=post_obs - post_syn,
        effect_mean_pct=100.0 * (post_obs - post_syn) / post_syn if post_syn > 0 else np.nan,
    )


def placebo_inference(panel: pd.DataFrame, treated: str, intervention: str,
                      *, donors: list[str] | None = None,
                      min_pre_fit_ratio: float = 5.0, **kwargs) -> dict:
    """Rank the treated unit's RMSPE ratio against donor-as-treated placebos.

    Donors whose own pre-period fit is much worse than the treated unit's are
    excluded (``min_pre_fit_ratio``), following Abadie et al.: a placebo the
    method could not fit before the intervention will show a large post-period
    gap for reasons that have nothing to do with any treatment, and leaving those
    in makes the test conservative to the point of being uninformative.
    """
    real = synthetic_control(panel, treated, intervention, donors=donors, **kwargs)

    pool = donors if donors is not None else [
        u for u in panel["entity"].unique() if u != treated
    ]
    ratios, skipped = {}, []
    for unit in pool:
        try:
            alt_donors = [u for u in pool if u != unit] + [treated]
            res = synthetic_control(panel, unit, intervention, donors=alt_donors, **kwargs)
        except Exception:  # noqa: BLE001 - donors with gaps simply drop out
            skipped.append(unit)
            continue
        if real.pre_rmspe > 0 and res.pre_rmspe > min_pre_fit_ratio * real.pre_rmspe:
            skipped.append(unit)
            continue
        ratios[unit] = res.rmspe_ratio

    all_ratios = np.asarray([real.rmspe_ratio, *ratios.values()])
    rank = int(np.sum(all_ratios >= real.rmspe_ratio))
    p = rank / all_ratios.size

    return {
        "result": real,
        "placebo_ratios": ratios,
        "n_placebos": len(ratios),
        "n_skipped": len(skipped),
        "rank": rank,
        "p_value": float(p),
        "significant_at_10pct": bool(p <= 0.10),
    }


def placebo_in_time(panel: pd.DataFrame, treated: str, intervention: str,
                    *, shift_days: int = 30, **kwargs) -> dict:
    """Re-run with the intervention date moved earlier, into the pre-period.

    Nothing happened then, so a well-behaved design must find no effect. A large
    "effect" at a fake date means the method is picking up ordinary divergence
    between the unit and its donors, not the policy.
    """
    fake = pd.Timestamp(intervention) - pd.Timedelta(days=shift_days)
    res = synthetic_control(panel, treated, fake, **kwargs)
    return {
        "fake_intervention": str(fake.date()),
        "rmspe_ratio": res.rmspe_ratio,
        "effect_mean_pct": res.effect_mean_pct,
    }
