"""Composite Data Reliability Index (DRI).

Seven independent detectors are folded into one 0-100 score per country, where
100 means "nothing suspicious found" and low scores mean the series should not
be fed to a model without repair.

Two things keep it from being an arbitrary index:

* every component is capped at a stated severity, so one extreme metric cannot
  dominate and the weights keep their intended meaning
* the ranking is stress-tested against its own weights. ``weight_sensitivity``
  redraws the weight vector from a Dirichlet hundreds of times and reports how
  much the ranking moves. An index whose bottom 20 reshuffles under small weight
  changes is not measuring anything, so it is worth checking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pandemic.config import get_logger
from pandemic.forensics.digits import first_digit_test, round_number_excess, terminal_digit_test
from pandemic.forensics.flags import summarise_country

log = get_logger(__name__)


# component -> (cap, weight, human-readable failure mode)
COMPONENTS: dict[str, tuple[float, float, str]] = {
    "neg_rate":       (0.02, 0.20, "impossible negative daily counts"),
    "gap_severity":   (14.0, 0.15, "reporting stopped mid-epidemic"),
    "fill_severity":  (14.0, 0.15, "series padded with a constant"),
    "dump_rate":      (6.00, 0.15, "batch releases of withheld cases"),
    "weekday_amp":    (1.00, 0.15, "cases reported in weekday batches"),
    "heaping":        (0.50, 0.15, "counts rounded to 0 or 5"),
    "benford_mad":    (0.03, 0.05, "leading digits depart from Benford"),
}


def _penalty(value: float, cap: float) -> float:
    """Scale a raw metric into [0, 1], saturating at ``cap``."""
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value / cap, 0.0, 1.0))


def build_scorecard(panel: pd.DataFrame, *, value_col: str = "new_cases",
                    min_active_days: int = 180, n_permutations: int = 500) -> pd.DataFrame:
    """Run every detector on every country and assemble the index.

    ``panel`` must have columns ``entity``, ``date`` and ``value_col``.
    Countries with fewer than ``min_active_days`` of real transmission are
    dropped -- the detectors need a series to work on, and scoring a country
    with 40 active days against one with 1,200 is not a comparison.
    """
    rows = []
    excluded: list[dict] = []
    for entity, g in panel.groupby("entity", sort=True):
        g = g.sort_values("date")
        stats_ = summarise_country(g, value_col=value_col, n_permutations=n_permutations)
        if stats_["active_days"] < min_active_days:
            # Not a pass: a country that barely reported cannot be *scored*, but
            # that silence is itself a finding, so it is recorded rather than
            # dropped. Tanzania and Nicaragua land here.
            excluded.append({
                "entity": entity,
                "active_days": stats_["active_days"],
                "peak_daily": stats_["peak_daily"],
                "weekday_amplitude": stats_["weekday_amplitude"],
            })
            continue

        values = g[value_col].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values >= 0)]
        benford = first_digit_test(values)
        terminal = terminal_digit_test(values)

        years_active = max(stats_["active_days"] / 365.25, 1e-9)
        rows.append({
            "entity": entity,
            **{k: v for k, v in stats_.items() if k != "weekday_multipliers"},
            "weekday_multipliers": stats_["weekday_multipliers"],
            # normalised raw metrics
            "neg_rate": stats_["neg_days"] / max(stats_["active_days"], 1),
            "gap_severity": float(stats_["gap_longest"]),
            "fill_severity": float(stats_["fill_longest"]),
            "dump_rate": stats_["dump_days"] / years_active,
            "weekday_amp": stats_["weekday_amplitude"],
            "heaping": round_number_excess(values),
            "benford_mad": benford.mad,
            # test detail, kept for the report
            "benford_p": benford.p_value,
            "benford_verdict": benford.verdict,
            "terminal_p": terminal.p_value,
            "terminal_verdict": terminal.verdict,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no country met the min_active_days threshold")

    # Penalties, then the weighted index.
    weights = {k: w for k, (_, w, _) in COMPONENTS.items()}
    for name, (cap, _, _) in COMPONENTS.items():
        df[f"pen_{name}"] = df[name].apply(lambda v, c=cap: _penalty(v, c))

    pen_cols = [f"pen_{k}" for k in COMPONENTS]
    w = np.array([weights[k] for k in COMPONENTS])

    pen = df[pen_cols].to_numpy(dtype=float)
    # Renormalise weights row-wise over the components that are actually
    # available, so a country missing one test is not silently rewarded.
    avail = np.isfinite(pen)
    pen_filled = np.where(avail, pen, 0.0)
    w_row = np.where(avail, w[None, :], 0.0)
    w_sum = w_row.sum(axis=1)
    total_penalty = np.divide((pen_filled * w_row).sum(axis=1), w_sum,
                              out=np.zeros(len(df)), where=w_sum > 0)

    df["penalty"] = total_penalty
    df["reliability_index"] = 100.0 * (1.0 - total_penalty)
    df["worst_component"] = [
        max(COMPONENTS, key=lambda k, i=i: (df[f"pen_{k}"].iloc[i] * weights[k])
            if np.isfinite(df[f"pen_{k}"].iloc[i]) else -1)
        for i in range(len(df))
    ]
    df["worst_component_desc"] = df["worst_component"].map({k: v[2] for k, v in COMPONENTS.items()})

    out = df.sort_values("reliability_index").reset_index(drop=True)
    out.attrs["excluded"] = sorted(excluded, key=lambda r: -(r["weekday_amplitude"] or 0))
    out.attrs["min_active_days"] = min_active_days
    return out


def weight_sensitivity(scorecard: pd.DataFrame, n_draws: int = 500,
                       concentration: float = 20.0, seed: int = 0) -> dict:
    """How much does the ranking depend on the weights we chose?

    Draws weight vectors from ``Dirichlet(concentration * w0)`` -- centred on the
    chosen weights, with spread controlled by ``concentration`` -- and measures
    Spearman rank correlation against the published ranking, plus how stable the
    bottom-20 membership is.

    A median rank correlation near 1 means the ranking is a property of the data
    rather than of our judgement about weights.
    """
    pen_cols = [f"pen_{k}" for k in COMPONENTS]
    pen = scorecard[pen_cols].to_numpy(dtype=float)
    avail = np.isfinite(pen)
    pen_filled = np.where(avail, pen, 0.0)

    w0 = np.array([w for _, w, _ in COMPONENTS.values()])
    base = scorecard["reliability_index"].to_numpy()
    base_bottom = set(np.argsort(base)[:20].tolist())

    rng = np.random.default_rng(seed)
    correlations = np.empty(n_draws)
    overlaps = np.empty(n_draws)

    for i in range(n_draws):
        w = rng.dirichlet(concentration * w0)
        w_row = np.where(avail, w[None, :], 0.0)
        w_sum = w_row.sum(axis=1)
        p = np.divide((pen_filled * w_row).sum(axis=1), w_sum,
                      out=np.zeros(len(scorecard)), where=w_sum > 0)
        score = 100.0 * (1.0 - p)
        correlations[i] = stats.spearmanr(base, score).statistic
        overlaps[i] = len(base_bottom & set(np.argsort(score)[:20].tolist())) / 20.0

    return {
        "spearman_median": float(np.median(correlations)),
        "spearman_p05": float(np.percentile(correlations, 5)),
        "bottom20_overlap_median": float(np.median(overlaps)),
        "bottom20_overlap_p05": float(np.percentile(overlaps, 5)),
        "n_draws": n_draws,
    }
