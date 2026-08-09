"""Scoring rules for probabilistic forecasts.

Main metric is the Weighted Interval Score (Bracher, Ray, Gneiting & Reich, 2021,
PLoS Comput Biol 17(2):e1008618), used by the US and European COVID-19 Forecast
Hubs. It is a proper scoring rule, and it decomposes into

    WIS = dispersion + underprediction penalty + overprediction penalty

so a bad score can be read as "too confident" or "biased low" rather than just
"worse". MAE on the point forecast would not distinguish those, and would not
notice a model that is usually right but occasionally wildly overconfident.
"""

from __future__ import annotations

import numpy as np


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float) -> np.ndarray:
    """Interval score for a central (1 - alpha) prediction interval.

    Sharpness plus a penalty of ``2/alpha`` times the distance by which the
    observation falls outside. Lower is better.
    """
    y, lower, upper = np.asarray(y, float), np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    under = (2.0 / alpha) * np.clip(lower - y, 0, None)
    over = (2.0 / alpha) * np.clip(y - upper, 0, None)
    return width + under + over


def weighted_interval_score(y: np.ndarray, quantile_levels: np.ndarray,
                            quantile_preds: np.ndarray,
                            *, decompose: bool = False):
    """WIS for forecasts supplied as a set of predictive quantiles.

    Parameters
    ----------
    y : (n,) observations
    quantile_levels : (q,) increasing levels, which must include 0.5 and pair up
        symmetrically around it (e.g. 0.025/0.975, 0.10/0.90, 0.25/0.75)
    quantile_preds : (n, q) predicted values at those levels
    decompose : also return the sharpness / under / over components

    Returns
    -------
    (n,) array of scores, or a dict of component arrays when ``decompose``.
    """
    levels = np.asarray(quantile_levels, float)
    preds = np.asarray(quantile_preds, float)
    y = np.asarray(y, float)

    if preds.ndim != 2 or preds.shape[1] != levels.size:
        raise ValueError(f"quantile_preds must be (n, {levels.size}), got {preds.shape}")
    if not np.isclose(levels, 0.5).any():
        raise ValueError("quantile_levels must contain the median (0.5)")

    # Enforce monotone quantiles: crossing is a bug in the forecaster, and
    # sorting is the standard, score-improving repair.
    preds = np.sort(preds, axis=1)

    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    median = preds[:, median_idx]

    lower_levels = levels[levels < 0.5]
    alphas, scores, widths, unders, overs = [], [], [], [], []
    for lo in lower_levels:
        hi = 1.0 - lo
        hi_idx = int(np.argmin(np.abs(levels - hi)))
        lo_idx = int(np.argmin(np.abs(levels - lo)))
        if not np.isclose(levels[hi_idx], hi):
            continue  # unpaired level: not part of a central interval
        alpha = 2.0 * lo
        low, up = preds[:, lo_idx], preds[:, hi_idx]
        alphas.append(alpha)
        scores.append(interval_score(y, low, up, alpha))
        widths.append(up - low)
        unders.append((2.0 / alpha) * np.clip(low - y, 0, None))
        overs.append((2.0 / alpha) * np.clip(y - up, 0, None))

    k = len(alphas)
    if k == 0:
        raise ValueError("no symmetric quantile pairs found")

    weights = np.array(alphas) / 2.0
    denom = k + 0.5

    total = (0.5 * np.abs(y - median) + sum(w * s for w, s in zip(weights, scores, strict=True))) / denom

    if not decompose:
        return total
    return {
        "wis": total,
        "sharpness": sum(w * v for w, v in zip(weights, widths, strict=True)) / denom,
        "underprediction": sum(w * v for w, v in zip(weights, unders, strict=True)) / denom,
        "overprediction": sum(w * v for w, v in zip(weights, overs, strict=True)) / denom,
        "median_ae": 0.5 * np.abs(y - median) / denom,
    }


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical share of observations inside the interval.

    A calibrated 95% interval covers 95% of the time. Forecast intervals in
    practice cover far less, which is the single most common way a forecast
    misleads its reader.
    """
    y, lower, upper = np.asarray(y, float), np.asarray(lower, float), np.asarray(upper, float)
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if ok.sum() == 0:
        return np.nan
    return float(((y >= lower) & (y <= upper))[ok].mean())


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    return float(np.abs(y[ok] - pred[ok]).mean()) if ok.any() else np.nan


def mase_scale(history: np.ndarray, season: int = 7) -> float:
    """Denominator for the mean absolute scaled error: in-sample seasonal-naive MAE.

    Scaling by this makes errors comparable across countries whose case counts
    differ by four orders of magnitude -- without it, a national average is just
    a ranking of population sizes.
    """
    h = np.asarray(history, float)
    h = h[np.isfinite(h)]
    if h.size <= season:
        return np.nan
    d = np.abs(h[season:] - h[:-season])
    m = float(np.mean(d)) if d.size else np.nan
    return m if m and m > 0 else np.nan


def relative_skill(scores: np.ndarray, baseline_scores: np.ndarray) -> float:
    """Geometric-mean ratio of model score to baseline score.

    The geometric mean is the right average for a ratio: it is symmetric under
    inversion, so "twice as good" and "twice as bad" are equal and opposite,
    which the arithmetic mean gets wrong.
    """
    s, b = np.asarray(scores, float), np.asarray(baseline_scores, float)
    ok = np.isfinite(s) & np.isfinite(b) & (b > 0) & (s >= 0)
    if ok.sum() == 0:
        return np.nan
    # +epsilon keeps a single perfect forecast from collapsing the mean to zero
    eps = 1e-9
    return float(np.exp(np.mean(np.log((s[ok] + eps) / (b[ok] + eps)))))
