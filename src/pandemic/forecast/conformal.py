"""Conformal calibration: point forecasts to intervals.

Every model here emits a single number. The interval comes from the model's own
past errors, on the log scale, since errors in epidemic counts are multiplicative
and a fixed +/- 2,000 band is meaningless at both ends of a wave.

For a given model, country and horizon:

1. take residuals ``e = log1p(actual) - log1p(predicted)`` from forecast origins
   strictly before the one being scored
2. read off empirical quantiles of ``e``
3. place them around the current point forecast and invert the transform

This is split-conformal prediction. The coverage guarantee is distribution-free,
so it holds without assuming Gaussian errors, which these are not.

The assumption that does bite is exchangeability. Epidemic residuals are not
exchangeable across regimes, so intervals calibrated on a quiet period come out
too narrow when a new variant arrives. The backtest measures how big that gap is;
see the coverage column in the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LOCAL_RESIDUALS = 12  # below this, borrow the pooled distribution


def _empirical_quantiles(residuals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.zeros(levels.size)
    return np.quantile(r, levels, method="higher")


def calibrate(point: float, residuals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Place residual quantiles around a point forecast, on the log scale."""
    if not np.isfinite(point):
        return np.full(levels.size, np.nan)
    offsets = _empirical_quantiles(residuals, levels)
    return np.clip(np.expm1(np.log1p(max(point, 0.0)) + offsets), 0.0, None)


class ConformalCalibrator:
    """Accumulates residuals per (model, entity, horizon) and issues quantiles.

    Residuals are only ever *added* for origins already in the past relative to
    the forecast being calibrated; the backtest enforces that ordering, which is
    where the leak-free guarantee actually lives.
    """

    def __init__(self, levels: tuple[float, ...], max_history: int = 60):
        self.levels = np.asarray(levels, float)
        self.max_history = max_history
        self._local: dict[tuple[str, str, int], list[float]] = {}
        self._pooled: dict[tuple[str, int], list[float]] = {}

    def add_residual(self, model: str, entity: str, horizon: int,
                     actual: float, predicted: float) -> None:
        if not (np.isfinite(actual) and np.isfinite(predicted)):
            return
        e = float(np.log1p(max(actual, 0.0)) - np.log1p(max(predicted, 0.0)))
        if not np.isfinite(e):
            return
        loc = self._local.setdefault((model, entity, horizon), [])
        loc.append(e)
        if len(loc) > self.max_history:
            del loc[0]
        pool = self._pooled.setdefault((model, horizon), [])
        pool.append(e)
        if len(pool) > self.max_history * 40:
            del pool[0]

    def residuals_for(self, model: str, entity: str, horizon: int) -> tuple[np.ndarray, str]:
        """Local residuals when there are enough of them, pooled otherwise."""
        loc = self._local.get((model, entity, horizon), [])
        if len(loc) >= MIN_LOCAL_RESIDUALS:
            return np.asarray(loc), "local"
        pool = self._pooled.get((model, horizon), [])
        if len(pool) >= MIN_LOCAL_RESIDUALS:
            return np.asarray(pool), "pooled"
        return np.asarray(loc + pool), "sparse"

    def quantiles(self, model: str, entity: str, horizon: int,
                  point: float) -> tuple[np.ndarray, str]:
        residuals, source = self.residuals_for(model, entity, horizon)
        if residuals.size == 0:
            return np.full(self.levels.size, np.nan), "none"
        return calibrate(point, residuals, self.levels), source


def summarise_calibration(scored: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    """Nominal versus empirical coverage for every central interval, by model.

    The column that matters: if the 95% interval covers 70% of the time, the
    forecast is not 95% confident about anything and should not be presented as
    though it were.
    """
    lv = np.asarray(levels)
    rows = []
    for model, g in scored.groupby("model", sort=True):
        y = g["actual"].to_numpy(float)
        for lo in lv[lv < 0.5]:
            hi = 1.0 - lo
            lo_col, hi_col = f"q{lo:g}", f"q{hi:g}"
            if lo_col not in g or hi_col not in g:
                continue
            low, up = g[lo_col].to_numpy(float), g[hi_col].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(low) & np.isfinite(up)
            if ok.sum() == 0:
                continue
            rows.append({
                "model": model,
                "nominal": round(hi - lo, 3),
                "empirical": float(((y >= low) & (y <= up))[ok].mean()),
                "n": int(ok.sum()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap"] = out["empirical"] - out["nominal"]
    return out
