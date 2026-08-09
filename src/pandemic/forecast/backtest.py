"""Rolling-origin backtest.

Four things this protocol guarantees, spelled out because the evaluation is what
decides whether the rest of the project means anything.

No lookahead in the features. Models read a :class:`SeriesCache` only at indices
``<= i``, and every cached quantity is causal by construction.

No lookahead in the training set. ``PanelGBM`` is refit on an expanding window
holding only rows whose *target* date precedes the origin, not just rows whose
feature date does. The second is the subtler and more common leak.

No lookahead in the intervals. A residual from origin ``t`` only becomes
observable at ``t + horizon``, so it sits in a pending queue until the clock
reaches its target date. Calibrating on residuals you could not have seen yet
gives coverage numbers that fall apart in production.

Scale-free aggregation. Country case counts span four orders of magnitude, so a
raw mean WIS across countries would rank populations. Results aggregate as a
geometric-mean ratio against the persistence baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pandemic.config import QUANTILES, get_logger
from pandemic.forecast.conformal import ConformalCalibrator
from pandemic.forecast.metrics import mase_scale, relative_skill, weighted_interval_score
from pandemic.forecast.models import Forecaster, SeriesCache, default_models

log = get_logger(__name__)


def build_caches(panel: pd.DataFrame, entities: Sequence[str]) -> dict[str, SeriesCache]:
    """Precompute causal features once per entity."""
    caches: dict[str, SeriesCache] = {}
    for entity in entities:
        g = panel[panel["entity"] == entity].sort_values("date")
        if len(g) < 120:
            continue
        pop = float(g["population"].iloc[0]) if "population" in g.columns else np.nan
        caches[entity] = SeriesCache(entity, g["date"].to_numpy(),
                                     g["new_cases"].to_numpy(), pop)
    return caches


def make_origins(caches: dict[str, SeriesCache], *, start: str, end: str,
                 every_days: int = 14) -> list[pd.Timestamp]:
    """Evenly spaced forecast origins within the window covered by the data."""
    lo = max(pd.Timestamp(start), min(c.dates[0] for c in caches.values()))
    hi = min(pd.Timestamp(end), max(c.dates[-1] for c in caches.values()))
    return list(pd.date_range(lo, hi, freq=f"{every_days}D"))


def run_backtest(panel: pd.DataFrame, entities: Sequence[str], *,
                 horizons: Sequence[int], start: str, end: str,
                 every_days: int = 14, eval_from: str | None = None,
                 quantiles: Sequence[float] = QUANTILES,
                 models_factory=default_models) -> pd.DataFrame:
    """Score every model at every (origin, entity, horizon). Returns tidy rows."""
    caches = build_caches(panel, entities)
    if not caches:
        raise ValueError("no entity had enough history to backtest")
    origins = make_origins(caches, start=start, end=end, every_days=every_days)
    eval_start = pd.Timestamp(eval_from) if eval_from else origins[0]
    levels = np.asarray(quantiles, float)

    log.info("backtest: %d entities x %d origins x %d horizons",
             len(caches), len(origins), len(horizons))

    # Index lookup per entity, so an origin maps to a position in O(1).
    positions = {e: pd.Series(np.arange(c.n), index=c.dates) for e, c in caches.items()}
    scales = {e: mase_scale(c.avg[np.isfinite(c.avg)]) for e, c in caches.items()}

    rows: list[dict] = []
    for horizon in horizons:
        models: list[Forecaster] = models_factory(horizon)
        calibrator = ConformalCalibrator(tuple(quantiles))
        pending: list[tuple[pd.Timestamp, str, str, float, float]] = []

        for origin in origins:
            # 1. Release residuals that have become observable by now.
            still_pending = []
            for item in pending:
                if item[0] <= origin:
                    _, mname, entity, actual, point = item
                    calibrator.add_residual(mname, entity, horizon, actual, point)
                else:
                    still_pending.append(item)
            pending = still_pending

            # 2. Refit global models on data strictly before the origin.
            for m in models:
                if m.needs_panel_fit:
                    m.fit_panel(caches, origin)

            # 3. Forecast.
            for entity, cache in caches.items():
                pos = positions[entity]
                if origin not in pos.index:
                    continue
                i = int(pos.loc[origin])
                j = i + horizon
                if j >= cache.n:
                    continue
                actual = cache.avg[j]
                if not np.isfinite(actual) or not np.isfinite(cache.avg[i]):
                    continue
                # Skip periods with no meaningful epidemic: forecasting 2 cases
                # a day is a division-by-noise exercise, not a forecast.
                if cache.avg[i] < 10:
                    continue

                for m in models:
                    point = m.predict(cache, i, horizon)
                    if not np.isfinite(point):
                        continue
                    qs, source = calibrator.quantiles(m.name, entity, horizon, point)
                    target_date = cache.dates[j]

                    if origin >= eval_start:
                        row = {
                            "origin": origin, "target_date": target_date,
                            "entity": entity, "horizon": horizon, "model": m.name,
                            "point": point, "actual": float(actual),
                            "baseline_level": cache.level(i),
                            "calibration_source": source,
                            "mase_scale": scales.get(entity, np.nan),
                        }
                        row.update({f"q{q:g}": v for q, v in zip(levels, qs, strict=True)})
                        rows.append(row)

                    pending.append((target_date, m.name, entity, float(actual), point))

        log.info("  horizon %2dd complete", horizon)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("backtest produced no rows")
    return df


def score(results: pd.DataFrame, quantiles: Sequence[float] = QUANTILES,
          baseline: str = "persistence") -> pd.DataFrame:
    """Aggregate raw backtest rows into a per-model, per-horizon table."""
    levels = np.asarray(quantiles, float)
    qcols = [f"q{q:g}" for q in levels]

    df = results.dropna(subset=qcols + ["actual", "point"]).copy()
    if df.empty:
        raise ValueError("no fully-calibrated rows to score")

    parts = weighted_interval_score(df["actual"].to_numpy(), levels,
                                   df[qcols].to_numpy(), decompose=True)
    for k, v in parts.items():
        df[k] = v
    df["abs_error"] = np.abs(df["actual"] - df["point"])
    df["scaled_wis"] = df["wis"] / df["mase_scale"].replace(0, np.nan)

    lo95, hi95 = f"q{levels.min():g}", f"q{levels.max():g}"
    df["covered_95"] = ((df["actual"] >= df[lo95]) & (df["actual"] <= df[hi95]))
    lo50, hi50 = "q0.25", "q0.75"
    if lo50 in df and hi50 in df:
        df["covered_50"] = ((df["actual"] >= df[lo50]) & (df["actual"] <= df[hi50]))

    # Align each model's rows with the baseline's on the same forecast tasks, so
    # the ratio compares like with like.
    key = ["origin", "entity", "horizon"]
    base = df[df["model"] == baseline].set_index(key)["wis"]

    out = []
    for (model, horizon), g in df.groupby(["model", "horizon"], sort=True):
        aligned = g.set_index(key)
        b = base.reindex(aligned.index)
        out.append({
            "model": model,
            "horizon": horizon,
            "n": int(len(g)),
            "rel_wis": relative_skill(aligned["wis"].to_numpy(), b.to_numpy()),
            # Median, not mean: scaled WIS is a heavy-tailed ratio, and an
            # undamped extrapolation that explodes on two countries would
            # otherwise set the average for all forty.
            "scaled_wis_median": float(np.nanmedian(g["scaled_wis"])),
            "scaled_wis_p90": float(np.nanpercentile(g["scaled_wis"], 90)),
            "wis": float(np.nanmean(g["wis"])),
            "sharpness": float(np.nanmean(g["sharpness"])),
            "underprediction": float(np.nanmean(g["underprediction"])),
            "overprediction": float(np.nanmean(g["overprediction"])),
            "mae": float(np.nanmean(g["abs_error"])),
            "coverage_95": float(g["covered_95"].mean()),
            "coverage_50": float(g["covered_50"].mean()) if "covered_50" in g else np.nan,
        })

    return (pd.DataFrame(out)
            .sort_values(["horizon", "rel_wis"])
            .reset_index(drop=True))


def score_by_regime(results: pd.DataFrame, quantiles: Sequence[float] = QUANTILES,
                    baseline: str = "persistence") -> pd.DataFrame:
    """Skill split by whether the epidemic was growing, flat, or receding.

    An average over all conditions hides the only thing a forecast is for. Models
    look interchangeable in flat periods and separate sharply at turning points.
    """
    levels = np.asarray(quantiles, float)
    qcols = [f"q{q:g}" for q in levels]
    df = results.dropna(subset=qcols + ["actual", "point"]).copy()
    df["wis"] = weighted_interval_score(df["actual"].to_numpy(), levels, df[qcols].to_numpy())

    ratio = df["actual"] / df["baseline_level"].replace(0, np.nan)
    df["regime"] = pd.cut(ratio, [-np.inf, 0.8, 1.25, np.inf],
                          labels=["receding", "flat", "growing"])

    key = ["origin", "entity", "horizon"]
    base = df[df["model"] == baseline].set_index(key)["wis"]

    out = []
    for (model, horizon, regime), g in df.groupby(["model", "horizon", "regime"],
                                                  sort=True, observed=True):
        aligned = g.set_index(key)
        out.append({
            "model": model, "horizon": horizon, "regime": str(regime),
            "n": int(len(g)),
            "rel_wis": relative_skill(aligned["wis"].to_numpy(),
                                      base.reindex(aligned.index).to_numpy()),
        })
    return pd.DataFrame(out).sort_values(["horizon", "regime", "rel_wis"]).reset_index(drop=True)
