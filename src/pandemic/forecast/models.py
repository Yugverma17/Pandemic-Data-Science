"""Forecasting models, from trivial baselines to a global gradient-boosted panel.

Same contract for every model: given a :class:`SeriesCache` and the integer index
of the forecast origin, return a point prediction of the 7-day trailing average
``horizon`` days later. Quantiles are not each model's job. They come from one
shared conformal layer (:mod:`pandemic.forecast.conformal`), so a WIS difference
between models reflects the forecast and not a better-tuned error bar.

Target is the 7-day average, not the raw daily count. The forensics module shows
the day-to-day signal is mostly reporting rhythm, and forecasting that means
predicting which weekday a health ministry clears its queue.

On the cache: every quantity these models use (trailing mean, R_t, running peak)
is causal, so its value at index ``i`` depends only on ``incidence[:i+1]``. Each
can therefore be computed once over the whole series and indexed into, instead of
recomputed at every origin. That is what makes the rolling-origin backtest
affordable. ``tests/test_forecast.py`` checks the causality rather than assuming it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pandemic.forecast.rt import discretise_serial_interval, estimate_rt, project_renewal

SMOOTH = 7  # trailing window defining the target


def trailing_average(x: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """Trailing mean with a full-window requirement, NaN before it is defined."""
    s = pd.Series(np.clip(np.asarray(x, float), 0, None))
    return s.rolling(window, min_periods=window).mean().to_numpy()


class SeriesCache:
    """Causal features for one entity, precomputed over the full series."""

    __slots__ = ("dates", "incidence", "population", "entity", "avg", "rt", "runmax", "n")

    def __init__(self, entity: str, dates: np.ndarray, incidence: np.ndarray,
                 population: float):
        self.entity = entity
        self.dates = pd.DatetimeIndex(dates)
        self.incidence = np.nan_to_num(np.clip(np.asarray(incidence, float), 0, None))
        self.population = float(population) if np.isfinite(population) else 1e7
        self.n = self.incidence.size

        self.avg = trailing_average(self.incidence)
        self.rt = estimate_rt(self.incidence, tau=7).mean
        # Running maximum of the trailing average, NaN-safe and causal.
        filled = np.where(np.isfinite(self.avg), self.avg, 0.0)
        self.runmax = np.maximum.accumulate(filled)

    def level(self, i: int) -> float:
        v = self.avg[i]
        return float(v) if np.isfinite(v) else np.nan

    def features(self, i: int) -> dict[str, float] | None:
        """Scale-free feature vector at origin ``i``, or None if under-determined."""
        if i < 30 or not np.isfinite(self.avg[i]):
            return None
        y = self.avg
        log_now = float(np.log1p(y[i]))

        def growth(days: int) -> float:
            j = i - days
            if j < 0 or not np.isfinite(y[j]):
                return 0.0
            return log_now - float(np.log1p(y[j]))

        g7, g14, g28 = growth(7), growth(14), growth(28)
        r = self.rt[i]
        if not np.isfinite(r):
            r = 1.0

        return {
            "log_level": log_now,
            "growth_7": g7,
            "growth_14": g14,
            "growth_28": g28,
            "rt": float(np.clip(r, 0, 5)),
            "log_rel_peak": log_now - float(np.log1p(self.runmax[i])),
            "log_days_since_start": float(np.log1p(i)),
            "log_pop": float(np.log1p(self.population)),
            # Second difference of log level: is growth itself accelerating?
            "curvature": g7 - (g14 - g7),
        }


FEATURES = ["log_level", "growth_7", "growth_14", "growth_28", "rt",
            "log_rel_peak", "log_days_since_start", "log_pop", "curvature"]


class Forecaster(ABC):
    """Common interface for every model in the comparison."""

    name: str = "base"
    needs_panel_fit: bool = False

    def fit_panel(self, caches: dict[str, SeriesCache], cutoff: pd.Timestamp) -> None:  # noqa: B027
        """Global models override this; local models need no training.

        Deliberately concrete and empty rather than abstract: every local model
        would otherwise carry an identical do-nothing override, and the backtest
        can call it unconditionally.
        """

    @abstractmethod
    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        """Point forecast of the 7-day trailing average at origin ``i + horizon``."""


# ------------------------------------------------------------------- baselines


class Persistence(Forecaster):
    """"Tomorrow looks like today." The level stays where it is.

    Weak-looking, and remarkably hard to beat once the horizon approaches the
    epidemic's own timescale. Any model that cannot beat this is adding nothing.
    """

    name = "persistence"

    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        return cache.level(i)


class LogLinearDrift(Forecaster):
    """Current exponential growth rate persists, optionally damped.

    Fits ``log1p(7-day average) ~ a + b t`` over the trailing window and
    extrapolates. Epidemics grow multiplicatively, so the log scale is the right
    one. Fitting a polynomial to the *cumulative* curve -- the common shortcut --
    yields a superb in-sample R-squared and no forecasting value at all, because
    a monotone series is trivially fittable and the fit says nothing about the
    increment, which is the quantity anyone actually wants.
    """

    def __init__(self, window: int = 14, damping: float = 1.0, name: str | None = None):
        self.window = window
        self.damping = damping
        self.name = name or (f"drift{window}" if damping == 1.0
                             else f"drift{window}_damp{damping:g}")

    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        lo = i - self.window + 1
        if lo < 0:
            return cache.level(i)
        y = cache.avg[lo: i + 1]
        if not np.isfinite(y).all():
            return cache.level(i)

        logy = np.log1p(y)
        t = np.arange(self.window, dtype=float)
        b, a = np.polyfit(t, logy, 1)

        if self.damping == 1.0:
            growth = b * horizon
        else:
            d = self.damping
            growth = b * d * (1 - d**horizon) / (1 - d)
        # Anchor on the fitted value at the origin, not the raw last point, so a
        # single noisy day cannot swing the whole extrapolation.
        return float(np.clip(np.expm1(a + b * (self.window - 1) + growth), 0, None))


# ------------------------------------------------------------------ mechanistic


class RenewalRt(Forecaster):
    """Estimate R_t (Cori et al.) and project forward with the renewal equation.

    The only model here carrying an epidemiological mechanism, which makes it
    auditable: it can be wrong for a reason a domain expert can name, rather
    than wrong for reasons buried inside a fitted function.
    """

    def __init__(self, damping: float = 0.95, name: str | None = None):
        self.damping = damping
        self._w = discretise_serial_interval()
        self.name = name or (f"renewal_rt_damp{damping:g}" if damping != 1.0 else "renewal_rt")

    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        r = cache.rt[i]
        if not np.isfinite(r) or cache.incidence[max(0, i - 30): i + 1].sum() <= 0:
            return cache.level(i)

        hist = cache.incidence[: i + 1]
        projected = project_renewal(hist, float(r), horizon, w=self._w, damping=self.damping)
        # The target is a trailing 7-day mean ending at i + horizon, so for short
        # horizons it still contains observed days.
        combined = np.concatenate([hist, projected])
        window = combined[i + horizon - SMOOTH + 1: i + horizon + 1]
        return float(np.clip(np.mean(window), 0, None)) if window.size else cache.level(i)


# ---------------------------------------------------------------- learned model


class PanelGBM(Forecaster):
    """Gradient-boosted trees trained across all countries at once.

    Predicts the *log growth* from origin to target rather than the level, which
    matters more than the choice of learner: predicting a level forces the model
    to spend capacity re-learning each country's scale, whereas predicting a
    ratio lets every country's wave contribute to one shared question -- given
    this growth pattern, what happens next?

    Refit on an expanding window at fixed intervals, only ever on rows whose
    target date precedes the cutoff.
    """

    needs_panel_fit = True

    def __init__(self, horizon: int, refit_every_days: int = 56,
                 min_train_rows: int = 400, stride: int = 5, seed: int = 0):
        self.horizon = horizon
        self.refit_every_days = refit_every_days
        self.min_train_rows = min_train_rows
        self.stride = stride
        self.seed = seed
        self.name = "panel_gbm"
        self._model: HistGradientBoostingRegressor | None = None
        self._fitted_through: pd.Timestamp | None = None

    def fit_panel(self, caches: dict[str, SeriesCache], cutoff: pd.Timestamp) -> None:
        if (self._fitted_through is not None
                and (cutoff - self._fitted_through).days < self.refit_every_days):
            return

        rows, targets = [], []
        for cache in caches.values():
            avg = cache.avg
            # Last index whose *target* is still strictly before the cutoff.
            valid = np.flatnonzero(cache.dates < cutoff)
            if valid.size == 0:
                continue
            last_target = int(valid[-1])
            for i in range(30, last_target - self.horizon + 1, self.stride):
                y_now, y_next = avg[i], avg[i + self.horizon]
                if not (np.isfinite(y_now) and np.isfinite(y_next)) or y_now < 5:
                    continue
                feats = cache.features(i)
                if feats is None:
                    continue
                rows.append([feats[f] for f in FEATURES])
                targets.append(np.log1p(y_next) - np.log1p(y_now))

        self._fitted_through = cutoff
        if len(rows) < self.min_train_rows:
            self._model = None
            return

        self._model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=self.seed,
        )
        self._model.fit(np.asarray(rows), np.asarray(targets))

    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        fallback = cache.level(i)
        if self._model is None:
            return fallback
        feats = cache.features(i)
        if feats is None or not np.isfinite(fallback):
            return fallback
        x = np.asarray([[feats[f] for f in FEATURES]])
        # Clip to +-e^2: a 7x move in a fortnight is already extreme, and an
        # unclipped tree extrapolation can produce absurd levels on thin data.
        log_growth = float(np.clip(self._model.predict(x)[0], -2.0, 2.0))
        return float(np.clip(np.expm1(np.log1p(fallback) + log_growth), 0, None))


def default_models(horizon: int) -> list[Forecaster]:
    """The model set used in the published backtest.

    A seasonal-naive baseline is deliberately absent: the target is already a
    7-day trailing average, so "the value one season ago" collapses onto
    persistence and would only pad the results table with a duplicate row.
    """
    return [
        Persistence(),
        LogLinearDrift(window=14),
        LogLinearDrift(window=14, damping=0.9, name="drift14_damped"),
        RenewalRt(damping=1.0, name="renewal_rt_undamped"),
        RenewalRt(damping=0.95),
        PanelGBM(horizon=horizon),
    ]
