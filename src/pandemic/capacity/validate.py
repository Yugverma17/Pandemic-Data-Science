"""Check the cases-to-ICU model against observed occupancy.

About thirty countries reported daily ICU occupancy alongside cases, so the
convolution model can actually be tested rather than just assumed.

Two things are checked separately, because one R-squared would mix them up:

Shape and timing: does predicted occupancy rise, peak and fall when the observed
series does? Scale-free, so an error in the assumed hospitalisation rate does not
affect it.

Level: one fitted multiplier per country. Under the model this is proportional to
IHR x ascertainment, and the IHR is held common across countries, so the spread in
multipliers estimates how differently countries detected infections. A country
needing 5x was finding roughly a fifth as many of its infections as one needing 1x.

Validation has to be period-specific. The model assumes a fixed IHR and
vaccination changes it, so the test runs on the pre-vaccination era with the
post-vaccination period reported separately. Pooling the two drops median r from
0.94 to 0.37, which looks like a broken model but is a broken evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.capacity.convolve import cases_to_icu_census
from pandemic.config import get_logger

log = get_logger(__name__)

MIN_OBSERVED_DAYS = 120
MAX_SHIFT_DAYS = 21

# Mass vaccination of the vulnerable was substantially complete in the reporting
# countries by this date, which is when a fixed infection-hospitalisation ratio
# stops being defensible.
VACCINE_CUTOFF = "2021-02-01"
POST_VACCINE_START = "2021-07-01"


def _best_shift(pred: np.ndarray, obs: np.ndarray, max_shift: int = MAX_SHIFT_DAYS) -> int:
    """Lag (in days) maximising correlation. Positive = prediction leads reality."""
    best_shift, best_r = 0, -np.inf
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b = pred[: len(pred) - s], obs[s:]
        else:
            a, b = pred[-s:], obs[: len(obs) + s]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 60 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if r > best_r:
            best_shift, best_r = s, r
    return best_shift


def validate_country(g: pd.DataFrame, *, start: str | None = None,
                     end: str | None = None,
                     min_days: int = MIN_OBSERVED_DAYS) -> dict | None:
    """Compare predicted with observed ICU occupancy for one country.

    ``start`` / ``end`` restrict the *evaluation* window only. The convolution
    still runs on the full case history, because occupancy today depends on
    admissions from the preceding weeks -- truncating the input as well would
    start the model from an empty ICU and manufacture an error at the boundary.
    """
    g = g.sort_values("date")
    obs = g["icu_patients"].to_numpy(float)
    if np.isfinite(obs).sum() < min_days:
        return None

    cases = np.nan_to_num(g["new_cases"].to_numpy(float).clip(0))
    pred = cases_to_icu_census(cases)

    dates = pd.DatetimeIndex(g["date"])
    window = np.ones(len(g), dtype=bool)
    if start is not None:
        window &= np.asarray(dates >= pd.Timestamp(start))
    if end is not None:
        window &= np.asarray(dates < pd.Timestamp(end))

    obs = np.where(window, obs, np.nan)

    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < min_days or pred[ok].sum() <= 0:
        return None

    # Scalar level calibration: least squares through the origin.
    k = float(np.sum(pred[ok] * obs[ok]) / np.sum(pred[ok] ** 2))
    scaled = k * pred

    r = float(np.corrcoef(pred[ok], obs[ok])[0, 1])
    shift = _best_shift(pred, obs)

    denom = float(np.sum((obs[ok] - obs[ok].mean()) ** 2))
    r2 = 1.0 - float(np.sum((obs[ok] - scaled[ok]) ** 2)) / denom if denom > 0 else np.nan

    # Peak timing, on the smoothed series so a single spike cannot set it.
    # Both series must be restricted to the evaluation window: the prediction
    # runs over the full history, so comparing its global peak against an
    # observed peak inside a truncated window measures the distance between two
    # different waves rather than a timing error.
    s_obs = pd.Series(obs).rolling(7, min_periods=4).mean().to_numpy()
    s_pred = pd.Series(np.where(window, scaled, np.nan)).rolling(
        7, min_periods=4).mean().to_numpy()
    peak_err = np.nan
    if np.isfinite(s_obs).any() and np.isfinite(s_pred).any():
        peak_err = float(np.nanargmax(s_pred) - np.nanargmax(s_obs))

    return {
        "entity": g["entity"].iloc[0],
        "n_days": int(ok.sum()),
        "correlation": r,
        "r_squared_after_scaling": r2,
        "level_multiplier": k,
        "best_shift_days": shift,
        "peak_timing_error_days": peak_err,
        "observed_peak": float(np.nanmax(obs)),
        "predicted_peak_scaled": float(np.nanmax(scaled)),
    }


def validate_all(panel: pd.DataFrame, *, start: str | None = None,
                 end: str | None = None,
                 min_days: int = MIN_OBSERVED_DAYS) -> pd.DataFrame:
    """Run the validation for every country reporting ICU occupancy."""
    if "icu_patients" not in panel.columns:
        raise ValueError("panel has no icu_patients column")

    rows = []
    for _, g in panel.groupby("entity", sort=True):
        try:
            res = validate_country(g, start=start, end=end, min_days=min_days)
        except Exception as exc:  # noqa: BLE001
            log.warning("validation failed for %s: %s", g["entity"].iloc[0], exc)
            continue
        if res:
            rows.append(res)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("correlation", ascending=False).reset_index(drop=True)


def summarise(validation: pd.DataFrame) -> dict:
    """Headline validation numbers, plus what the multipliers imply."""
    if validation.empty:
        return {"n_countries": 0}
    k = validation["level_multiplier"]
    return {
        "n_countries": int(len(validation)),
        "median_correlation": float(validation["correlation"].median()),
        "correlation_p25": float(validation["correlation"].quantile(0.25)),
        "correlation_p75": float(validation["correlation"].quantile(0.75)),
        "share_above_0.8": float((validation["correlation"] > 0.8).mean()),
        "median_peak_timing_error_days": float(
            validation["peak_timing_error_days"].median(skipna=True)),
        "level_multiplier_median": float(k.median()),
        "level_multiplier_iqr": [float(k.quantile(0.25)), float(k.quantile(0.75))],
        "level_multiplier_spread_ratio": float(k.quantile(0.9) / max(k.quantile(0.1), 1e-9)),
        "interpretation": (
            "Shape and timing transfer across countries with a fixed parameter set; "
            "the level does not. The spread in fitted multipliers is an estimate of "
            "how differently countries ascertained infections, since the assumed "
            "hospitalisation rate is held common."
        ),
    }
