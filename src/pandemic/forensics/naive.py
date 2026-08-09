"""The z-score detector, and what it actually finds.

A z-score on raw daily cases is the standard first move and it does not work. The
series is non-stationary by three orders of magnitude, so "days more than k
standard deviations above the mean" is a roundabout way of asking "which days
were near a wave peak".

``decompose_flags`` shows this. Each flagged day gets attributed to the first
mechanism that explains it:

``trend``     the 7-day level was already this high, so the day is unremarkable
``weekday``   the deviation is the country's normal day-of-week rhythm
``dump``      a batch release, which is a real reporting anomaly
``residual``  none of the above, so a genuine unexplained jump

Only the last two are anomalies in any useful sense, and for most countries they
are a small minority of what the z-score returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.forensics.flags import backlog_dumps


def naive_zscore_flags(new_cases: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag days whose raw value is > ``threshold`` global SDs above the mean.

    Deliberately naive -- this reproduces the common approach so we can measure
    it, not because it is recommended.
    """
    x = new_cases.astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(False, index=x.index)
    return ((x - mu) / sd) > threshold


def rolling_zscore_flags(new_cases: pd.Series, window: int = 14,
                         threshold: float = 3.0) -> pd.Series:
    """Rolling-window z-score -- the usual "improved" version.

    Better than the global z-score, but still blind to the weekday rhythm, so on
    a country with strong batch reporting it flags most Mondays.
    """
    x = new_cases.astype(float)
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0)
    z = (x - mu) / sd.replace(0, np.nan)
    return (z > threshold).fillna(False)


def decompose_flags(new_cases: pd.Series, flags: pd.Series,
                    *, weekday_tolerance: float = 1.5,
                    residual_threshold: float = 3.0) -> pd.DataFrame:
    """Attribute each flagged day to the mechanism that explains it.

    Works on the multiplicative scale, because case counts are multiplicative:
        log1p(cases) = trend + weekday effect + residual
    where trend is a centred 7-day mean of log1p (which cancels the weekday
    cycle exactly) and the weekday effect is the country's mean log-deviation for
    that day of week.
    """
    x = new_cases.astype(float)
    logx = np.log1p(x.clip(lower=0))

    trend = logx.rolling(7, center=True, min_periods=7).mean()
    dev = logx - trend
    dow = pd.Series(x.index.dayofweek, index=x.index)

    # Weekday effect estimated only where the trend is defined.
    wk_effect = dev.groupby(dow).transform("mean")
    resid = dev - wk_effect

    rs = resid.std(ddof=0)
    resid_z = resid / rs if np.isfinite(rs) and rs > 0 else resid * 0.0

    dumps = backlog_dumps(x)
    dump_dates = set(dumps.loc[dumps["is_dump"], "date"]) if len(dumps) else set()

    # Is the day merely "high because the epidemic was high"? Compare the day's
    # own trend level against the distribution of trend levels overall.
    trend_pct = trend.rank(pct=True)

    rows = []
    for date in x.index[flags.fillna(False).to_numpy()]:
        d_resid_z = float(resid_z.get(date, np.nan))
        d_wk = float(wk_effect.get(date, np.nan))
        d_trend_pct = float(trend_pct.get(date, np.nan))

        if date in dump_dates:
            mechanism = "dump"
        elif np.isfinite(d_resid_z) and abs(d_resid_z) > residual_threshold:
            mechanism = "residual"
        elif np.isfinite(d_wk) and abs(d_wk) > np.log(weekday_tolerance):
            mechanism = "weekday"
        else:
            mechanism = "trend"

        rows.append({
            "date": date,
            "value": float(x.get(date, np.nan)),
            "trend_pctile": d_trend_pct,
            "weekday_log_effect": d_wk,
            "residual_z": d_resid_z,
            "mechanism": mechanism,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_true_anomaly"] = out["mechanism"].isin(["dump", "residual"])
    return out


def compare_detectors(new_cases: pd.Series) -> dict:
    """Summarise how much of each detector's output survives decomposition."""
    result = {}
    for label, flags in (
        ("global_z", naive_zscore_flags(new_cases)),
        ("rolling_z", rolling_zscore_flags(new_cases)),
    ):
        dec = decompose_flags(new_cases, flags)
        counts = (dec["mechanism"].value_counts().to_dict() if not dec.empty else {})
        n = int(len(dec))
        result[label] = {
            "n_flagged": n,
            "n_true": int(dec["is_true_anomaly"].sum()) if not dec.empty else 0,
            "precision": (float(dec["is_true_anomaly"].mean()) if n else np.nan),
            "by_mechanism": counts,
        }
    return result
