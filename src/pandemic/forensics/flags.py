"""Structural reporting anomalies in daily count series.

An epidemiological anomaly is a real change in transmission. A reporting anomaly
is an artefact of how the number got into the spreadsheet. A z-score on daily
cases flags both without distinguishing them, and mostly just finds wave peaks.

Each detector below targets one specific failure mode of a surveillance system,
so every flag comes with a mechanism attached instead of just a large residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A series only counts as "actively transmitting" above this trailing mean.
# Below it, gaps and zeros are unremarkable rather than evidence of failure.
ACTIVE_THRESHOLD = 10.0


def _trailing_mean(x: pd.Series, window: int = 28) -> pd.Series:
    return x.rolling(window, min_periods=7).mean()


def negative_revisions(new_cases: pd.Series) -> pd.DataFrame:
    """Days where the cumulative count went *down*.

    Physically impossible for an incidence count, so every one of these is a
    retroactive correction: deduplication, a reclassified death, or a transfer
    between jurisdictions. Magnitude is the size of the correction.
    """
    neg = new_cases[new_cases < 0]
    return pd.DataFrame({"magnitude": -neg.to_numpy()}, index=neg.index)


def frozen_runs(new_cases: pd.Series, min_run: int = 3) -> pd.DataFrame:
    """Consecutive days reporting an identical value while transmission is active.

    Two distinct pathologies share this signature:
      * runs of exact zeros  -> reporting stopped (holiday, system outage, war)
      * runs of identical non-zero values -> the series was interpolated or
        back-filled with a constant, i.e. those days were never really measured
    """
    active = _trailing_mean(new_cases) >= ACTIVE_THRESHOLD
    vals = new_cases.to_numpy(dtype=float)

    # Label maximal runs of equal consecutive values.
    change = np.ones(len(vals), dtype=bool)
    change[1:] = vals[1:] != vals[:-1]
    run_id = np.cumsum(change)

    runs = pd.DataFrame({"value": vals, "run_id": run_id, "active": active.to_numpy()},
                        index=new_cases.index)
    agg = runs.groupby("run_id").agg(
        value=("value", "first"),
        length=("value", "size"),
        active_share=("active", "mean"),
        start=("value", lambda s: s.index[0]),
        end=("value", lambda s: s.index[-1]),
    )
    agg = agg[(agg["length"] >= min_run) & (agg["active_share"] > 0.5)]
    agg["kind"] = np.where(agg["value"] == 0, "reporting-gap", "constant-fill")
    return agg.reset_index(drop=True)


def backlog_dumps(new_cases: pd.Series, *, ratio: float = 5.0, floor: float = 50.0,
                  window: int = 15) -> pd.DataFrame:
    """Single-day spikes that are batch releases of previously withheld cases.

    A dump is identified by two properties, not one:
      1. the day towers over its local level (``ratio`` x the centred rolling
         median, which is immune to being dragged up by the spike itself), and
      2. mass is approximately conserved -- the days *before* the spike sit
         below the local level, because those cases were being held back.

    Requiring (2) is what separates a data dump from a genuine explosive
    outbreak, where the surrounding days are elevated rather than depressed.
    """
    x = new_cases.astype(float)
    med = x.rolling(window, center=True, min_periods=window // 2).median()

    is_spike = (x > floor) & (med > 0) & (x / med.replace(0, np.nan) > ratio)
    if not is_spike.any():
        return pd.DataFrame(columns=["date", "value", "local_median", "ratio",
                                     "prior_deficit", "conservation"])

    rows = []
    values = x.to_numpy()
    medians = med.to_numpy()
    for pos in np.flatnonzero(is_spike.to_numpy()):
        lo = max(0, pos - 14)
        prior = values[lo:pos]
        prior_med = medians[lo:pos]
        excess = values[pos] - medians[pos]
        deficit = float(np.nansum(np.clip(prior_med - prior, 0, None)))
        rows.append({
            "date": x.index[pos],
            "value": values[pos],
            "local_median": medians[pos],
            "ratio": values[pos] / medians[pos],
            "prior_deficit": deficit,
            # 1.0 => the spike is exactly the accumulated shortfall
            "conservation": deficit / excess if excess > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out["is_dump"] = out["conservation"] >= 0.25
    return out


def weekday_profile(new_cases: pd.Series, n_permutations: int = 1000,
                    seed: int = 0) -> dict:
    """Quantify batch (weekday-driven) reporting.

    Each day is divided by its centred 7-day mean, which removes the epidemic
    trend and leaves the reporting rhythm. Averaging those ratios by day of week
    gives seven multipliers; a system reporting continuously has all seven near
    1.0, while one that batches on weekdays shows a deep Sunday trough.

    Significance comes from a permutation test -- day-of-week labels are shuffled
    within the series and the amplitude recomputed. This needs no distributional
    assumption, which matters because the ratios are heavy-tailed and would
    violate the assumptions of an F-test.
    """
    x = new_cases.astype(float)
    centred = x.rolling(7, center=True, min_periods=7).mean()
    ratio = (x / centred).replace([np.inf, -np.inf], np.nan)

    mask = ratio.notna() & (centred >= ACTIVE_THRESHOLD)
    r = ratio[mask].to_numpy()
    dow = np.asarray(x.index[mask].dayofweek)

    if r.size < 60:
        return {"amplitude": np.nan, "p_value": np.nan, "n": int(r.size),
                "multipliers": np.full(7, np.nan)}

    def amplitude_of(vals: np.ndarray) -> tuple[np.ndarray, float]:
        total = np.bincount(dow, weights=vals, minlength=7)
        count = np.bincount(dow, minlength=7)
        mult = np.divide(total, count, out=np.ones(7), where=count > 0)
        return mult, float(mult.max() - mult.min())

    multipliers, observed = amplitude_of(r)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = amplitude_of(rng.permutation(r))[1]
    # +1 in numerator and denominator: never report p = 0 from a finite test.
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return {"amplitude": observed, "p_value": p_value, "n": int(r.size),
            "multipliers": multipliers}


def summarise_country(g: pd.DataFrame, value_col: str = "new_cases",
                      n_permutations: int = 500, seed: int = 0) -> dict:
    """Run every structural detector on one country's series."""
    s = g.set_index("date")[value_col].sort_index()

    neg = negative_revisions(s)
    runs = frozen_runs(s)
    dumps = backlog_dumps(s)
    wk = weekday_profile(s, n_permutations=n_permutations, seed=seed)

    active_days = int((_trailing_mean(s) >= ACTIVE_THRESHOLD).sum())
    gaps = runs[runs["kind"] == "reporting-gap"]
    fills = runs[runs["kind"] == "constant-fill"]

    return {
        "n_days": int(s.notna().sum()),
        "active_days": active_days,
        "peak_daily": float(s.max()) if s.notna().any() else np.nan,
        "neg_days": int(len(neg)),
        "neg_max_magnitude": float(neg["magnitude"].max()) if len(neg) else 0.0,
        "gap_runs": int(len(gaps)),
        "gap_longest": int(gaps["length"].max()) if len(gaps) else 0,
        "fill_runs": int(len(fills)),
        "fill_longest": int(fills["length"].max()) if len(fills) else 0,
        "dump_days": int(dumps["is_dump"].sum()) if len(dumps) else 0,
        "dump_max_ratio": float(dumps.loc[dumps["is_dump"], "ratio"].max())
        if len(dumps) and dumps["is_dump"].any() else 0.0,
        "weekday_amplitude": wk["amplitude"],
        "weekday_p": wk["p_value"],
        "weekday_multipliers": wk["multipliers"],
    }
