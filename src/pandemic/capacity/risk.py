"""Turning a forecast into a decision: who needs beds, and how soon.

A projected ICU census is not an answer on its own. The operational question is
how likely a region is to exceed what it can handle, and how many days of warning
it has. That is a probability and a deadline, not a point estimate.

Two sources of uncertainty are propagated separately, because they behave
differently.

Epidemiological parameters. Hospitalisation rate, critical-care share, admission
lag and length of stay are uncertain within published ranges. This uncertainty is
large but roughly multiplicative and persistent: if the true IHR sits at the top
of its range it stays there for the whole projection, so it widens the level
without moving the timing.

Transmission. Where cases go next, projected with the renewal model under a
distribution over R_t. This is what drives the timing of a breach.

The capacity benchmark is each region's own observed peak occupancy rather than a
published bed count. Bed counts are unreliable, hard to compare across countries,
and say nothing about staffing. A level a region already sustained is a defensible
floor for what it can sustain again, and it needs no extra data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.capacity.convolve import cases_to_icu_census
from pandemic.config import EPI, get_logger, rng
from pandemic.forecast.rt import discretise_serial_interval, estimate_rt, project_renewal

log = get_logger(__name__)


def _sample_params(r, n: int) -> dict[str, np.ndarray]:
    """Draw epidemiological parameters from their published ranges.

    Triangular distributions over [low, mean, high]: they respect the bounds
    exactly and put most weight on the central estimate, which is the honest
    reading of a literature range without pretending to know its shape.
    """
    return {
        "ihr": r.triangular(EPI.ihr_low, EPI.ihr_mean, EPI.ihr_high, n),
        "icu_share": r.triangular(EPI.icu_share_low, EPI.icu_share_mean,
                                  EPI.icu_share_high, n),
        "lag_mean": r.triangular(4.0, EPI.onset_to_admission_mean, 11.0, n),
        "los_mean": r.triangular(7.0, EPI.icu_los_mean, 18.0, n),
    }


def simulate_census(cases: np.ndarray, *, horizon: int = 21, n_draws: int = 400,
                    rt_sd: float = 0.15, damping: float = 0.95,
                    seed: int = 0) -> np.ndarray:
    """Monte Carlo ICU census paths over history plus ``horizon`` days.

    Returns an array of shape ``(n_draws, len(cases) + horizon)``.
    """
    r = rng(seed)
    hist = np.nan_to_num(np.clip(np.asarray(cases, float), 0, None))
    w = discretise_serial_interval()

    rt_est = estimate_rt(hist, tau=7, w=w)
    r_point = rt_est.last_valid()
    if not np.isfinite(r_point):
        r_point = 1.0

    params = _sample_params(r, n_draws)
    # Lognormal around the point estimate keeps R positive and is right-skewed,
    # matching the asymmetry of the Gamma posterior it approximates.
    r_draws = np.clip(r_point * np.exp(r.normal(0, rt_sd, n_draws)), 0.05, 5.0)

    out = np.empty((n_draws, hist.size + horizon))
    for i in range(n_draws):
        future = project_renewal(hist, float(r_draws[i]), horizon, w=w, damping=damping)
        full = np.concatenate([hist, future])
        out[i] = cases_to_icu_census(
            full,
            ihr=float(params["ihr"][i]),
            icu_share=float(params["icu_share"][i]),
            lag_mean=float(params["lag_mean"][i]),
            los_mean=float(params["los_mean"][i]),
        )
    return out


def exceedance_curve(paths: np.ndarray, threshold: float,
                     start_index: int) -> np.ndarray:
    """Per-day probability that the census exceeds ``threshold`` from ``start_index``."""
    if not np.isfinite(threshold) or threshold <= 0:
        return np.full(paths.shape[1] - start_index, np.nan)
    return (paths[:, start_index:] > threshold).mean(axis=0)


def lead_time(exceedance: np.ndarray, probability: float = 0.5) -> float:
    """Days until the breach probability first reaches ``probability``.

    Returns NaN when the threshold is never crossed within the horizon -- which
    is the answer "no action needed yet", and is reported as such rather than
    silently becoming the horizon length.
    """
    hits = np.flatnonzero(np.asarray(exceedance) >= probability)
    return float(hits[0]) if hits.size else np.nan


def assess_region(cases: np.ndarray, population: float, *, horizon: int = 21,
                  n_draws: int = 400, capacity: float | None = None,
                  seed: int = 0) -> dict:
    """Full risk assessment for one region as of the last observed day."""
    hist = np.nan_to_num(np.clip(np.asarray(cases, float), 0, None))
    if hist.size < 60 or hist[-60:].sum() < 50:
        return {}

    paths = simulate_census(hist, horizon=horizon, n_draws=n_draws, seed=seed)
    now = hist.size - 1

    observed_peak = float(np.median(paths[:, :now + 1].max(axis=1)))
    benchmark = capacity if capacity is not None else observed_peak

    future = paths[:, now + 1:]
    exceed = exceedance_curve(paths, benchmark, now + 1)

    per_100k = 1e5 / population if population and population > 0 else np.nan
    projected_peak = float(np.median(future.max(axis=1)))
    benchmark_per_100k = benchmark * per_100k

    # A region that never had a wave has a prior peak near zero, and dividing by
    # it turns a small outbreak into an "infinite" utilisation. Below this floor
    # the ratio is not reported at all rather than reported as a huge number:
    # the honest statement is "no prior peak worth comparing to".
    benchmark_usable = np.isfinite(benchmark_per_100k) and benchmark_per_100k >= 1.0

    return {
        "current_census_median": float(np.median(paths[:, now])),
        "projected_peak_median": projected_peak,
        "projected_peak_p90": float(np.percentile(future.max(axis=1), 90)),
        "projected_peak_per_100k": float(projected_peak * per_100k),
        "benchmark_peak": benchmark,
        "benchmark_per_100k": float(benchmark_per_100k),
        "benchmark_usable": bool(benchmark_usable),
        "utilisation_median": (float(projected_peak / benchmark)
                               if benchmark_usable and benchmark > 0 else np.nan),
        "prob_exceed_benchmark": (float(exceed.max())
                                  if exceed.size and benchmark_usable else np.nan),
        "lead_time_days": lead_time(exceed, 0.5) if benchmark_usable else np.nan,
        "lead_time_days_p25": lead_time(exceed, 0.25) if benchmark_usable else np.nan,
        "horizon": horizon,
    }


def triage_table(panel: pd.DataFrame, *, as_of: str, horizon: int = 21,
                 n_draws: int = 300, min_population: float = 1e6,
                 seed: int = 0) -> pd.DataFrame:
    """Rank regions by projected ICU pressure as of a chosen date.

    ``as_of`` is respected strictly -- nothing after it enters any calculation --
    so the table is exactly what could have been produced on the day.
    """
    cutoff = pd.Timestamp(as_of)
    rows = []
    for entity, g in panel.groupby("entity", sort=True):
        g = g[g["date"] <= cutoff].sort_values("date")
        if g.empty:
            continue
        population = float(g["population"].iloc[0]) if "population" in g else np.nan
        if not np.isfinite(population) or population < min_population:
            continue

        res = assess_region(g["new_cases"].to_numpy(float), population,
                            horizon=horizon, n_draws=n_draws, seed=seed)
        if not res:
            continue
        rows.append({"entity": entity, "as_of": cutoff, "population": population, **res})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Rank by absolute per-capita pressure, not by the ratio to each region's own
    # history. The ratio answers "is this unusual *here*", which sounds like the
    # better question but is dominated by regions that previously had almost no
    # epidemic -- a place going from 0.1 to 2 ICU patients per 100k tops a ratio
    # ranking while needing no help at all. Per-capita census is what determines
    # whether a health system is actually in trouble, and it is comparable across
    # regions. Utilisation is retained as a severity qualifier.
    return (out.sort_values("projected_peak_per_100k", ascending=False)
            .reset_index(drop=True))
