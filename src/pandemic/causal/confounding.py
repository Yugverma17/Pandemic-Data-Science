"""Check whether a correlation is really just population scaling.

An easy way to get a regional analysis wrong: correlate two absolute quantities
that both scale with population, get r > 0.9, and report it as a relationship
between the variables. Big regions have more of everything.

This is not the "always use per capita" rule of thumb. It is a decomposition.
Estimate the correlation, re-estimate holding log population fixed (partial
correlation), then re-estimate per capita. A correlation surviving none of the
three was measuring population.

On this project's data it finds nothing. Cases against deaths, across countries
and across Indian states, survives every adjustment (India: raw 0.91, partial
0.89, per capita 0.84). The relationship is real and the tool says so, which is
the point of having it.

The real pathology is narrower: when one variable is *defined* as a fixed share of
population, as "projected infections" in a hospital planning scorecard is, its
correlation with any other population-scaled quantity approaches 1 regardless of
the underlying rates. See :func:`mechanical_scaling_demo`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Pearson correlation of x and y after linearly removing z from both.

    Returns (r, p). Degrees of freedom account for the covariates removed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    x, y, z = x[ok], y[ok], z[ok]
    n = x.size
    if n < z.shape[1] + 4:
        return np.nan, np.nan

    design = np.column_stack([np.ones(n), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - z.shape[1] - 2
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return r, np.nan
    t = r * np.sqrt(dof / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof))


def scaling_diagnostic(df: pd.DataFrame, x_col: str, y_col: str,
                       population_col: str = "population") -> dict:
    """Decompose a correlation into what survives population adjustment.

    ``x_col`` and ``y_col`` must be *absolute* counts. The three views reported:

    ``raw``          the naive correlation of the two totals
    ``partial``      correlation holding log population fixed
    ``per_capita``   correlation of the two rates
    """
    d = df[[x_col, y_col, population_col]].dropna()
    d = d[d[population_col] > 0]
    if len(d) < 12:
        raise ValueError("not enough complete rows for the diagnostic")

    x, y = d[x_col].to_numpy(float), d[y_col].to_numpy(float)
    pop = d[population_col].to_numpy(float)
    log_pop = np.log(pop)

    r_raw, p_raw = stats.pearsonr(x, y)
    r_partial, p_partial = partial_correlation(x, y, log_pop)

    xc, yc = x / pop, y / pop
    r_pc, p_pc = stats.pearsonr(xc, yc)

    # How much of each variable is simply population?
    r_x_pop = float(stats.pearsonr(x, pop).statistic)
    r_y_pop = float(stats.pearsonr(y, pop).statistic)

    return {
        "n": int(len(d)),
        "x": x_col, "y": y_col,
        "raw": {"r": float(r_raw), "p": float(p_raw), "r_squared": float(r_raw**2)},
        "partial_given_log_population": {"r": float(r_partial), "p": float(p_partial)},
        "per_capita": {"r": float(r_pc), "p": float(p_pc)},
        "x_vs_population_r": r_x_pop,
        "y_vs_population_r": r_y_pop,
        "variance_explained_lost": float(r_raw**2 - r_partial**2),
        "verdict": _verdict(r_raw, r_partial, r_pc),
    }


def _verdict(r_raw: float, r_partial: float, r_pc: float) -> str:
    if not np.isfinite(r_partial):
        return "inconclusive"
    if abs(r_raw) > 0.7 and abs(r_partial) < 0.3 and abs(r_pc) < 0.3:
        return ("the raw correlation is population scaling; it does not survive "
                "either adjustment")
    if abs(r_raw) > 0.7 and abs(r_partial) < 0.5:
        return "substantially attenuated by population; treat the raw figure as misleading"
    if np.sign(r_raw) != np.sign(r_pc) and abs(r_pc) > 0.2:
        return "sign reverses on the per-capita scale -- a Simpson's-paradox pattern"
    return "the association survives population adjustment"


def mechanical_scaling_demo(population: np.ndarray, beds_per_thousand: np.ndarray,
                            projected_attack_rate: float = 0.20) -> dict:
    """Show that a planning-scorecard correlation is true by construction.

    Hospital-capacity scorecards typically pair *total beds* with *projected
    infections*, where projected infections is an epidemiological attack-rate
    assumption multiplied by population. Both columns therefore contain
    population as a factor, and their correlation measures how unequal the
    regions' populations are -- nothing about healthcare or the virus.

    The demonstration reports what happens under a constant assumed attack rate
    (correlation is exactly the beds-vs-population correlation, and identically
    1.0 if bed density is also constant) and how little the picture changes when
    the rate is allowed to vary by +/- 50%.
    """
    pop = np.asarray(population, float)
    bpt = np.asarray(beds_per_thousand, float)
    ok = np.isfinite(pop) & np.isfinite(bpt) & (pop > 0)
    pop, bpt = pop[ok], bpt[ok]
    if pop.size < 12:
        raise ValueError("not enough regions for the demonstration")

    total_beds = bpt * pop / 1000.0
    projected_constant = projected_attack_rate * pop

    r_constant = float(stats.pearsonr(total_beds, projected_constant).statistic)

    # Same construction, but letting the attack rate vary substantially across
    # regions -- the correlation barely moves, because population dominates.
    r = np.random.default_rng(0)
    varying_rate = projected_attack_rate * r.uniform(0.5, 1.5, size=pop.size)
    r_varying = float(stats.pearsonr(total_beds, varying_rate * pop).statistic)

    # The identical-density case: correlation is exactly 1.
    r_identical = float(stats.pearsonr(pop / 1000.0, projected_constant).statistic)

    return {
        "n_regions": int(pop.size),
        "assumed_attack_rate": projected_attack_rate,
        "r_beds_vs_projected_constant_rate": r_constant,
        "r_beds_vs_projected_varying_rate": r_varying,
        "r_if_bed_density_identical": r_identical,
        "r_beds_vs_population": float(stats.pearsonr(total_beds, pop).statistic),
        "note": ("Both columns contain population as a factor. The correlation is a "
                 "property of the population distribution, not of hospital capacity, "
                 "and survives even large variation in the assumed attack rate."),
    }


def build_absolute_frame(design: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the absolute counts that invite the error.

    Hospital beds are published per thousand people, so total beds is literally
    ``beds_per_thousand * population / 1000`` -- population appears on both sides
    of the correlation by construction. That is the cleanest possible illustration
    of why the naive figure is guaranteed to look impressive.
    """
    d = design.dropna(subset=["hospital_beds_per_thousand", "population",
                              "cases_per_million_1y", "deaths_per_million_1y"]).copy()
    d["total_hospital_beds"] = d["hospital_beds_per_thousand"] * d["population"] / 1000.0
    d["total_cases_1y"] = d["cases_per_million_1y"] * d["population"] / 1e6
    d["total_deaths_1y"] = d["deaths_per_million_1y"] * d["population"] / 1e6
    return d
