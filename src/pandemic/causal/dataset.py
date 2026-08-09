"""Study design: country-day panel to one row per country.

Three design choices, all of which matter more than the estimator applied later.

Epidemic time, not calendar time. Every window is indexed to the date a country
hit its 100th case. Comparing Italy's March 2020 with New Zealand's compares a
country three weeks into an outbreak against one with almost no cases, and that
difference in stage would show up as a difference in policy. Indexing on epidemic
time also fixes the case count at t = 0, which removes the largest single source
of reverse causality.

Deaths, not cases, as the outcome. Confirmed cases measure testing capacity as
much as infection, since a country testing ten times more finds more cases at the
same true prevalence. Deaths are far from clean but much less sensitive to
surveillance effort.

A pre-vaccination window. The outcome accrues over the year after each country's
100th case, which closes before mass vaccination almost everywhere. Vaccination is
post-treatment, on the path from policy to death, so conditioning on it would
block part of the effect being estimated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.config import get_logger
from pandemic.data.load import load_country_covariates, load_owid

log = get_logger(__name__)

TREATMENT_WINDOW_DAYS = 60
OUTCOME_WINDOW_DAYS = 365
MIN_STRINGENCY_COVERAGE = 0.5  # share of treatment-window days needing a value

STRUCTURAL_CONFOUNDERS = [
    "median_age",
    "aged_65_older",
    "log_gdp_per_capita",
    "log_population_density",
    "hospital_beds_per_thousand",
    "human_development_index",
    "diabetes_prevalence",
    "cardiovasc_death_rate",
    "life_expectancy",
    "log_population",
]

TIMING_CONFOUNDERS = [
    "seeding_delay_days",   # how late the country was seeded -> how much warning it had
    "onset_growth_rate",    # how fast cases were rising when the clock started
]

CONFOUNDERS = STRUCTURAL_CONFOUNDERS + TIMING_CONFOUNDERS


def build_country_design(refresh: bool = False) -> pd.DataFrame:
    """One row per country: treatment, outcome, confounders, and diagnostics."""
    panel = load_owid(refresh=refresh)
    cov = load_country_covariates(refresh=refresh)

    rows = []
    for (iso, entity), g in panel.groupby(["iso_code", "entity"], sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        base = cov[cov["iso_code"] == iso]
        if base.empty:
            continue
        base = base.iloc[0]

        t0 = base["date_100_cases"]
        population = base["population"]
        if pd.isna(t0) or pd.isna(population) or population <= 0:
            continue

        t_end = t0 + pd.Timedelta(days=TREATMENT_WINDOW_DAYS)
        y_end = t0 + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)

        window = g[(g["date"] >= t0) & (g["date"] < t_end)]
        outcome_window = g[(g["date"] >= t0) & (g["date"] < y_end)]
        if window.empty or outcome_window.empty:
            continue

        # --- treatment: how strict was the response, in the first two months
        stringency = window["stringency_index"]
        coverage = float(stringency.notna().mean())
        if coverage < MIN_STRINGENCY_COVERAGE:
            continue
        treatment = float(stringency.mean(skipna=True))

        # Speed of response: days until stringency first reached 50, from t0.
        reached = window.loc[window["stringency_index"] >= 50, "date"]
        days_to_strict = float((reached.iloc[0] - t0).days) if len(reached) else np.nan

        # Outbreak size *during* the policy window. This is what governments were
        # watching when they chose how hard to clamp down, so it is the variable
        # that carries the simultaneity. It is emphatically NOT a control -- it is
        # partly a consequence of the treatment -- and is recorded only to test
        # whether the design is identified at all.
        window_cases_per_m = (float(window["new_cases"].clip(lower=0).sum(skipna=True))
                              / (population / 1e6))

        # --- outcome: deaths per million over the following year
        deaths = float(outcome_window["new_deaths"].clip(lower=0).sum(skipna=True))
        deaths_per_m = deaths / (population / 1e6)

        # --- falsification outcome: deaths in the first 21 days.
        # Policy cannot have caused these. Infection to death runs about three
        # weeks, and policy needs one to two more to change infections, so
        # anything dying inside 21 days of t0 was already infected when the
        # window opened. An association here is confounding, by construction.
        early = g[(g["date"] >= t0) & (g["date"] < t0 + pd.Timedelta(days=21))]
        early_deaths_per_m = (float(early["new_deaths"].clip(lower=0).sum(skipna=True))
                              / (population / 1e6)) if not early.empty else np.nan

        # --- confounder measured at t0: how fast was the epidemic moving?
        pre = g[g["date"] <= t0].tail(14)
        onset_growth = np.nan
        if len(pre) >= 14:
            early = float(pre["new_cases"].head(7).clip(lower=0).sum())
            late = float(pre["new_cases"].tail(7).clip(lower=0).sum())
            onset_growth = float(np.log1p(late) - np.log1p(early))

        # --- descriptive extras, not used as controls
        tests = outcome_window["new_tests"].sum(skipna=True)
        rows.append({
            "iso_code": iso,
            "entity": entity,
            "continent": base["continent"],
            "t0": t0,
            "stringency_mean_60d": treatment,
            "stringency_coverage": coverage,
            "days_to_stringency_50": days_to_strict,
            "deaths_per_million_1y": deaths_per_m,
            "deaths_per_million_first21d": early_deaths_per_m,
            "deaths_1y": deaths,
            "cases_per_million_1y": float(
                outcome_window["new_cases"].clip(lower=0).sum(skipna=True)) / (population / 1e6),
            "tests_per_million_1y": (float(tests) / (population / 1e6)) if tests > 0 else np.nan,
            "onset_growth_rate": onset_growth,
            "cases_per_million_window": window_cases_per_m,
            "seeding_delay_days": base["seeding_delay_days"],
            "population": population,
            "median_age": base["median_age"],
            "aged_65_older": base["aged_65_older"],
            "gdp_per_capita": base["gdp_per_capita"],
            "population_density": base["population_density"],
            "hospital_beds_per_thousand": base["hospital_beds_per_thousand"],
            "human_development_index": base["human_development_index"],
            "diabetes_prevalence": base["diabetes_prevalence"],
            "cardiovasc_death_rate": base["cardiovasc_death_rate"],
            "life_expectancy": base["life_expectancy"],
        })

    df = pd.DataFrame(rows)

    # Log transforms where the raw variable spans orders of magnitude; a linear
    # term in GDP would let the United States dominate the fit single-handedly.
    df["log_gdp_per_capita"] = np.log(df["gdp_per_capita"].clip(lower=1))
    df["log_population_density"] = np.log(df["population_density"].clip(lower=0.1))
    df["log_population"] = np.log(df["population"].clip(lower=1))
    # Outcome on the log scale: deaths per million are strongly right-skewed, and
    # the coefficient then reads as an approximate proportional change.
    df["log_deaths_per_million_1y"] = np.log1p(df["deaths_per_million_1y"])
    df["log_deaths_per_million_first21d"] = np.log1p(df["deaths_per_million_first21d"])
    df["log_cases_per_million_window"] = np.log1p(df["cases_per_million_window"])

    log.info("design: %d countries before filtering", len(df))
    return df


def analysis_sample(df: pd.DataFrame, confounders: list[str] | None = None,
                    min_deaths: float = 0.0) -> pd.DataFrame:
    """Complete-case sample for the estimators, with the attrition recorded.

    Dropping incomplete rows is a choice with consequences -- poorer countries
    are missing covariates more often, so the surviving sample skews rich. That
    is stated in the report rather than hidden, and it is why the estimate is
    described as an effect *among countries with complete data* rather than a
    global one.
    """
    confounders = confounders or CONFOUNDERS
    needed = ["stringency_mean_60d", "log_deaths_per_million_1y", *confounders]
    out = df.dropna(subset=needed).copy()
    if min_deaths > 0:
        out = out[out["deaths_per_million_1y"] >= min_deaths]
    out.attrs["n_before"] = len(df)
    out.attrs["n_after"] = len(out)
    out.attrs["dropped"] = len(df) - len(out)
    log.info("analysis sample: %d of %d countries (%d dropped for missing data)",
             len(out), len(df), len(df) - len(out))
    return out.reset_index(drop=True)
