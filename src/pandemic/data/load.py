"""Tidy loaders. Raw upstream files in, analysis-ready frames out.

Each loader caches its result as parquet under ``data/processed`` so the
expensive CSV parse (the OWID file is ~100 MB) happens once. Pass
``refresh=True`` to rebuild.

Conventions used everywhere downstream:
  * one row per (entity, date), sorted, no duplicates
  * ``entity`` is the unit of analysis (country or Indian state)
  * daily counts keep their raw sign -- negatives are *data*, not errors to
    clip away, because they are exactly what the forensics module hunts for
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pandemic.config import MIN_CUMULATIVE_CASES, MIN_POPULATION, PROCESSED, get_logger
from pandemic.data.sources import fetch

log = get_logger(__name__)


# --------------------------------------------------------------------------- cache


def _cached(name: str, builder: Callable[[], pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
    """Memoise a builder to parquet."""
    path = PROCESSED / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    log.info("built %-22s %6d rows x %3d cols", name, len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------- OWID

# Time-invariant country attributes. These are the confounders in the causal DAG.
COVARIATES = [
    "population",
    "population_density",
    "median_age",
    "aged_65_older",
    "aged_70_older",
    "gdp_per_capita",
    "extreme_poverty",
    "cardiovasc_death_rate",
    "diabetes_prevalence",
    "hospital_beds_per_thousand",
    "life_expectancy",
    "human_development_index",
]

# Time-varying series we care about.
SERIES = [
    "new_cases",
    "new_deaths",
    "total_cases",
    "total_deaths",
    "icu_patients",
    "hosp_patients",
    "new_tests",
    "positive_rate",
    "stringency_index",
    "people_vaccinated_per_hundred",
    "excess_mortality_cumulative",
    "excess_mortality",
]


def load_owid(refresh: bool = False) -> pd.DataFrame:
    """Country-day panel from Our World in Data.

    Drops OWID's aggregate rows (``World``, continents, income groups), which
    carry synthetic ``OWID_*`` codes and would otherwise be silently treated as
    countries -- a common and badly distorting mistake in this dataset.
    """

    def build() -> pd.DataFrame:
        path = fetch("owid")
        header = pd.read_csv(path, nrows=0).columns.tolist()
        wanted = ["iso_code", "continent", "location", "date"]
        wanted += [c for c in SERIES + COVARIATES if c in header]
        missing = [c for c in SERIES + COVARIATES if c not in header]
        if missing:
            log.warning("OWID schema drift, columns absent: %s", missing)

        df = pd.read_csv(path, usecols=wanted, low_memory=False)
        df["date"] = pd.to_datetime(df["date"])

        # OWID_* codes are aggregates, not countries.
        df = df[~df["iso_code"].astype("string").str.startswith("OWID_", na=True)]
        df = df[df["continent"].notna()]

        df = df.rename(columns={"location": "entity"})
        df = df.sort_values(["entity", "date"]).reset_index(drop=True)
        return df

    return _cached("owid_panel", build, refresh)


def load_country_covariates(refresh: bool = False) -> pd.DataFrame:
    """One row per country of time-invariant attributes, plus epidemic summaries.

    ``first_case_date`` and ``days_to_100_cases`` capture *epidemic timing*, which
    is the single most important confounder in cross-country policy comparisons:
    countries hit early had neither the warning nor the option to act early.
    """

    def build() -> pd.DataFrame:
        panel = load_owid()
        cov = (
            panel.groupby(["iso_code", "entity", "continent"], as_index=False)[
                [c for c in COVARIATES if c in panel.columns]
            ]
            .median(numeric_only=True)
        )

        # Epidemic timing, derived from the cumulative series.
        timing = []
        for (iso, entity), g in panel.groupby(["iso_code", "entity"], sort=False):
            g = g.sort_values("date")
            total = g["total_cases"].ffill()
            hit1 = g.loc[total >= 1, "date"]
            hit100 = g.loc[total >= 100, "date"]
            first = hit1.iloc[0] if len(hit1) else pd.NaT
            hundred = hit100.iloc[0] if len(hit100) else pd.NaT
            timing.append(
                {
                    "iso_code": iso,
                    "entity": entity,
                    "first_case_date": first,
                    "date_100_cases": hundred,
                    "days_first_to_100": (
                        (hundred - first).days if pd.notna(first) and pd.notna(hundred) else np.nan
                    ),
                }
            )
        timing_df = pd.DataFrame(timing)

        out = cov.merge(timing_df, on=["iso_code", "entity"], how="left")

        # Days after the global epidemic onset that this country was seeded.
        anchor = out["date_100_cases"].min()
        out["seeding_delay_days"] = (out["date_100_cases"] - anchor).dt.days
        return out

    return _cached("country_covariates", build, refresh)


def analysis_countries(min_cases: int = MIN_CUMULATIVE_CASES,
                       min_pop: int = MIN_POPULATION) -> list[str]:
    """Countries with a large enough epidemic and population to model."""
    panel = load_owid()
    peak = panel.groupby("entity")["total_cases"].max()
    pop = panel.groupby("entity")["population"].median()
    keep = peak[(peak >= min_cases) & (pop.reindex(peak.index) >= min_pop)]
    return sorted(keep.index.tolist())


# ----------------------------------------------------------------------------- JHU


def _tidy_jhu_wide(path, value_name: str) -> pd.DataFrame:
    """Melt JHU's wide date-column layout into a country-day long frame."""
    wide = pd.read_csv(path)
    id_cols = ["Province/State", "Country/Region", "Lat", "Long"]
    date_cols = [c for c in wide.columns if c not in id_cols]

    long = wide.melt(
        id_vars=["Country/Region"], value_vars=date_cols,
        var_name="date", value_name=value_name,
    )
    long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
    # Sub-national rows (provinces, overseas territories) roll up to the country.
    long = (
        long.groupby(["Country/Region", "date"], as_index=False)[value_name]
        .sum()
        .rename(columns={"Country/Region": "entity"})
    )
    return long


def load_jhu(refresh: bool = False) -> pd.DataFrame:
    """JHU CSSE cumulative counts with *unsmoothed, unclipped* daily differences.

    OWID silently repairs some negative daily values. JHU does not, which makes
    it the right substrate for reporting forensics: a negative ``new_cases`` is a
    downward revision, and those are the events we want to find.
    """

    def build() -> pd.DataFrame:
        conf = _tidy_jhu_wide(fetch("jhu_confirmed"), "cum_cases")
        dead = _tidy_jhu_wide(fetch("jhu_deaths"), "cum_deaths")
        df = conf.merge(dead, on=["entity", "date"], how="outer")
        df = df.sort_values(["entity", "date"]).reset_index(drop=True)

        grp = df.groupby("entity", sort=False)
        df["new_cases"] = grp["cum_cases"].diff()
        df["new_deaths"] = grp["cum_deaths"].diff()
        df["dow"] = df["date"].dt.dayofweek
        return df

    return _cached("jhu_panel", build, refresh)


# --------------------------------------------------------------------------- India


def load_india_states(refresh: bool = False) -> pd.DataFrame:
    """Daily confirmed / recovered / deceased by Indian state.

    The upstream file is doubly wide: one column per state, and a ``Status``
    column cycling Confirmed/Recovered/Deceased across three rows per date.
    """

    def build() -> pd.DataFrame:
        raw = pd.read_csv(fetch("india_states"))
        raw["Date_YMD"] = pd.to_datetime(
            raw["Date_YMD"] if "Date_YMD" in raw.columns else raw["Date"],
            format="mixed", dayfirst=True,
        )
        drop = {"Date", "Date_YMD", "Status", "TT"}  # TT = all-India total
        state_cols = [c for c in raw.columns if c not in drop]

        long = raw.melt(
            id_vars=["Date_YMD", "Status"], value_vars=state_cols,
            var_name="state_code", value_name="value",
        )
        wide = (
            long.pivot_table(
                index=["Date_YMD", "state_code"], columns="Status",
                values="value", aggfunc="sum",
            )
            .reset_index()
            .rename(columns={
                "Date_YMD": "date", "Confirmed": "new_cases",
                "Deceased": "new_deaths", "Recovered": "new_recovered",
            })
        )
        wide.columns.name = None
        wide["entity"] = wide["state_code"].map(INDIA_STATE_NAMES).fillna(wide["state_code"])
        wide = wide.sort_values(["entity", "date"]).reset_index(drop=True)

        grp = wide.groupby("entity", sort=False)
        wide["cum_cases"] = grp["new_cases"].cumsum()
        wide["cum_deaths"] = grp["new_deaths"].cumsum()
        wide["population"] = wide["state_code"].map(INDIA_STATE_POPULATION)
        return wide

    return _cached("india_states", build, refresh)


# State codes used by the covid19india API, with 2021 projected populations
# (Unique Identification Authority of India projections, in persons).
INDIA_STATE_NAMES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh",
    "CT": "Chhattisgarh", "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana",
    "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "JH": "Jharkhand",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
    "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OR": "Odisha",
    "PY": "Puducherry", "PB": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim",
    "TN": "Tamil Nadu", "TG": "Telangana", "TR": "Tripura",
    "UP": "Uttar Pradesh", "UT": "Uttarakhand", "WB": "West Bengal",
}

INDIA_STATE_POPULATION = {
    "AN": 400_000, "AP": 53_900_000, "AR": 1_570_000, "AS": 35_600_000,
    "BR": 124_800_000, "CH": 1_160_000, "CT": 29_400_000, "DN": 800_000,
    "DL": 20_600_000, "GA": 1_540_000, "GJ": 70_000_000, "HR": 28_900_000,
    "HP": 7_450_000, "JK": 13_600_000, "JH": 38_600_000, "KA": 67_600_000,
    "KL": 35_700_000, "LA": 290_000, "LD": 70_000, "MP": 85_000_000,
    "MH": 123_100_000, "MN": 3_100_000, "ML": 3_360_000, "MZ": 1_240_000,
    "NL": 2_200_000, "OR": 46_400_000, "PY": 1_600_000, "PB": 30_100_000,
    "RJ": 81_000_000, "SK": 690_000, "TN": 77_800_000, "TG": 38_500_000,
    "TR": 4_100_000, "UP": 231_500_000, "UT": 11_400_000, "WB": 99_100_000,
}
