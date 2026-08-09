"""Stage 4 -- hospital capacity: validation, projection, and triage.

Usage:
    python scripts/run_capacity.py [--as-of 2020-11-01]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pandemic.capacity import (
    cases_to_icu_census,
    gamma_kernel,
    los_survival,
    peak_lag_days,
    simulate_census,
    summarise,
    triage_table,
    validate_all,
)
from pandemic.capacity.validate import POST_VACCINE_START, VACCINE_CUTOFF
from pandemic.config import EPI, TABLES, get_logger, set_seed
from pandemic.data.load import load_owid
from pandemic.viz import plots_capacity as pcap
from pandemic.viz.theme import render

log = get_logger("capacity")

# Autumn 2020: Europe's second wave is building, vaccines are months away, and
# the decision "who needs surge capacity" is live. A date chosen for being a real
# decision point rather than for flattering the model.
DEFAULT_AS_OF = "2020-10-25"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    set_seed()

    panel = load_owid()

    # ------------------------------------------------------------- validation
    # Evaluated separately either side of vaccination, because the model assumes
    # a fixed infection-hospitalisation ratio and vaccination changes it.
    val = validate_all(panel, end=VACCINE_CUTOFF)
    val_post = validate_all(panel, start=POST_VACCINE_START, min_days=90)

    if val.empty:
        log.warning("no country had enough ICU data to validate against")
    else:
        val.to_csv(TABLES / "capacity_validation.csv", index=False)
        if not val_post.empty:
            val_post.to_csv(TABLES / "capacity_validation_postvax.csv", index=False)

        stats = summarise(val)
        log.info("pre-vaccination: %d countries, median r = %.3f, %.0f%% above 0.8",
                 stats["n_countries"], stats["median_correlation"],
                 100 * stats["share_above_0.8"])
        if not val_post.empty:
            log.info("post-vaccination: %d countries, median r = %.3f (degradation is "
                     "the expected consequence of a fixed IHR assumption)",
                     len(val_post), float(val_post["correlation"].median()))
        log.info("level multiplier median %.2f, 10-90 spread %.1fx",
                 stats["level_multiplier_median"], stats["level_multiplier_spread_ratio"])

        render("capacity_validation", pcap.validation(val, val_post), figsize=(12, 6.8))

        # One country's curve, as the concrete illustration.
        best = val.iloc[0]
        g = panel[(panel["entity"] == best["entity"])
                  & (panel["date"] < pd.Timestamp(VACCINE_CUTOFF))].sort_values("date")
        pred = cases_to_icu_census(g["new_cases"].to_numpy(float))
        render("capacity_validation_example",
               pcap.validation_example(g["date"], g["icu_patients"].to_numpy(float),
                                       best["level_multiplier"] * pred,
                                       best["entity"], best["correlation"],
                                       best["level_multiplier"]),
               figsize=(10.5, 5.2))

    # ---------------------------------------------------------------- kernels
    lag_k = gamma_kernel(EPI.onset_to_admission_mean, EPI.onset_to_admission_sd)
    los_c = los_survival(EPI.icu_los_mean, EPI.icu_los_sd)
    render("capacity_kernels", pcap.kernels(lag_k, los_c), figsize=(11, 4.6))

    # The headline mechanical number: how long the warning is.
    lags = []
    for _, g in panel.groupby("entity", sort=False):
        c = np.nan_to_num(g.sort_values("date")["new_cases"].to_numpy(float).clip(0))
        if c.sum() < 10_000:
            continue
        lags.append(peak_lag_days(c, cases_to_icu_census(c)))
    median_peak_lag = float(np.median(lags)) if lags else np.nan
    log.info("median case-peak to ICU-peak lag: %.0f days (n=%d)", median_peak_lag, len(lags))

    # ----------------------------------------------------------------- triage
    table = triage_table(panel, as_of=args.as_of, horizon=21, n_draws=args.draws)
    if table.empty:
        log.warning("triage table empty for as-of %s", args.as_of)
    else:
        table.to_csv(TABLES / "capacity_triage.csv", index=False)
        cols = ["entity", "projected_peak_per_100k", "utilisation_median",
                "prob_exceed_benchmark", "lead_time_days"]
        log.info("\n%s", table.head(12)[cols].to_string(index=False))
        render("capacity_triage", pcap.triage(table, args.as_of), figsize=(12, 6.4))

        # Fan chart for the most-pressured region in the table.
        top = table.iloc[0]["entity"]
        g = panel[(panel["entity"] == top) & (panel["date"] <= pd.Timestamp(args.as_of))]
        g = g.sort_values("date")
        cases = g["new_cases"].to_numpy(float)
        paths = simulate_census(cases, horizon=21, n_draws=args.draws)
        dates = list(g["date"]) + list(
            pd.date_range(g["date"].iloc[-1] + pd.Timedelta(days=1), periods=21))
        render("capacity_fan",
               pcap.fan(paths, dates, len(cases) - 1, top,
                        float(table.iloc[0]["benchmark_peak"])),
               figsize=(11, 5.6))

    # ---------------------------------------------------------------- summary
    summary = {
        "as_of": args.as_of,
        "median_case_peak_to_icu_peak_lag_days": median_peak_lag,
        "n_countries_with_lag": len(lags),
        "validation_pre_vaccination": summarise(val) if not val.empty else {"n_countries": 0},
        "validation_post_vaccination": (
            summarise(val_post) if not val_post.empty else {"n_countries": 0}),
        "vaccine_cutoff": VACCINE_CUTOFF,
        "validation_top10": (val.head(10).to_dict("records") if not val.empty else []),
        "triage_top10": (table.head(10).to_dict("records") if not table.empty else []),
        "parameters": {
            "ihr": [EPI.ihr_low, EPI.ihr_mean, EPI.ihr_high],
            "icu_share": [EPI.icu_share_low, EPI.icu_share_mean, EPI.icu_share_high],
            "onset_to_admission_mean": EPI.onset_to_admission_mean,
            "icu_los_mean": EPI.icu_los_mean,
        },
    }
    (TABLES / "capacity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("stage 4 complete")


if __name__ == "__main__":
    main()
