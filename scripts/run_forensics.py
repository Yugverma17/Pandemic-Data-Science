"""Stage 1 -- reporting forensics.

Scores every country's reporting series, benchmarks the textbook z-score
detector against the decomposition, and writes figures + tables.

Usage:
    python scripts/run_forensics.py
"""

from __future__ import annotations

import json

import pandas as pd

from pandemic.config import TABLES, get_logger, set_seed
from pandemic.data.load import load_jhu
from pandemic.forensics import (
    backlog_dumps,
    build_scorecard,
    first_digit_test,
    negative_revisions,
    terminal_digit_test,
    weekday_profile,
    weight_sensitivity,
)
from pandemic.forensics.naive import compare_detectors, decompose_flags, naive_zscore_flags
from pandemic.viz import plots_forensics as pf
from pandemic.viz.theme import render

log = get_logger("forensics")

# Countries whose reporting rhythm illustrates a distinct failure mode.
FINGERPRINT_COUNTRIES = ["Nicaragua", "Tanzania", "Spain",
                         "United Kingdom", "Sweden", "India"]

# Countries where the naive detector is benchmarked in detail.
DECOMPOSE_COUNTRIES = ["India", "United Kingdom", "Brazil", "US"]


def main() -> None:
    set_seed()
    jhu = load_jhu()

    # ---------------------------------------------------------------- scorecard
    log.info("scoring %d entities ...", jhu["entity"].nunique())
    scorecard = build_scorecard(jhu, n_permutations=500)
    log.info("scored %d countries meeting the activity threshold", len(scorecard))

    keep = [c for c in scorecard.columns if c != "weekday_multipliers"]
    scorecard[keep].to_csv(TABLES / "reliability_scorecard.csv", index=False)

    sens = weight_sensitivity(scorecard, n_draws=500)
    log.info("weight sensitivity: spearman median %.4f (p05 %.4f), "
             "bottom-20 overlap median %.2f",
             sens["spearman_median"], sens["spearman_p05"], sens["bottom20_overlap_median"])

    # ------------------------------------------------- naive detector benchmark
    comparisons = {}
    decomposed_frames = {}
    for country in DECOMPOSE_COUNTRIES:
        g = jhu[jhu["entity"] == country].sort_values("date")
        if g.empty:
            log.warning("country not found: %s", country)
            continue
        s = g.set_index("date")["new_cases"]
        comparisons[country] = compare_detectors(s)
        decomposed_frames[country] = decompose_flags(s, naive_zscore_flags(s))

        # Recall: reporting events the forensic detectors find independently, and
        # how many of those the z-score also caught.
        dumps = backlog_dumps(s)
        event_days = set(dumps.loc[dumps["is_dump"], "date"]) if len(dumps) else set()
        event_days |= set(negative_revisions(s).index)
        flagged_days = set(s.index[naive_zscore_flags(s).to_numpy()])
        caught = len(event_days & flagged_days)
        comparisons[country]["forensic_events"] = {
            "n_events": len(event_days),
            "n_caught_by_zscore": caught,
            "recall": (caught / len(event_days)) if event_days else None,
        }

        c = comparisons[country]["global_z"]
        log.info("%-16s z-score flagged %3d days, %3d survive (precision %5.1f%%) | "
                 "%2d real reporting events, %d caught (recall %.0f%%)",
                 country, c["n_flagged"], c["n_true"], 100 * (c["precision"] or 0),
                 len(event_days), caught,
                 100 * (caught / len(event_days)) if event_days else 0.0)

    # ------------------------------------------------------------------ figures
    profiles = {}
    for country in FINGERPRINT_COUNTRIES:
        g = jhu[jhu["entity"] == country].sort_values("date")
        if g.empty:
            continue
        profiles[country] = weekday_profile(g.set_index("date")["new_cases"],
                                            n_permutations=500)

    render("forensics_weekday_fingerprints", pf.weekday_fingerprints(profiles),
           figsize=(11.5, 6.2))
    render("forensics_reliability_ranking", pf.reliability_ranking(scorecard),
           figsize=(10.5, 7.0))

    hero = "India"
    if hero in decomposed_frames:
        s = jhu[jhu["entity"] == hero].sort_values("date").set_index("date")["new_cases"]
        render("forensics_naive_decomposition",
               pf.naive_decomposition(s, decomposed_frames[hero], hero,
                                      panel=decomposed_frames),
               figsize=(11, 7.6))

    # Digit tests on the country with the strongest terminal-digit signal.
    flagged = scorecard[scorecard["terminal_verdict"].isin(["heaping", "non-uniform"])]
    target = (flagged.nsmallest(1, "terminal_p")["entity"].iloc[0]
              if len(flagged) else "Turkey")
    tv = jhu[jhu["entity"] == target]["new_cases"].to_numpy()
    render("forensics_digit_tests",
           pf.digit_tests(first_digit_test(tv), terminal_digit_test(tv), target),
           figsize=(11, 4.6))

    # ------------------------------------------------------------------- tables
    summary = {
        "n_countries_scored": int(len(scorecard)),
        "weight_sensitivity": sens,
        "least_reliable": scorecard.nsmallest(10, "reliability_index")[
            ["entity", "reliability_index", "worst_component", "worst_component_desc"]
        ].to_dict("records"),
        "most_reliable": scorecard.nlargest(10, "reliability_index")[
            ["entity", "reliability_index"]].to_dict("records"),
        "naive_detector_benchmark": comparisons,
        "digit_test_target": target,
        # Countries too sparse to score. The exclusion is not a clean bill of
        # health -- several reported so little that no detector has anything to
        # work with, which is the most severe reporting failure of all.
        "excluded_insufficient_data": {
            "n": len(scorecard.attrs.get("excluded", [])),
            "min_active_days": scorecard.attrs.get("min_active_days"),
            "most_irregular": scorecard.attrs.get("excluded", [])[:10],
        },
        "totals": {
            "countries_with_negative_days": int((scorecard["neg_days"] > 0).sum()),
            "countries_with_reporting_gaps": int((scorecard["gap_longest"] >= 3).sum()),
            "countries_with_constant_fill": int((scorecard["fill_longest"] >= 3).sum()),
            "countries_with_significant_weekday": int(
                ((scorecard["weekday_p"] < 0.01) & (scorecard["weekday_amplitude"] > 0.3)).sum()),
            "countries_failing_benford": int(
                (scorecard["benford_verdict"] == "nonconformity").sum()),
            "countries_with_heaping": int((scorecard["terminal_verdict"] == "heaping").sum()),
        },
    }
    (TABLES / "forensics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    pd.DataFrame(summary["least_reliable"]).to_csv(
        TABLES / "least_reliable_countries.csv", index=False)

    log.info("stage 1 complete: %d countries scored", len(scorecard))
    for k, v in summary["totals"].items():
        log.info("  %-38s %d", k, v)


if __name__ == "__main__":
    main()
