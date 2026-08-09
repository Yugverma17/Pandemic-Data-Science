"""Stage 3 -- causal analysis of why regional outcomes diverged.

Usage:
    python scripts/run_causal.py [--fast]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pandemic.causal import (
    analysis_sample,
    build_absolute_frame,
    build_country_design,
    build_dag,
    columns_for,
    dml_effect,
    minimal_backdoor_sets,
    ols_effect,
    propensity_weighted_effect,
    run_suite,
    scaling_diagnostic,
)
from pandemic.causal import describe as describe_dag
from pandemic.causal.confounding import mechanical_scaling_demo
from pandemic.causal.identification import assess
from pandemic.causal.synthetic_control import placebo_in_time, placebo_inference
from pandemic.config import SEED, TABLES, get_logger, set_seed
from pandemic.data.load import load_india_states
from pandemic.viz import plots_causal as pc
from pandemic.viz.theme import render

log = get_logger("causal")

OUTCOME = "log_deaths_per_million_1y"
TREATMENT = "stringency_mean_60d"

# Maharashtra announced weekend restrictions and a night curfew on 4-5 April
# 2021, well ahead of the other large states in the second wave. Delhi followed
# on 19 April, Uttar Pradesh on 25 April, Karnataka on 27 April, which is what
# bounds the usable post-window.
INDIA_INTERVENTION = "2021-04-05"
INDIA_TREATED = "Maharashtra"
INDIA_POST_DAYS = 21
INDIA_TRIO = ("Karnataka", "Maharashtra", "Delhi")


def _sc_verdict(result, placebo: dict, in_time: dict) -> str:
    """State plainly whether the synthetic control detected anything.

    A synthetic control that fails its own placebo-in-time check has not found a
    small effect -- it has shown that the treated unit drifts from its donors for
    reasons unrelated to any intervention, so the post-period gap carries no
    information. Saying so is the result.
    """
    fake_bigger = in_time["rmspe_ratio"] >= result.rmspe_ratio
    not_significant = placebo["p_value"] > 0.10

    if fake_bigger or not_significant:
        return (
            f"NULL RESULT, and the design does not support interpreting it. The "
            f"post/pre RMSPE ratio is {result.rmspe_ratio:.2f} and ranks "
            f"{placebo['rank']} of {placebo['n_placebos'] + 1} placebos "
            f"(p = {placebo['p_value']:.2f})"
            + (f"; a fake intervention 45 days earlier produces a *larger* apparent "
               f"effect (ratio {in_time['rmspe_ratio']:.2f})." if fake_bigger else ".")
            + " With a 21-day post-window forced by donor contamination and a "
              "pre-period fit that is not tight, this design lacks the power to "
              "detect a plausible effect. It is reported as a failed identification "
              "attempt rather than as evidence of no effect."
        )
    return (
        f"Effect of {result.effect_mean_pct:+.1f}% with placebo p = "
        f"{placebo['p_value']:.3f}; the fake-date check is clean "
        f"(ratio {in_time['rmspe_ratio']:.2f} vs {result.rmspe_ratio:.2f})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="fewer refutation draws")
    args = parser.parse_args()
    set_seed()

    # -------------------------------------------------------------------- DAG
    graph = build_dag()
    dag_info = describe_dag()
    adjustment_nodes = set(minimal_backdoor_sets(graph)[0])
    log.info("adjustment set (graph nodes): %s", sorted(adjustment_nodes))
    log.info("mediators refused by the criterion: %s", dag_info["mediators_excluded"])

    # ---------------------------------------------------------------- design
    design = build_country_design()
    sample = analysis_sample(design)
    controls = [c for c in columns_for(adjustment_nodes) if c in sample.columns]
    log.info("controls: %s", controls)

    # ------------------------------------------------------------- estimates
    estimates = [
        ols_effect(sample, OUTCOME, TREATMENT, [], method="OLS, unadjusted"),
        ols_effect(sample, OUTCOME, TREATMENT, controls, method="OLS, back-door adjusted"),
        dml_effect(sample, OUTCOME, TREATMENT, controls,
                   n_repeats=5 if args.fast else 20),
        propensity_weighted_effect(sample, OUTCOME, TREATMENT, controls),
    ]
    for e in estimates:
        log.info("%-38s %+.5f  CI [%+.4f, %+.4f]  p=%.4g",
                 e.method, e.estimate, e.ci_low, e.ci_high, e.p_value)

    headline = estimates[2]  # DML

    # ------------------------------------------------- identification checks
    ident = assess(sample, TREATMENT, controls, headline)
    log.info("response channel: %s", ident["response_channel"]["interpretation"])
    log.info("effect-before-cause: %s", ident["effect_before_cause"]["interpretation"])
    log.info("IDENTIFIED: %s", ident["identified"])

    # ------------------------------------------------------ refutation suite
    #
    # The suite refits the estimator ~136 times, so it uses a deliberately
    # lighter forest than the headline estimate: 150 trees rather than 400, and
    # 2 cross-fitting repeats rather than 20. That is a precision/compute
    # trade-off, and it is the right way round -- refutations ask whether the
    # estimate *moves*, a question that tolerates far more Monte Carlo noise than
    # the point estimate itself. With the full-size learner this stage took over
    # twenty minutes, which is long enough that people stop running it.
    refute_learner = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=5, max_features=0.6,
        random_state=SEED, n_jobs=1,
    )
    refutation = run_suite(
        sample, TREATMENT,
        estimator=lambda d, c: dml_effect(d, OUTCOME, TREATMENT, c, n_repeats=2,
                                          learner=refute_learner),
        controls=controls, estimate=headline,
        n_placebo=40 if args.fast else 150,
    )
    log.info("refutations passed: %d/%d", refutation["n_checks_passed"],
             refutation["n_checks"])
    log.info("E-value %.2f (CI limit %.2f)",
             refutation["e_value"]["e_value"], refutation["e_value"].get("e_value_ci", 1.0))

    # -------------------------------------------------- population scaling
    absolute = build_absolute_frame(design)
    scaling = {
        "countries_beds_vs_cases": scaling_diagnostic(
            absolute, "total_hospital_beds", "total_cases_1y"),
        "mechanical_demo": mechanical_scaling_demo(
            absolute["population"].to_numpy(),
            absolute["hospital_beds_per_thousand"].to_numpy()),
    }

    # ---------------------------------------------------------------- India
    india = load_india_states()
    agg = (india.groupby("entity")
           .agg(cases=("new_cases", "sum"), deaths=("new_deaths", "sum"),
                population=("population", "first"))
           .dropna())
    agg = agg[agg["population"] > 2e6].reset_index()
    agg["deaths_per_million"] = agg["deaths"] / agg["population"] * 1e6
    agg["cases_per_million"] = agg["cases"] / agg["population"] * 1e6
    agg["case_fatality_pct"] = 100 * agg["deaths"] / agg["cases"].replace(0, np.nan)
    agg = agg.sort_values("deaths_per_million", ascending=False)
    agg.to_csv(TABLES / "india_state_outcomes.csv", index=False)

    trio = agg[agg["entity"].isin(INDIA_TRIO)][
        ["entity", "cases_per_million", "deaths_per_million", "case_fatality_pct"]]
    log.info("\n%s", trio.to_string(index=False))

    india_scaling = scaling_diagnostic(agg, "cases", "deaths")

    # --------------------------------------------------- synthetic control
    sc_panel = india.copy()
    sc_panel["value"] = (sc_panel["new_cases"] / sc_panel["population"] * 1e6)
    sc_panel = sc_panel[sc_panel["population"] > 2e6]
    sc_panel = sc_panel[["entity", "date", "value"]].dropna()

    sc_report: dict = {}
    try:
        placebo = placebo_inference(
            sc_panel, INDIA_TREATED, INDIA_INTERVENTION,
            pre_days=60, post_days=INDIA_POST_DAYS)
        res = placebo["result"]
        in_time = placebo_in_time(sc_panel, INDIA_TREATED, INDIA_INTERVENTION,
                                  shift_days=45, pre_days=45, post_days=INDIA_POST_DAYS)
        log.info("synthetic control: effect %.1f%% (p=%.3f, %d placebos); "
                 "placebo-in-time ratio %.2f",
                 res.effect_mean_pct, placebo["p_value"], placebo["n_placebos"],
                 in_time["rmspe_ratio"])
        sc_report = {
            "treated": INDIA_TREATED,
            "intervention": INDIA_INTERVENTION,
            "post_days": INDIA_POST_DAYS,
            "donor_weights": {k: round(float(v), 4) for k, v in res.weights.items()},
            "pre_rmspe": res.pre_rmspe,
            "post_rmspe": res.post_rmspe,
            "rmspe_ratio": res.rmspe_ratio,
            "effect_mean_pct": res.effect_mean_pct,
            "placebo_p_value": placebo["p_value"],
            "n_placebos": placebo["n_placebos"],
            "placebo_in_time": in_time,
            "caveat": (
                "Other large states restricted within three weeks of the treated "
                "date, so the donor pool is only clean for a 21-day post-window. "
                "This bounds what the design can detect and is reported rather "
                "than worked around."
            ),
            "verdict": _sc_verdict(res, placebo, in_time),
        }
        render("causal_synthetic_control",
               pc.synthetic_control_plot(res, placebo, INDIA_TREATED), figsize=(10.5, 7.0))
    except Exception as exc:  # noqa: BLE001 - donor pool may be too thin
        log.warning("synthetic control unavailable: %s", exc)
        sc_report = {"error": str(exc)}

    # -------------------------------------------------------------- figures
    render("causal_dag",
           pc.causal_dag(graph, adjustment_nodes, set(dag_info["mediators_excluded"])),
           figsize=(11, 7.0))

    placebo_row = dict(ident["effect_before_cause"]["placebo_pre_period"])
    placebo_row["method"] = "Falsification: deaths in first 21 days"
    placebo_row["_placebo"] = True
    render("causal_estimate_ladder",
           pc.estimate_ladder([e.as_row() for e in estimates], placebo_row),
           figsize=(10.5, 5.0))

    render("causal_india_regional", pc.india_regional(agg, INDIA_TRIO), figsize=(9.5, 8.0))

    # -------------------------------------------------------------- summary
    summary = {
        "dag": dag_info,
        "controls_used": controls,
        "sample": {"n_analysed": int(len(sample)),
                   "n_designed": int(sample.attrs.get("n_before", len(sample))),
                   "n_dropped_missing": int(sample.attrs.get("dropped", 0))},
        "estimates": [e.as_row() | {"detail": e.detail} for e in estimates],
        "identification": ident,
        "refutation": refutation,
        "population_scaling": scaling,
        "india": {
            "trio": trio.to_dict("records"),
            "top_10_by_deaths_per_million": agg.head(10)[
                ["entity", "deaths_per_million", "cases_per_million",
                 "case_fatality_pct"]].to_dict("records"),
            "scaling_diagnostic": india_scaling,
            "synthetic_control": sc_report,
        },
    }
    (TABLES / "causal_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    pd.DataFrame([e.as_row() for e in estimates]).to_csv(
        TABLES / "causal_estimates.csv", index=False)
    log.info("stage 3 complete")


if __name__ == "__main__":
    main()
