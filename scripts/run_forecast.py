"""Stage 2 -- probabilistic forecasting and rolling-origin backtest.

Usage:
    python scripts/run_forecast.py [--entities 40] [--quick]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from pandemic.config import HORIZONS, QUANTILES, TABLES, get_logger, set_seed
from pandemic.data.load import load_jhu, load_owid
from pandemic.forecast import run_backtest, score, score_by_regime, summarise_calibration
from pandemic.viz import plots_forecast as pf
from pandemic.viz.theme import render

log = get_logger("forecast")


def top_entities(jhu: pd.DataFrame, n: int) -> list[str]:
    """Countries with the largest epidemics -- where a forecast has stakes."""
    totals = jhu.groupby("entity")["new_cases"].sum()
    return totals.nlargest(n).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=40)
    parser.add_argument("--quick", action="store_true",
                        help="fewer origins; for smoke-testing the pipeline")
    parser.add_argument("--rescore", action="store_true",
                        help="reuse the cached backtest and only redo scoring and figures")
    args = parser.parse_args()

    set_seed()
    raw_path = TABLES / "backtest_raw.parquet"

    if args.rescore and raw_path.exists():
        results = pd.read_parquet(raw_path)
        entities = sorted(results["entity"].unique())
        log.info("re-scoring cached backtest: %d rows, %d countries",
                 len(results), len(entities))
    else:
        jhu = load_jhu()
        owid = load_owid()
        pop = owid.groupby("entity")["population"].median()
        jhu = jhu.merge(pop.rename("population"), left_on="entity",
                        right_index=True, how="left")

        entities = top_entities(jhu, args.entities)
        log.info("backtesting %d countries", len(entities))

        results = run_backtest(
            jhu, entities,
            horizons=list(HORIZONS),
            start="2020-05-01", end="2023-03-01",
            every_days=28 if args.quick else 14,
            eval_from="2020-09-01",
        )
        results.to_parquet(raw_path, index=False)
        log.info("backtest rows: %d", len(results))

    # ------------------------------------------------------------------ scoring
    scores = score(results, QUANTILES)
    scores.to_csv(TABLES / "forecast_scores.csv", index=False)
    log.info("\n%s", scores.to_string(index=False))

    regime = score_by_regime(results, QUANTILES)
    regime.to_csv(TABLES / "forecast_scores_by_regime.csv", index=False)

    cal = summarise_calibration(results.dropna(subset=["q0.5"]), QUANTILES)
    cal.to_csv(TABLES / "forecast_calibration.csv", index=False)

    # ------------------------------------- cross-pillar: data quality vs skill
    quality_link = {}
    merged = pd.DataFrame()
    try:
        sc = pd.read_csv(TABLES / "reliability_scorecard.csv")
        best_model = scores[scores["horizon"] == HORIZONS[0]].iloc[0]["model"]
        per_entity = _per_entity_skill(results, best_model, HORIZONS[0])
        merged = per_entity.merge(sc[["entity", "reliability_index"]], on="entity")
        if len(merged) >= 8:
            rho, pv = stats.spearmanr(merged["reliability_index"], merged["rel_wis"])
            quality_link = {"model": best_model, "horizon": int(HORIZONS[0]),
                            "spearman_rho": float(rho), "p_value": float(pv),
                            "n_countries": int(len(merged))}
            log.info("data quality vs forecast skill: rho=%.3f p=%.4f (n=%d)",
                     rho, pv, len(merged))
            render("forecast_quality_vs_skill",
                   pf.quality_vs_skill(merged, rho, pv, HORIZONS[0]), figsize=(9.5, 6.0))
    except FileNotFoundError:
        log.warning("run_forensics.py has not been run; skipping the cross-pillar link")

    # ------------------------------------------------------------------ figures
    render("forecast_skill_ranking", pf.skill_ranking(scores), figsize=(12.5, 4.8))
    render("forecast_calibration", pf.calibration(cal), figsize=(10, 5.4))
    render("forecast_regime_skill", pf.regime_skill(regime, HORIZONS[-1]), figsize=(10, 5.4))

    best = scores[scores["horizon"] == HORIZONS[-1]].iloc[0]["model"]
    track_entity = _pick_track_entity(results, HORIZONS[-1])
    render("forecast_track_record",
           pf.forecast_track(results, track_entity, HORIZONS[-1], ("persistence", best)),
           figsize=(11, 7.0))

    # ------------------------------------------------------------------- summary
    summary = {
        "n_entities": len(entities),
        "n_forecast_rows": int(len(results)),
        "horizons": list(HORIZONS),
        "quantiles": list(QUANTILES),
        "scores": scores.to_dict("records"),
        "calibration": cal.to_dict("records"),
        "regime": regime.to_dict("records"),
        "quality_vs_skill": quality_link,
        "track_entity": track_entity,
        "best_model_by_horizon": {
            int(h): scores[scores["horizon"] == h].iloc[0]["model"] for h in HORIZONS
        },
    }
    (TABLES / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("stage 2 complete")


def _per_entity_skill(results: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    """Relative WIS of ``model`` against persistence, computed per country."""
    from pandemic.forecast.metrics import relative_skill, weighted_interval_score

    levels = np.asarray(QUANTILES, float)
    qcols = [f"q{q:g}" for q in levels]
    d = results[results["horizon"] == horizon].dropna(subset=qcols + ["actual"]).copy()
    d["wis"] = weighted_interval_score(d["actual"].to_numpy(), levels, d[qcols].to_numpy())

    key = ["origin", "entity"]
    base = d[d["model"] == "persistence"].set_index(key)["wis"]
    rows = []
    for entity, g in d[d["model"] == model].groupby("entity"):
        aligned = g.set_index(key)
        rows.append({
            "entity": entity,
            "rel_wis": relative_skill(aligned["wis"].to_numpy(),
                                      base.reindex(aligned.index).to_numpy()),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


def _pick_track_entity(results: pd.DataFrame, horizon: int) -> str:
    """A country with a long, complete record and real wave structure.

    Selection is on the dynamic range of the *observed* series, not on how far
    any model swung. Ranking by model swing picks whichever country a model blew
    up on, which makes for one dramatic panel and no information about the track
    record the figure is supposed to show.
    """
    d = results[results["horizon"] == horizon]
    stats_ = d.groupby("entity")["actual"].agg(
        n="count",
        lo=lambda s: float(np.nanpercentile(s, 5)),
        hi=lambda s: float(np.nanpercentile(s, 95)),
        median="median",
    )
    stats_ = stats_[(stats_["n"] >= 40) & (stats_["median"] > 500)]
    if stats_.empty:
        return d["entity"].iloc[0]
    stats_["log_range"] = np.log10(stats_["hi"].clip(lower=1) / stats_["lo"].clip(lower=1))
    return stats_.sort_values("log_range", ascending=False).index[0]


if __name__ == "__main__":
    main()
