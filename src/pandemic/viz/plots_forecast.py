"""Figures for Pillar 2 (forecasting)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.viz.theme import Palette, footnote, suptitle, title

PRETTY = {
    "persistence": "Persistence (baseline)",
    "drift14": "Log-linear drift",
    "drift14_damped": "Log-linear drift, damped",
    "renewal_rt_undamped": "Renewal / $R_t$, undamped",
    "renewal_rt_damp0.95": "Renewal / $R_t$, damped",
    "panel_gbm": "Gradient boosting (global panel)",
}


def label_of(model: str) -> str:
    return PRETTY.get(model, model)


def skill_ranking(scores: pd.DataFrame):
    """Relative WIS by model and horizon. Below 1.0 beats the baseline."""

    def draw(fig, p: Palette):
        horizons = sorted(scores["horizon"].unique())
        axes = np.atleast_1d(fig.subplots(1, len(horizons), sharey=True))

        order = (scores[scores["horizon"] == horizons[0]]
                 .sort_values("rel_wis")["model"].tolist())

        for ax, h in zip(axes, horizons, strict=False):
            d = scores[scores["horizon"] == h].set_index("model").reindex(order).iloc[::-1]
            y = np.arange(len(d))
            colors = [p.status["good"] if v < 1.0 else p.muted for v in d["rel_wis"]]
            ax.barh(y, d["rel_wis"], color=colors, height=0.66, zorder=2)
            ax.axvline(1.0, color=p.status["critical"], lw=1.6, zorder=3)
            ax.set_yticks(y, [label_of(m) for m in d.index], fontsize=9)
            ax.grid(axis="x", color=p.grid, lw=0.7)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(1.15, float(scores["rel_wis"].max()) * 1.08))
            for yi, v in enumerate(d["rel_wis"]):
                ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=8.5,
                        color=p.ink_secondary)
            ax.set_title(f"{h}-day horizon", loc="left", fontsize=10.5)
            ax.set_xlabel("Relative WIS vs persistence")

        suptitle(fig, "The mechanistic model wins at both horizons",
                 "Weighted Interval Score relative to persistence, geometric mean over "
                 "matched forecast tasks. Lower is better; 1.0 is the baseline.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        footnote(fig, "Rolling-origin backtest, leak-free conformal intervals. "
                      "Green = beats the baseline.", p)

    return draw


def calibration(cal: pd.DataFrame):
    """Nominal versus empirical coverage -- the honesty check on the intervals."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        models = sorted(cal["model"].unique())
        nominals = sorted(cal["nominal"].unique())

        width = 0.8 / max(len(models), 1)
        x = np.arange(len(nominals))
        for k, m in enumerate(models):
            d = cal[cal["model"] == m].set_index("nominal").reindex(nominals)
            ax.bar(x + k * width - 0.4 + width / 2, d["empirical"], width=width * 0.9,
                   color=p.series[k % len(p.series)], label=label_of(m), zorder=2)

        for xi, nom in zip(x, nominals, strict=True):
            ax.plot([xi - 0.45, xi + 0.45], [nom, nom], color=p.status["critical"],
                    lw=2.0, zorder=4)

        ax.set_xticks(x, [f"{n:.0%}" for n in nominals])
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("Nominal interval")
        ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=8.5, ncols=2, loc="upper left")
        title(ax, "Every model's prediction intervals are too narrow",
              "Red bar = nominal level. Bars below it mean the stated confidence is overstated.",
              palette=p)
        fig.tight_layout()
        footnote(fig, "Conformal intervals calibrated only on residuals observable at the "
                      "forecast origin. Under-coverage is the cost of regime change.", p)

    return draw


def regime_skill(regime: pd.DataFrame, horizon: int):
    """Where the skill actually comes from: growing, flat, or receding epidemics."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        d = regime[regime["horizon"] == horizon]
        regimes = ["receding", "flat", "growing"]
        models = [m for m in d["model"].unique() if m != "persistence"]
        models = sorted(models, key=lambda m: d[d["model"] == m]["rel_wis"].mean())

        width = 0.8 / max(len(models), 1)
        x = np.arange(len(regimes))
        for k, m in enumerate(models):
            sub = d[d["model"] == m].set_index("regime").reindex(regimes)
            ax.bar(x + k * width - 0.4 + width / 2, sub["rel_wis"], width=width * 0.9,
                   color=p.series[k % len(p.series)], label=label_of(m), zorder=2)

        ax.axhline(1.0, color=p.status["critical"], lw=1.6, zorder=3)
        ax.set_xticks(x, [r.capitalize() for r in regimes])
        ax.set_ylabel("Relative WIS vs persistence")
        ax.set_xlabel("Epidemic regime at the target date")
        ax.legend(fontsize=8.5, ncols=2)
        title(ax, f"Skill concentrates where it matters ({horizon}-day horizon)",
              "Models are near-indistinguishable in flat periods and separate at turning points.",
              palette=p)
        fig.tight_layout()
        footnote(fig, "Regime defined by actual / last-observed level: <0.8 receding, "
                      "0.8-1.25 flat, >1.25 growing.", p)

    return draw


def forecast_track(results: pd.DataFrame, entity: str, horizon: int,
                   models: tuple[str, str]):
    """Every forecast a model made, plotted at the date it was forecasting.

    Honest in a way a single hand-picked fan chart is not: it shows the whole
    track record, misses included.
    """

    def draw(fig, p: Palette):
        axes = fig.subplots(len(models), 1, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        # Log scale, because case counts are multiplicative and a linear axis
        # lets a single blown-up forecast flatten everything else to the floor.
        floor = 1.0
        for ax, model in zip(axes, models, strict=True):
            d = (results[(results["entity"] == entity)
                         & (results["horizon"] == horizon)
                         & (results["model"] == model)]
                 .sort_values("target_date"))
            if d.empty:
                continue

            lo95 = d["q0.025"].clip(lower=floor)
            hi95 = d["q0.975"].clip(lower=floor)
            lo50 = d["q0.25"].clip(lower=floor)
            hi50 = d["q0.75"].clip(lower=floor)

            ax.fill_between(d["target_date"], lo95, hi95,
                            color=p.series[0], alpha=0.18, lw=0, zorder=2,
                            label="95% prediction interval")
            ax.fill_between(d["target_date"], lo50, hi50,
                            color=p.series[0], alpha=0.32, lw=0, zorder=3,
                            label="50% prediction interval")
            ax.plot(d["target_date"], d["q0.5"].clip(lower=floor), color=p.series[0],
                    lw=1.9, zorder=4, label="Forecast median")
            ax.plot(d["target_date"], d["actual"].clip(lower=floor), color=p.ink,
                    lw=2.0, zorder=5, label="Actual (7-day average)")

            ax.set_yscale("log")
            ax.set_title(label_of(model), loc="left", fontsize=10.5)
            ax.margins(x=0.01)

        axes[0].legend(fontsize=8.5, ncols=2, loc="upper left")
        suptitle(fig, f"{horizon}-day-ahead track record: {entity}",
                 "Each point was forecast a fortnight earlier, using only data "
                 "available at that time. Log scale.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        footnote(fig, "Intervals are conformal, calibrated on residuals observable at "
                      "each forecast origin.", p)

    return draw


def quality_vs_skill(merged: pd.DataFrame, rho: float, pval: float, horizon: int):
    """Does poor reporting make a country harder to forecast? Links Pillars 1 and 2."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        ax.scatter(merged["reliability_index"], merged["rel_wis"], s=44,
                   color=p.series[0], edgecolor=p.surface, linewidth=1.0, zorder=3)

        if len(merged) >= 3:
            b, a = np.polyfit(merged["reliability_index"], merged["rel_wis"], 1)
            xs = np.linspace(merged["reliability_index"].min(),
                             merged["reliability_index"].max(), 50)
            ax.plot(xs, a + b * xs, color=p.status["critical"], lw=1.8,
                    ls=(0, (5, 3)), zorder=4)

        # Label only the extremes; a name on every point is unreadable.
        extremes = pd.concat([merged.nsmallest(3, "reliability_index"),
                              merged.nlargest(2, "rel_wis"),
                              merged.nsmallest(2, "rel_wis")]).drop_duplicates("entity")
        for _, r in extremes.iterrows():
            ax.annotate(r["entity"], (r["reliability_index"], r["rel_wis"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=8.3,
                        color=p.ink_secondary)

        ax.axhline(1.0, color=p.axis, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel("Data Reliability Index (Pillar 1)")
        ax.set_ylabel(f"Relative WIS at {horizon} days (Pillar 2)")
        title(ax, "Countries with worse reporting are harder to forecast",
              f"Spearman rho = {rho:.3f}, p = {pval:.4f}, n = {len(merged)} countries.",
              palette=p)
        fig.tight_layout()
        footnote(fig, "Relative WIS of the best mechanistic model per country. "
                      "Higher means the model helps less there.", p)

    return draw
