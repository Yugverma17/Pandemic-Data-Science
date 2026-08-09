"""Figures for Pillar 4 (hospital capacity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandemic.viz.theme import Palette, footnote, suptitle, thousands, title


def kernels(lag_kernel: np.ndarray, los_curve: np.ndarray):
    """The two delay distributions that make census lag cases."""

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(1, 2)

        d1 = np.arange(lag_kernel.size)
        ax1.bar(d1, lag_kernel, color=p.series[0], width=0.8, zorder=2)
        ax1.set_xlim(-0.5, 30)
        ax1.set_xlabel("Days from case report to admission")
        ax1.set_ylabel("Probability")
        mean_lag = float(np.sum(d1 * lag_kernel))
        title(ax1, "Admission delay", f"Gamma, mean {mean_lag:.1f} days", palette=p)

        d2 = np.arange(los_curve.size)
        ax2.fill_between(d2, 0, los_curve, color=p.series[2], alpha=0.35, lw=0, zorder=2)
        ax2.plot(d2, los_curve, color=p.series[2], lw=2.2, zorder=3)
        ax2.set_xlim(0, 45)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("Days since ICU admission")
        ax2.set_ylabel("Still in ICU")
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        title(ax2, "Length-of-stay survival",
              f"Area = mean stay of {los_curve.sum():.1f} days", palette=p)

        suptitle(fig, "Why ICU occupancy peaks two weeks after cases do",
                 "Census is admissions convolved with the probability a patient is "
                 "still present. Long right-skewed stays keep occupancy climbing "
                 "after admissions turn.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        footnote(fig, "Docherty et al. (2020) BMJ 369:m1985; Rees et al. (2020) "
                      "BMC Medicine 18:270.", p)

    return draw


def validation(pre: pd.DataFrame, post: pd.DataFrame):
    """Model accuracy before and after vaccination, plus the level multipliers.

    The two panels together are the finding: the mechanistic chain is accurate
    exactly where its fixed-hospitalisation-rate assumption holds, and degrades
    where vaccination breaks it.
    """

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.2, 1.0])

        d = pre.sort_values("correlation", ascending=True)
        y = np.arange(len(d))
        post_map = post.set_index("entity")["correlation"] if not post.empty else {}

        ax1.barh(y, d["correlation"], color=p.series[0], height=0.66, zorder=2,
                 label="Pre-vaccination")
        matched = [post_map.get(e, np.nan) if len(post_map) else np.nan
                   for e in d["entity"]]
        ax1.scatter(matched, y, s=26, color=p.status["critical"], zorder=4,
                    edgecolor=p.surface, linewidth=0.9, label="Post-vaccination")

        ax1.set_yticks(y, d["entity"], fontsize=7.4)
        ax1.set_xlim(0, 1.0)
        ax1.grid(axis="x", color=p.grid, lw=0.7)
        ax1.grid(axis="y", visible=False)
        ax1.set_xlabel("Correlation, predicted vs observed ICU census")
        ax1.legend(fontsize=8.4, loc="lower right")
        title(ax1, "Accurate until vaccination changes the ratio",
              f"Median r = {pre['correlation'].median():.2f} pre-vaccination, "
              f"{post['correlation'].median():.2f} after."
              if not post.empty else
              f"Median r = {pre['correlation'].median():.2f}", palette=p)

        k = pre["level_multiplier"]
        ax2.hist(k, bins=14, color=p.series[0], zorder=2)
        ax2.axvline(float(k.median()), color=p.status["critical"], lw=2.0, zorder=3)
        ax2.set_xlabel("Fitted level multiplier")
        ax2.set_ylabel("Countries")
        title(ax2, "The level needs a country-specific scalar",
              f"{float(k.quantile(0.9) / max(k.quantile(0.1), 1e-9)):.0f}x spread "
              "between the 10th and 90th percentile", palette=p)

        suptitle(fig, "Validating the cases-to-ICU model against observed occupancy",
                 "One fixed literature parameter set; only a single scalar is fitted "
                 "per country, and never the shape.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        footnote(fig, "Countries reporting daily ICU occupancy for at least 120 days "
                      "(Our World in Data). Pre-vaccination window ends 2021-02-01.", p)

    return draw


def validation_example(dates, observed, predicted, entity: str, r: float, k: float):
    """One country's predicted versus observed ICU curve."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        ax.plot(dates, observed, color=p.ink, lw=2.2, zorder=3, label="Observed ICU patients")
        ax.plot(dates, predicted, color=p.series[0], lw=2.0, ls=(0, (5, 3)), zorder=2,
                label="Predicted from cases (level-calibrated)")
        thousands(ax)
        ax.legend(fontsize=9)
        ax.set_ylabel("ICU patients")
        title(ax, f"Cases predict ICU occupancy: {entity}",
              f"r = {r:.3f}. Only a single scalar was fitted; the shape is "
              "entirely mechanistic.", palette=p)
        fig.tight_layout()
        footnote(fig, "Parameters from the literature, not fitted to this series.", p)

    return draw


def triage(table: pd.DataFrame, as_of: str, n: int = 15):
    """Which regions face ICU pressure, and how much of it is already unprecedented."""

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(1, 2, sharey=True, width_ratios=[1.4, 1.0])
        d = table.head(n).iloc[::-1]
        y = np.arange(len(d))

        peak = d["projected_peak_per_100k"].to_numpy(float)
        current = (d["current_census_median"].to_numpy(float)
                   * 1e5 / d["population"].to_numpy(float))
        ax1.barh(y, peak, color=p.series[0], height=0.68, zorder=2,
                 label="Projected peak, next 21 days")
        ax1.barh(y, current, color=p.series[2], height=0.40, zorder=3,
                 label="Today")
        ax1.set_yticks(y, d["entity"], fontsize=8.6)
        ax1.grid(axis="x", color=p.grid, lw=0.7)
        ax1.grid(axis="y", visible=False)
        ax1.set_xlabel("Modelled ICU patients per 100,000")
        ax1.legend(fontsize=8.4, loc="lower right")
        # Subtitles stay short: on a two-panel figure a long one runs under its
        # neighbour's, and the reasoning belongs in the footnote anyway.
        title(ax1, "Projected ICU pressure", "Ranked by per-capita occupancy",
              palette=p)

        util = d["utilisation_median"].to_numpy(float)
        finite = np.isfinite(util)
        colors = [p.status["critical"] if u >= 1.0 else
                  (p.status["serious"] if u >= 0.7 else p.series[3])
                  for u in util[finite]]
        ax2.barh(y[finite], util[finite], color=colors, height=0.68, zorder=2)
        ax2.axvline(1.0, color=p.status["critical"], lw=1.6, zorder=3)
        for yi in y[~finite]:
            ax2.text(0.05, yi, "no prior peak to compare", va="center", fontsize=7.6,
                     color=p.muted, style="italic")
        ax2.grid(axis="x", color=p.grid, lw=0.7)
        ax2.grid(axis="y", visible=False)
        ax2.set_xscale("symlog", linthresh=1.0)
        ax2.xaxis.set_major_formatter(lambda v, _: f"{v:g}x")
        ax2.set_xlabel("Projected peak, as a multiple of the prior peak")
        title(ax2, "Is this unprecedented?", "Log scale; 1x = prior peak", palette=p)

        suptitle(fig, f"Who should be adding ICU capacity, as of {as_of}",
                 "Nothing after the as-of date enters the calculation, so this is "
                 "exactly the table that could have been produced on the day.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        footnote(fig, "Ranked by absolute per-capita occupancy, which is what determines "
                      "whether a system copes; the ratio to a region's own history "
                      "favours places that never had a wave. Monte Carlo over published "
                      "parameter ranges and a distribution over R_t; 21-day horizon.", p)

    return draw


def fan(paths: np.ndarray, dates, split_index: int, entity: str, benchmark: float,
        history_days: int = 90):
    """Simulated ICU census with uncertainty, history and projection.

    Only the recent history is drawn. Plotting the full series squeezes the
    projection -- the part the figure exists for -- into the last inch of the
    axes, behind ten months of flat line.
    """
    start = max(0, split_index - history_days)
    paths = paths[:, start:]
    dates = list(dates)[start:]
    split_index -= start

    def draw(fig, p: Palette):
        ax = fig.subplots()
        q = np.percentile(paths, [2.5, 25, 50, 75, 97.5], axis=0)

        ax.fill_between(dates, q[0], q[4], color=p.series[0], alpha=0.16, lw=0, zorder=2,
                        label="95% interval")
        ax.fill_between(dates, q[1], q[3], color=p.series[0], alpha=0.30, lw=0, zorder=3,
                        label="50% interval")
        ax.plot(dates, q[2], color=p.series[0], lw=2.0, zorder=4, label="Median")

        if np.isfinite(benchmark) and benchmark > 0:
            ax.axhline(benchmark, color=p.status["critical"], lw=1.8, ls=(0, (5, 3)),
                       zorder=5, label="Prior peak occupancy")

        split_date = dates[split_index]
        ax.axvline(split_date, color=p.axis, lw=1.2, zorder=1)
        ax.annotate("forecast starts", xy=(split_date, 0.97),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.4, color=p.ink_secondary,
                    va="top")

        thousands(ax)
        ax.legend(fontsize=8.6, ncols=2, loc="upper left")
        ax.set_ylabel("ICU patients (modelled)")
        title(ax, f"Projected ICU census: {entity}",
              "Uncertainty combines epidemiological parameters with the "
              "transmission outlook.", palette=p)
        fig.tight_layout()
        footnote(fig, "Parameter ranges from published sources; see config.EpiParams.", p)

    return draw
