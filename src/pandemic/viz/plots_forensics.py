"""Figures for Pillar 1 (reporting forensics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pandemic.forensics.digits import BENFORD_P
from pandemic.viz.theme import Palette, footnote, suptitle, thousands, title

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekday_fingerprints(profiles: dict[str, dict]):
    """Small multiples of day-of-week reporting multipliers.

    A continuously-reporting system sits flat at 1.0. Everything else is the
    shape of the bureaucracy behind the number.
    """

    def draw(fig, p: Palette):
        n = len(profiles)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        axes = fig.subplots(nrows, ncols, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, (name, prof) in zip(axes, profiles.items(), strict=False):
            mult = np.asarray(prof["multipliers"], dtype=float)
            ax.axhline(1.0, color=p.axis, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.bar(DAYS, mult, color=p.series[0], width=0.68, zorder=2)
            ax.set_title(name, loc="left", fontsize=10.5)
            amp, pv = prof["amplitude"], prof["p_value"]
            ax.text(0.98, 0.94, f"amplitude {amp:.2f}\np {pv:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=p.ink_secondary,
                    linespacing=1.4)
            ax.tick_params(axis="x", labelsize=8)
            ax.margins(x=0.03)

        for ax in axes[n:]:
            ax.set_visible(False)
        for ax in axes[:n]:
            ax.set_ylim(0, max(2.2, min(6.5, max(
                float(np.nanmax(pr["multipliers"])) for pr in profiles.values()) * 1.12)))

        suptitle(fig, "Reported cases by day of week, relative to the 7-day trend",
                 "1.0 = the day carries its fair share. Deviation is the reporting system, not the virus.",
                 p)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        footnote(fig, "Source: JHU CSSE. p-value from a 500-draw permutation test on day-of-week labels.", p)

    return draw


def reliability_ranking(scorecard: pd.DataFrame, n: int = 20):
    """The ``n`` least reliable series, annotated with the dominant failure mode."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        d = scorecard.nsmallest(n, "reliability_index").iloc[::-1]
        y = np.arange(len(d))

        ax.barh(y, d["reliability_index"], color=p.series[0], height=0.68, zorder=2)
        ax.set_yticks(y, d["entity"], fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color=p.grid, lw=0.7)
        ax.grid(axis="y", visible=False)

        for yi, (score, desc) in enumerate(zip(d["reliability_index"],
                                               d["worst_component_desc"], strict=False)):
            ax.text(score + 1.4, yi, f"{score:.0f}  ·  {desc}", va="center",
                    fontsize=8.5, color=p.ink_secondary)

        title(ax, "Least reliable reporting series",
              f"Data Reliability Index, 0-100. Bottom {n} of {len(scorecard)} countries scored.",
              palette=p)
        ax.set_xlabel("Data Reliability Index")
        fig.tight_layout()
        footnote(fig, "Seven detectors, severity-capped and weighted; see README for weights and sensitivity check.", p)

    return draw


MECHANISM_LABELS = {
    "trend": "epidemic was already high",
    "weekday": "normal weekday rhythm",
    "dump": "batch release (real anomaly)",
    "residual": "unexplained jump (real anomaly)",
}


def naive_decomposition(series: pd.Series, decomposed: pd.DataFrame, country: str,
                        panel: dict[str, pd.DataFrame] | None = None):
    """What the textbook z-score detector actually flags.

    Top panel: one country's series, with flagged days split into explained vs
    genuinely anomalous. Bottom panel: the mechanism breakdown across every
    benchmarked country, which is where the pattern becomes undeniable.
    """
    panel = panel or {country: decomposed}

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(2, 1, height_ratios=[3.4, 1.0])

        ax1.plot(series.index, series.to_numpy(), color=p.muted, lw=1.1, zorder=1,
                 label="Daily reported cases")
        smooth = series.rolling(7, center=True, min_periods=4).mean()
        ax1.plot(smooth.index, smooth.to_numpy(), color=p.ink_secondary, lw=1.8, zorder=2,
                 label="7-day average")

        explained = decomposed[~decomposed["is_true_anomaly"]]
        genuine = decomposed[decomposed["is_true_anomaly"]]

        ax1.scatter(explained["date"], explained["value"], s=26, zorder=3,
                    facecolor="none", edgecolor=p.series[0], linewidth=1.4,
                    label=f"Flagged, explained by trend or weekday  (n={len(explained)})")
        ax1.scatter(genuine["date"], genuine["value"], s=52, zorder=4,
                    color=p.status["critical"], edgecolor=p.surface, linewidth=1.2,
                    label=f"Flagged, genuinely anomalous  (n={len(genuine)})")

        thousands(ax1)
        ax1.legend(loc="upper left", fontsize=8.8)
        title(ax1, f"The textbook z-score detector on {country}",
              f"{len(decomposed)} days exceed 3 SD above the mean. "
              f"{len(genuine)} survive decomposition.", palette=p)

        order = ("trend", "weekday", "dump", "residual")
        colors = {"trend": p.muted, "weekday": p.series[1],
                  "dump": p.status["critical"], "residual": p.series[6]}

        names = list(panel.keys())
        for row, name in enumerate(names):
            dec = panel[name]
            counts = dec["mechanism"].value_counts() if not dec.empty else pd.Series(dtype=int)
            total = max(int(counts.sum()), 1)
            left = 0.0
            for m in order:
                w = int(counts.get(m, 0))
                if w == 0:
                    continue
                ax2.barh([row], [100 * w / total], left=left, color=colors[m],
                         height=0.62, zorder=2)
                if w / total > 0.10:
                    ax2.text(left + 50 * w / total, row, str(w), ha="center", va="center",
                             fontsize=8.5, color=p.surface, weight="600")
                left += 100 * w / total

        ax2.set_yticks(range(len(names)), names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)
        ax2.grid(visible=False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.tick_params(axis="x", labelsize=8.5)
        ax2.xaxis.set_major_formatter(lambda v, _: f"{v:g}%")
        ax2.set_xlabel("Share of z-score flags, attributed to mechanism")
        ax2.legend(handles=[Line2D([0], [0], marker="s", color="none", markersize=8,
                                   markerfacecolor=colors[m], label=MECHANISM_LABELS[m])
                            for m in order],
                   loc="upper center", bbox_to_anchor=(0.5, -0.42), ncols=2, fontsize=8.8)

        fig.tight_layout()
        footnote(fig, "Decomposition: log1p(cases) = centred 7-day trend + day-of-week effect + residual.", p)

    return draw


def digit_tests(first, terminal, country: str):
    """Observed leading- and final-digit distributions against their nulls."""

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(1, 2)

        obs1 = first.observed / first.observed.sum()
        x1 = np.arange(1, 10)
        ax1.bar(x1, obs1, color=p.series[0], width=0.66, zorder=2, label="Observed")
        ax1.plot(x1, BENFORD_P, ls="none", marker="_", markersize=17,
                 markeredgewidth=2.4, color=p.status["critical"], zorder=3,
                 label="Benford expectation")
        ax1.set_xticks(x1)
        ax1.set_xlabel("Leading digit")
        ax1.set_ylabel("Share of days")
        ax1.legend(fontsize=8.8)
        title(ax1, "Leading digit",
              f"MAD {first.mad:.4f} · {first.verdict} · n={first.n}", palette=p)

        obs2 = terminal.observed / terminal.observed.sum()
        x2 = np.arange(10)
        ax2.bar(x2, obs2, color=p.series[0], width=0.66, zorder=2, label="Observed")
        ax2.axhline(0.1, color=p.status["critical"], lw=2.0, zorder=3,
                    label="Uniform expectation")
        ax2.set_xticks(x2)
        ax2.set_xlabel("Final digit")
        ax2.legend(fontsize=8.8)
        pv = terminal.p_value
        title(ax2, "Final digit",
              f"chi-square p {pv:.1e} · {terminal.verdict} · n={terminal.n}", palette=p)

        suptitle(fig, f"Digit-distribution forensics: {country}", None, p)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        footnote(fig, "Non-conformity flags a series for investigation; it is not evidence of fabrication.", p)

    return draw
