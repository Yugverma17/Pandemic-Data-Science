"""Figures for Pillar 3 (causal analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from pandemic.viz.theme import Palette, footnote, suptitle, title

# Hand-placed so the graph reads left-to-right as cause-to-effect; a spring
# layout is unreadable at this size and hides the structure that matters.
DAG_POSITIONS = {
    "median_age":        (0.06, 0.86),
    "wealth":            (0.28, 0.94),
    "density":           (0.50, 0.86),
    "seeding_delay":     (0.72, 0.94),
    "onset_growth":      (0.92, 0.86),
    "comorbidity":       (0.14, 0.58),
    "health_capacity":   (0.38, 0.58),
    "stringency":        (0.06, 0.22),
    "transmission":      (0.44, 0.22),
    "testing":           (0.34, 0.02),
    "observed_cases":    (0.62, 0.02),
    "vaccination_speed": (0.72, 0.40),
    "deaths":            (0.94, 0.22),
}

# Kept to two short lines each: anything longer overflows the node circle, and a
# label that spills past its marker is worse than an abbreviation.
DAG_LABELS = {
    "median_age": "Age\nstructure",
    "wealth": "Wealth",
    "density": "Pop.\ndensity",
    "seeding_delay": "Seeding\ndelay",
    "onset_growth": "Onset\ngrowth",
    "comorbidity": "Chronic\ndisease",
    "health_capacity": "Health\ncapacity",
    "stringency": "NPI\nstringency",
    "transmission": "Trans-\nmission",
    "testing": "Testing",
    "observed_cases": "Observed\ncases",
    "vaccination_speed": "Vaccine\nrollout",
    "deaths": "Deaths",
}


def causal_dag(graph, adjustment: set[str], mediators: set[str],
               treatment: str = "stringency", outcome: str = "deaths"):
    """Draw the graph, colour-coded by each node's role in identification."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.12, 1.08)
        ax.axis("off")
        ax.grid(visible=False)

        def role_colour(n: str) -> tuple[str, str]:
            if n == treatment:
                return p.series[0], "Treatment"
            if n == outcome:
                return p.status["critical"], "Outcome"
            if n in mediators:
                return p.muted, "Mediator (must NOT adjust)"
            if n in adjustment:
                return p.series[2], "Adjusted for (back-door)"
            return p.series[3], "Other"

        for u, v in graph.edges():
            if u not in DAG_POSITIONS or v not in DAG_POSITIONS:
                continue
            x1, y1 = DAG_POSITIONS[u]
            x2, y2 = DAG_POSITIONS[v]
            is_mediating = (u == treatment) or (v in mediators and u in mediators)
            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle="arc3,rad=0.08",
                arrowstyle="-|>", mutation_scale=11,
                linewidth=1.1,
                color=p.muted if is_mediating else p.axis,
                alpha=0.85, zorder=1,
                shrinkA=19, shrinkB=19,
            ))

        for n, (x, y) in DAG_POSITIONS.items():
            if n not in graph:
                continue
            colour, _ = role_colour(n)
            ax.scatter([x], [y], s=2000, color=colour, zorder=2,
                       edgecolor=p.surface, linewidth=2.0)
            ax.text(x, y, DAG_LABELS.get(n, n), ha="center", va="center",
                    fontsize=7.2, color=p.surface, weight="600", linespacing=1.1,
                    zorder=3)

        seen, handles = set(), []
        for n in DAG_POSITIONS:
            if n not in graph:
                continue
            colour, label = role_colour(n)
            if label not in seen:
                seen.add(label)
                handles.append(Line2D([0], [0], marker="o", color="none", markersize=9,
                                      markerfacecolor=colour, label=label))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncols=4, fontsize=8.6)

        suptitle(fig, "The causal graph, and what it forbids",
                 "Adjustment set chosen by Pearl's back-door criterion, verified in code. "
                 "Grey nodes are consequences of the policy: controlling for them would "
                 "remove part of the effect.", p)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        footnote(fig, "Every edge is a substantive claim, published so it can be disputed.", p)

    return draw


def estimate_ladder(rows: list[dict], placebo_row: dict | None = None):
    """Forest plot of every estimator, with the falsification test alongside."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        items = list(rows)
        if placebo_row:
            items = [*items, placebo_row]

        y = np.arange(len(items))[::-1]
        for yi, r in zip(y, items, strict=True):
            is_placebo = r.get("_placebo", False)
            colour = p.status["critical"] if is_placebo else p.series[0]
            ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=colour, lw=2.4,
                    solid_capstyle="round", zorder=2)
            ax.scatter([r["estimate"]], [yi], s=64, color=colour, zorder=3,
                       edgecolor=p.surface, linewidth=1.4)
            ax.text(r["ci_high"] + 0.0035, yi,
                    f"{r['estimate']:+.3f}  (p={r['p_value']:.3g})",
                    va="center", fontsize=8.4, color=p.ink_secondary)

        ax.axvline(0.0, color=p.axis, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(y, [r["method"] for r in items], fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=p.grid, lw=0.7)
        ax.set_xlabel("Effect on log deaths per million, per stringency point")

        title(ax, "Every estimator agrees -- and that is the problem",
              "The falsification test (red) uses an outcome the policy could not have "
              "affected, and reproduces the same association.", palette=p)
        fig.tight_layout()
        footnote(fig, "A positive coefficient means stricter response is associated with "
                      "MORE deaths, which is reverse causality, not effect.", p)

    return draw


def india_regional(agg: pd.DataFrame, highlight: tuple[str, ...] = ()):
    """Deaths per million by Indian state, with the problem statement's trio marked."""

    def draw(fig, p: Palette):
        ax = fig.subplots()
        d = agg.sort_values("deaths_per_million", ascending=True)
        y = np.arange(len(d))
        colors = [p.status["critical"] if e in highlight else p.series[0]
                  for e in d["entity"]]
        ax.barh(y, d["deaths_per_million"], color=colors, height=0.7, zorder=2)
        ax.set_yticks(y, d["entity"], fontsize=8.2)
        ax.grid(axis="x", color=p.grid, lw=0.7)
        ax.grid(axis="y", visible=False)

        for yi, (v, e) in enumerate(zip(d["deaths_per_million"], d["entity"], strict=True)):
            if e in highlight:
                ax.text(v + 18, yi, f"{v:,.0f}", va="center", fontsize=8.6,
                        color=p.ink, weight="600")

        ax.set_xlabel("Reported COVID-19 deaths per million, to Dec 2021")
        title(ax, "The comparison the brief asks for",
              "Karnataka (Bengaluru) recorded half the death rate of Delhi and "
              "Maharashtra (Mumbai).", palette=p)
        fig.tight_layout()
        footnote(fig, "Reported deaths only. Indian COVID mortality is substantially "
                      "under-recorded and the degree varies by state, so these gaps are "
                      "a lower bound on the real ones.", p)

    return draw


def synthetic_control_plot(result, placebo: dict, unit_label: str):
    """Observed versus synthetic, the gap, and the placebo distribution."""

    def draw(fig, p: Palette):
        ax1, ax2 = fig.subplots(2, 1, height_ratios=[2.0, 1.2], sharex=True)

        ax1.plot(result.observed.index, result.observed.to_numpy(), color=p.ink,
                 lw=2.2, zorder=4, label=f"{unit_label} (observed)")
        ax1.plot(result.synthetic.index, result.synthetic.to_numpy(),
                 color=p.series[0], lw=2.0, ls=(0, (5, 3)), zorder=3,
                 label=f"Synthetic {unit_label}")
        ax1.axvline(result.intervention, color=p.status["critical"], lw=1.6, zorder=2)
        ax1.annotate("intervention", xy=(result.intervention, 0.96),
                     xycoords=("data", "axes fraction"), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.4,
                     color=p.status["critical"], va="top")
        ax1.legend(fontsize=8.8, loc="upper left")
        ax1.set_ylabel("Daily cases per million")
        title(ax1, f"Synthetic control: {unit_label}",
              f"Donor weights fitted on the pre-period only. "
              f"Pre-period RMSPE {result.pre_rmspe:.2f}, post/pre ratio "
              f"{result.rmspe_ratio:.2f}.", palette=p)

        ax2.axhline(0, color=p.axis, lw=1.0, zorder=1)
        ax2.plot(result.gap.index, result.gap.to_numpy(), color=p.series[1], lw=2.0,
                 zorder=3)
        ax2.fill_between(result.gap.index, 0, result.gap.to_numpy(),
                         color=p.series[1], alpha=0.22, lw=0, zorder=2)
        ax2.axvline(result.intervention, color=p.status["critical"], lw=1.6, zorder=2)
        ax2.set_ylabel("Gap")
        ax2.set_xlabel(
            f"Placebo permutation p = {placebo.get('p_value', float('nan')):.3f} "
            f"({placebo.get('n_placebos', 0)} donor placebos)")

        fig.tight_layout()
        footnote(fig, "Weights are non-negative and sum to one, so the counterfactual "
                      "never extrapolates beyond the donor pool.", p)

    return draw
