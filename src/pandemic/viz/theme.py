"""Matplotlib theme.

Every figure renders twice, once on a light surface and once on a dark one, so
the README works under either GitHub theme via a ``<picture>`` element. Plot code
never names a colour. It takes a ``Palette`` and asks for a role (``series[0]``,
``critical``, ``grid``), which keeps the two modes in step.

On the palette: the categorical slot order is validated. Worst adjacent-pair
colour-vision-deficiency separation is dE 9.1 light / 8.4 dark in OKLab x100,
against a target of 8. Only the first three slots clear the stricter all-pairs
gate, so scatter plots never use more than three categorical colours. Past that
the code folds to "Other" or facets.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pandemic.config import FIGURES, get_logger

log = get_logger(__name__)

MODES = ("light", "dark")


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: Sequence[str]
    sequential: Sequence[str]
    diverging_low: str
    diverging_mid: str
    diverging_high: str
    status: dict[str, str] = field(default_factory=dict)

    @property
    def cmap_sequential(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(f"seq_{self.mode}", list(self.sequential))

    @property
    def cmap_diverging(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            f"div_{self.mode}", [self.diverging_low, self.diverging_mid, self.diverging_high]
        )


_STATUS = {  # fixed across modes, by design: status must never impersonate a series
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_SEQ_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

PALETTES: dict[str, Palette] = {
    "light": Palette(
        mode="light",
        surface="#fcfcfb", page="#f9f9f7",
        ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
        sequential=_SEQ_LIGHT,
        diverging_low="#2a78d6", diverging_mid="#f0efec", diverging_high="#d03b3b",
        status=_STATUS,
    ),
    "dark": Palette(
        mode="dark",
        surface="#1a1a19", page="#0d0d0d",
        ink="#ffffff", ink_secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=("#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767"),
        sequential=_SEQ_DARK,
        diverging_low="#3987e5", diverging_mid="#383835", diverging_high="#d03b3b",
        status=_STATUS,
    ),
}


@contextlib.contextmanager
def use_theme(mode: str):
    """Apply the theme's rcParams for the duration of the block."""
    p = PALETTES[mode]
    rc = {
        "figure.facecolor": p.surface,
        "figure.edgecolor": p.surface,
        "savefig.facecolor": p.surface,
        "savefig.edgecolor": p.surface,
        "axes.facecolor": p.surface,
        "axes.edgecolor": p.axis,
        "axes.labelcolor": p.ink_secondary,
        "axes.titlecolor": p.ink,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": p.grid,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "xtick.labelcolor": p.ink_secondary,
        "ytick.labelcolor": p.ink_secondary,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": p.ink,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": p.ink_secondary,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=list(p.series)),
    }
    with mpl.rc_context(rc):
        yield p


def render(name: str, draw: Callable[[plt.Figure, Palette], None], *,
           figsize: tuple[float, float] = (10.0, 5.5),
           modes: Sequence[str] = MODES) -> list[Path]:
    """Draw one figure in every theme mode and write it to ``reports/figures``.

    ``draw(fig, palette)`` builds the figure; it must take all colours from
    ``palette`` so both renders stay in step.
    """
    paths = []
    for mode in modes:
        with use_theme(mode) as p:
            fig = plt.figure(figsize=figsize)
            try:
                draw(fig, p)
                out = FIGURES / f"{name}.{mode}.png"
                fig.savefig(out, facecolor=p.surface)
                paths.append(out)
            finally:
                plt.close(fig)
    log.info("figure %-42s -> %s", name, " + ".join(m for m in modes))
    return paths


def title(ax, text: str, subtitle: str | None = None, *, palette: Palette | None = None) -> None:
    """Left-aligned title with an optional recessive subtitle beneath it.

    The subtitle is placed in *offset points* rather than axes fractions so the
    gap between the two lines does not shrink as the axes get taller, which is
    what makes them collide on multi-panel figures.
    """
    p = palette
    ax.set_title(text, loc="left", pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5,
                    color=p.ink_secondary if p else "#52514e")


def suptitle(fig, text: str, subtitle: str | None, palette: Palette,
             *, size: float = 13.5, sub_size: float = 9.5) -> None:
    """Figure-level title block, laid out top-down.

    The gap between the two lines is derived from the font size and the figure's
    actual height in inches rather than being a fixed figure fraction. A fixed
    fraction is the same distance in *relative* terms on every figure, which
    means it shrinks in points as the figure grows -- so a title block tuned on a
    5-inch figure collides on an 8-inch one.
    """
    fig.text(0.005, 1.0, text, ha="left", va="top", fontsize=size, color=palette.ink,
             weight="600")
    if subtitle:
        height_pts = fig.get_size_inches()[1] * 72.0
        gap = (size * 1.45) / height_pts
        fig.text(0.005, 1.0 - gap, subtitle, ha="left", va="top", fontsize=sub_size,
                 color=palette.ink_secondary)


def footnote(fig, text: str, palette: Palette) -> None:
    """Source / caveat line at the bottom-left of the figure."""
    fig.text(0.005, -0.015, text, ha="left", va="top", fontsize=8, color=palette.muted)


def thousands(ax, axis: str = "y") -> None:
    """Format large counts as 1.2k / 3.4M instead of scientific notation."""
    def fmt(v, _pos):
        av = abs(v)
        if av >= 1e9:
            return f"{v / 1e9:g}B"
        if av >= 1e6:
            return f"{v / 1e6:g}M"
        if av >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
