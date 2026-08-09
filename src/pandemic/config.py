"""Project-wide paths, constants, and reproducibility settings.

Every module imports paths from here rather than hard-coding them, so the whole
pipeline can be relocated or pointed at a scratch directory by changing one file.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

for _p in (RAW, INTERIM, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- reproducibility

SEED = 20200311  # WHO pandemic declaration date, as good a seed as any


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rng(seed: int = SEED) -> np.random.Generator:
    """Preferred RNG: an explicit Generator beats global state."""
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ epi constants
#
# Literature-derived parameters used by the renewal-equation and hospital-capacity
# models. Each carries its source so a reviewer can check the number rather than
# trust it. Uncertainty ranges drive the Monte Carlo in `pandemic.capacity`.


@dataclass(frozen=True)
class EpiParams:
    """Epidemiological parameters with published uncertainty ranges."""

    # Serial interval, gamma-distributed. Mean 4.7d (95% CrI 3.7-6.0), SD 2.9d.
    # Nishiura, Linton & Akhmetzhanov (2020), Int J Infect Dis 93:284-286.
    serial_interval_mean: float = 4.7
    serial_interval_sd: float = 2.9

    # Case -> hospital admission lag, gamma. Mean ~7d.
    # Docherty et al. (2020), BMJ 369:m1985 (ISARIC-4C cohort).
    onset_to_admission_mean: float = 7.0
    onset_to_admission_sd: float = 4.0

    # Infection-hospitalisation ratio, population-averaged, pre-vaccination.
    # Range brackets age-structure differences across countries.
    ihr_mean: float = 0.030
    ihr_low: float = 0.015
    ihr_high: float = 0.060

    # Share of hospitalised patients requiring critical care.
    icu_share_mean: float = 0.17
    icu_share_low: float = 0.10
    icu_share_high: float = 0.26

    # ICU length of stay, gamma. Mean ~12d, heavily right-skewed.
    # Rees et al. (2020), BMC Medicine 18:270 (systematic review).
    icu_los_mean: float = 12.0
    icu_los_sd: float = 8.0


EPI = EpiParams()


# ------------------------------------------------------------------ analysis knobs

# A country enters the panel only once it has a real epidemic; below this the
# series is mostly reporting noise and every model looks identical.
MIN_CUMULATIVE_CASES = 5_000
MIN_POPULATION = 1_000_000

# Forecast horizons in days (targets are 7-day trailing averages at that offset).
HORIZONS = (7, 14)

# Quantile levels for probabilistic forecasts. This is the COVID-19 Forecast Hub
# set, which makes the WIS numbers here directly comparable to published work.
QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


# ----------------------------------------------------------------------- logging


def get_logger(name: str) -> logging.Logger:
    """Console logger with a consistent format; safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
