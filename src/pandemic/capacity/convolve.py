"""From reported cases to ICU census, by convolution.

The chain is deliberately mechanistic rather than fitted:

    cases_t  --(IHR, admission lag)-->  admissions_t
    admissions_t  --(critical-care share)-->  ICU admissions_t
    ICU admissions  --(length-of-stay survival)-->  ICU census_t

The last step is the one a regression on cases cannot reproduce. Census is not
proportional to admissions. It is admissions convolved with the probability that
a patient admitted ``s`` days ago is still there. COVID ICU stays are long and
right-skewed (mean ~12 days, SD ~8), so census keeps climbing for one to two
weeks after admissions peak. Treating occupancy as a scaled copy of cases calls
the turning point about a fortnight early, which is exactly when the decision to
open surge capacity gets made.

Every parameter has a published source and an uncertainty range in
:class:`pandemic.config.EpiParams`. :mod:`pandemic.capacity.risk` propagates those
ranges instead of reporting one deterministic path.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from pandemic.config import EPI


def gamma_kernel(mean: float, sd: float, max_days: int = 45) -> np.ndarray:
    """Discrete PMF over delays 0..max_days for a Gamma(mean, sd) distribution.

    Bins are centred on integers (mass over ``[s-0.5, s+0.5)``) so the discrete
    mean matches the continuous one instead of drifting half a day later -- the
    same bias corrected in the serial-interval discretisation.
    """
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    dist = stats.gamma(a=shape, scale=scale)

    days = np.arange(max_days + 1)
    upper = dist.cdf(days + 0.5)
    lower = dist.cdf(np.clip(days - 0.5, 0, None))
    w = np.clip(upper - lower, 0, None)
    total = w.sum()
    return w / total if total > 0 else w


def los_survival(mean: float, sd: float, max_days: int = 60) -> np.ndarray:
    """``S[s] = P(length of stay > s)`` -- the share of admissions still present.

    This is the occupancy kernel. Summing it gives the mean length of stay, which
    is the factor converting a steady admission rate into a steady census.
    """
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    dist = stats.gamma(a=shape, scale=scale)
    days = np.arange(max_days + 1)
    return np.clip(dist.sf(days), 0, 1)


def _convolve_causal(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution: ``y[t] = sum_s x[t-s] * kernel[s]``, same length as x."""
    x = np.nan_to_num(np.clip(np.asarray(x, float), 0, None))
    full = np.convolve(x, np.asarray(kernel, float), mode="full")
    return full[: x.size]


def cases_to_admissions(cases: np.ndarray, *, ihr: float = EPI.ihr_mean,
                        lag_mean: float = EPI.onset_to_admission_mean,
                        lag_sd: float = EPI.onset_to_admission_sd) -> np.ndarray:
    """Hospital admissions implied by a reported-case series."""
    return ihr * _convolve_causal(cases, gamma_kernel(lag_mean, lag_sd))


def admissions_to_icu_census(admissions: np.ndarray, *,
                             icu_share: float = EPI.icu_share_mean,
                             los_mean: float = EPI.icu_los_mean,
                             los_sd: float = EPI.icu_los_sd) -> np.ndarray:
    """ICU census from hospital admissions, via the length-of-stay survival curve."""
    icu_admissions = icu_share * np.asarray(admissions, float)
    return _convolve_causal(icu_admissions, los_survival(los_mean, los_sd))


def cases_to_icu_census(cases: np.ndarray, *, ihr: float = EPI.ihr_mean,
                        icu_share: float = EPI.icu_share_mean,
                        lag_mean: float = EPI.onset_to_admission_mean,
                        lag_sd: float = EPI.onset_to_admission_sd,
                        los_mean: float = EPI.icu_los_mean,
                        los_sd: float = EPI.icu_los_sd) -> np.ndarray:
    """The full chain, cases to ICU occupancy."""
    adm = cases_to_admissions(cases, ihr=ihr, lag_mean=lag_mean, lag_sd=lag_sd)
    return admissions_to_icu_census(adm, icu_share=icu_share,
                                    los_mean=los_mean, los_sd=los_sd)


def peak_lag_days(cases: np.ndarray, census: np.ndarray) -> int:
    """Days between the case peak and the resulting ICU-census peak.

    The quantity that makes the model operationally useful: it is the warning
    time a hospital gets from watching case counts.
    """
    c = np.nan_to_num(np.asarray(cases, float))
    v = np.nan_to_num(np.asarray(census, float))
    if c.size == 0 or v.size == 0 or c.max() <= 0 or v.max() <= 0:
        return 0
    return int(np.argmax(v) - np.argmax(c))
