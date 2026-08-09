"""Time-varying reproduction number and forward projection.

Cori, Ferguson, Fraser & Cauchemez (2013), Am J Epidemiol 178(9):1505-1512.
Same method as EpiEstim.

The renewal equation:

    I_t  ~  Poisson( R_t * Lambda_t ),      Lambda_t = sum_s I_{t-s} * w_s

``w`` is the discretised serial interval, i.e. the chance a secondary case shows
up ``s`` days after its infector. A Gamma(a, b) prior on R makes the posterior
over a trailing window conjugate, so R_t has a closed form and needs no sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from pandemic.config import EPI


def discretise_serial_interval(mean: float = EPI.serial_interval_mean,
                               sd: float = EPI.serial_interval_sd,
                               max_days: int = 30) -> np.ndarray:
    """Discrete serial-interval PMF ``w[1..max_days]``.

    Uses the discretisation from Cori et al. (2013), Web Appendix 11 -- the same
    one EpiEstim's ``discr_si`` implements. It is not the obvious approach, and
    the obvious approach is wrong: binning a Gamma CDF at integer edges
    (``w_s = F(s) - F(s-1)``) yields a distribution whose mean is half a day too
    large, because it assigns the mass of ``[s-1, s)`` to the point ``s``. That
    half-day bias propagates straight into R_t.

    This version fits the Gamma to ``mean - 1`` and integrates against a triangular
    kernel, which returns ``w[0] = 0`` by construction -- a case cannot infect
    anyone on the day it is infected -- and recovers the intended mean.
    """
    if mean <= 1 or sd <= 0:
        raise ValueError("serial interval mean must exceed 1 day and sd must be positive")

    shape = ((mean - 1) / sd) ** 2
    scale = sd**2 / (mean - 1)

    def f(x, a):  # CDF of Gamma(a, scale), clamped at zero
        return stats.gamma.cdf(np.clip(x, 0, None), a=a, scale=scale)

    k = np.arange(0, max_days + 1, dtype=float)
    w = (k * f(k, shape)
         + (k - 2) * f(k - 2, shape)
         - 2 * (k - 1) * f(k - 1, shape)
         + shape * scale * (2 * f(k - 1, shape + 1)
                            - f(k - 2, shape + 1)
                            - f(k, shape + 1)))
    w = np.clip(w, 0, None)
    w[0] = 0.0
    total = w.sum()
    return w / total if total > 0 else w


def total_infectiousness(incidence: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Lambda_t = sum_{s>=1} I_{t-s} w_s, by direct convolution."""
    inc = np.asarray(incidence, float)
    inc = np.nan_to_num(np.clip(inc, 0, None))
    n = inc.size
    lam = np.zeros(n)
    max_s = min(w.size - 1, n)
    for s in range(1, max_s + 1):
        lam[s:] += inc[:-s] * w[s]
    return lam


@dataclass(frozen=True)
class RtEstimate:
    """Posterior summary of R_t, aligned to the input series."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    shape: np.ndarray
    scale: np.ndarray

    def last_valid(self) -> float:
        finite = self.mean[np.isfinite(self.mean)]
        return float(finite[-1]) if finite.size else np.nan


def estimate_rt(incidence: np.ndarray, *, tau: int = 7,
                prior_shape: float = 1.0, prior_scale: float = 5.0,
                w: np.ndarray | None = None, ci: float = 0.95) -> RtEstimate:
    """Posterior mean and credible interval for R_t over a trailing window.

    With a Gamma(a, b) prior and Poisson likelihood, the posterior over the
    window ending at ``t`` is Gamma with

        shape = a + sum(I)             over the window
        scale = 1 / (1/b + sum(Lambda)) over the window

    Defaults ``a=1, b=5`` give a prior with mean 5 and SD 5 -- weak enough to be
    dominated by a handful of cases, which is what Cori et al. recommend.
    """
    inc = np.nan_to_num(np.clip(np.asarray(incidence, float), 0, None))
    n = inc.size
    if w is None:
        w = discretise_serial_interval()
    lam = total_infectiousness(inc, w)

    shape = np.full(n, np.nan)
    scale = np.full(n, np.nan)

    # Cumulative sums turn the sliding window into O(n) vector arithmetic.
    cinc = np.concatenate([[0.0], np.cumsum(inc)])
    clam = np.concatenate([[0.0], np.cumsum(lam)])

    t = np.arange(tau, n)
    inc_sum = cinc[t + 1] - cinc[t + 1 - tau]
    lam_sum = clam[t + 1] - clam[t + 1 - tau]

    # Where no infectious pressure has accumulated, R_t is not identified and is
    # left as NaN rather than silently returning the prior mean.
    ok = lam_sum > 0
    shape[t[ok]] = prior_shape + inc_sum[ok]
    scale[t[ok]] = 1.0 / (1.0 / prior_scale + lam_sum[ok])

    mean = shape * scale
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    with np.errstate(invalid="ignore"):
        lower = stats.gamma.ppf(lo_q, a=shape, scale=scale)
        upper = stats.gamma.ppf(hi_q, a=shape, scale=scale)

    return RtEstimate(mean=mean, lower=lower, upper=upper, shape=shape, scale=scale)


def project_renewal(incidence: np.ndarray, r: float, horizon: int,
                    *, w: np.ndarray | None = None, damping: float = 1.0) -> np.ndarray:
    """Project incidence forward under a (possibly damped) reproduction number.

    ``damping`` pulls R geometrically toward 1 as the horizon grows:

        R_k = 1 + (R - 1) * damping**k

    Damping is not a fudge factor -- it encodes that neither explosive growth nor
    free-fall persists, because susceptible depletion and behavioural response
    both push R toward 1. Setting ``damping=1`` recovers the naive
    "current R persists" projection, which is what the backtest compares against.
    """
    if horizon <= 0:
        return np.empty(0)
    if w is None:
        w = discretise_serial_interval()
    if not np.isfinite(r) or r < 0:
        r = 1.0

    hist = np.nan_to_num(np.clip(np.asarray(incidence, float), 0, None))
    n = hist.size
    lag = w.size - 1

    # One preallocated buffer plus a reversed weight vector turns the inner
    # convolution into a single dot product; growing a Python list and calling
    # np.asarray on it each step was the dominant cost of the whole backtest.
    buf = np.empty(n + horizon)
    buf[:n] = hist
    w_rev = w[1: lag + 1][::-1]  # [w_lag, ..., w_1]

    for k in range(horizon):
        t = n + k
        lo = max(0, t - lag)
        window = buf[lo:t]                       # [I_{t-m}, ..., I_{t-1}]
        lam = float(np.dot(window, w_rev[-window.size:])) if window.size else 0.0
        r_k = 1.0 + (r - 1.0) * (damping**k)
        buf[t] = max(r_k * lam, 0.0)

    return buf[n:]
