"""
Context vector construction for conditional forecasting.

Given a factor time series xi_t and (optionally) a stock price time series S_t,
build a context vector Y_k that summarizes the recent history at each k.

The context is *exogenous and observable*: it is a deterministic function of
the path up to time k. Adding it to the NSDE input

    d xi_t = mu(xi_t, Y_t) dt + sigma(xi_t, Y_t) dW_t

upgrades the model from Markovian-in-xi to a conditional forecaster, in the
sense of Jin-Agarwal: the network drift and diffusion are reminded about
path-dependence they would otherwise have to recover from xi alone.

For Heston (which is itself Markov in (S, v) where v ~ xi_1), the context
should add little — that's a useful sanity check.

Default context features:
  Y[0..d-1]   : EWMA over a short window (5 steps) of xi
  Y[d..2d-1]  : EWMA over a longer window (20 steps) of xi
  Y[2d]       : realized variance of log S over last 20 steps  (if S provided)

Total context dim d_Y = 2d or 2d + 1.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    """
    Exponentially-weighted moving average of x along axis 0.
    Uses recursive form with alpha = 1 - exp(-ln(2)/halflife).
    Returns array of the same shape as x.
    """
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, len(x)):
        out[k] = alpha * x[k] + (1.0 - alpha) * out[k - 1]
    return out


def _realized_var_logS(S: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling realized variance of log(S) over a trailing window.
    Returns an array of length len(S) where the first `window` entries are
    zero-padded (insufficient history).
    """
    log_S = np.log(S)
    r = np.diff(log_S, prepend=log_S[0])     # length len(S), r[0]=0
    rv = np.zeros_like(r)
    # cumulative sum trick for rolling sum-of-squares
    r2 = r * r
    cumsum = np.cumsum(r2)
    for k in range(len(r)):
        if k < window:
            rv[k] = r2[:k+1].sum()
        else:
            rv[k] = cumsum[k] - cumsum[k - window]
    return rv


@dataclass
class ContextBuilder:
    """
    Builds context vector Y_k from xi (and optionally S) time series.

    For training, we want Y at *every* time index k where we observe
    (xi_k -> xi_{k+1}), i.e. k = 0..L-1.
    """
    halflife_short: float = 5.0
    halflife_long: float = 20.0
    rv_window: int = 20
    include_rv: bool = True

    def fit_transform(self, xi: np.ndarray, S: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        """
        xi : (L+1, d)
        S  : (L+1,) optional spot path
        returns Y : (L+1, d_Y)
        """
        ewma_s = _ewma(xi, self.halflife_short)
        ewma_l = _ewma(xi, self.halflife_long)
        parts = [ewma_s, ewma_l]
        if self.include_rv and S is not None:
            rv = _realized_var_logS(S, self.rv_window)
            parts.append(rv[:, None])
        return np.concatenate(parts, axis=1)

    def context_dim(self, d: int) -> int:
        n = 2 * d
        if self.include_rv:
            n += 1
        return n


def normalize_context(Y_train: np.ndarray, Y_test: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, dict]:
    """
    Standardize context features using training-set mean/std.
    Returns (Y_train_norm, stats) and optionally (Y_test_norm).
    """
    mean = Y_train.mean(axis=0)
    std = Y_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Y_train_norm = (Y_train - mean) / std
    stats = dict(mean=mean, std=std)
    if Y_test is None:
        return Y_train_norm, stats
    Y_test_norm = (Y_test - mean) / std
    return Y_train_norm, Y_test_norm, stats
