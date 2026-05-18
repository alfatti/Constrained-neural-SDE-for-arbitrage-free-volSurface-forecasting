"""
Factor decoding from a time series of normalized call surfaces.

Implements a parsimonious version of Cohen-Reisinger-Wang (2021) Algorithm 1:
given a (L+1, N) matrix C of normalized call prices over time, decompose

    c_t = G_0 + G^T xi_t + residual

with xi_t in R^d. We use PCA on (C - mean) to get the basis, then linearly
project the static-arbitrage polyhedron {A c >= hat_b} into factor space:

    P = { xi in R^d : (A G^T) xi >= hat_b - A G_0 }
      = { xi in R^d : V xi >= b }

where V := A G^T and b := hat_b - A G_0.

For full CRW: they construct factors to minimize three objectives in order
(dynamic arbitrage residual via z_t PCA, then statistical accuracy via
residual PCA, then static arbitrage violations via a heuristic search). We
omit the dynamic-arbitrage z_t-PCA step for parsimony — pure PCA on prices
already yields very few static-arbitrage violations on Heston synthetic
data when d >= 3 (since the ground-truth process is 2-factor).

This module is fully self-contained: no dependency on the CRW repo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FactorModel:
    """Decoded factor representation: c = G0 + G^T xi."""
    G0: np.ndarray              # (N,)         constant
    G: np.ndarray               # (d, N)       basis rows = G_i^T
    d: int                      # number of factors
    xi: Optional[np.ndarray] = None        # (L+1, d) factor time series
    V: Optional[np.ndarray] = None         # (R, d)   polytope coefficient
    b: Optional[np.ndarray] = None         # (R,)     polytope offset
    A_orig: Optional[np.ndarray] = None    # (R, N)   original price-space A
    b_orig: Optional[np.ndarray] = None    # (R,)     original price-space b
    scale: Optional[np.ndarray] = None     # (d,)     per-factor scale (std)


def normalize_factors(fm: FactorModel, target_std: float = 1.0
                      ) -> FactorModel:
    """
    Rescale factors so each component has standard deviation = target_std.
    This is critical for neural-net training: PCA factors have wildly
    different scales (PC1 variance >> PC2 variance), and uniform learning
    rates work poorly across them.

    The rescaling is: xi_new = xi_old / scale, G_new = scale * G,
    V_new = V * scale (so V_new xi_new = V xi).
    """
    if fm.xi is None:
        raise ValueError("FactorModel has no xi to normalize")
    scale = fm.xi.std(axis=0) / target_std         # (d,)
    scale = np.where(scale < 1e-12, 1.0, scale)    # avoid div by zero

    xi_new = fm.xi / scale[None, :]                # (T, d)
    G_new = (scale[:, None] * fm.G)                # (d, N): G_i is scaled by scale_i
    V_new = fm.V * scale[None, :] if fm.V is not None else None
    # b is unchanged: V_new xi_new = V xi (sanity check)

    return FactorModel(
        G0=fm.G0, G=G_new, d=fm.d, xi=xi_new,
        V=V_new, b=fm.b, A_orig=fm.A_orig, b_orig=fm.b_orig,
        scale=scale,
    )


def decode_factors(C: np.ndarray, d: int,
                   A: Optional[np.ndarray] = None,
                   b_arb: Optional[np.ndarray] = None,
                   orthonormalize: bool = True
                   ) -> FactorModel:
    """
    Decode d factors from price time series C.

    Parameters
    ----------
    C : (L+1, N) time series of normalized call prices
    d : number of factors
    A, b_arb : (R, N), (R,) optional price-space arbitrage constraints. If
               provided, these are projected to factor space.
    orthonormalize : if True, the rows of G are orthonormal (standard PCA).
                     The corresponding factors xi are orthogonal but may
                     have differing scales (PC variances).

    Returns
    -------
    FactorModel with G0, G, xi, and (if A,b_arb provided) V, b.
    """
    L_plus_1, N = C.shape
    assert d <= min(L_plus_1, N), f"d={d} too large for data shape {C.shape}"

    G0 = C.mean(axis=0)                            # (N,)
    residual = C - G0[None, :]                     # (L+1, N)

    # SVD-based PCA: residual = U S V^T, take top d
    # rows of V^T = principal components in price space = basis vectors G_i^T
    U, s, Vt = np.linalg.svd(residual, full_matrices=False)
    G = Vt[:d, :]                                  # (d, N)
    xi = residual @ G.T                            # (L+1, d): xi_t = G (c_t - G0)

    fm = FactorModel(G0=G0, G=G, d=d, xi=xi, A_orig=A, b_orig=b_arb)

    if A is not None and b_arb is not None:
        # Pull back the polyhedron: {c : A c >= b_arb} -> {xi : (A G^T) xi >= b_arb - A G0}
        V = A @ G.T                                # (R, d)
        b_pulled = b_arb - A @ G0                  # (R,)
        fm.V = V
        fm.b = b_pulled

    return fm


def reconstruction_error(fm: FactorModel, C: np.ndarray) -> dict:
    """Compute reconstruction metrics."""
    C_recon = fm.G0[None, :] + fm.xi @ fm.G        # (L+1, N)
    err = C - C_recon
    mape = np.mean(np.abs(err) / np.maximum(np.abs(C), 1e-8))
    rmse = np.sqrt(np.mean(err ** 2))
    rel_rmse = rmse / np.sqrt(np.mean(C ** 2))
    return dict(mape=mape, rmse=rmse, rel_rmse=rel_rmse)


def psas(fm: FactorModel, A: np.ndarray, b: np.ndarray,
         tol: float = 1e-8) -> float:
    """
    Proportion of Statically Arbitrageable Samples after factor reconstruction.
    Following CRW notation.
    """
    C_recon = fm.G0[None, :] + fm.xi @ fm.G        # (L+1, N)
    # Each sample t: arbitrageable if any A c - b < -tol
    slacks = (C_recon @ A.T) - b[None, :]          # (L+1, R)
    violations_per_t = (slacks < -tol).any(axis=1)
    return float(violations_per_t.mean())


def filter_to_polytope(fm: FactorModel, tol: float = 1e-8
                       ) -> tuple[FactorModel, np.ndarray]:
    """
    Restrict the factor time series to those samples that lie strictly inside
    the no-arbitrage polytope. CRW remove ~0.4% of samples this way before
    training. Returns the filtered FactorModel and the boolean mask of kept
    samples.

    NB: Filtering breaks consecutive-time-step adjacency in the filtered xi.
    For SDE training the caller should use valid_pair_mask() on the
    unfiltered xi to find pairs (k, k+1) where both endpoints are inside.
    """
    if fm.V is None or fm.b is None:
        return fm, np.ones(fm.xi.shape[0], dtype=bool)
    slacks = fm.xi @ fm.V.T - fm.b[None, :]        # (L+1, R)
    mask = (slacks > tol).all(axis=1)
    fm_filt = FactorModel(
        G0=fm.G0, G=fm.G, d=fm.d, xi=fm.xi[mask],
        V=fm.V, b=fm.b, A_orig=fm.A_orig, b_orig=fm.b_orig,
        scale=fm.scale,
    )
    return fm_filt, mask


def valid_pair_mask(fm: FactorModel, tol: float = 1e-8) -> np.ndarray:
    """
    Returns a boolean mask of length L over indices k in {0..L-1} such that
    BOTH xi[k] and xi[k+1] lie strictly inside the polytope. Use this to
    select training pairs (xi_t, xi_{t+1}) that are truly adjacent in time.
    """
    if fm.V is None or fm.b is None:
        L = fm.xi.shape[0] - 1
        return np.ones(L, dtype=bool)
    slacks = fm.xi @ fm.V.T - fm.b[None, :]        # (L+1, R)
    inside = (slacks > tol).all(axis=1)
    return inside[:-1] & inside[1:]                # (L,)
