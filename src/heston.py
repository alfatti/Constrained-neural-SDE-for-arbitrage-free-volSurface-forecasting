"""
Heston model simulation and option pricing.

We use vanilla Heston (no stochastic local volatility) for parsimony. The
Heston model under the risk-neutral measure is:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

Option prices are computed via the Carr-Madan FFT method using the Heston
characteristic function. This gives us closed-form-quality call prices with
no Monte Carlo noise, which is essential for clean factor decoding.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float = 0.04        # initial variance
    kappa: float = 3.0      # mean reversion speed
    theta: float = 0.04     # long-run variance
    sigma: float = 0.3      # vol of vol  (Feller: 2*3*0.04 = 0.24 > 0.09)
    rho: float = -0.7       # spot-vol correlation
    r: float = 0.0          # risk-free rate
    q: float = 0.0          # dividend yield
    S0: float = 100.0       # initial spot

    def feller(self) -> float:
        """Feller condition: 2 kappa theta > sigma^2 keeps v > 0."""
        return 2 * self.kappa * self.theta - self.sigma ** 2


def heston_cf(u: np.ndarray, tau: float, S: float, v: float,
              params: HestonParams) -> np.ndarray:
    """
    Heston characteristic function evaluated at u, for log-spot ln(S_T).

    Uses the formulation that is numerically stable (avoids the so-called
    "Heston trap" via the convention of Albrecher et al. / Kahl-Jackel).
    """
    r, q, kappa, theta, sigma, rho = (
        params.r, params.q, params.kappa, params.theta, params.sigma, params.rho
    )
    x = np.log(S)

    d = np.sqrt((rho * sigma * 1j * u - kappa) ** 2
                + sigma ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)

    # "little Heston trap" formulation: use g (not 1/g)
    C = (r - q) * 1j * u * tau + (kappa * theta / sigma ** 2) * (
        (kappa - rho * sigma * 1j * u - d) * tau
        - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )
    D = ((kappa - rho * sigma * 1j * u - d) / sigma ** 2) * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )

    return np.exp(C + D * v + 1j * u * x)


def heston_call_carrmadan(K: np.ndarray, tau: float, S: float, v: float,
                          params: HestonParams,
                          alpha: float = 1.5, N: int = 4096,
                          eta: float = 0.25) -> np.ndarray:
    """
    Carr-Madan FFT pricing of European call options.

    Standard formulation:
        C(K) = exp(-alpha * k) / pi  *  Re[ int_0^inf exp(-i u k) psi(u) du ]
    where k = log(K), and
        psi(u) = exp(-r tau) * phi(u - i(alpha+1)) /
                 (alpha^2 + alpha - u^2 + i (2 alpha + 1) u)

    The FFT discretization with frequency step eta gives N values of C at
    log-strikes ku = -b + lam * j, j=0..N-1, with lam = 2*pi/(N*eta) and
    b = N*lam/2 (so the grid is symmetric around 0).

    Reference: Carr & Madan (1999), "Option valuation using the fast Fourier
    transform".
    """
    K = np.atleast_1d(K).astype(float)

    # Frequency and log-strike grids
    lam = 2 * np.pi / (N * eta)            # log-strike spacing
    b = N * lam / 2                         # half-width
    u_grid = eta * np.arange(N)             # u_j = j * eta
    ku = -b + lam * np.arange(N)            # log-strike grid centered at 0

    # Carr-Madan modified function. phi is the CF of log(S_T).
    phi = heston_cf(u_grid - (alpha + 1) * 1j, tau, S, v, params)
    denom = alpha ** 2 + alpha - u_grid ** 2 + 1j * (2 * alpha + 1) * u_grid
    psi = np.exp(-params.r * tau) * phi / denom

    # Simpson weights: 1/3, 4/3, 2/3, 4/3, ..., 4/3, 1/3
    simpson = np.ones(N)
    simpson[1::2] = 4.0
    simpson[2:-1:2] = 2.0
    simpson = simpson / 3.0

    integrand = np.exp(1j * b * u_grid) * psi * eta * simpson
    fft_vals = np.real(np.fft.fft(integrand))
    call_grid = np.exp(-alpha * ku) * fft_vals / np.pi

    # Interpolate onto requested strikes
    log_K = np.log(K)
    call = np.interp(log_K, ku, call_grid)

    # Clip to no-arbitrage bounds for numerical robustness
    disc = np.exp(-params.r * tau)
    forward = S * np.exp((params.r - params.q) * tau)
    intrinsic = np.maximum(disc * (forward - K), 0.0)
    call = np.maximum(call, intrinsic)
    call = np.minimum(call, disc * forward)

    return call


def simulate_heston(params: HestonParams, T: float, n_steps: int,
                    n_paths: int = 1, dt: Optional[float] = None,
                    rng: Optional[np.random.Generator] = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston paths using full-truncation Euler.

    Returns
    -------
    S : (n_paths, n_steps+1)
    v : (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if dt is None:
        dt = T / n_steps

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = params.S0
    v[:, 0] = params.v0

    sqrt_dt = np.sqrt(dt)
    chol = np.array([[1.0, 0.0], [params.rho, np.sqrt(1 - params.rho ** 2)]])

    for k in range(n_steps):
        z = rng.standard_normal((n_paths, 2))
        dW = z @ chol.T * sqrt_dt
        vk = np.maximum(v[:, k], 0.0)  # full truncation
        sqv = np.sqrt(vk)

        S[:, k + 1] = S[:, k] * np.exp(
            (params.r - params.q - 0.5 * vk) * dt + sqv * dW[:, 0]
        )
        v[:, k + 1] = v[:, k] + params.kappa * (params.theta - vk) * dt \
                    + params.sigma * sqv * dW[:, 1]

    return S, v


def build_option_grid(taus: np.ndarray, moneyness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (tau, m) grid as flat arrays. moneyness m = K/S (relative strike).
    Returns arrays of length N = n_tau * n_m.
    """
    T, M = np.meshgrid(taus, moneyness, indexing='ij')
    return T.flatten(), M.flatten()


def price_surface(S: float, v: float, taus_flat: np.ndarray,
                  moneyness_flat: np.ndarray,
                  params: HestonParams) -> np.ndarray:
    """
    Price a vector of call options, one per (tau, m) pair.
    Returns normalized prices c = C/S so they are roughly O(1).

    For each tau, we batch the strikes (K = m * S) into one FFT call.
    """
    K = moneyness_flat * S
    out = np.empty_like(K, dtype=float)
    unique_taus = np.unique(taus_flat)
    for tau in unique_taus:
        mask = taus_flat == tau
        C = heston_call_carrmadan(K[mask], tau, S, v, params)
        out[mask] = C
    return out / S  # normalized: c = C/S


def implied_vol_bs(price: float, S: float, K: float, tau: float,
                   r: float = 0.0, q: float = 0.0,
                   tol: float = 1e-8) -> float:
    """
    Black-Scholes implied volatility by bracketed root finding (brentq).

    Returns np.nan if the price is outside the no-arbitrage bounds or if
    the option is so deep OTM that vega is numerically zero.
    """
    from math import log, sqrt
    from scipy.stats import norm
    from scipy.optimize import brentq

    F = S * np.exp((r - q) * tau)
    disc = np.exp(-r * tau)
    intrinsic = max(disc * (F - K), 0.0)
    upper = disc * F

    if price <= intrinsic + 1e-12 or price >= upper - 1e-12:
        return np.nan

    def bs_call(sigma):
        if sigma <= 0:
            return -np.inf
        d1 = (log(F / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt(tau))
        d2 = d1 - sigma * sqrt(tau)
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2)) - price

    # brentq needs a bracket
    try:
        return brentq(bs_call, 1e-6, 5.0, xtol=tol, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_surface(prices_norm: np.ndarray, S: float,
                        taus_flat: np.ndarray, moneyness_flat: np.ndarray,
                        r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Convert a vector of normalized call prices c = C/S into an implied vol vector.
    """
    iv = np.empty_like(prices_norm)
    for i, (c, tau, m) in enumerate(zip(prices_norm, taus_flat, moneyness_flat)):
        iv[i] = implied_vol_bs(c * S, S, m * S, tau, r=r, q=q)
    return iv
