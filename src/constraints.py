"""
Static arbitrage constraints for a finite lattice of call options.

For a lattice {(tau_j, m_j)}_{j=1..N} of options with normalized prices
c_j = C_j / S, the no-arbitrage region is defined by linear inequalities:

    A c >= b

We construct A and b from the following families of constraints (Carr & Madan
2005, Cousot 2007, Cohen-Reisinger-Wang 2020):

  (i)   Bounds: max(1 - m, 0) <= c <= 1   (intrinsic and upper bound)
  (ii)  Calendar monotonicity (in m-coordinates): for same moneyness m, c is
        non-decreasing in tau. We work with the moneyness convention m = K/S.
  (iii) Butterfly convexity: in the moneyness direction, c is convex in m.
        For three consecutive moneynesses m_{i-1} < m_i < m_{i+1} at the
        same tau, we require:
            c_{i+1}(m_{i+1} - m_{i-1}) - c_i(m_{i+1} - m_{i-1})
              >= c_i(m_{i-1} - m_{i+1}) - c_{i-1}(m_{i-1} - m_{i+1})
        equivalently the second divided difference is non-negative.
  (iv)  Monotonicity in m: c is non-increasing in m at each tau.

We use a *rectangular* lattice for simplicity (n_tau x n_m). This matches
Heston synthetic data; real OTC FX/SPX would need an arbitrarily-shaped
lattice (Cohen-Reisinger-Wang 2021 handle this case).
"""

import numpy as np


def build_constraints(taus: np.ndarray, moneynesses: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build A, b such that A c >= b encodes static no-arbitrage on the
    rectangular lattice (taus x moneynesses), with c flattened in
    (tau, m) C-order (tau varies slowest).

    Returns
    -------
    A : (R, N) constraint matrix
    b : (R,)  constant vector
    """
    n_tau = len(taus)
    n_m = len(moneynesses)
    N = n_tau * n_m

    def idx(i_tau, j_m):
        return i_tau * n_m + j_m

    rows_A = []
    rows_b = []

    # (i) Lower bound: c >= max(1 - m, 0)
    for i_tau in range(n_tau):
        for j_m in range(n_m):
            row = np.zeros(N)
            row[idx(i_tau, j_m)] = 1.0
            rows_A.append(row)
            rows_b.append(max(1.0 - moneynesses[j_m], 0.0))

    # (i') Upper bound: -c >= -1, i.e. c <= 1
    for i_tau in range(n_tau):
        for j_m in range(n_m):
            row = np.zeros(N)
            row[idx(i_tau, j_m)] = -1.0
            rows_A.append(row)
            rows_b.append(-1.0)

    # (ii) Calendar monotonicity: c(tau_{i+1}, m) - c(tau_i, m) >= 0
    for i_tau in range(n_tau - 1):
        for j_m in range(n_m):
            row = np.zeros(N)
            row[idx(i_tau + 1, j_m)] = 1.0
            row[idx(i_tau, j_m)] = -1.0
            rows_A.append(row)
            rows_b.append(0.0)

    # (iii) Butterfly: c convex in m. Second divided diff non-negative.
    # For consecutive m_{j-1}, m_j, m_{j+1}:
    #   (m_{j+1} - m_{j-1}) c_j <= (m_{j+1} - m_j) c_{j-1} + (m_j - m_{j-1}) c_{j+1}
    # Rearranged as A row >= 0:
    #   (m_{j+1} - m_j) c_{j-1} - (m_{j+1} - m_{j-1}) c_j + (m_j - m_{j-1}) c_{j+1} >= 0
    for i_tau in range(n_tau):
        for j_m in range(1, n_m - 1):
            mm = moneynesses[j_m - 1]
            m0 = moneynesses[j_m]
            mp = moneynesses[j_m + 1]
            row = np.zeros(N)
            row[idx(i_tau, j_m - 1)] = mp - m0
            row[idx(i_tau, j_m)] = -(mp - mm)
            row[idx(i_tau, j_m + 1)] = m0 - mm
            rows_A.append(row)
            rows_b.append(0.0)

    # (iv) Monotonicity in m: c(tau, m_j) - c(tau, m_{j+1}) >= 0
    for i_tau in range(n_tau):
        for j_m in range(n_m - 1):
            row = np.zeros(N)
            row[idx(i_tau, j_m)] = 1.0
            row[idx(i_tau, j_m + 1)] = -1.0
            rows_A.append(row)
            rows_b.append(0.0)

    A = np.vstack(rows_A)
    b = np.array(rows_b)
    return A, b


def remove_redundant(A: np.ndarray, b: np.ndarray,
                     tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove obviously redundant constraints by deduplication. A full LP-based
    redundancy elimination (Caron-McDonald-Ponic) is overkill for our purposes;
    deduplication suffices to clean up.
    """
    # Normalize rows then dedupe
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < tol] = 1.0
    A_norm = A / norms
    b_norm = b / norms.squeeze()

    # Build hashable representation (row, b)
    rep = np.hstack([A_norm, b_norm[:, None]])
    rep_rounded = np.round(rep, decimals=8)
    _, unique_idx = np.unique(rep_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return A[unique_idx], b[unique_idx]


def violations(A: np.ndarray, b: np.ndarray, c: np.ndarray,
               tol: float = 1e-10) -> np.ndarray:
    """
    For a price vector c, return the per-constraint slack (A c - b).
    Negative entries indicate violations.
    """
    return A @ c - b


def is_arbitrage_free(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                      tol: float = 1e-8) -> bool:
    """Check whether c lies in the no-arbitrage region."""
    return np.all(violations(A, b, c) >= -tol)


def project_to_NA(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                  tol: float = 1e-8) -> np.ndarray:
    """
    Project a price vector onto the no-arbitrage polyhedron {c : A c >= b}.
    Solves min ||c - c_hat||^2 s.t. A c_hat >= b via cvxpy.

    Use sparingly — for the constrained NSDE we don't need projection since
    the SDE stays in the region by construction. This is here for safety
    checks and for the DDPM variant.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required for projection; install via pip.")
    n = c.size
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - c)),
                       [A @ x >= b])
    prob.solve(solver=cp.OSQP, eps_abs=tol, eps_rel=tol)
    return np.asarray(x.value)
