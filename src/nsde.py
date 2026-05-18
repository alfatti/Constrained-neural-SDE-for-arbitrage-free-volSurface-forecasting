"""
Constrained neural SDE for factor dynamics inside a polytope P = {xi : V xi >= b}.

We model
    d xi_t = mu(xi_t) dt + sigma(xi_t) dW_t,
where mu, sigma are neural networks, and xi_t must remain inside P a.s.

Following Cohen-Reisinger-Wang (2021) Section 3, we use Friedman-Pinsky-style
operators to enforce:
    v_k^T sigma sigma^T v_k -> 0    at v_k^T xi = b_k    (no normal diffusion)
    v_k^T mu >= 0                   at v_k^T xi = b_k    (inward drift)

PARSIMONIOUS IMPLEMENTATION
---------------------------
We use simpler smooth operators than CRW's Gram-Schmidt construction:

  Diffusion gate: scale the entire diffusion matrix by a smooth function
  of the minimum normalized signed distance to the boundary:
      g(xi) = sigmoid(beta * (d_min(xi) / rho_star - 1))
  -> 0 as xi -> boundary, -> 1 in interior.

  Drift correction: add a Chebyshev-center attractive force that ramps up
  only near the boundary:
      mu = mu_hat + c0 * relu(1 - d_min/rho_star) * (center - xi)

Together with a "safety projection" step (project escapees back to the
nearest facet), this gives a robust SDE that stays inside P with probability
close to 1. Less elegant than CRW's pure architectural guarantee, but
dimension-free and easy to implement and debug.

Training: maximum likelihood under Euler-Maruyama discretization.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import linprog


# ----------------------------------------------------------------------
# Polytope geometry
# ----------------------------------------------------------------------

def chebyshev_center(V: np.ndarray, b: np.ndarray
                     ) -> tuple[np.ndarray, float]:
    """
    Compute the Chebyshev center of P = {x : V x >= b}.
    Returns (center, radius).
    """
    R, d = V.shape
    norms = np.linalg.norm(V, axis=1)
    c_obj = np.zeros(d + 1)
    c_obj[-1] = -1.0
    A_ub = np.hstack([-V, norms[:, None]])
    b_ub = -b
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * d + [(0, None)],
                  method='highs')
    if not res.success:
        raise RuntimeError(f"Chebyshev center LP failed: {res.message}")
    return res.x[:d], float(res.x[-1])


# ----------------------------------------------------------------------
# Neural network building blocks
# ----------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        last = d_in
        for _ in range(depth):
            layers.append(nn.Linear(last, width))
            layers.append(nn.Tanh())     # smooth activations help SDE training
            last = width
        layers.append(nn.Linear(last, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Constrained NSDE
# ----------------------------------------------------------------------

class ConstrainedNSDE(nn.Module):
    """
    Drift and diffusion networks composed with boundary-aware operators.

    The network input is (xi, Y) where Y is an optional exogenous context
    vector summarizing path history (Way 1, exogenous context augmentation).
    Boundary operators (diffusion gate, drift pull) act on xi only, since
    only xi is constrained to the polytope.

    Output dim: d  for mu;  d*(d+1)/2  Cholesky entries for sigma (lower-tri).
    """

    def __init__(self, d: int, V: np.ndarray, b: np.ndarray,
                 center: np.ndarray, rho_star: float,
                 d_Y: int = 0,
                 width: int = 64, depth: int = 3,
                 boundary_beta: float = 10.0,
                 drift_correction_strength: float = 1.0):
        super().__init__()
        self.d = d
        self.d_Y = int(d_Y)
        V_t = torch.from_numpy(V).float()
        b_t = torch.from_numpy(b).float()
        center_t = torch.from_numpy(center).float()
        v_norms = torch.linalg.norm(V_t, dim=1, keepdim=True)
        self.register_buffer('V', V_t)
        self.register_buffer('b', b_t)
        self.register_buffer('center', center_t)
        self.register_buffer('v_norms', v_norms.squeeze())
        self.rho_star = float(rho_star)
        self.boundary_beta = float(boundary_beta)
        self.drift_correction_strength = float(drift_correction_strength)

        n_chol = d * (d + 1) // 2
        self.mlp = MLP(d + self.d_Y, d + n_chol, width=width, depth=depth)

        tril_idx = torch.tril_indices(d, d)
        self.register_buffer('tril_row', tril_idx[0])
        self.register_buffer('tril_col', tril_idx[1])
        diag_in_tril = (tril_idx[0] == tril_idx[1]).nonzero(as_tuple=True)[0]
        self.register_buffer('diag_in_tril', diag_in_tril)

    def min_distance(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Min normalized signed distance to any facet.
        Positive when xi is in the interior, zero on a facet.
        Returns (B,).
        """
        signed = (xi @ self.V.T - self.b[None, :]) / self.v_norms[None, :]
        return signed.min(dim=1).values

    def diffusion_gate(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Smooth gate g(xi) in (0,1): -> 0 at boundary, -> 1 in interior.
        """
        d_min = self.min_distance(xi)
        return torch.sigmoid(self.boundary_beta * (d_min / self.rho_star - 1.0))

    def drift_pull(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Drift correction: c0 * relu(1 - d_min/rho_star) * (center - xi).
        """
        d_min = self.min_distance(xi)
        ramp = torch.relu(1.0 - d_min / self.rho_star)
        return self.drift_correction_strength * ramp[:, None] * (self.center[None, :] - xi)

    def forward(self, xi: torch.Tensor, Y: torch.Tensor = None):
        """
        Returns (mu, L) where mu : (B,d) is the constrained drift, and
        L : (B,d,d) is the lower-triangular Cholesky factor of sigma sigma^T,
        already gated by the boundary operator.

        Y : (B, d_Y) exogenous context. If d_Y > 0 and Y is None, an error
        is raised. If d_Y == 0, Y is ignored.
        """
        B = xi.shape[0]
        if self.d_Y > 0:
            if Y is None:
                raise ValueError(f"Context Y required (d_Y={self.d_Y}) but None given")
            inp = torch.cat([xi, Y], dim=-1)
        else:
            inp = xi
        out = self.mlp(inp)
        mu_hat = out[:, :self.d]
        chol_raw = out[:, self.d:]

        L_hat = xi.new_zeros(B, self.d, self.d)
        L_hat[:, self.tril_row, self.tril_col] = chol_raw
        diag_vals = chol_raw[:, self.diag_in_tril]
        L_hat[:, torch.arange(self.d), torch.arange(self.d)] = \
            torch.exp(torch.clamp(diag_vals, min=-6.0, max=4.0))

        # Diffusion gate (acts on xi only)
        gate = self.diffusion_gate(xi)
        L = L_hat * gate[:, None, None]

        # Drift correction (acts on xi only)
        mu = mu_hat + self.drift_pull(xi)
        return mu, L


# ----------------------------------------------------------------------
# Training: Euler-Maruyama negative log-likelihood
# ----------------------------------------------------------------------

def euler_neg_log_likelihood(xi_t: torch.Tensor, xi_tp1: torch.Tensor,
                              mu: torch.Tensor, L: torch.Tensor,
                              dt: float, sigma_floor: float = 1e-3
                              ) -> torch.Tensor:
    """
    Negative Euler log-likelihood. A small floor on the diagonal of L avoids
    singular covariance near the boundary (where the gate -> 0).
    """
    d = xi_t.shape[1]
    eye = torch.eye(d, device=L.device)[None, :, :]
    L_eff = L + sigma_floor * eye

    diff = xi_tp1 - xi_t - mu * dt
    y = torch.linalg.solve_triangular(L_eff, diff.unsqueeze(-1), upper=False)
    quad = (y.squeeze(-1) ** 2).sum(dim=1) / dt
    logdet_a = 2 * torch.log(torch.diagonal(L_eff, dim1=1, dim2=2)).sum(dim=1)
    nll = 0.5 * (d * np.log(2 * np.pi * dt) + logdet_a + quad)
    return nll.mean()


@dataclass
class TrainConfig:
    n_epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    print_every: int = 50
    val_frac: float = 0.1
    sigma_floor: float = 1e-3


def train_nsde(model: ConstrainedNSDE,
               xi_t: np.ndarray, xi_tp1: np.ndarray,
               dt: float,
               Y_t: np.ndarray = None,
               cfg: TrainConfig = TrainConfig(),
               device: str = 'cpu') -> dict:
    """
    Train the constrained neural SDE by Euler MLE on explicit pairs.

    Parameters
    ----------
    xi_t   : (n_pairs, d) starting points of one-step transitions
    xi_tp1 : (n_pairs, d) ending points, one timestep dt later
    dt     : time step between paired observations
    Y_t    : (n_pairs, d_Y) optional context at the starting time. Required
             iff model.d_Y > 0.

    The caller must ensure xi_t[i] -> xi_tp1[i] is a true single-step
    transition (i.e. adjacent in the original time index). Use
    factors.valid_pair_mask() to filter out pairs straddling polytope-
    excluded samples.
    """
    model = model.to(device)
    xs = torch.from_numpy(xi_t).float().to(device)
    ys = torch.from_numpy(xi_tp1).float().to(device)

    if model.d_Y > 0:
        if Y_t is None:
            raise ValueError("Y_t required when model.d_Y > 0")
        Ys = torch.from_numpy(Y_t).float().to(device)
    else:
        Ys = None

    n = xs.shape[0]
    n_val = max(1, int(n * cfg.val_frac))
    n_train = n - n_val
    train_x, train_y = xs[:n_train], ys[:n_train]
    val_x, val_y = xs[n_train:], ys[n_train:]
    if Ys is not None:
        train_Y = Ys[:n_train]
        val_Y = Ys[n_train:]
    else:
        train_Y = val_Y = None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    history = {'train': [], 'val': []}

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = []
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x_b, y_b = train_x[idx], train_y[idx]
            Y_b = train_Y[idx] if train_Y is not None else None
            mu, L = model(x_b, Y_b)
            loss = euler_neg_log_likelihood(x_b, y_b, mu, L, dt,
                                            sigma_floor=cfg.sigma_floor)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        model.eval()
        with torch.no_grad():
            mu_v, L_v = model(val_x, val_Y)
            val_loss = euler_neg_log_likelihood(val_x, val_y, mu_v, L_v, dt,
                                                 sigma_floor=cfg.sigma_floor).item()
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if (epoch + 1) % cfg.print_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}  train={train_loss:.4f}  val={val_loss:.4f}")

    return history


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def _project_to_polytope_step(xi: torch.Tensor, V: torch.Tensor,
                              b: torch.Tensor, v_norms: torch.Tensor):
    """
    For each path, if xi violates any facet, project to the most-violated
    facet's boundary by a single half-space projection.
    """
    slacks = xi @ V.T - b[None, :]                          # (B, R)
    min_slack, min_idx = slacks.min(dim=1)
    violating_mask = min_slack < 0
    if not violating_mask.any():
        return xi
    # For each violating path, project to that facet
    out = xi.clone()
    for path_idx in violating_mask.nonzero(as_tuple=True)[0]:
        k = min_idx[path_idx].item()
        v_k = V[k]
        # Projection: xi_new = xi + (b_k - v_k . xi) * v_k / ||v_k||^2
        s = (b[k] - v_k @ xi[path_idx]) / (v_norms[k] ** 2)
        out[path_idx] = xi[path_idx] + s * v_k
    return out


@torch.no_grad()
def simulate(model: ConstrainedNSDE, xi0: np.ndarray, n_steps: int,
             dt: float, rng_seed: int = 0,
             Y0: np.ndarray = None,
             Y_update: callable = None,
             device: str = 'cpu', sigma_floor: float = 0.0,
             clip_to_polytope: bool = True) -> np.ndarray:
    """
    Tamed-Euler simulation of the SDE starting from xi0.

    Context handling:
      - If model.d_Y == 0, Y0 is ignored.
      - If Y0 is given and Y_update is None, Y is held fixed at Y0 throughout
        the rollout. This is the natural choice for one-day-ahead forecasting
        where Y_k is today's market context.
      - If Y_update is given, it should be a callable
            Y_update(xi_history) -> Y_next
        where xi_history is (n_paths, k+1, d) and the return is (n_paths, d_Y).
        This permits self-consistent rollouts where Y is recomputed from the
        simulated path.

    Returns trajectory:
        (n_paths, n_steps+1, d) if xi0 is (n_paths, d)
        (n_steps+1, d)         if xi0 is (d,)
    """
    model = model.to(device).eval()
    single = (xi0.ndim == 1)
    if single:
        xi0 = xi0[None, :]
        if Y0 is not None and Y0.ndim == 1:
            Y0 = Y0[None, :]
    xi_t = torch.from_numpy(xi0).float().to(device)
    n_paths, d = xi_t.shape

    if model.d_Y > 0:
        if Y0 is None:
            raise ValueError("Y0 required when model.d_Y > 0")
        Y_t = torch.from_numpy(np.asarray(Y0)).float().to(device)
        if Y_t.dim() == 1:
            Y_t = Y_t.unsqueeze(0).expand(n_paths, -1).contiguous()
    else:
        Y_t = None

    torch.manual_seed(rng_seed)
    out = torch.zeros(n_paths, n_steps + 1, d, device=device)
    out[:, 0] = xi_t
    sqrt_dt = np.sqrt(dt)
    eye = torch.eye(d, device=device)[None, :, :]

    for k in range(n_steps):
        mu, L = model(xi_t, Y_t)
        L_eff = L + sigma_floor * eye
        z = torch.randn(n_paths, d, device=device)
        mu_norm = torch.linalg.norm(mu, dim=1, keepdim=True)
        L_norm = torch.linalg.matrix_norm(L_eff, dim=(1, 2), keepdim=True).squeeze(-1)
        tame_mu = mu / (1 + mu_norm * sqrt_dt)
        tame_L = L_eff / (1 + L_norm.unsqueeze(-1) * sqrt_dt)
        increment = tame_mu * dt + torch.bmm(tame_L, z.unsqueeze(-1)).squeeze(-1) * sqrt_dt
        xi_t = xi_t + increment

        if clip_to_polytope:
            xi_t = _project_to_polytope_step(xi_t, model.V, model.b, model.v_norms)

        out[:, k + 1] = xi_t

        if Y_update is not None and model.d_Y > 0:
            Y_new = Y_update(out[:, :k + 2].cpu().numpy())  # (n_paths, d_Y)
            Y_t = torch.from_numpy(np.asarray(Y_new)).float().to(device)

    arr = out.cpu().numpy()
    return arr[0] if single else arr
