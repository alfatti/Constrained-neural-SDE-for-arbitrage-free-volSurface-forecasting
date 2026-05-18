"""
Conditional Diffusion Guidance via Martingale Loss (CDG-ML) for the
constrained neural SDE.

Following Guo-Tang-Xu (2026): given a pretrained SDE
    d xi_t = mu(xi_t) dt + sigma(xi_t) dW_t
and a positive-measure rare-event set S subset of R^d, define
    h(t, xi) := P(xi_T in S | xi_t = xi).
The h-transform gives a guided SDE
    d xi_t^S = [mu(xi_t^S) + a(xi_t^S) grad log h(t, xi_t^S)] dt + sigma dW_t
where a = sigma sigma^T, whose terminal law equals the conditioned target.

CDG-ML learns h by exploiting that {h(t, xi_t)} is a (local) martingale
under the pretrained measure, yielding the L^2 projection objective:

    min_phi  E [ int_0^T (h_phi(t, xi_t) - 1{xi_T in S})^2 dt ]

where the expectation is over trajectories of the pretrained SDE.

For our forecasting application T is the calendar horizon (e.g., one day
ahead), so trajectories used for training the guidance function are simulated
from the pretrained constrained NSDE on [0, T] with the empirical initial
distribution.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional

from .nsde import ConstrainedNSDE, MLP


class HFunction(nn.Module):
    """
    Approximates h(t, xi, Y) in [0, 1]. We parametrize the logit and apply
    sigmoid, so h is automatically in (0, 1).

    Input: (B, 1 + d + d_Y) concatenated [t, xi, Y].
    """
    def __init__(self, d: int, d_Y: int = 0, width: int = 64, depth: int = 3):
        super().__init__()
        self.d = d
        self.d_Y = int(d_Y)
        self.net = MLP(1 + d + self.d_Y, 1, width=width, depth=depth)

    def _stack(self, t: torch.Tensor, xi: torch.Tensor,
               Y: torch.Tensor = None) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        parts = [t, xi]
        if self.d_Y > 0:
            if Y is None:
                raise ValueError(f"Y required (d_Y={self.d_Y})")
            parts.append(Y)
        return torch.cat(parts, dim=-1)

    def forward(self, t: torch.Tensor, xi: torch.Tensor,
                Y: torch.Tensor = None) -> torch.Tensor:
        logit = self.net(self._stack(t, xi, Y)).squeeze(-1)
        return torch.sigmoid(logit)

    def log_h(self, t: torch.Tensor, xi: torch.Tensor,
              Y: torch.Tensor = None) -> torch.Tensor:
        logit = self.net(self._stack(t, xi, Y)).squeeze(-1)
        return nn.functional.logsigmoid(logit)


# ----------------------------------------------------------------------
# Off-policy trajectory simulation
# ----------------------------------------------------------------------

@torch.no_grad()
def simulate_trajectories_with_indicator(
    nsde: ConstrainedNSDE, xi0_batch: torch.Tensor,
    n_steps: int, dt: float,
    in_S: Callable[[torch.Tensor], torch.Tensor],
    Y0_batch: torch.Tensor = None,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate n_paths trajectories from initial states xi0_batch (n_paths, d)
    and frozen context Y0_batch (n_paths, d_Y) under the pretrained NSDE.

    Return:
        traj          : (n_paths, n_steps+1, d)
        Y_traj        : (n_paths, n_steps+1, d_Y)  -- frozen at Y0 (or zeros if d_Y=0)
        terminal_in_S : (n_paths,) float indicator 1{xi_T in S}

    Y is held fixed at Y0 throughout the rollout — appropriate for one-day-
    ahead conditional forecasting where the context is observed at t=0.
    """
    nsde = nsde.to(device).eval()
    xi = xi0_batch.to(device).float()
    n_paths, d = xi.shape

    if nsde.d_Y > 0:
        if Y0_batch is None:
            raise ValueError("Y0_batch required when nsde.d_Y > 0")
        Y = Y0_batch.to(device).float()
    else:
        Y = None

    traj = torch.zeros(n_paths, n_steps + 1, d, device=device)
    traj[:, 0] = xi
    sqrt_dt = np.sqrt(dt)

    for k in range(n_steps):
        mu, L = nsde(xi, Y)
        z = torch.randn(n_paths, d, device=device)
        mu_norm = torch.linalg.norm(mu, dim=1, keepdim=True)
        L_norm = torch.linalg.matrix_norm(L, dim=(1, 2), keepdim=True).squeeze(-1)
        tame_mu = mu / (1 + mu_norm * sqrt_dt)
        tame_L = L / (1 + L_norm.unsqueeze(-1) * sqrt_dt)
        xi = xi + tame_mu * dt + torch.bmm(tame_L, z.unsqueeze(-1)).squeeze(-1) * sqrt_dt
        traj[:, k + 1] = xi

    if Y is not None:
        Y_traj = Y[:, None, :].expand(n_paths, n_steps + 1, nsde.d_Y).contiguous()
    else:
        Y_traj = None
    terminal_in_S = in_S(traj[:, -1]).float()
    return traj, Y_traj, terminal_in_S


def cdg_ml_loss(h_net: HFunction, traj: torch.Tensor,
                terminal_indicator: torch.Tensor,
                t_grid: torch.Tensor,
                Y_traj: torch.Tensor = None) -> torch.Tensor:
    """
    Martingale loss:
        L = E[ int_0^T (h(t, xi_t, Y) - 1{xi_T in S})^2 dt ]

    traj  : (n_paths, n_steps+1, d)
    Y_traj: (n_paths, n_steps+1, d_Y) or None
    """
    n_paths, n_steps_plus_1, d = traj.shape
    t_rep = t_grid[None, :].expand(n_paths, -1).reshape(-1)
    xi_rep = traj.reshape(-1, d)
    if Y_traj is not None:
        Y_rep = Y_traj.reshape(-1, Y_traj.shape[-1])
    else:
        Y_rep = None
    h_vals = h_net(t_rep, xi_rep, Y_rep).reshape(n_paths, n_steps_plus_1)
    target = terminal_indicator[:, None].expand(-1, n_steps_plus_1)
    return ((h_vals - target) ** 2).mean()


@dataclass
class CDGConfig:
    n_paths_per_epoch: int = 256
    n_epochs: int = 50
    n_steps: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    print_every: int = 5


def train_h(h_net: HFunction, nsde: ConstrainedNSDE,
            sample_xi0_Y0: Callable[[int], tuple],
            in_S: Callable[[torch.Tensor], torch.Tensor],
            T: float, cfg: CDGConfig = CDGConfig(),
            device: str = 'cpu') -> dict:
    """
    Train h via CDG-ML off-policy.

    Parameters
    ----------
    h_net : HFunction to train
    nsde  : pretrained constrained NSDE (frozen during this stage)
    sample_xi0_Y0 : callable returning (xi0, Y0) where
                    xi0 is (n, d) and Y0 is (n, d_Y) or None
    in_S : indicator function on R^d (B, d) -> (B,) bool, defining S
    T : terminal horizon
    """
    h_net = h_net.to(device)
    for p in nsde.parameters():
        p.requires_grad_(False)
    nsde.eval()

    opt = torch.optim.AdamW(h_net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    dt = T / cfg.n_steps
    t_grid = torch.linspace(0.0, T, cfg.n_steps + 1, device=device)
    history = {'loss': [], 'hit_rate': []}

    for epoch in range(cfg.n_epochs):
        out = sample_xi0_Y0(cfg.n_paths_per_epoch)
        if isinstance(out, tuple):
            xi0_np, Y0_np = out
        else:
            xi0_np, Y0_np = out, None
        xi0 = torch.from_numpy(xi0_np).float().to(device)
        Y0 = torch.from_numpy(Y0_np).float().to(device) if Y0_np is not None else None
        traj, Y_traj, term = simulate_trajectories_with_indicator(
            nsde, xi0, cfg.n_steps, dt, in_S,
            Y0_batch=Y0, device=device
        )
        loss = cdg_ml_loss(h_net, traj, term, t_grid, Y_traj=Y_traj)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h_net.parameters(), 5.0)
        opt.step()

        hit_rate = term.mean().item()
        history['loss'].append(loss.item())
        history['hit_rate'].append(hit_rate)
        if (epoch + 1) % cfg.print_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.5f}  "
                  f"unconditional hit rate={hit_rate:.4f}")

    return history


# ----------------------------------------------------------------------
# Guided simulation
# ----------------------------------------------------------------------

def simulate_guided(nsde: ConstrainedNSDE, h_net: HFunction,
                    xi0: np.ndarray, T: float, n_steps: int,
                    Y0: np.ndarray = None,
                    guidance_scale: float = 1.0,
                    rng_seed: int = 0,
                    device: str = 'cpu') -> np.ndarray:
    """
    Simulate the guided SDE
        d xi = [mu(xi, Y) + eta * a * grad_xi log h(t, xi, Y)] dt + sigma dW
    where eta = guidance_scale. Y is held fixed at Y0 (one-day-ahead context).

    grad_xi log h is computed by autograd on h_net.
    """
    nsde = nsde.to(device).eval()
    h_net = h_net.to(device).eval()

    single = (xi0.ndim == 1)
    if single:
        xi0 = xi0[None, :]
        if Y0 is not None and Y0.ndim == 1:
            Y0 = Y0[None, :]
    xi = torch.from_numpy(xi0).float().to(device)
    n_paths, d = xi.shape

    if nsde.d_Y > 0:
        if Y0 is None:
            raise ValueError("Y0 required when nsde.d_Y > 0")
        Y = torch.from_numpy(np.asarray(Y0)).float().to(device)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0).expand(n_paths, -1).contiguous()
    else:
        Y = None

    torch.manual_seed(rng_seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    out = torch.zeros(n_paths, n_steps + 1, d, device=device)
    out[:, 0] = xi

    for k in range(n_steps):
        t_k = torch.full((n_paths,), k * dt, device=device)
        xi_g = xi.detach().requires_grad_(True)
        log_h = h_net.log_h(t_k, xi_g, Y).sum()
        grad_log_h, = torch.autograd.grad(log_h, xi_g)
        grad_log_h = grad_log_h.detach()

        with torch.no_grad():
            mu, L = nsde(xi, Y)
            a = torch.bmm(L, L.transpose(1, 2))
            guidance = torch.bmm(a, grad_log_h.unsqueeze(-1)).squeeze(-1)
            full_drift = mu + guidance_scale * guidance

            z = torch.randn(n_paths, d, device=device)
            drift_norm = torch.linalg.norm(full_drift, dim=1, keepdim=True)
            L_norm = torch.linalg.matrix_norm(L, dim=(1, 2), keepdim=True).squeeze(-1)
            tame_drift = full_drift / (1 + drift_norm * sqrt_dt)
            tame_L = L / (1 + L_norm.unsqueeze(-1) * sqrt_dt)
            xi = xi + tame_drift * dt + torch.bmm(tame_L, z.unsqueeze(-1)).squeeze(-1) * sqrt_dt
            out[:, k + 1] = xi

    arr = out.cpu().numpy()
    return arr[0] if single else arr
