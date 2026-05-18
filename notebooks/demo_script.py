"""
End-to-end demo of CONTEXT-CONDITIONED arbitrage-free forecasting on Heston.

The forecaster:
    d xi_t = mu(xi_t, Y_t) dt + sigma(xi_t, Y_t) dW_t
where Y_t = exogenous context (EWMAs of xi, realized log-S variance).

Held-out test trajectory (different seed) for proper forecast evaluation.
Compares:
    (a) no-context (Markov-in-xi) baseline
    (b) context-conditioned NSDE
    (c) random-walk sanity baseline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import numpy as np
import torch
from src import heston, constraints, factors, context, nsde, cdg


def simulate_data(seed, params, T_horizon, n_steps, taus_flat, m_flat):
    """Return one Heston trajectory (S, v) plus the call surface time series."""
    S, V = heston.simulate_heston(params, T_horizon, n_steps, n_paths=1,
                                  rng=np.random.default_rng(seed))
    S, V = S[0], V[0]
    C = np.zeros((n_steps + 1, taus_flat.size))
    for k in range(n_steps + 1):
        C[k] = heston.price_surface(S[k], V[k], taus_flat, m_flat, params)
    return S, V, C


def main():
    # 1. data
    print("=" * 64)
    print("1. Train + test Heston trajectories")
    print("=" * 64)
    params = heston.HestonParams()
    T_horizon, n_steps = 1.0, 500
    dt = T_horizon / n_steps
    taus = np.array([0.05, 0.1, 0.25, 0.5])
    moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    taus_flat, m_flat = heston.build_option_grid(taus, moneyness)
    S_tr, V_tr, C_tr = simulate_data(42, params, T_horizon, n_steps,
                                     taus_flat, m_flat)
    S_te, V_te, C_te = simulate_data(43, params, T_horizon, n_steps,
                                     taus_flat, m_flat)
    print(f"Train C: {C_tr.shape},  Test C: {C_te.shape}")

    # 2. constraints + factor decoding on train
    print("\n" + "=" * 64)
    print("2. Constraints + factor decoding (TRAIN only)")
    print("=" * 64)
    A, b_arb = constraints.build_constraints(taus, moneyness)
    A, b_arb = constraints.remove_redundant(A, b_arb)
    fm = factors.decode_factors(C_tr, d=2, A=A, b_arb=b_arb)
    fm = factors.normalize_factors(fm, target_std=1.0)
    print(f"Polytope V: {fm.V.shape}")

    # Project test data into the same basis (least-squares)
    xi_test = np.linalg.lstsq(fm.G.T, (C_te - fm.G0).T, rcond=None)[0].T
    print(f"Test xi: {xi_test.shape}")

    # 3. context
    print("\n" + "=" * 64)
    print("3. Context vectors")
    print("=" * 64)
    ctx_builder = context.ContextBuilder()
    Y_tr_raw = ctx_builder.fit_transform(fm.xi, S_tr)
    Y_te_raw = ctx_builder.fit_transform(xi_test, S_te)
    Y_tr, Y_te, ctx_stats = context.normalize_context(Y_tr_raw, Y_te_raw)
    d_Y = Y_tr.shape[1]
    print(f"d_Y = {d_Y}")

    # 4. pair masks
    pair_mask_tr = factors.valid_pair_mask(fm, tol=1e-6)
    fm_test = factors.FactorModel(G0=fm.G0, G=fm.G, d=2, xi=xi_test,
                                  V=fm.V, b=fm.b, A_orig=A, b_orig=b_arb,
                                  scale=fm.scale)
    pair_mask_te = factors.valid_pair_mask(fm_test, tol=1e-6)
    print(f"Train pairs: {pair_mask_tr.sum()}/{len(pair_mask_tr)}, "
          f"Test pairs: {pair_mask_te.sum()}/{len(pair_mask_te)}")

    xi_t_tr = fm.xi[:-1][pair_mask_tr]
    xi_tp1_tr = fm.xi[1:][pair_mask_tr]
    Y_t_tr = Y_tr[:-1][pair_mask_tr]
    xi_t_te_np = xi_test[:-1][pair_mask_te]
    xi_tp1_te_np = xi_test[1:][pair_mask_te]
    Y_t_te_np = Y_te[:-1][pair_mask_te]

    # 5. auto-tune rho_star
    slacks = fm.xi @ fm.V.T - fm.b[None, :]
    v_norms = np.linalg.norm(fm.V, axis=1)
    d_min = (slacks / v_norms[None, :]).min(axis=1)
    d_min_pos = d_min[d_min > 0]
    rho_star = max(1e-3, 0.5 * np.percentile(d_min_pos, 5))
    print(f"rho_star = {rho_star:.4f}")
    center, cheb_r = nsde.chebyshev_center(fm.V, fm.b)

    # 6. train
    print("\n" + "=" * 64)
    print("6. Training")
    print("=" * 64)
    cfg = nsde.TrainConfig(n_epochs=80, batch_size=64, lr=3e-3,
                            print_every=20)

    print("\n[Model A] No-context NSDE:")
    torch.manual_seed(0)
    model_no = nsde.ConstrainedNSDE(d=2, V=fm.V, b=fm.b, center=center,
                                     rho_star=rho_star, d_Y=0,
                                     width=32, depth=2,
                                     boundary_beta=5.0)
    hist_no = nsde.train_nsde(model_no, xi_t_tr, xi_tp1_tr, dt, cfg=cfg)

    print(f"\n[Model B] Context NSDE (d_Y={d_Y}):")
    torch.manual_seed(0)
    # Smaller width for context model to avoid overfitting the d_Y=5 extra inputs
    model_c = nsde.ConstrainedNSDE(d=2, V=fm.V, b=fm.b, center=center,
                                    rho_star=rho_star, d_Y=d_Y,
                                    width=16, depth=2,
                                    boundary_beta=5.0)
    hist_c = nsde.train_nsde(model_c, xi_t_tr, xi_tp1_tr, dt,
                              Y_t=Y_t_tr, cfg=cfg)

    # 7. test evaluation
    print("\n" + "=" * 64)
    print("7. Test-set forecast evaluation")
    print("=" * 64)
    xi_t_te = torch.from_numpy(xi_t_te_np).float()
    xi_tp1_te = torch.from_numpy(xi_tp1_te_np).float()
    Y_t_te = torch.from_numpy(Y_t_te_np).float()

    model_no.eval()
    model_c.eval()
    with torch.no_grad():
        mu_no, L_no = model_no(xi_t_te, None)
        nll_no = nsde.euler_neg_log_likelihood(xi_t_te, xi_tp1_te,
                                                mu_no, L_no, dt,
                                                sigma_floor=1e-3).item()
        mu_c, L_c = model_c(xi_t_te, Y_t_te)
        nll_c = nsde.euler_neg_log_likelihood(xi_t_te, xi_tp1_te,
                                               mu_c, L_c, dt,
                                               sigma_floor=1e-3).item()

    # Random-walk baseline
    diff_train = xi_tp1_tr - xi_t_tr
    sigma_emp_var = diff_train.var(axis=0) / dt
    diff_test = xi_tp1_te_np - xi_t_te_np
    rw_quad = ((diff_test ** 2) / (sigma_emp_var[None, :] * dt)).sum(axis=1)
    rw_logdet = np.log(sigma_emp_var * dt).sum()
    rw_nll = 0.5 * (2 * np.log(2 * np.pi) + rw_logdet + rw_quad).mean()

    print(f"\nTest one-step NLL (lower better):")
    print(f"  Random-walk baseline:     {rw_nll:>8.4f}")
    print(f"  No-context NSDE  (A):     {nll_no:>8.4f}")
    print(f"  Context NSDE     (B):     {nll_c:>8.4f}")

    # Price MAPE
    def price_mape(model, Y):
        with torch.no_grad():
            mu, L = model(xi_t_te, Y)
            xi_pred = (xi_t_te + mu * dt).numpy()
        c_pred = fm.G0[None, :] + xi_pred @ fm.G
        c_true = fm.G0[None, :] + xi_tp1_te_np @ fm.G
        mask_liq = np.abs(c_true) > 1e-3
        return np.mean(np.abs(c_pred[mask_liq] - c_true[mask_liq]) /
                       np.abs(c_true[mask_liq]))

    mape_no = price_mape(model_no, None)
    mape_c = price_mape(model_c, Y_t_te)
    c_pred_rw = fm.G0[None, :] + xi_t_te_np @ fm.G
    c_true = fm.G0[None, :] + xi_tp1_te_np @ fm.G
    mask_liq = np.abs(c_true) > 1e-3
    mape_rw = np.mean(np.abs(c_pred_rw[mask_liq] - c_true[mask_liq]) /
                      np.abs(c_true[mask_liq]))

    print(f"\nTest one-step price MAPE on liquid options:")
    print(f"  Random-walk baseline:     {mape_rw*100:>7.4f}%")
    print(f"  No-context NSDE  (A):     {mape_no*100:>7.4f}%")
    print(f"  Context NSDE     (B):     {mape_c*100:>7.4f}%")

    # 8. structural checks
    print("\n" + "=" * 64)
    print("8. Structural checks (polytope, arbitrage)")
    print("=" * 64)
    xi0 = np.tile(xi_test[0], (50, 1))
    Y0 = np.tile(Y_te[0], (50, 1))
    sim_no = nsde.simulate(model_no, xi0, n_steps=100, dt=dt,
                            sigma_floor=1e-3, clip_to_polytope=True)
    sim_c = nsde.simulate(model_c, xi0, n_steps=100, dt=dt,
                           Y0=Y0, sigma_floor=1e-3, clip_to_polytope=True)
    for name, tr in [("A", sim_no), ("B", sim_c)]:
        flat = tr.reshape(-1, 2)
        slacks_ = flat @ fm.V.T - fm.b[None, :]
        n_out = (slacks_.min(axis=1) < -1e-8).sum()
        xi_T = tr[:, -1, :]
        c_T = fm.G0[None, :] + xi_T @ fm.G
        n_arb = sum(1 for c in c_T if not constraints.is_arbitrage_free(A, b_arb, c))
        print(f"  Model {name}:  polytope OK {len(flat)-n_out}/{len(flat)},  "
              f"terminal NA OK {len(c_T)-n_arb}/{len(c_T)}")

    # 9. CDG guidance using context-conditioned base
    print("\n" + "=" * 64)
    print("9. CDG-ML guidance with context")
    print("=" * 64)
    atm_short_idx = np.argmin(np.abs(taus_flat - 0.05) + np.abs(m_flat - 1.0))
    G_atm = fm.G[:, atm_short_idx]
    G0_atm = float(fm.G0[atm_short_idx])
    VIX_train = G0_atm + fm.xi @ G_atm
    q90 = float(np.quantile(VIX_train, 0.9))
    G_atm_t = torch.from_numpy(G_atm).float()

    def in_S(xi_b):
        return (G0_atm + xi_b @ G_atm_t) > q90

    valid_xi = fm.xi[:-1][pair_mask_tr]
    valid_Y = Y_tr[:-1][pair_mask_tr]

    def sample_xi0_Y0(n):
        idxs = np.random.choice(len(valid_xi), size=n, replace=True)
        return valid_xi[idxs], valid_Y[idxs]

    torch.manual_seed(7)
    h_net = cdg.HFunction(d=2, d_Y=d_Y, width=32, depth=2)
    T_cdg = 30 * dt
    n_cdg = 30
    cfg_cdg = cdg.CDGConfig(n_paths_per_epoch=256, n_epochs=30,
                             n_steps=n_cdg, lr=3e-3, print_every=10)
    cdg.train_h(h_net, model_c, sample_xi0_Y0, in_S, T_cdg, cfg_cdg)

    xi_start = np.tile(xi_test[0], (200, 1))
    Y_start = np.tile(Y_te[0], (200, 1))
    unguided = nsde.simulate(model_c, xi_start, n_cdg, dt,
                              rng_seed=11, Y0=Y_start, sigma_floor=1e-3)
    guided = cdg.simulate_guided(model_c, h_net, xi_start, T_cdg, n_cdg,
                                  Y0=Y_start, guidance_scale=1.0, rng_seed=11)
    guided3 = cdg.simulate_guided(model_c, h_net, xi_start, T_cdg, n_cdg,
                                   Y0=Y_start, guidance_scale=3.0, rng_seed=11)

    print(f"\n{'Sampler':22s} {'mean VIX':>10s} {'P(VIX>q90)':>12s}")
    for name, sim in [("Unguided", unguided), ("Guided eta=1", guided),
                      ("Guided eta=3", guided3)]:
        vix = G0_atm + sim[:, -1, :] @ G_atm
        print(f"{name:22s} {vix.mean():>10.4f} {(vix>q90).mean():>11.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
