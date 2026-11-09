"""Regression Monte Carlo reference estimates for nonlinear default risk.

Provides backward regression (GLW-style) and forward Picard (Bender--Denk-style)
schemes for the configured minimum-payoff problem. Neither routine is a neural
deep-BSDE baseline. Their minimum-only polynomial basis is an approximation;
reported path standard errors omit regression and time-discretisation bias.

Importing this module does not simulate paths or train models. The CLI defaults
to the no-jump case, with 400,000 paths and 1,000 backward/100 Picard time steps.
Use --config to take equation parameters from a Black--Scholes experiment YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import PDEConfig


def _r_fn(u_arr: np.ndarray, cfg) -> np.ndarray:
    """r(u) = (1-δ)Q(u) + R — path-dependent discount rate."""
    slope = (cfg.gamma_h - cfg.gamma_l) / (cfg.v_h - cfg.v_l)
    Q = np.where(
        u_arr < cfg.v_h, cfg.gamma_h,
        np.where(u_arr >= cfg.v_l, cfg.gamma_l,
                 cfg.gamma_h + slope * (u_arr - cfg.v_h))
    )
    return (1.0 - cfg.delta_recovery) * Q + cfg.R_rate


def _poly_features(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Ten degree-three polynomial features in (min_X, t).

    This is a reduced regression basis, not an exact state reduction.
    Exchangeability does not imply that the current minimum determines
    the continuation value of the minimum terminal payoff.
    """
    return np.column_stack([
        np.ones_like(m),
        m,      t,
        m**2,   m*t,   t**2,
        m**3,   m**2*t, m*t**2, t**3,
    ])


def _simulate_paths(cfg, x0, n_paths, n_steps, rng):
    """
    Simulate n_paths antithetic GBM+jump paths.
    Returns min_X [n_paths, n_steps+1] and dW [n_paths, n_steps].

    dW is stored for Z_t estimation (needed in full GLW, optional here).
    """
    d      = cfg.dimen
    T      = cfg.T
    dt     = T / n_steps
    n_half = n_paths // 2

    min_X = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    dW    = np.empty((n_paths, n_steps),     dtype=np.float32)
    min_X[:, 0] = x0

    X = np.full((n_paths, d), x0, dtype=np.float32)

    for s in range(n_steps):
        Z     = rng.standard_normal((n_half, d)).astype(np.float32)
        Z_all = np.concatenate([Z, -Z], axis=0)          # antithetic

        X *= np.exp(
            (cfg.mu_bar_adjusted - 0.5 * cfg.sigma_bar**2) * dt
            + cfg.sigma_bar * np.sqrt(dt) * Z_all
        )

        # scalar dW for Z_t regression  (avg over assets, proxy)
        dW[:, s] = Z_all.mean(axis=1) * np.sqrt(dt)

        if cfg.use_jumps and cfg.jump_intensity > 0:
            mask  = (rng.uniform(size=(n_paths, 1)) < cfg.jump_intensity * dt
                     ).astype(np.float32)
            xi_c  = rng.standard_normal((n_paths, 1)).astype(np.float32)
            xi_i  = rng.standard_normal((n_paths, d)).astype(np.float32)
            log_J = cfg.jump_mean + cfg.jump_std * (
                np.sqrt(cfg.jump_corr)       * xi_c
                + np.sqrt(1. - cfg.jump_corr) * xi_i
            )
            X *= np.exp(log_J * mask)

        min_X[:, s + 1] = X.min(axis=1)

    del X
    return min_X, dW


# ══════════════════════════════════════════════════════════════════════
# Algorithm 1 — GLW (2005)
# Gobet, Lemor, Warin — true backward induction
# ══════════════════════════════════════════════════════════════════════

def glw_bsde(
    cfg,
    x0:            float = 100.0,
    n_paths:       int   = 400_000,
    n_steps:       int   = 1000,
    n_reg_per_step: int  = 4_000,
    seed:          int   = 42,
) -> tuple[float, float]:
    """
    GLW backward induction (Gobet-Lemor-Warin 2005).

    At each time step k (backward from T to 0):

      1. Build regression matrix A_k from basis(min_X_k, t_k)
      2. Fit:  Y_{k+1} ≈ A_k @ β_k   (least squares)
         → gives Ê[Y_{k+1} | X_k]  for all paths
      3. Apply driver explicitly:
         Y_k = Ê[Y_{k+1}|X_k]  -  r(Ê[Y_{k+1}|X_k]) · Ê[Y_{k+1}|X_k] · dt

    No global iteration — the nonlinearity is handled locally at each step.
    Computational cost: O(n_paths × n_steps) with small constant.
    """
    assert n_paths % 2 == 0
    rng    = np.random.default_rng(seed)
    T      = cfg.T
    dt     = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"  GLW (Gobet-Lemor-Warin 2005) — backward induction")
    print(f"  n_paths={n_paths:,}  n_steps={n_steps}  dt={dt:.4f}")
    print(f"{'='*60}")

    min_X, dW = _simulate_paths(cfg, x0, n_paths, n_steps, rng)
    payoff = min_X[:, -1].copy()

    print(f"  payoff: mean={payoff.mean():.4f}  "
          f"p5={np.percentile(payoff,5):.2f}  "
          f"p10={np.percentile(payoff,10):.2f}")

    # ── Backward pass ──────────────────────────────────────────────
    # Y[i] = current estimate of u(t_k, X_{i,t_k})
    Y = payoff.copy()   # initialise at terminal: Y_n = φ(X_T)

    for k in range(n_steps - 1, -1, -1):
        t_k = float(t_grid[k])

        # sub-sample for regression (keeps memory bounded)
        idx = rng.integers(0, n_paths, n_reg_per_step)
        A   = _poly_features(min_X[idx, k],
                             np.full(n_reg_per_step, t_k, dtype=np.float32))
        coeffs, _, _, _ = np.linalg.lstsq(A, Y[idx], rcond=None)

        # conditional expectation on ALL paths
        A_full  = _poly_features(min_X[:, k],
                                 np.full(n_paths, t_k, dtype=np.float32))
        E_Y_next = np.clip(A_full @ coeffs, 0.0, 2.0 * x0)

        # apply driver: Y_k = E[Y_{k+1}|X_k] - r(E[Y_{k+1}|X_k])·E[Y_{k+1}|X_k]·dt
        # (explicit Euler — driver evaluated at the regressed value)
        Y = E_Y_next - _r_fn(E_Y_next, cfg) * E_Y_next * dt

    # Y now holds estimates of u(0, X_{i,0}) = u(0, x0) for all paths
    u_val  = float(Y.mean())
    se_val = float(Y.std() / np.sqrt(n_paths))

    print(f"\n  >>> u(0, x₀) = {u_val:.4f} ± {se_val:.4f}")
    print(f"  >>> 95% CI:    [{u_val-1.96*se_val:.4f}, {u_val+1.96*se_val:.4f}]")
    return u_val, se_val


# ══════════════════════════════════════════════════════════════════════
# Algorithm 2 — Bender-Denk (2008)
# Forward Picard on the Feynman-Kac representation
# 
# ══════════════════════════════════════════════════════════════════════

def bender_denk_picard(
    cfg,
    x0:             float = 100.0,
    n_paths:        int   = 400_000,
    n_steps:        int   = 100,
    n_picard:       int   = 30,
    n_reg_per_step: int   = 4_000,
    seed:           int   = 42,
) -> tuple[float, float]:
    """
    Bender-Denk (2008) forward Picard scheme.

    Iterates on the Feynman-Kac representation:

      u^{k+1}(0, x₀) = E[ φ(X_T) · exp(-∫₀ᵀ r(u^k(s,Xₛ)) ds) ]

    At each Picard step:
      1. Compute path-dependent discount using current u^k along paths
      2. Update global estimate u^{k+1}
      3. Fit new u^{k+1}(s,x) via backward-pass regression
         (backward only to build the regression targets, not to price)

    Key difference from GLW: this is a GLOBAL fixed-point iteration
    on the full trajectory, not a LOCAL step-by-step backward induction.
    This reduced regression approximation must be checked for sensitivity
    to time discretisation, regression basis, path count and iteration count.
    """
    assert n_paths % 2 == 0
    rng    = np.random.default_rng(seed)
    T      = cfg.T
    dt     = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"  Bender-Denk (2008) — forward Picard iteration")
    print(f"  n_paths={n_paths:,}  n_steps={n_steps}  n_picard={n_picard}")
    print(f"{'='*60}")

    min_X, _ = _simulate_paths(cfg, x0, n_paths, n_steps, rng)
    payoff    = min_X[:, -1].copy()

    print(f"  payoff: mean={payoff.mean():.4f}  "
          f"p5={np.percentile(payoff,5):.2f}  "
          f"p10={np.percentile(payoff,10):.2f}")

    # Initialise u^0 as constant
    u_path = np.full((n_paths, n_steps + 1), float(payoff.mean()),
                     dtype=np.float32)

    print(f"\n  {'k':>3}   {'u':>10}   {'SE':>8}   {'|Δu|':>9}   r̄")

    u_val  = np.nan
    se_val = np.nan
    u_prev = np.inf

    for k in range(n_picard):

        # ── Forward pass: compute discount ──
        u_mid  = 0.5 * (u_path[:, :-1] + u_path[:, 1:])
        r_mid  = _r_fn(u_mid, cfg)
        log_d  = -(r_mid * dt).sum(axis=1)
        samples = payoff * np.exp(log_d)

        u_val  = float(samples.mean())
        se_val = float(samples.std() / np.sqrt(n_paths))
        r_bar  = float(r_mid.mean())
        delta  = abs(u_val - u_prev)
        u_prev = u_val

        print(f"  {k+1:>3}   {u_val:>10.5f}   {se_val:>8.5f}   "
              f"{delta:>9.5f}   {r_bar:.5f}")

        if delta < 5e-5 and k >= 1:
            print("  ✅ Converged!")
            break

        # ── Backward pass: build regression targets ──
        disc_back = np.ones(n_paths, dtype=np.float32)
        reg_m, reg_t, reg_g = [], [], []

        for s in range(n_steps, -1, -1):
            if s < n_steps:
                disc_back *= np.exp(-_r_fn(u_path[:, s], cfg) * dt)
            target_s = payoff * disc_back
            idx = rng.integers(0, n_paths, n_reg_per_step)
            reg_m.append(min_X[idx, s])
            reg_t.append(np.full(n_reg_per_step, t_grid[s], dtype=np.float32))
            reg_g.append(target_s[idx])

        A      = _poly_features(np.concatenate(reg_m), np.concatenate(reg_t))
        coeffs, _, _, _ = np.linalg.lstsq(A, np.concatenate(reg_g), rcond=None)

        for s in range(n_steps + 1):
            A_s = _poly_features(
                min_X[:, s],
                np.full(n_paths, t_grid[s], dtype=np.float32)
            )
            u_path[:, s] = np.clip(A_s @ coeffs, 0.0, 2.0 * x0)

    print(f"\n  >>> u(0, x₀) = {u_val:.4f} ± {se_val:.4f}")
    print(f"  >>> 95% CI:    [{u_val-1.96*se_val:.4f}, {u_val+1.96*se_val:.4f}]")
    return u_val, se_val


def main(argv=None):
    """Run reference routines explicitly, optionally across independent seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Black--Scholes experiment YAML")
    parser.add_argument("--method", choices=("both", "glw", "picard"), default="both")
    parser.add_argument("--n-paths", type=int, default=400_000)
    parser.add_argument("--n-steps", type=int, help="Override the time-step count for both methods")
    parser.add_argument("--glw-steps", type=int, default=1000)
    parser.add_argument("--picard-steps", type=int, default=100)
    parser.add_argument("--n-picard", type=int, default=30)
    parser.add_argument("--n-reg-per-step", type=int, default=4_000)
    parser.add_argument("--x0", type=float, help="Initial asset value (from YAML, otherwise 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)
    if args.n_paths < 2 or args.n_paths % 2:
        parser.error("--n-paths must be a positive even number for antithetic sampling.")
    for name in ("glw_steps", "picard_steps", "n_picard", "n_reg_per_step", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.n_steps is not None and args.n_steps < 1:
        parser.error("--n-steps must be positive.")
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.config is None:
        cfg = PDEConfig(use_jumps=False, jump_intensity=0.1, jump_mean=-0.2,
                        jump_std=0.15, jump_corr=0.5, v_h=50.0, v_l=70.0)
        x0 = 100.0
    else:
        from ..configuration import from_document, read_yaml, validate
        run = from_document(read_yaml(args.config), experiment="black_scholes")
        validate(run)
        cfg = run.parameters.problem
        x0 = run.parameters.evaluation.x0_value
    if args.x0 is not None:
        x0 = args.x0
    if not np.isfinite(x0) or x0 <= 0:
        parser.error("--x0 must be a finite positive value.")

    records = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        shared = dict(x0=x0, n_paths=args.n_paths,
                      n_reg_per_step=args.n_reg_per_step, seed=seed)
        if args.method in ("both", "glw"):
            value, se = glw_bsde(cfg, n_steps=args.n_steps or args.glw_steps, **shared)
            records.append(dict(method="glw", seed=seed, estimate=value, path_se=se))
        if args.method in ("both", "picard"):
            value, se = bender_denk_picard(
                cfg, n_steps=args.n_steps or args.picard_steps,
                n_picard=args.n_picard, **shared
            )
            records.append(dict(method="picard", seed=seed, estimate=value, path_se=se))
    print("\nReference estimates (not a validated ground truth):")
    for row in records:
        print(f"  {row['method']:7s} seed={row['seed']} "
              f"estimate={row['estimate']:.6f} path_SE={row['path_se']:.6f}")
    print("Path SEs omit regression, discretisation and fixed-point bias. "
          "Use repeated runs and convergence checks.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    main()
