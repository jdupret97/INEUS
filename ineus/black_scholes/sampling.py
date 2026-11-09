"""Collocation-point sampling: uniform and Residual-Adaptive Distribution."""

import numpy as np
from .problem import residual_pointwise


# ==============================================================
# Uniform samplers 
# ==============================================================

def sample_xt(num_points, T, sample_b, dimen):
    """Uniform sample from [0, sample_b]^d × [0, T]."""
    t = np.random.uniform(0.0, T, num_points).reshape(-1, 1).astype(np.float32)
    x = np.random.uniform(0.0, sample_b, (num_points, dimen)).astype(np.float32)
    return x, t


def sample_x_terminal(num_points, sample_b, dimen):
    """Sample spatial points for terminal-condition loss."""
    return np.random.uniform(0.0, sample_b, (num_points, dimen)).astype(np.float32)


def rad_resample(model, pde_cfg, train_cfg, eval_batch_size=2048,
                 sample_fn=None):
    """Residual-Adaptive Distribution (RAD) resampling.

    Parameters
    ----------
    sample_fn : callable(num_points) -> (x, t), optional
        If None, uses uniform sample_xt.
        Pass a custom function to change candidate generation.
    """
    T          = pde_cfg.T
    dimen      = pde_cfg.dimen
    num_points = train_cfg.num_samples
    cand_size  = train_cfg.rad_candidate_size
    k, c      = train_cfg.rad_k, train_cfg.rad_c
    sample_b   = train_cfg.sample_b
    eps        = 1e-12

    # --- generate candidates ---
    if sample_fn is not None:
        x_cand, t_cand = sample_fn(cand_size)
    else:
        x_cand, t_cand = sample_xt(cand_size, T, sample_b, dimen)

    # --- evaluate residuals ---
    res_abs = np.empty(cand_size, dtype=np.float32)
    for lo in range(0, cand_size, eval_batch_size):
        hi = min(lo + eval_batch_size, cand_size)
        r, _ = residual_pointwise(model, x_cand[lo:hi], t_cand[lo:hi], pde_cfg)
        res_abs[lo:hi] = np.abs(r.numpy())

    scores = np.power(np.maximum(res_abs, eps), k)
    scores = scores / (scores.mean() + eps) + c
    probs  = scores / scores.sum()

    idx = np.random.choice(cand_size, size=num_points, replace=False, p=probs)

    stats = dict(mean_residual=float(res_abs.mean()),
                 max_residual=float(res_abs.max()),
                 min_residual=float(res_abs.min()))
    return x_cand[idx], t_cand[idx], stats


# ==============================================================
# Path-based samplers (optional)
# ==============================================================

def sample_xt_paths(num_points, T, x0_value, mu, sigma, dimen):
    """Sample (x,t) along GBM paths from x₀.

    X_t = x₀ · exp((μ - σ²/2)t + σ√t·Z)
    """
    t = np.random.uniform(0.0, T, num_points).reshape(-1, 1).astype(np.float32)
    Z = np.random.randn(num_points, dimen).astype(np.float32)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t) * Z
    x = x0_value * np.exp(drift + diffusion)
    return x.astype(np.float32), t


def sample_x_terminal_paths(num_points, T, x0_value, mu, sigma, dimen):
    """Terminal spatial points along GBM paths at t=T."""
    Z = np.random.randn(num_points, dimen).astype(np.float32)
    drift = (mu - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    x = x0_value * np.exp(drift + diffusion)
    return x.astype(np.float32)


def make_path_sampler(T, x0_value, mu, sigma, dimen):
    """Return a sample_fn compatible with rad_resample's interface.

    Usage
    -----
        sampler = make_path_sampler(T, 100.0, 0.02, 0.20, 100)
        rad_resample(model, pde_cfg, train_cfg, sample_fn=sampler)
    """
    def _sample(num_points):
        return sample_xt_paths(num_points, T, x0_value, mu, sigma, dimen)
    return _sample




def sample_x_terminal_with_jumps(num_points, T, x0_value, mu, sigma, dimen,
                                   jump_mean=-0.5, jump_std=0.15, jump_corr=0.5,
                                   jump_frac=0.5):
    """Terminal points: GBM endpoints blended with post-jump endpoints."""
    n_jump = int(jump_frac * num_points)
    n_gbm  = num_points - n_jump

    Z = np.random.randn(num_points, dimen).astype(np.float32)
    X = x0_value * np.exp(
        (mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z
    )
    xi_c = np.random.randn(n_jump, 1).astype(np.float32)
    xi_i = np.random.randn(n_jump, dimen).astype(np.float32)
    log_J = jump_mean + jump_std * (
        np.sqrt(jump_corr) * xi_c + np.sqrt(1.0 - jump_corr) * xi_i
    )
    X[n_gbm:] *= np.exp(log_J)

    return X[np.random.permutation(num_points)]


def sample_xt_with_jumps(num_points, T, x0_value, mu, sigma, dimen,
                          jump_mean=-0.5, jump_std=0.15, jump_corr=0.5,
                          jump_frac=0.5):
    """Blend GBM paths with post-jump states.

    The jump term λ·[u(t,x·eᶻ) - u(t,x)] requires the model to be
    accurate at x·eᶻ, which with μ_J=-0.5 lands near x*0.61 — a region
    GBM-only sampling almost never visits. This blends in those states.
    """
    n_jump = int(jump_frac * num_points)
    n_gbm  = num_points - n_jump

    # --- standard GBM paths ---
    t_all = np.random.uniform(0, T, num_points).reshape(-1, 1).astype(np.float32)
    Z_gbm = np.random.randn(num_points, dimen).astype(np.float32)
    X_all = x0_value * np.exp(
        (mu - 0.5*sigma**2)*t_all + sigma*np.sqrt(t_all)*Z_gbm
    )

    # --- apply a jump to the second half to get post-jump states ---
    xi_c = np.random.randn(n_jump, 1).astype(np.float32)
    xi_i = np.random.randn(n_jump, dimen).astype(np.float32)
    log_J = jump_mean + jump_std * (
        np.sqrt(jump_corr) * xi_c + np.sqrt(1.0 - jump_corr) * xi_i
    )
    X_all[n_gbm:] *= np.exp(log_J)

    idx = np.random.permutation(num_points)
    return X_all[idx], t_all[idx]
