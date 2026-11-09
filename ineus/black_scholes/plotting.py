"""Prediction trajectories and sample-path plots for Black--Scholes runs.

The checkpoint loader uses pickle: load only checkpoints you trust. Importing
this module does not create figures, change the plotting backend or load a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 7,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
        "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    })
    return plt


def _call_model(model, t_vals, x_vals):
    import tensorflow as tf

    t_tf = tf.constant(t_vals, dtype=tf.float32)
    x_tf = tf.constant(x_vals, dtype=tf.float32)
    return model((x_tf, t_tf), training=False).numpy().ravel()


# ══════════════════════════════════════════════════════
# 1(b)  Prediction trajectory
# ══════════════════════════════════════════════════════
def plot_1b(history, u_ref=55.810, save_path="fig1b_prediction.png"):
    """Plot student predictions at each completed distillation block."""
    plt = _pyplot()
    epochs = np.asarray(history["distill_epoch"])
    preds  = np.asarray(history["student_pred"])

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(epochs, preds, "o-", color="navy", ms=3, lw=1.2,
            label=r"$u(0,\,x_0)$")
    if u_ref is not None:
        ax.axhline(u_ref, color="firebrick", ls="--", lw=1, alpha=0.7,
                   label=f"Reference = {u_ref:.3f}")

    ax.set_xlabel("Epoch k")
    ax.legend(loc="best", framealpha=0.85, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════
# 2(a)  Solution along GBM sample paths
# ══════════════════════════════════════════════════════
def plot_2a(model, T, dimen, mu, sigma, x0_val,
            n_paths=50, n_steps=200, eval_batch=2048,
            save_path="fig2a_solution.png", pde_cfg=None, seed=42):
    """Plot model values along GBM paths with optional Bernoulli jump steps."""
    if min(n_paths, n_steps, eval_batch) < 1:
        raise ValueError("n_paths, n_steps and eval_batch must be positive.")
    plt = _pyplot()
    dt     = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1).astype(np.float32)

    rng = np.random.RandomState(seed)
    X   = np.zeros((n_paths, n_steps + 1, dimen), dtype=np.float32)
    X[:, 0, :] = x0_val
    
    use_jumps = pde_cfg is not None and getattr(pde_cfg, 'use_jumps', False)
    
    for i in range(n_steps):
        # log-Euler for GBM (more stable than arithmetic Euler)
        Z  = rng.randn(n_paths, dimen).astype(np.float32)
        X[:, i+1, :] = X[:, i, :] * np.exp(
            (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )
    
        # Bernoulli approximation: at most one jump event per time step
        if use_jumps and pde_cfg.jump_intensity > 0:
            lam   = pde_cfg.jump_intensity
            mu_J  = pde_cfg.jump_mean
            sig_J = pde_cfg.jump_std
            rho_J = pde_cfg.jump_corr
    
            mask  = (rng.uniform(size=(n_paths, 1)) < lam * dt).astype(np.float32)
            xi_c  = rng.randn(n_paths, 1).astype(np.float32)
            xi_i  = rng.randn(n_paths, dimen).astype(np.float32)
            log_J = mu_J + sig_J * (
                np.sqrt(rho_J) * xi_c + np.sqrt(1. - rho_J) * xi_i)
            X[:, i+1, :] *= np.exp(log_J * mask)
    

    all_x = X.reshape(-1, dimen)
    all_t = np.tile(t_grid, n_paths).reshape(-1, 1)

    all_u = []
    for s in range(0, len(all_x), eval_batch):
        e = min(s + eval_batch, len(all_x))
        all_u.append(_call_model(model, all_t[s:e], all_x[s:e]))
    all_u = np.concatenate(all_u).reshape(n_paths, n_steps + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    cmap = plt.cm.viridis
    for j in range(n_paths):
        col = cmap(j / max(n_paths - 1, 1))
        ax.plot(t_grid, all_u[j], color=col, alpha=0.8, lw=1.3,
                label=f"Path {j + 1}" if n_paths <= 10 else "")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u(t,\, X_t)$")
    if n_paths <= 10:
        ax.legend(fontsize=7, ncol=2, loc="best",
                  framealpha=0.85, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


def main(argv=None):
    """Rebuild the configured architecture and plot a trusted checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted model checkpoint (.pkl)")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/black_scholes"))
    parser.add_argument("--n-paths", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42, help="Sample-path seed")
    parser.add_argument("--u-ref", type=float, help="Override the plotted reference value")
    args = parser.parse_args(argv)
    if min(args.n_paths, args.n_steps, args.eval_batch) < 1:
        parser.error("--n-paths, --n-steps and --eval-batch must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32).")

    from ..configuration import from_document, read_yaml, validate
    from .models import build_model
    from .trainer import load_checkpoint

    run = from_document(read_yaml(args.config), experiment="black_scholes")
    validate(run)
    config = run.parameters
    model = build_model(config.model, config.problem, config.train)
    checkpoint = load_checkpoint(args.checkpoint)
    model.set_weights(checkpoint["model_weights"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.u_ref if args.u_ref is not None else config.evaluation.u_ref
    plot_1b(checkpoint["history"], u_ref=reference,
            save_path=args.output_dir / "fig1b_prediction.png")
    plot_2a(model, T=config.problem.T, dimen=config.problem.dimen,
            mu=config.problem.mu_bar_adjusted, sigma=config.problem.sigma_bar,
            x0_val=config.evaluation.x0_value, pde_cfg=config.problem,
            n_paths=args.n_paths, n_steps=args.n_steps, eval_batch=args.eval_batch,
            seed=args.seed, save_path=args.output_dir / "fig2a_solution.png")


if __name__ == "__main__":
    main()
