"""Headless plotting and multi-run aggregation for the nonlinear experiment."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_local_bsde_surrogate_history_nl(history, save_path, *, problem=None):
    plt.figure(figsize=(8, 5))
    epochs = history["epoch"]
    train_loss = np.maximum(np.asarray(history["train_loss"]), 1e-30)
    val_loss = np.asarray(history["val_loss"])

    plt.plot(epochs, np.log(train_loss), lw=2, label="train")
    if not np.all(np.isnan(val_loss)):
        plt.plot(
            epochs,
            np.log(np.maximum(val_loss, 1e-30)),
            lw=2,
            ls="--",
            label="validation",
        )

    plt.xlabel("Surrogate epoch")
    plt.ylabel("log MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bellman_vs_pinn_nl(hist_bellman, hist_pinn, save_path, legend_fontsize=9, *, problem=None):
    epochs_b = hist_bellman["epoch"]
    epochs_p = hist_pinn["epoch"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # MAE on left axis
    ax1.plot(epochs_b, hist_bellman["mae"], color="blue", lw=2, ls="-")
    ax1.plot(epochs_p, hist_pinn["mae"],    color="blue", lw=2, ls="--")
    ax1.set_xlabel("Epoch $k$", fontsize=13)
    ax1.set_ylabel("MAE", color="blue", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="blue")

    # runtime on right axis
    ax2 = ax1.twinx()
    ax2.plot(epochs_b, hist_bellman["time"], color="green", lw=2, ls="-")
    ax2.plot(epochs_p, hist_pinn["time"],    color="green", lw=2, ls="--")
    ax2.set_ylabel("Runtime (sec)", color="teal", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="teal")

    legend_lines = [
        Line2D([0], [0], color="black", lw=2, ls="-",  label="Bellman update"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="Standard PINN residual"),
    ]
    ax1.legend(handles=legend_lines, loc="upper center", fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def aggregate_histories_over_time(histories, metric_key="mae", n_grid=250):
    """
    Interpolate several histories onto a common runtime grid and compute
    mean/std of the chosen metric over the overlapping time interval.
    """
    processed = []

    for hist in histories:
        t = np.asarray(hist["time"], dtype=np.float64)
        y = np.asarray(hist[metric_key], dtype=np.float64)

        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]

        if len(t) < 2:
            continue

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        t_unique, idx_unique = np.unique(t, return_index=True)
        y_unique = y[idx_unique]

        if len(t_unique) < 2:
            continue

        processed.append((t_unique, y_unique))

    if len(processed) == 0:
        raise ValueError(f"No valid histories found for metric '{metric_key}'.")

    t_start = max(t[0] for t, _ in processed)
    t_end = min(t[-1] for t, _ in processed)

    if t_end <= t_start:
        raise ValueError("The runs do not have a non-empty overlapping runtime interval.")

    t_grid = np.linspace(t_start, t_end, n_grid)
    y_interp = np.vstack([np.interp(t_grid, t, y) for t, y in processed])

    return {
        "time": t_grid,
        "mean": np.mean(y_interp, axis=0),
        "std": np.std(y_interp, axis=0),
        "all_interp": y_interp,
    }


def plot_ineus_average_nl(aggregate, save_path, n_runs, *, problem=None):
    """Plot the common-runtime mean/std calculation for INEUS alone."""

    mean = np.maximum(aggregate["mean"], 1e-16)
    lower = np.maximum(aggregate["mean"] - aggregate["std"], 1e-16)
    upper = np.maximum(aggregate["mean"] + aggregate["std"], 1e-16)
    plt.figure(figsize=(10, 6))
    plt.plot(aggregate["time"], mean, lw=2, label=f"INEUS ({n_runs} runs)")
    plt.fill_between(aggregate["time"], lower, upper, alpha=0.20)
    plt.xlabel("Runtime (sec)", fontsize=13)
    plt.ylabel("MAE", fontsize=13)
    plt.yscale("log")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mae_vs_time_three_methods_avg_std_nl(
    bellman_histories,
    pinn_histories,
    bsde_results,
    save_path,
    n_grid=250,
    *,
    problem=None,
):
    """
    Plot mean +/- std MAE versus runtime across runs.

    Bellman and PINN have full training trajectories, so they are interpolated
    onto common runtime grids.  The local Deep BSDE baseline only yields a
    globally comparable value surface after all local solves and the surrogate
    stage are complete, so it is represented by one mean final point with
    runtime/MAE error bars across runs.
    """
    agg_bellman = aggregate_histories_over_time(
        bellman_histories,
        metric_key="mae",
        n_grid=n_grid,
    )
    agg_pinn = aggregate_histories_over_time(
        pinn_histories,
        metric_key="mae",
        n_grid=n_grid,
    )

    bsde_times = np.asarray(
        [res["total_time"] for res in bsde_results],
        dtype=np.float64,
    )
    bsde_maes = np.asarray(
        [res["test_mae"] for res in bsde_results],
        dtype=np.float64,
    )

    agg_bsde = {
        "time": np.asarray([np.mean(bsde_times)], dtype=np.float64),
        "mean": np.asarray([np.mean(bsde_maes)], dtype=np.float64),
        "std": np.asarray([np.std(bsde_maes)], dtype=np.float64),
        "time_std": np.asarray([np.std(bsde_times)], dtype=np.float64),
        "all_times": bsde_times,
        "all_maes": bsde_maes,
    }

    plt.figure(figsize=(10, 6))

    def plot_band(agg, label, ls):
        mean = np.maximum(agg["mean"], 1e-16)
        lower = np.maximum(agg["mean"] - agg["std"], 1e-16)
        upper = np.maximum(agg["mean"] + agg["std"], 1e-16)

        plt.plot(agg["time"], mean, lw=2, ls=ls, label=label)
        plt.fill_between(agg["time"], lower, upper, alpha=0.20)

    plot_band(
        agg_bellman,
        f"Bellman ({len(bellman_histories)} runs)",
        "-",
    )
    plot_band(
        agg_pinn,
        f"PINN ({len(pinn_histories)} runs)",
        "--",
    )

    plt.errorbar(
        agg_bsde["time"][0],
        max(agg_bsde["mean"][0], 1e-16),
        xerr=agg_bsde["time_std"][0],
        yerr=agg_bsde["std"][0],
        fmt="o",
        markersize=8,
        capsize=4,
        label=f"Local Deep BSDE + surrogate ({len(bsde_results)} runs)",
    )

    plt.xlabel("Runtime (sec)", fontsize=13)
    plt.ylabel("MAE", fontsize=13)
    plt.yscale("log")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return agg_bellman, agg_pinn, agg_bsde
