"""Headless plotting helpers for the linear comparison."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_surrogate_history(history, save_path):
    plt.figure(figsize=(8, 5))

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.plot(epochs, np.log(np.asarray(train_loss)), lw=2, label="train")

    if not np.all(np.isnan(val_loss)):
        plt.plot(epochs, np.log(np.asarray(val_loss)), lw=2, ls="--", label="validation")

    plt.xlabel("Surrogate epoch")
    plt.ylabel("log MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_local_label_errors(label_df, save_path):
    if "exact_y0" not in label_df.columns:
        return

    abs_err = np.abs(label_df["y0_local_bsde"].values - label_df["exact_y0"].values)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(abs_err)), abs_err, marker="o", lw=1)
    plt.xlabel("Local model index")
    plt.ylabel("|Y0_hat - exact|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mae_vs_time_three_methods(hist_bellman, hist_pinn, res_bsde, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(
        hist_bellman["time"],
        np.log(hist_bellman["mae"]),
        lw=2,
        ls="-",
        label="Bellman update (pinn_loss1bis)",
    )

    plt.plot(
        hist_pinn["time"],
        np.log(hist_pinn["mae"]),
        lw=2,
        ls="--",
        label="Standard PINN (rhs_square)",
    )

    # The classical local Deep BSDE baseline only produces a global approximation
    # after (i) all local solves and (ii) training the supervised surrogate.
    # Therefore its globally comparable MAE is shown at the total pipeline runtime.
    plt.plot(
        [res_bsde["total_time"]],
        [np.log(max(res_bsde["test_mae"], 1e-16))],
        marker="o",
        markersize=8,
        linestyle="None",
        label="Classical local Deep BSDE + surrogate",
    )

    plt.xlabel("Runtime (sec)", fontsize=13)
    plt.ylabel("log(MAE)", fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
