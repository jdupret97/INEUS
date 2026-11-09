"""Per-run tables, curve arrays and dimension-wise summary statistics."""

import os

import numpy as np
import pandas as pd

from .utils import ensure_dir


def save_single_run_outputs_bs(eval_out, run_dir, run_record):
    ensure_dir(run_dir)

    df_t0 = pd.DataFrame(
        {
            "x1": eval_out["x1_grid"],
            "prediction": eval_out["pred_t0"],
            "reference_mc": eval_out["true_t0"],
            "reference_mc_stderr": eval_out["se_t0"],
            "abs_error": np.abs(eval_out["pred_t0"] - eval_out["true_t0"]),
        }
    )
    df_t0.to_csv(os.path.join(run_dir, "curve_t0.csv"), index=False)

    df_time = pd.DataFrame(
        {
            "t": eval_out["t_grid"],
            "prediction": eval_out["pred_time"],
            "reference_mc": eval_out["true_time"],
            "reference_mc_stderr": eval_out["se_time"],
            "abs_error": np.abs(eval_out["pred_time"] - eval_out["true_time"]),
        }
    )
    df_time.to_csv(os.path.join(run_dir, "curve_xfixed_time.csv"), index=False)

    pd.DataFrame([run_record]).to_csv(os.path.join(run_dir, "run_summary.csv"), index=False)

    np.savez(
        os.path.join(run_dir, "run_outputs.npz"),
        x1_grid=eval_out["x1_grid"],
        pred_t0=eval_out["pred_t0"],
        true_t0=eval_out["true_t0"],
        se_t0=eval_out["se_t0"],
        t_grid=eval_out["t_grid"],
        pred_time=eval_out["pred_time"],
        true_time=eval_out["true_time"],
        se_time=eval_out["se_time"],
        x1_fixed=np.array([eval_out["x1_fixed"]], dtype=np.float32),
        x_other_fixed=np.array([eval_out["x_other_fixed"]], dtype=np.float32),
    )


def aggregate_results(df_runs):
    summary_df = (
        df_runs.groupby("dimen", as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            mean_MAE=("mae_full_test", "mean"),
            std_MAE=("mae_full_test", "std"),
            mean_loss_function=("last_epoch_pde_loss", "mean"),
            std_loss_function=("last_epoch_pde_loss", "std"),
            mean_time_seconds=("time_seconds", "mean"),
            std_time_seconds=("time_seconds", "std"),
            mean_mae_curve_t0=("mae_curve_t0", "mean"),
            mean_mae_curve_time=("mae_curve_time", "mean"),
        )
        .sort_values("dimen")
        .reset_index(drop=True)
    )
    return summary_df
