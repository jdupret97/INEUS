"""Local Deep BSDE to global-surrogate orchestration and public imports.

Architectures and optimizer loops live in their dedicated subpackages; this
module exposes the models, training functions, and comparison pipeline.
"""

import os
import timeit

import numpy as np
import pandas as pd

from .architectures.bsde import (
    ClassicalLocalDeepBSDEJump,
    DirectJumpUNet,
    DirectZNet,
)
from .architectures.surrogate import GlobalValueSurrogateFromLocalBSDE
from .training.local_bsde import (
    make_local_bsde_label_dataset,
    train_one_classical_local_bsde_jump,
)
from .training.surrogate import (
    predict_u_local_bsde_surrogate,
    train_global_surrogate_from_local_bsde_labels,
)
from .plotting import plot_local_label_errors, plot_surrogate_history
from .problem import resolve_problem


def run_classical_local_bsde_to_global_surrogate(
    xtest,
    ttest,
    values_true,
    output_dir,
    num_local_models=50,
    local_epochs=2000,
    local_batch_size=1024,
    local_lr=3e-4,
    local_optimizer_name="adam",
    N_time=20,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    clipnorm=1.0,
    local_eval_every=500,
    surrogate_epochs=5000,
    surrogate_batch_size=256,
    surrogate_lr=1e-3,
    surrogate_optimizer_name="adam",
    surrogate_hidden_dim=128,
    surrogate_n_hidden=3,
    surrogate_eval_every=500,
    *,
    problem=None,
):
    """
    Full classical-local-to-global pipeline.

    Stage 1:
        Train many independent local classical Deep BSDE models.

    Stage 2:
        Train a global supervised surrogate u_psi(t,x) from the local labels.
    """
    problem = resolve_problem(problem)
    os.makedirs(output_dir, exist_ok=True)

    tic_total = timeit.default_timer()

    # --------------------------------------------------------
    # Stage 1: local classical BSDE label generation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("Stage 1: Training independent local classical Deep BSDE models")
    print("=" * 80)

    tic_labels = timeit.default_timer()

    labels = make_local_bsde_label_dataset(
        num_local_models=num_local_models,
        local_epochs=local_epochs,
        local_batch_size=local_batch_size,
        lr=local_lr,
        optimizer_name=local_optimizer_name,
        N_time=N_time,
        hidden_dim_z=hidden_dim_z,
        n_hidden_z=n_hidden_z,
        hidden_dim_u=hidden_dim_u,
        n_hidden_u=n_hidden_u,
        K_jump_expect=K_jump_expect,
        clipnorm=clipnorm,
        eval_every=local_eval_every,
        problem=problem,
    )

    label_time = timeit.default_timer() - tic_labels

    label_df = pd.DataFrame({
        "t0": labels["t0"].reshape(-1),
        "y0_local_bsde": labels["y0"].reshape(-1),
        "exact_y0": labels["exact_y0"].reshape(-1),
        "abs_error_local": np.abs(
            labels["y0"].reshape(-1) - labels["exact_y0"].reshape(-1)
        ),
        "local_best_loss": labels["local_losses"].reshape(-1),
    })

    for j in range(problem.dimen):
        label_df[f"x0_{j}"] = labels["x0"][:, j]

    label_path = os.path.join(output_dir, "classical_local_bsde_labels.csv")
    label_df.to_csv(label_path, index=False)

    local_label_error_mean = float(label_df["abs_error_local"].mean())
    local_label_error_max = float(label_df["abs_error_local"].max())

    print("\nLocal label generation complete.")
    print(f"Mean local label abs error against exact solution: {local_label_error_mean:.6e}")
    print(f"Max local label abs error against exact solution : {local_label_error_max:.6e}")
    print(f"Label generation time: {label_time:.2f}s")

    label_error_plot_path = os.path.join(output_dir, "local_label_errors.png")
    plot_local_label_errors(label_df, label_error_plot_path)

    # --------------------------------------------------------
    # Stage 2: global surrogate training
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("Stage 2: Training global surrogate from local BSDE labels")
    print("=" * 80)

    surrogate_res = train_global_surrogate_from_local_bsde_labels(
        x_labels=labels["x0"],
        t_labels=labels["t0"],
        y_labels=labels["y0"],
        epochs=surrogate_epochs,
        batch_size=surrogate_batch_size,
        lr=surrogate_lr,
        optimizer_name=surrogate_optimizer_name,
        hidden_dim=surrogate_hidden_dim,
        n_hidden=surrogate_n_hidden,
        constrained=True,
        eval_every=surrogate_eval_every,
        validation_fraction=0.2,
        problem=problem,
    )

    global_model = surrogate_res["model"]

    surrogate_history_path = os.path.join(output_dir, "surrogate_history.csv")
    pd.DataFrame(surrogate_res["history"]).to_csv(surrogate_history_path, index=False)

    surrogate_plot_path = os.path.join(output_dir, "surrogate_training_history.png")
    plot_surrogate_history(surrogate_res["history"], surrogate_plot_path)

    # --------------------------------------------------------
    # Evaluate global surrogate on full test set
    # --------------------------------------------------------
    preds = predict_u_local_bsde_surrogate(global_model, xtest, ttest)
    test_mae = float(np.mean(np.abs(preds - values_true)))
    test_mse = float(np.mean(np.square(preds - values_true)))

    total_time = timeit.default_timer() - tic_total

    total_local_optimizer_steps = num_local_models * local_epochs
    total_surrogate_optimizer_steps = surrogate_epochs

    summary = pd.DataFrame([{
        "num_local_models": num_local_models,
        "local_epochs_per_model": local_epochs,
        "total_local_optimizer_steps": total_local_optimizer_steps,
        "surrogate_epochs": surrogate_epochs,
        "total_surrogate_optimizer_steps": total_surrogate_optimizer_steps,
        "N_time": N_time,
        "K_jump_expect": K_jump_expect,
        "local_batch_size": local_batch_size,
        "surrogate_batch_size": surrogate_batch_size,
        "local_lr": local_lr,
        "surrogate_lr": surrogate_lr,
        "mean_local_label_abs_error": local_label_error_mean,
        "max_local_label_abs_error": local_label_error_max,
        "global_surrogate_test_mae": test_mae,
        "global_surrogate_test_mse": test_mse,
        "label_generation_time": label_time,
        "total_time": total_time,
    }])

    summary_path = os.path.join(
        output_dir,
        "classical_local_bsde_global_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    preds_df = pd.DataFrame({
        "t": ttest.reshape(-1),
        "u_pred": preds.reshape(-1),
        "u_true": values_true.reshape(-1),
        "abs_error": np.abs(preds.reshape(-1) - values_true.reshape(-1)),
    })

    for j in range(problem.dimen):
        preds_df[f"x_{j}"] = xtest[:, j]

    preds_path = os.path.join(output_dir, "classical_local_bsde_global_predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    print("\n" + "=" * 80)
    print("Classical local BSDE -> global surrogate complete")
    print("=" * 80)
    print(f"Global surrogate test MAE: {test_mae:.6e}")
    print(f"Global surrogate test MSE: {test_mse:.6e}")
    print(f"Total local BSDE optimizer steps: {total_local_optimizer_steps}")
    print(f"Total surrogate optimizer steps: {total_surrogate_optimizer_steps}")
    print(f"Total wall-clock time: {total_time:.2f}s")
    print("\nSaved:")
    print(f"  {label_path}")
    print(f"  {summary_path}")
    print(f"  {preds_path}")
    print(f"  {surrogate_history_path}")
    print(f"  {label_error_plot_path}")
    print(f"  {surrogate_plot_path}")

    return {
        "labels": labels,
        "label_df": label_df,
        "global_model": global_model,
        "surrogate_res": surrogate_res,
        "summary": summary,
        "predictions": preds_df,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "total_time": total_time,
    }
