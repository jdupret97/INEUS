"""Orchestration and command-line entry point for the nonlinear experiment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .averaging import copy_weights
from .bsde import (
    run_classical_local_bsde_to_global_surrogate_nl,
)
from .config import ExperimentConfig, default_config
from .architectures.value import make_model, strategy
from .plotting import (
    aggregate_histories_over_time,
    plot_ineus_average_nl,
    plot_mae_vs_time_three_methods_avg_std_nl,
)
from .problem import (
    g_fin,
    make_problem,
    resolve_problem,
    u_pairs,
)
from .sampling import ReplayBuffer, sample_xt, set_all_seeds
from .training.ineus import train_bellman_ct_nl
from .training.pinn import train_standard_pinn_nl


def ensure_dir(path):
    """Create an output directory and all missing parents."""

    Path(path).mkdir(parents=True, exist_ok=True)


def run_three_way_comparison_nl(
    make_model_fn,
    xtest,
    ttest,
    values,
    output_dir,
    epochs_pinn,
    epochs_bellman,
    num_samples,
    buffer_size,
    batch_size_pinn,
    lr_bellman,
    lr_pinn,
    rho1,
    n_inner,
    num_batches_N1,
    h_scale,
    zeta,
    bellman_m=10,
    distill_steps=200,
    student_lr=1e-3,
    replace_with_student=True,
    jump_samples=200,
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    log_every_pinn=10,
    num_t0_each_epoch=0,
    num_local_models=100,
    local_bsde_epochs=5000,
    local_bsde_batch_size=1024,
    local_bsde_lr=3e-4,
    local_bsde_eval_every=500,
    local_bsde_clipnorm=1.0,
    local_bsde_x_scale=2.0,
    N_time=50,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    surrogate_epochs=5000,
    surrogate_batch_size=256,
    surrogate_lr=1e-3,
    surrogate_hidden_dim=128,
    surrogate_n_hidden=3,
    surrogate_eval_every=500,
    test_jump_samples=50,
    rad_jump_samples=25,
    learning_rate_decay_steps=1000,
    learning_rate_decay_ineus=0.995,
    learning_rate_decay_pinn=0.991,
    learning_rate_staircase=True,
    mode="comparison",
    *,
    problem=None,
):
    """Run INEUS alone or the three-method comparison.

    Model initialization is identical in both modes, preserving INEUS's
    random-number stream. INEUS-only mode exits before baseline training.
    """

    problem = resolve_problem(problem)
    T = problem.horizon
    if mode not in {"comparison", "ineus"}:
        raise ValueError("mode must be 'comparison' or 'ineus'.")

    os.makedirs(output_dir, exist_ok=True)

    with strategy.scope():
        reference_model = make_model_fn()
        ineus_model = make_model_fn()
        ineus_target = make_model_fn()
        pinn_model = make_model_fn()

    initial_weights = copy_weights(reference_model)
    ineus_model.set_weights(initial_weights)
    ineus_target.set_weights(initial_weights)
    pinn_model.set_weights(initial_weights)

    x_init, t_init = sample_xt(num_samples, T, problem=problem)
    replay_buffer_ineus = ReplayBuffer(buffer_size)
    replay_buffer_ineus.extend(x_init, t_init)
    replay_buffer_pinn = ReplayBuffer(buffer_size)
    replay_buffer_pinn.extend(x_init, t_init)

    print("\n" + "=" * 80)
    print("Training nonlinear INEUS")
    print("=" * 80)

    res_bellman = train_bellman_ct_nl(
        model1=ineus_model,
        model1targ=ineus_target,
        make_model=make_model_fn,
        T=T,
        epochs=epochs_bellman,
        num_samples=num_samples,
        lr=lr_bellman,
        rho1=rho1,
        n_inner=n_inner,
        num_batches_N1=num_batches_N1,
        h_scale=h_scale,
        zeta=zeta,
        replay_buffer=replay_buffer_ineus,
        xtest=xtest,
        ttest=ttest,
        values=values,
        batch_size=batch_size_pinn,
        m=bellman_m,
        distill_steps=distill_steps,
        student_lr=student_lr,
        replace_with_student=replace_with_student,
        use_rad=use_rad,
        rad_period=rad_period,
        rad_k=rad_k,
        rad_c=rad_c,
        rad_candidate_size=rad_candidate_size,
        rad_eval_batch_size=rad_eval_batch_size,
        optimizer_name="adam",
        optimizer_kwargs=None,
        log_every=log_every_pinn,
        K_test=test_jump_samples,
        K_rad=rad_jump_samples,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay_ineus,
        staircase=learning_rate_staircase,
        num_t0_each_epoch=num_t0_each_epoch,
        problem=problem,
    )

    if mode == "ineus":
        from ..reporting import save_ineus_result

        saved = save_ineus_result(res_bellman, output_dir)
        return {
            "mode": mode,
            "ineus": res_bellman,
            "model": ineus_model,
            "history": saved["history"],
            "summary": saved["summary"],
        }

    print("\n" + "=" * 80)
    print("Training nonlinear standard PINN")
    print("=" * 80)

    res_pinn = train_standard_pinn_nl(
        model1=pinn_model,
        T=T,
        epochs=epochs_pinn,
        num_samples=num_samples,
        lr=lr_pinn,
        n_inner=n_inner,
        num_batches_N1=num_batches_N1,
        replay_buffer=replay_buffer_pinn,
        xtest=xtest,
        ttest=ttest,
        values=values,
        batch_size=batch_size_pinn,
        use_rad=use_rad,
        rad_period=rad_period,
        rad_k=rad_k,
        rad_c=rad_c,
        rad_candidate_size=rad_candidate_size,
        rad_eval_batch_size=rad_eval_batch_size,
        optimizer_name="adam",
        optimizer_kwargs=None,
        log_every=log_every_pinn,
        K_train=jump_samples,
        K_test=test_jump_samples,
        K_rad=rad_jump_samples,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay_pinn,
        staircase=learning_rate_staircase,
        num_t0_each_epoch=num_t0_each_epoch,
        problem=problem,
    )

    print("\n" + "=" * 80)
    print(
        "Training local transformed Deep BSDE models "
        "+ global u surrogate"
    )
    print("=" * 80)

    bsde_output_dir = os.path.join(
        output_dir,
        "classical_local_transformed_bsde",
    )
    res_bsde = run_classical_local_bsde_to_global_surrogate_nl(
        xtest=xtest,
        ttest=ttest,
        values_true=values,
        output_dir=bsde_output_dir,
        T=T,
        x_scale=local_bsde_x_scale,
        num_local_models=num_local_models,
        local_epochs=local_bsde_epochs,
        local_batch_size=local_bsde_batch_size,
        local_lr=local_bsde_lr,
        local_optimizer_name="adam",
        N_time=N_time,
        hidden_dim_z=hidden_dim_z,
        n_hidden_z=n_hidden_z,
        hidden_dim_u=hidden_dim_u,
        n_hidden_u=n_hidden_u,
        K_jump_expect=K_jump_expect,
        clipnorm=local_bsde_clipnorm,
        local_eval_every=local_bsde_eval_every,
        surrogate_epochs=surrogate_epochs,
        surrogate_batch_size=surrogate_batch_size,
        surrogate_lr=surrogate_lr,
        surrogate_optimizer_name="adam",
        surrogate_hidden_dim=surrogate_hidden_dim,
        surrogate_n_hidden=surrogate_n_hidden,
        surrogate_eval_every=surrogate_eval_every,
        problem=problem,
    )

    history = pd.DataFrame(
        {
            "epoch_bellman": pd.Series(
                res_bellman["history"]["epoch"]
            ),
            "time_bellman": pd.Series(
                res_bellman["history"]["time"]
            ),
            "mae_bellman": pd.Series(
                res_bellman["history"]["mae"]
            ),
            "train_loss_bellman": pd.Series(
                res_bellman["history"]["train_loss"]
            ),
            "test_rhs_square_bellman": pd.Series(
                res_bellman["history"]["test_residual"]
            ),
            "epoch_pinn": pd.Series(
                res_pinn["history"]["epoch"]
            ),
            "time_pinn": pd.Series(
                res_pinn["history"]["time"]
            ),
            "mae_pinn": pd.Series(
                res_pinn["history"]["mae"]
            ),
            "train_loss_pinn": pd.Series(
                res_pinn["history"]["train_loss"]
            ),
            "test_rhs_square_pinn": pd.Series(
                res_pinn["history"]["test_residual"]
            ),
            "time_bsde": pd.Series([res_bsde["total_time"]]),
            "mae_bsde": pd.Series([res_bsde["test_mae"]]),
            "mse_bsde": pd.Series([res_bsde["test_mse"]]),
        }
    )
    history_path = os.path.join(
        output_dir,
        "three_method_history.csv",
    )
    history.to_csv(history_path, index=False)

    bsde_summary_row = res_bsde["summary"].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "final_mae_bellman": (
                    res_bellman["history"]["mae"][-1]
                ),
                "final_mae_pinn": (
                    res_pinn["history"]["mae"][-1]
                ),
                "final_mae_bsde": res_bsde["test_mae"],
                "best_logged_mae_bellman": float(
                    np.min(res_bellman["history"]["mae"])
                ),
                "best_logged_mae_pinn": float(
                    np.min(res_pinn["history"]["mae"])
                ),
                "best_loss_bellman": res_bellman["best_loss"],
                "best_loss_pinn": res_pinn["best_loss"],
                "best_surrogate_loss_bsde": (
                    res_bsde["surrogate_res"]["best_loss"]
                ),
                "mean_local_terminal_loss_bsde": float(
                    bsde_summary_row[
                        "mean_local_terminal_loss"
                    ]
                ),
                "runtime_bellman": (
                    res_bellman["history"]["time"][-1]
                ),
                "runtime_pinn": (
                    res_pinn["history"]["time"][-1]
                ),
                "runtime_bsde_total": res_bsde["total_time"],
                "runtime_bsde_local_labels": (
                    res_bsde["label_time"]
                ),
                "num_local_bsde_models": num_local_models,
                "local_bsde_epochs_per_model": (
                    local_bsde_epochs
                ),
                "surrogate_epochs": surrogate_epochs,
            }
        ]
    )
    summary_path = os.path.join(
        output_dir,
        "three_method_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  {history_path}")
    print(f"  {summary_path}")
    print(f"  {bsde_output_dir}")
    return history, res_bellman, res_pinn, res_bsde


def run_three_way_comparison_nl_n_runs(
    n_runs,
    base_seed,
    make_model_fn,
    xtest,
    ttest,
    values,
    output_dir,
    epochs_pinn,
    epochs_bellman,
    num_samples,
    buffer_size,
    batch_size_pinn,
    lr_bellman,
    lr_pinn,
    rho1,
    n_inner,
    num_batches_N1,
    h_scale,
    zeta,
    bellman_m=10,
    distill_steps=200,
    student_lr=1e-3,
    replace_with_student=True,
    jump_samples=200,
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    log_every_pinn=10,
    num_t0_each_epoch=0,
    num_local_models=100,
    local_bsde_epochs=5000,
    local_bsde_batch_size=1024,
    local_bsde_lr=3e-4,
    local_bsde_eval_every=500,
    local_bsde_clipnorm=1.0,
    local_bsde_x_scale=2.0,
    N_time=50,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    surrogate_epochs=5000,
    surrogate_batch_size=256,
    surrogate_lr=1e-3,
    surrogate_hidden_dim=128,
    surrogate_n_hidden=3,
    surrogate_eval_every=500,
    test_jump_samples=50,
    rad_jump_samples=25,
    learning_rate_decay_steps=1000,
    learning_rate_decay_ineus=0.995,
    learning_rate_decay_pinn=0.991,
    learning_rate_staircase=True,
    mode="comparison",
    *,
    problem=None,
):
    """Repeat INEUS or the complete comparison across successive random seeds."""

    problem = resolve_problem(problem)
    if mode not in {"comparison", "ineus"}:
        raise ValueError("mode must be 'comparison' or 'ineus'.")

    os.makedirs(output_dir, exist_ok=True)

    bellman_histories = []
    pinn_histories = []
    bsde_results = []
    run_summaries = []
    ineus_results = []

    for run_index in range(n_runs):
        run_seed = base_seed + run_index
        print("\n" + "#" * 90)
        print(
            f"GLOBAL RUN {run_index + 1}/{n_runs} | "
            f"seed={run_seed}"
        )
        print("#" * 90)

        tf.keras.backend.clear_session()
        set_all_seeds(run_seed)

        run_dir = os.path.join(
            output_dir,
            f"run_{run_index + 1:02d}",
        )
        run_result = run_three_way_comparison_nl(
            make_model_fn=make_model_fn,
            xtest=xtest,
            ttest=ttest,
            values=values,
            output_dir=run_dir,
            epochs_pinn=epochs_pinn,
            epochs_bellman=epochs_bellman,
            num_samples=num_samples,
            buffer_size=buffer_size,
            batch_size_pinn=batch_size_pinn,
            lr_bellman=lr_bellman,
            lr_pinn=lr_pinn,
            rho1=rho1,
            n_inner=n_inner,
            num_batches_N1=num_batches_N1,
            h_scale=h_scale,
            zeta=zeta,
            bellman_m=bellman_m,
            distill_steps=distill_steps,
            student_lr=student_lr,
            replace_with_student=replace_with_student,
            jump_samples=jump_samples,
            test_jump_samples=test_jump_samples,
            rad_jump_samples=rad_jump_samples,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_ineus=learning_rate_decay_ineus,
            learning_rate_decay_pinn=learning_rate_decay_pinn,
            learning_rate_staircase=learning_rate_staircase,
            use_rad=use_rad,
            rad_period=rad_period,
            rad_k=rad_k,
            rad_c=rad_c,
            rad_candidate_size=rad_candidate_size,
            rad_eval_batch_size=rad_eval_batch_size,
            log_every_pinn=log_every_pinn,
            num_t0_each_epoch=num_t0_each_epoch,
            num_local_models=num_local_models,
            local_bsde_epochs=local_bsde_epochs,
            local_bsde_batch_size=local_bsde_batch_size,
            local_bsde_lr=local_bsde_lr,
            local_bsde_eval_every=local_bsde_eval_every,
            local_bsde_clipnorm=local_bsde_clipnorm,
            local_bsde_x_scale=local_bsde_x_scale,
            N_time=N_time,
            hidden_dim_z=hidden_dim_z,
            n_hidden_z=n_hidden_z,
            hidden_dim_u=hidden_dim_u,
            n_hidden_u=n_hidden_u,
            K_jump_expect=K_jump_expect,
            surrogate_epochs=surrogate_epochs,
            surrogate_batch_size=surrogate_batch_size,
            surrogate_lr=surrogate_lr,
            surrogate_hidden_dim=surrogate_hidden_dim,
            surrogate_n_hidden=surrogate_n_hidden,
            surrogate_eval_every=surrogate_eval_every,
            mode=mode,
            problem=problem,
        )

        if mode == "ineus":
            ineus_results.append(run_result)
            bellman_histories.append(run_result["ineus"]["history"])
            run_summaries.append({
                "run": run_index + 1,
                "seed": run_seed,
                **run_result["summary"].iloc[0].to_dict(),
            })
            continue

        comparison_df, res_bellman, res_pinn, res_bsde = run_result
        bellman_histories.append(res_bellman["history"])
        pinn_histories.append(res_pinn["history"])
        bsde_results.append(res_bsde)

        bsde_summary_row = res_bsde["summary"].iloc[0]
        run_summaries.append(
            {
                "run": run_index + 1,
                "seed": run_seed,
                "final_mae_bellman": (
                    res_bellman["history"]["mae"][-1]
                ),
                "final_mae_pinn": (
                    res_pinn["history"]["mae"][-1]
                ),
                "final_mae_bsde": res_bsde["test_mae"],
                "best_logged_mae_bellman": float(
                    np.min(res_bellman["history"]["mae"])
                ),
                "best_logged_mae_pinn": float(
                    np.min(res_pinn["history"]["mae"])
                ),
                "best_loss_bellman": res_bellman["best_loss"],
                "best_loss_pinn": res_pinn["best_loss"],
                "best_surrogate_loss_bsde": (
                    res_bsde["surrogate_res"]["best_loss"]
                ),
                "mean_local_terminal_loss_bsde": float(
                    bsde_summary_row[
                        "mean_local_terminal_loss"
                    ]
                ),
                "runtime_bellman": (
                    res_bellman["history"]["time"][-1]
                ),
                "runtime_pinn": (
                    res_pinn["history"]["time"][-1]
                ),
                "runtime_bsde_total": res_bsde["total_time"],
                "runtime_bsde_local_labels": (
                    res_bsde["label_time"]
                ),
            }
        )

    if mode == "ineus":
        return _save_ineus_runs(
            ineus_results, bellman_histories, run_summaries, output_dir,
            problem=problem,
        )

    aggregate_figure_path = os.path.join(
        output_dir,
        "mae_vs_time_three_methods_avg_std_"
        f"{n_runs}runs.png",
    )
    (
        aggregate_bellman,
        aggregate_pinn,
        aggregate_bsde,
    ) = plot_mae_vs_time_three_methods_avg_std_nl(
        bellman_histories=bellman_histories,
        pinn_histories=pinn_histories,
        bsde_results=bsde_results,
        save_path=aggregate_figure_path,
        problem=problem,
        n_grid=300,
    )

    aggregate_curves = pd.DataFrame(
        {
            "time_bellman": pd.Series(
                aggregate_bellman["time"]
            ),
            "mae_mean_bellman": pd.Series(
                aggregate_bellman["mean"]
            ),
            "mae_std_bellman": pd.Series(
                aggregate_bellman["std"]
            ),
            "time_pinn": pd.Series(
                aggregate_pinn["time"]
            ),
            "mae_mean_pinn": pd.Series(
                aggregate_pinn["mean"]
            ),
            "mae_std_pinn": pd.Series(
                aggregate_pinn["std"]
            ),
            "time_bsde": pd.Series(
                aggregate_bsde["time"]
            ),
            "time_std_bsde": pd.Series(
                aggregate_bsde["time_std"]
            ),
            "mae_mean_bsde": pd.Series(
                aggregate_bsde["mean"]
            ),
            "mae_std_bsde": pd.Series(
                aggregate_bsde["std"]
            ),
        }
    )
    aggregate_csv_path = os.path.join(
        output_dir,
        f"aggregated_curves_over_time_{n_runs}runs.csv",
    )
    aggregate_curves.to_csv(
        aggregate_csv_path,
        index=False,
    )

    summary_runs = pd.DataFrame(run_summaries)
    summary_runs_path = os.path.join(
        output_dir,
        f"summary_over_{n_runs}_runs.csv",
    )
    summary_runs.to_csv(summary_runs_path, index=False)

    print("\nSaved aggregated results:")
    print(f"  {aggregate_figure_path}")
    print(f"  {aggregate_csv_path}")
    print(f"  {summary_runs_path}")

    return {
        "bellman_histories": bellman_histories,
        "pinn_histories": pinn_histories,
        "bsde_results": bsde_results,
        "agg_bellman": aggregate_bellman,
        "agg_pinn": aggregate_pinn,
        "agg_bsde": aggregate_bsde,
        "summary_runs": summary_runs,
    }


def _save_ineus_runs(results, histories, summaries, output_dir, *, problem=None):
    """Save INEUS-only runs; interpolate only when the logged data permit it."""

    n_runs = len(results)
    summary_runs = pd.DataFrame(summaries)
    summary_path = Path(output_dir) / f"ineus_summary_over_{n_runs}_runs.csv"
    summary_runs.to_csv(summary_path, index=False)

    aggregate = None
    try:
        aggregate = aggregate_histories_over_time(
            histories, metric_key="mae", n_grid=300,
        )
    except ValueError as error:
        # One-epoch smoke tests still produce valid per-run histories and
        # summaries, but cannot support common-runtime interpolation.
        print(f"INEUS aggregate curve skipped: {error}")
    else:
        pd.DataFrame({
            "time_ineus": aggregate["time"],
            "mae_mean_ineus": aggregate["mean"],
            "mae_std_ineus": aggregate["std"],
        }).to_csv(
            Path(output_dir) / f"ineus_aggregated_curves_{n_runs}runs.csv",
            index=False,
        )
        plot_ineus_average_nl(
            aggregate,
            Path(output_dir) / f"ineus_mae_vs_time_{n_runs}runs.png",
            n_runs=n_runs,
            problem=problem,
        )

    print(f"Saved INEUS run summaries: {summary_path}")
    return {
        "mode": "ineus",
        "runs": results,
        "ineus_histories": histories,
        "agg_ineus": aggregate,
        "summary_runs": summary_runs,
    }


def build_fixed_test_set(config: ExperimentConfig, *, problem=None):
    """Build the common test set and its Monte Carlo reference values."""

    problem = make_problem(config.problem) if problem is None else resolve_problem(problem)
    evaluation = config.evaluation
    test_rng = np.random.default_rng(evaluation.test_seed)

    ttest = test_rng.uniform(
        0.0,
        problem.horizon,
        evaluation.test_samples,
    ).reshape(-1, 1).astype(np.float32)
    xtest = test_rng.uniform(
        -evaluation.test_space_bound,
        evaluation.test_space_bound,
        (evaluation.test_samples, problem.state_dim),
    ).astype(np.float32)

    values = u_pairs(
        x_pts=xtest,
        t_pts=ttest.reshape(-1),
        T=problem.horizon,
        eta=problem.eta,
        lam=problem.jump_intensity,
        muJ=problem.mu_j,
        SigmaJ=problem.cov_j,
        g=g_fin,
        n_paths=evaluation.reference_paths,
        paths_chunk=evaluation.reference_chunk_size,
        f_const=problem.driver_constant,
        seed=evaluation.reference_seed,
        problem=problem,
    ).reshape(-1, 1)
    return xtest, ttest, values


def run_experiment(
    *,
    seed: int | None = None,
    output_dir="rPINN/nonlinear/nonlinear_three_methods",
    runs: int | None = None,
    config: ExperimentConfig | None = None,
    mode: str = "comparison",
):
    """Run the configured nonlinear experiment without altering its algorithms."""

    if mode not in {"comparison", "ineus"}:
        raise ValueError("mode must be 'comparison' or 'ineus'.")
    config = config or default_config()
    seed = config.base_seed if seed is None else seed
    runs = config.runs if runs is None else runs
    if config.problem.dimensions:
        raise ValueError(
            "Use the CLI to run problem.dimensions sweeps; run_experiment handles one dimension."
        )
    if runs < 1:
        raise ValueError("runs must be at least 1.")

    ensure_dir(output_dir)
    problem = make_problem(config.problem)
    # Set initialization and pre-run random seeds.
    set_all_seeds(23)
    xtest, ttest, values = build_fixed_test_set(config, problem=problem)

    model_config = config.model
    training = config.train
    bsde = config.bsde
    evaluation = config.evaluation

    jump_std_max = np.sqrt(np.max(np.diag(problem.cov_j)))
    x_bound_norm = (
        max(3.0 * 0.8, evaluation.test_space_bound)
        + 3.0 * jump_std_max
    )

    def make_model_fn():
        return make_model(
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
            T=problem.horizon,
            dimen=problem.state_dim,
            constrained=(
                model_config.constrained_terminal_value
            ),
            x_min=-x_bound_norm,
            x_max=x_bound_norm,
            t_min=0.0,
            t_max=problem.horizon,
            problem=problem,
        )

    tf.keras.backend.clear_session()
    set_all_seeds(10)

    return run_three_way_comparison_nl_n_runs(
        n_runs=runs,
        base_seed=seed,
        make_model_fn=make_model_fn,
        xtest=xtest,
        ttest=ttest,
        values=values,
        output_dir=str(output_dir),
        epochs_pinn=training.epochs_pinn,
        epochs_bellman=training.epochs_ineus,
        num_samples=training.num_samples,
        buffer_size=training.buffer_size,
        batch_size_pinn=training.batch_size,
        lr_bellman=training.learning_rate_ineus,
        lr_pinn=training.learning_rate_pinn,
        rho1=training.target_mixing,
        n_inner=training.n_inner,
        num_batches_N1=training.num_batches,
        h_scale=training.h,
        zeta=training.alpha,
        bellman_m=training.averaging_block_size,
        distill_steps=training.distillation_steps,
        student_lr=training.distillation_learning_rate,
        replace_with_student=training.replace_with_student,
        jump_samples=training.jump_samples,
        test_jump_samples=training.test_jump_samples,
        rad_jump_samples=training.rad_jump_samples,
        learning_rate_decay_steps=training.learning_rate_decay_steps,
        learning_rate_decay_ineus=training.learning_rate_decay_ineus,
        learning_rate_decay_pinn=training.learning_rate_decay_pinn,
        learning_rate_staircase=training.learning_rate_staircase,
        use_rad=training.use_rad,
        rad_period=training.rad_period,
        rad_k=training.rad_exponent,
        rad_c=training.rad_offset,
        rad_candidate_size=training.rad_candidate_size,
        rad_eval_batch_size=training.rad_eval_batch_size,
        log_every_pinn=training.log_every,
        num_t0_each_epoch=training.num_t0_each_epoch,
        num_local_models=bsde.num_local_models,
        local_bsde_epochs=bsde.local_epochs,
        local_bsde_batch_size=bsde.local_batch_size,
        local_bsde_lr=bsde.local_learning_rate,
        local_bsde_eval_every=bsde.local_eval_every,
        local_bsde_clipnorm=bsde.gradient_clip_norm,
        local_bsde_x_scale=x_bound_norm,
        N_time=bsde.n_time_steps,
        hidden_dim_z=bsde.z_hidden_dim,
        n_hidden_z=bsde.z_hidden_layers,
        hidden_dim_u=bsde.jump_hidden_dim,
        n_hidden_u=bsde.jump_hidden_layers,
        K_jump_expect=bsde.jump_expectation_samples,
        surrogate_epochs=bsde.surrogate_epochs,
        surrogate_batch_size=bsde.surrogate_batch_size,
        surrogate_lr=bsde.surrogate_learning_rate,
        surrogate_hidden_dim=bsde.surrogate_hidden_dim,
        surrogate_n_hidden=bsde.surrogate_hidden_layers,
        surrogate_eval_every=bsde.surrogate_eval_every,
        mode=mode,
        problem=problem,
    )


# Alternate entry point for the configured nonlinear experiment.
run_paper_experiment = run_experiment


def parse_args(argv=None):
    """Parse the shared YAML-aware CLI for this experiment."""

    from ..cli import parse_args as shared_parse_args

    return shared_parse_args(argv, experiment="nonlinear")


def main(argv=None):
    """Delegate to the shared, YAML-aware command-line interface."""

    from ..cli import main as shared_main

    return shared_main(argv, experiment="nonlinear")


if __name__ == "__main__":
    main()
