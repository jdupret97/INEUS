"""End-to-end runner and command-line interface for the linear experiment."""

import gc
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from .averaging import copy_weights
from .bsde import run_classical_local_bsde_to_global_surrogate
from .config import DEFAULT_CONFIG, ExperimentConfig
from .architectures.value import make_model
from .plotting import plot_mae_vs_time_three_methods
from .problem import make_problem, resolve_problem
from .sampling import ReplayBuffer, sample_xt
from .training.ineus import train_bellman_ct
from .training.pinn import train_standard_pinn


def configure_runtime(seed):
    """Configure deterministic seeds and TensorFlow numeric behavior."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.set_floatx("float32")
    tf.config.experimental.enable_tensor_float_32_execution(False)


def make_fixed_test_set(config=DEFAULT_CONFIG, *, problem=None):
    """Draw the shared test set from the global NumPy stream.

    Call after configure_runtime. Drawing times before states fixes the order
    of test sampling and subsequent training random draws.
    """

    problem = make_problem(config.problem) if problem is None else problem
    evaluation = config.evaluation
    ttest = np.random.uniform(
        0.0,
        problem.T,
        size=(evaluation.test_samples, 1),
    ).astype(np.float32)
    xtest = np.random.uniform(
        -problem.test_box,
        problem.test_box,
        size=(
            evaluation.test_samples,
            problem.dimen,
        ),
    ).astype(np.float32)
    values = problem.exact_value(ttest, xtest).reshape(-1, 1)
    return xtest, ttest, values


def run_three_way_comparison(
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
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    log_every_pinn=10,
    K_jump=25,
    distill_steps=200,
    student_lr=1e-3,
    replace_with_student=True,
    num_local_models=100,
    local_bsde_epochs=5000,
    local_bsde_batch_size=1024,
    local_bsde_lr=3e-4,
    local_bsde_eval_every=500,
    N_time=20,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    local_clipnorm=1.0,
    surrogate_epochs=5000,
    surrogate_batch_size=256,
    surrogate_lr=1e-3,
    surrogate_hidden_dim=128,
    surrogate_n_hidden=3,
    surrogate_eval_every=500,
    K_test_bellman=50,
    K_train_pinn=100,
    K_test_pinn=50,
    K_rad_pinn=25,
    mode="comparison",
    *,
    problem=None,
):
    """Run the three-method comparison, or stop after its INEUS stage.

    Both modes use the same network initialization order so
    selecting a mode does not change the random stream used by INEUS.
    """

    problem = resolve_problem(problem)
    T = problem.T
    if mode not in ("comparison", "ineus"):
        raise ValueError("mode must be 'comparison' or 'ineus'")
    os.makedirs(output_dir, exist_ok=True)
    reference_model = make_model_fn()
    initial_weights = copy_weights(reference_model)

    ineus_model = make_model_fn()
    ineus_model.set_weights(initial_weights)
    ineus_target = make_model_fn()
    ineus_target.set_weights(initial_weights)
    # Build the unused PINN initialization even in INEUS-only mode: removing
    # this would change later seeded Keras initialization (e.g. distillation).
    pinn_model = make_model_fn()
    pinn_model.set_weights(initial_weights)

    x_init, t_init = sample_xt(num_samples, T, problem=problem)
    replay_buffer_ineus = ReplayBuffer(buffer_size)
    replay_buffer_ineus.extend(x_init, t_init)
    replay_buffer_pinn = ReplayBuffer(buffer_size)
    replay_buffer_pinn.extend(x_init, t_init)

    print("\n" + "=" * 80)
    print("Training INEUS")
    print("=" * 80)
    res_bellman = train_bellman_ct(
        model1=ineus_model,
        model1targ=ineus_target,
        make_model_fn=make_model_fn,
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
        K_rad=K_jump,
        K_test=K_test_bellman,
        problem=problem,
    )

    if mode == "ineus":
        from ineus.reporting import save_ineus_result

        saved = save_ineus_result(res_bellman, output_dir)
        return {"mode": mode, "ineus": res_bellman, **saved}

    print("\n" + "=" * 80)
    print("Training standard PINN")
    print("=" * 80)
    res_pinn = train_standard_pinn(
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
        K_train=K_train_pinn,
        K_test=K_test_pinn,
        K_rad=K_rad_pinn,
        problem=problem,
    )

    print("\n" + "=" * 80)
    print(
        "Training classical local Deep BSDE models "
        "+ global surrogate"
    )
    print("=" * 80)
    bsde_output_dir = os.path.join(
        output_dir,
        "classical_local_bsde",
    )
    res_bsde = run_classical_local_bsde_to_global_surrogate(
        xtest=xtest,
        ttest=ttest,
        values_true=values,
        output_dir=bsde_output_dir,
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
        clipnorm=local_clipnorm,
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

    figure_path = os.path.join(
        output_dir,
        "mae_vs_time_three_methods.png",
    )
    plot_mae_vs_time_three_methods(
        hist_bellman=res_bellman["history"],
        hist_pinn=res_pinn["history"],
        res_bsde=res_bsde,
        save_path=figure_path,
    )

    comparison = pd.DataFrame(
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
    comparison.to_csv(history_path, index=False)

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
                "best_loss_bellman": res_bellman["best_loss"],
                "best_loss_pinn": res_pinn["best_loss"],
                "best_surrogate_loss_bsde": (
                    res_bsde["surrogate_res"]["best_loss"]
                ),
                "runtime_bellman": (
                    res_bellman["history"]["time"][-1]
                ),
                "runtime_pinn": (
                    res_pinn["history"]["time"][-1]
                ),
                "runtime_bsde_total": res_bsde["total_time"],
                "runtime_bsde_local_labels": float(
                    bsde_summary_row["label_generation_time"]
                ),
                "num_local_bsde_models": int(
                    bsde_summary_row["num_local_models"]
                ),
                "local_bsde_epochs_per_model": int(
                    bsde_summary_row[
                        "local_epochs_per_model"
                    ]
                ),
                "surrogate_epochs": int(
                    bsde_summary_row["surrogate_epochs"]
                ),
                "mean_local_bsde_label_abs_error": float(
                    bsde_summary_row[
                        "mean_local_label_abs_error"
                    ]
                ),
                "max_local_bsde_label_abs_error": float(
                    bsde_summary_row[
                        "max_local_label_abs_error"
                    ]
                ),
            }
        ]
    )
    summary_path = os.path.join(
        output_dir,
        "three_method_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    for path in (
        figure_path,
        history_path,
        summary_path,
        bsde_output_dir,
    ):
        print(f"  {path}")

    return (
        comparison,
        res_bellman,
        res_pinn,
        res_bsde,
    )


def _run_single_experiment(
    seed=29,
    output_dir=None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    mode="comparison",
):
    """Run INEUS alone or the three-method comparison with validated settings."""

    if mode not in ("comparison", "ineus"):
        raise ValueError("mode must be 'comparison' or 'ineus'")
    if config.problem.dimensions:
        raise ValueError("Use the CLI to expand problem.dimensions into individual runs")
    problem = make_problem(config.problem)
    # The comparison shares its replay initialization and update
    # budget between INEUS and PINN. Reject conflicting overrides rather than
    # silently discarding values from PINNTrainConfig.
    shared_fields = (
        "num_samples", "buffer_size", "batch_size", "n_inner", "num_batches",
        "use_rad", "rad_period", "rad_exponent", "rad_offset",
        "rad_candidate_multiplier", "rad_eval_batch_size", "log_every",
    )
    conflicting = [
        name for name in shared_fields
        if getattr(config.ineus, name) != getattr(config.pinn, name)
    ]
    if mode == "comparison" and conflicting:
        raise ValueError(
            "The comparison shares these INEUS/PINN settings; "
            "apply overrides to both configurations: " + ", ".join(conflicting)
        )
    configure_runtime(seed)
    output_dir = (
        output_dir
        if output_dir is not None
        else config.evaluation.default_output_dir
    )
    print(
        f"dimen={config.problem.state_dim}, "
        f"dimend={config.problem.brownian_dim}"
    )
    print("sigma shape:", problem.sigma.shape)
    print("cov_j shape:", problem.cov_j.shape)
    xtest, ttest, values = make_fixed_test_set(config, problem=problem)
    model_config = config.model

    def make_model_fn():
        return make_model(
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
            T=config.problem.horizon,
            dimen=config.problem.state_dim,
            constrained=model_config.constrained,
            output_dim=model_config.output_dim,
        )

    ineus = config.ineus
    pinn = config.pinn
    bsde = config.bsde
    result = run_three_way_comparison(
        make_model_fn=make_model_fn,
        xtest=xtest,
        ttest=ttest,
        values=values,
        output_dir=output_dir,
        epochs_pinn=pinn.epochs,
        epochs_bellman=ineus.epochs,
        num_samples=ineus.num_samples,
        buffer_size=ineus.buffer_size,
        batch_size_pinn=ineus.batch_size,
        lr_bellman=ineus.learning_rate,
        lr_pinn=pinn.learning_rate,
        rho1=ineus.target_weight,
        n_inner=ineus.n_inner,
        num_batches_N1=ineus.num_batches,
        h_scale=ineus.h_scale,
        zeta=ineus.relaxation,
        bellman_m=ineus.block_size,
        use_rad=ineus.use_rad,
        rad_period=ineus.rad_period,
        rad_k=ineus.rad_exponent,
        rad_c=ineus.rad_offset,
        rad_candidate_size=(
            ineus.rad_candidate_multiplier
            * ineus.num_samples
        ),
        rad_eval_batch_size=ineus.rad_eval_batch_size,
        log_every_pinn=ineus.log_every,
        K_jump=ineus.jump_samples,
        distill_steps=ineus.distill_steps,
        student_lr=ineus.student_learning_rate,
        replace_with_student=ineus.replace_with_student,
        num_local_models=bsde.num_local_models,
        local_bsde_epochs=bsde.local_epochs,
        local_bsde_batch_size=bsde.local_batch_size,
        local_bsde_lr=bsde.local_learning_rate,
        local_bsde_eval_every=bsde.local_eval_every,
        N_time=bsde.time_steps,
        hidden_dim_z=bsde.hidden_dim_z,
        n_hidden_z=bsde.n_hidden_z,
        hidden_dim_u=bsde.hidden_dim_u,
        n_hidden_u=bsde.n_hidden_u,
        K_jump_expect=bsde.jump_expectation_samples,
        local_clipnorm=bsde.clipnorm,
        surrogate_epochs=bsde.surrogate_epochs,
        surrogate_batch_size=bsde.surrogate_batch_size,
        surrogate_lr=bsde.surrogate_learning_rate,
        surrogate_hidden_dim=bsde.surrogate_hidden_dim,
        surrogate_n_hidden=bsde.surrogate_n_hidden,
        surrogate_eval_every=bsde.surrogate_eval_every,
        K_test_bellman=ineus.test_jump_samples,
        K_train_pinn=pinn.jump_samples,
        K_test_pinn=pinn.test_jump_samples,
        K_rad_pinn=pinn.rad_jump_samples,
        mode=mode,
        problem=problem,
    )
    if mode == "ineus":
        print("\nINEUS summary:")
        print(result["summary"])
        return result
    comparison, _, _, res_bsde = result
    print("\nLast rows of merged history:")
    print(comparison.tail())
    print("\nClassical local Deep BSDE summary:")
    print(res_bsde["summary"])
    return result


def run_experiment(
    seed=29,
    output_dir=None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    mode="comparison",
    runs=10,
):
    """Run independently seeded repetitions without changing the training routine.

    A single run keeps the original output layout and return value. Repetitions
    write complete results below ``run_01``, ``run_02``, etc., plus scalar
    summaries and sample standard deviations in the parent directory.
    """

    if type(runs) is not int or runs < 1:
        raise ValueError("runs must be a positive integer")
    if config.problem.dimensions:
        raise ValueError("Use the CLI to expand problem.dimensions into individual runs")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if seed + runs - 1 >= 2**32:
        raise ValueError("All repetition seeds must be less than 2**32")
    if mode not in ("comparison", "ineus"):
        raise ValueError("mode must be 'comparison' or 'ineus'")
    if runs == 1:
        return _run_single_experiment(
            seed=seed, output_dir=output_dir, config=config, mode=mode,
        )

    from ineus.configuration import RunConfiguration, dump_yaml

    output_dir = Path(
        output_dir if output_dir is not None else config.evaluation.default_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summaries.csv"
    statistics_path = output_dir / "run_statistics.csv"
    rows = []
    tf.keras.backend.clear_session()
    gc.collect()
    for index in range(runs):
        run_seed = seed + index
        run_dir = output_dir / f"run_{index + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = RunConfiguration(
            experiment="linear", mode=mode, seed=run_seed, runs=1,
            output_dir=run_dir, parameters=config,
        )
        (run_dir / "config_resolved.yaml").write_text(
            dump_yaml(resolved), encoding="utf-8",
        )
        print(f"\nLinear PIDE run {index + 1}/{runs} (seed={run_seed})")
        result = None
        try:
            result = _run_single_experiment(
                seed=run_seed, output_dir=run_dir, config=config, mode=mode,
            )
            summary = (
                result["summary"] if mode == "ineus"
                else pd.read_csv(run_dir / "three_method_summary.csv")
            )
            rows.append({"run": index + 1, "seed": run_seed, **summary.iloc[0].to_dict()})
            # Keep the completed runs available even if a subsequent run fails.
            pd.DataFrame(rows).to_csv(summary_path, index=False)
        finally:
            del result
            tf.keras.backend.clear_session()
            gc.collect()

    summaries = pd.DataFrame(rows)
    metrics = summaries.drop(columns=["run", "seed"]).select_dtypes(include="number")
    statistics = pd.DataFrame({
        "metric": metrics.columns,
        "mean": metrics.mean().to_numpy(),
        "std": metrics.std(ddof=1).to_numpy(),
        "n_runs": metrics.count().to_numpy(),
    })
    statistics.to_csv(statistics_path, index=False)
    print("\nRepeated-run statistics:")
    print(statistics.to_string(index=False))
    return {
        "mode": mode, "runs": runs, "output_dir": output_dir,
        "run_summaries": summaries, "run_statistics": statistics,
        "summary_path": summary_path, "statistics_path": statistics_path,
    }


def build_parser():
    """Return the shared YAML-aware command-line parser for this experiment."""

    from ineus.cli import build_parser as shared_parser

    return shared_parser(experiment="linear")


def main(argv=None):
    """Delegate to the shared command-line configuration layer."""

    from ineus.cli import main as shared_main

    return shared_main(argv, experiment="linear")


if __name__ == "__main__":
    main()
