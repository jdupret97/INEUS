"""Seeded dimension sweeps for the linear Black–Scholes/Merton basket call."""

import gc
from pathlib import Path
import timeit

import pandas as pd
import tensorflow as tf

from .config import default_config
from .evaluation import build_evaluation_data, evaluate_mae_single_run_bs
from .models import make_model
from .problem import make_problem
from .replay_buffer import ReplayBuffer
from .reporting import aggregate_results, save_single_run_outputs_bs
from .trainer import train2
from .utils import ensure_dir, set_all_seeds


def run_single_experiment_bs(problem, evaluation, config, run_id, run_seed, output_dir):
    """Train one seed, preserving phase order and evaluating the final weights."""
    dimen_value = problem.dimen
    train = config.train
    print("\n" + "=" * 90)
    print(f"Dimension = {dimen_value}, run = {run_id}, seed = {run_seed}")
    print("=" * 90)

    tf.keras.backend.clear_session()
    gc.collect()
    set_all_seeds(run_seed)

    replay_buffer = ReplayBuffer(train.buffer_size)

    def make_model_fn():
        return make_model(
            hidden_dim_val=config.model.hidden_dim,
            n_layers_val=config.model.n_layers,
            T_val=problem.T,
            dimen_val=dimen_value,
            constrained=True,
            x_min=0.0,
            x_max=problem.max_S,
            t_min=0.0,
            t_max=problem.T,
            problem=problem,
        )

    model1 = make_model_fn()
    model1targ = make_model_fn()
    run_dir = Path(output_dir) / f"dimen_{dimen_value:03d}" / f"run_{run_id:02d}"
    ensure_dir(run_dir)

    training_args = dict(
        model1=model1,
        model1targ=model1targ,
        make_model_fn=make_model_fn,
        T_val=problem.T,
        epochs=train.epochs,
        num_samples_val=train.num_samples,
        rho1_val=train.rho1,
        n_inner_val=train.n_inner,
        num_batches_N1_val=train.num_batches,
        h_scale_val=train.h_scale,
        zeta_val=train.relaxation,
        replay_buffer=replay_buffer,
        w_bc_val=train.w_bc,
        w_int_val=train.w_int,
        batch_size_val=train.batch_size,
        m=train.distill_block_size,
        distill_steps=train.distill_steps,
        student_lr=train.distill_student_lr,
        replace_with_student=train.replace_with_student,
        use_distill=train.use_distill,
        use_rad=train.use_rad,
        rad_period=train.rad_period,
        rad_k=train.rad_k,
        rad_c=train.rad_c,
        rad_candidate_size=train.rad_candidate_multiplier * train.num_samples,
        rad_eval_batch_size=train.rad_eval_batch_size,
        sample_mode="paths",
        S0=problem.S0_train,
        optimizer_kwargs=None,
        problem=problem,
        evaluation=evaluation,
    )

    print(f"Phase 1: Adam, lr={train.lr1}, epochs={train.epochs}")
    tic1 = timeit.default_timer()
    phase1_out = train2(**training_args, lr=train.lr1, optimizer_name="adam")
    time1 = timeit.default_timer() - tic1
    print(f"Phase 1 time: {time1:.2f} s")

    phase2_out = None
    time2 = 0.0
    if train.run_phase2:
        print(f"Phase 2: Adagrad, lr={train.lr2}, epochs={train.epochs}")
        tic2 = timeit.default_timer()
        phase2_out = train2(**training_args, lr=train.lr2, optimizer_name="adagrad")
        time2 = timeit.default_timer() - tic2
        print(f"Phase 2 time: {time2:.2f} s")

    final_train_out = phase2_out if phase2_out is not None else phase1_out
    eval_out = evaluate_mae_single_run_bs(
        model1=model1,
        xtest_val=evaluation.xtest,
        ttest_val=evaluation.ttest,
        values_val=evaluation.values,
        ref_curves=evaluation.reference_curves,
    )
    run_record = {
        "dimen": dimen_value,
        "dimend": dimen_value,
        "run_id": run_id,
        "seed": run_seed,
        "sample_b": problem.sample_b,
        "h_scale": train.h_scale,
        "w_bc": train.w_bc,
        "w_int": train.w_int,
        "epochs1": train.epochs,
        "lr1": train.lr1,
        "lr2": train.lr2,
        "n_inner": train.n_inner,
        "num_batches_N1": train.num_batches,
        "alpha": train.relaxation,
        "xi": train.h_scale / (train.n_inner - 1),
        "batch_size": train.batch_size,
        "steps_per_epoch": (train.n_inner - 1) * train.num_batches,
        "mc_test_paths": config.evaluation.mc_test_paths,
        "mc_curve_paths": config.evaluation.mc_curve_paths,
        "n_x_curve": config.evaluation.n_x_curve,
        "n_t_curve": config.evaluation.n_t_curve,
        "mae_curve_t0": eval_out["mae_t0_curve"],
        "mae_curve_time": eval_out["mae_time_curve"],
        "mae_full_test": eval_out["mae_full"],
        "last_epoch_loss": final_train_out["last_epoch_loss"],
        "last_epoch_train_loss": final_train_out["last_epoch_train_loss"],
        "last_epoch_pde_loss": final_train_out["last_epoch_pde_loss"],
        "last_epoch_bc_loss": final_train_out["last_epoch_bc_loss"],
        "last_epoch_mae_during_training": final_train_out["last_epoch_mae"],
        "time_seconds": time1 + time2,
        "time_phase1_seconds": time1,
        "time_phase2_seconds": time2,
        "run_phase2": train.run_phase2,
        "use_rad": train.use_rad,
        "use_distill": train.use_distill,
    }
    save_single_run_outputs_bs(eval_out, run_dir, run_record)
    print(
        f"Done: dimen={dimen_value}, run={run_id}, "
        f"MAE(last)={run_record['mae_full_test']:.6e}, "
        f"Loss(last)={run_record['last_epoch_loss']:.6e}, "
        f"Time={run_record['time_seconds']:.2f}s"
    )

    del model1, model1targ, replay_buffer
    gc.collect()
    return run_record


def run_experiment(seed=23, mode="ineus", config=None, output_dir=None, runs=10):
    """Build fixed references per dimension, then run independent training seeds."""
    if mode != "ineus":
        raise ValueError("The linear Black–Scholes experiment supports INEUS only.")
    config = config if config is not None else default_config()
    output_dir = Path(output_dir or "rPINN/Black_Scholes/Black_Scholes_dimension_sweep")
    tf.keras.backend.set_floatx("float32")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    ensure_dir(output_dir)

    print("\n" + "=" * 90)
    print("BLACK-SCHOLES PIDE DIMENSION SWEEP")
    print("=" * 90)
    print(f"DIMENSIONS = {list(config.problem.dimensions)}")
    print(f"N_RUNS = {runs}")
    print("dimen=dimend=d for every d")
    print(f"epochs1={config.train.epochs}, batch_size={config.train.batch_size}, "
          f"num_samples={config.train.num_samples}")
    print(f"MC_TEST_PATHS={config.evaluation.mc_test_paths}, "
          f"MC_CURVE_PATHS={config.evaluation.mc_curve_paths}")
    print(f"OUTPUT_DIR={output_dir}")
    print("=" * 90)

    tag = config.evaluation.output_tag
    runs_csv = output_dir / f"all_runs_{tag}.csv"
    all_run_records = []
    for dimension in config.problem.dimensions:
        print("\n" + "=" * 90)
        print(f"Setting up Black-Scholes problem for dimen=dimend={dimension}")
        print("=" * 90)
        problem = make_problem(dimension, config.problem)
        evaluation = build_evaluation_data(problem, config.evaluation, seed)
        for run_id in range(1, runs + 1):
            record = run_single_experiment_bs(
                problem, evaluation, config, run_id, seed + run_id - 1, output_dir,
            )
            all_run_records.append(record)
            pd.DataFrame(all_run_records).to_csv(runs_csv, index=False)

    df_runs = pd.DataFrame(all_run_records)
    summary_df = aggregate_results(df_runs)
    summary_csv = output_dir / f"dimension_summary_{tag}.csv"
    summary_tex = output_dir / f"dimension_summary_{tag}.tex"
    df_runs.to_csv(runs_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_tex.write_text(
        summary_df.to_latex(index=False, float_format=lambda x: f"{x:.6e}"),
        encoding="utf-8",
    )
    print("\n" + "=" * 90)
    print("FINAL AGGREGATED TABLE")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print(f"Saved raw runs to:      {runs_csv}")
    print(f"Saved summary table to: {summary_csv}")
    print(f"Saved LaTeX table to:   {summary_tex}")
    return {"runs": df_runs, "summary": summary_df}
