"""Seeded INEUS training for the nonlinear Black–Scholes default-risk case."""

from dataclasses import replace
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .checkpointing import load_checkpoint
from .config import DEFAULT_CONFIG
from .models import build_model
from .replay_buffer import ReplayBuffer
from .sampling import sample_xt_paths
from .training.ineus import train
from .utils import evaluate_point, set_seeds


def make_fixed_test_set(config=DEFAULT_CONFIG):
    """Draw the fixed GBM test points before resetting the training seed.

    These evaluation points follow GBM even when the training collocation
    sampler additionally includes post-jump states.
    """
    return sample_xt_paths(
        config.train.test_samples,
        config.problem.T,
        x0_value=config.evaluation.x0_value,
        mu=config.problem.mu_bar_adjusted,
        sigma=config.problem.sigma_bar,
        dimen=config.problem.dimen,
    )


def run_experiment(seed=1, mode="ineus", config=None, output_dir=None, runs=5):
    """Run independent seeds and collect scalar summaries without retaining models.

    With ``runs=1``, preserve the single-run output layout and return value.
    Repeated runs use consecutive seeds in ``run_01``, ``run_02``, ... and
    sample standard deviations (ddof=1) in the aggregate statistics.
    """
    if mode != "ineus":
        raise ValueError("The Black–Scholes experiment supports mode='ineus' only.")
    if type(runs) is not int or runs < 1:
        raise ValueError("runs must be a positive integer.")
    if type(seed) is not int or not 0 <= seed < 2**32 or seed + runs - 1 >= 2**32:
        raise ValueError("All run seeds must be integers in [0, 2**32).")
    config = DEFAULT_CONFIG if config is None else config
    if runs > 1 and config.train.resume_path is not None:
        raise ValueError("Independent repeated runs cannot share a resume_path; "
                         "use runs=1 to resume a checkpoint.")
    if runs == 1:
        return _run_single_experiment(seed=seed, mode=mode, config=config,
                                      output_dir=output_dir)

    from ineus.configuration import RunConfiguration, dump_yaml

    root = Path(config.train.save_dir if output_dir is None else output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "run_summaries.csv"
    statistics_path = root / "run_statistics.csv"
    summaries = []
    tf.keras.backend.clear_session()
    gc.collect()
    for run_index in range(1, runs + 1):
        run_seed = seed + run_index - 1
        run_dir = root / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = RunConfiguration("black_scholes", mode, run_seed, 1, run_dir, config)
        (run_dir / "config_resolved.yaml").write_text(dump_yaml(resolved), encoding="utf-8")
        print(f"\nBlack–Scholes run {run_index}/{runs} | seed: {run_seed}")
        try:
            result = _run_single_experiment(seed=run_seed, mode=mode, config=config,
                                            output_dir=str(run_dir))
            summary = {name: value for name, value in result["summary"].items()
                       if value is None or np.isscalar(value)}
            summary.update(run=run_index, seed=run_seed)
            del result
            summaries.append(summary)
            frame = pd.DataFrame(summaries)
            frame.to_csv(summary_path, index=False)
            numeric = frame.select_dtypes(include="number").drop(columns=["run", "seed"])
            statistics = numeric.agg(["mean", "std", "count"]).T.rename(columns={"count": "n_runs"})
            statistics.index.name = "metric"
            statistics = statistics.reset_index()
            statistics["n_runs"] = statistics["n_runs"].astype(int)
            statistics.to_csv(statistics_path, index=False)
        finally:
            # Every distribution scope has exited before discarding Keras state.
            tf.keras.backend.clear_session()
            gc.collect()
    return {
        "mode": mode,
        "runs": runs,
        "output_dir": str(root),
        "run_summaries": frame,
        "run_statistics": statistics,
        "summary_path": str(summary_path),
        "statistics_path": str(statistics_path),
    }


def _run_single_experiment(seed=1, mode="ineus", config=None, output_dir=None):
    """Train one seeded INEUS model and save its weights and diagnostics.

    The experiment implements INEUS only, without PINN or deep-BSDE
    baselines. A resume_path loads trusted model weights and the epoch
    counter, not optimizer state, replay contents or the random streams.
    """
    if mode != "ineus":
        raise ValueError("The Black–Scholes experiment supports mode='ineus' only.")
    config = DEFAULT_CONFIG if config is None else config
    pde_cfg = config.problem
    eval_cfg = config.evaluation
    model_cfg = config.model
    run_dir = Path(config.train.save_dir if output_dir is None else output_dir)
    train_cfg = replace(config.train, save_dir=str(run_dir))

    tf.keras.backend.set_floatx("float32")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    strategy = tf.distribute.MirroredStrategy()
    if strategy.num_replicas_in_sync != 1:
        raise ValueError("Black-Scholes requires one visible GPU (or CPU) per process; "
                         "use separate jobs for independent seeds.")
    print(f"Replicas: {strategy.num_replicas_in_sync}")
    print(f"Seed: {seed}")

    set_seeds(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    xtest, ttest = make_fixed_test_set(config)

    # Reset the streams before model initialisation and training.
    set_seeds(seed)
    buffer = ReplayBuffer(train_cfg.buffer_size)

    def make_model_fn():
        return build_model(model_cfg, pde_cfg, train_cfg)

    with strategy.scope():
        model = make_model_fn()
        model_targ = make_model_fn()

    start_epoch = 0
    if train_cfg.resume_path is not None:
        checkpoint = load_checkpoint(train_cfg.resume_path)
        model.set_weights(checkpoint["model_weights"])
        model_targ.set_weights(checkpoint["model_weights"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded model weights at epoch {start_epoch}; optimizer and sampler restart.")

    result = train(
        model=model,
        model_targ=model_targ,
        make_model_fn=make_model_fn,
        pde_cfg=pde_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        replay_buffer=buffer,
        strategy=strategy,
        xtest=xtest,
        ttest=ttest,
        start_epoch=start_epoch,
    )

    prediction, error = evaluate_point(
        model, eval_cfg.x0_eval, eval_cfg.t0_eval, eval_cfg.u_ref,
    )
    print(f"\nu(0, x₀) = {prediction:.6f}", end="")
    print(f"   |error| = {error:.6e}" if error is not None else "")

    parameter_names = (
        "epochs", "h_scale", "lr", "n_inner", "num_batches", "sample_b",
        "w_bc", "w_int", "zeta_scale",
    )
    summary = {name: getattr(train_cfg, name) for name in parameter_names}
    summary.update(seed=seed, prediction=prediction,
                   abs_error=np.nan if error is None else error)
    frame = pd.DataFrame([summary])
    summary_path = run_dir / "hpinn_default_risk.csv"
    frame.to_csv(summary_path, index=False)
    print("\nRUN COMPLETE")
    print(frame.to_string(index=False))

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "prediction": prediction,
        "abs_error": error,
        "model_weights": result["model_weights"],
        "history": result["history"],
    }


def main(argv=None):
    """Run the shared YAML command-line interface."""
    from ineus.cli import main as shared_main

    return shared_main(argv, experiment="black_scholes")


if __name__ == "__main__":
    main()
