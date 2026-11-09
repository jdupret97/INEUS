"""Verify INEUS-only outputs and consistency with the comparison's first stage.

Run from the repository root with the experiment dependencies installed::

    python scripts/check_linear_modes.py

This trains two tiny, two-epoch INEUS runs (including RAD and distillation).
PINN and BSDE optimization must never execute. All output files are written to
a temporary directory under results/ and removed when the check completes.
This checks mode routing, not convergence or the full training budget.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import tensorflow as tf

from ineus.cli import execute
from ineus.configuration import RunConfiguration, read_yaml, validate
from ineus.linear import experiment
from ineus.linear.config import DEFAULT_CONFIG


class ComparisonReachedPINN(Exception):
    """Stop the comparison immediately after its completed INEUS stage."""


def tiny_config():
    """Shrink budgets while retaining block distillation and adaptive sampling."""
    shared = dict(
        epochs=2,
        num_samples=8,
        buffer_size=16,
        batch_size=4,
        n_inner=2,
        num_batches=1,
        use_rad=True,
        rad_period=1,
        rad_candidate_multiplier=2,
        rad_eval_batch_size=8,
        log_every=1,
        jump_samples=3,
        test_jump_samples=3,
    )
    return replace(
        DEFAULT_CONFIG,
        model=replace(DEFAULT_CONFIG.model, hidden_dim=4),
        ineus=replace(
            DEFAULT_CONFIG.ineus,
            **shared,
            block_size=1,
            distill_steps=2,
        ),
        pinn=replace(DEFAULT_CONFIG.pinn, **shared, rad_jump_samples=3),
        evaluation=replace(DEFAULT_CONFIG.evaluation, test_samples=4),
    )


def reset_runtime():
    """Make the sequential checks behave like separate CLI invocations."""
    tf.keras.backend.clear_session()


def check_written_outputs(output_dir, result):
    expected = {
        "config_resolved.yaml",
        "ineus_history.csv",
        "ineus_summary.csv",
        "mae_vs_time_ineus.png",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    history = pd.read_csv(output_dir / "ineus_history.csv")
    summary = pd.read_csv(output_dir / "ineus_summary.csv")
    pd.testing.assert_frame_equal(history, result["history"], check_exact=False)
    pd.testing.assert_frame_equal(summary, result["summary"], check_exact=False)
    assert len(history) == 2
    assert int(summary["final_logged_epoch"].iloc[0]) == 2
    assert summary["final_mae_ineus"].iloc[0] == history["mae"].iloc[-1]
    assert (output_dir / "mae_vs_time_ineus.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    document = read_yaml(output_dir / "config_resolved.yaml")
    assert document["experiment"] == "linear"
    assert document["mode"] == "ineus"
    assert document["seed"] == 731
    assert document["parameters"]["ineus"]["epochs"] == 2
    print("PASS INEUS-only CSVs, figure, and resolved YAML")


def compare_training_results(actual, expected):
    for name in expected["history"]:
        if name == "time":
            continue
        np.testing.assert_allclose(
            actual["history"][name], expected["history"][name],
            rtol=1e-6, atol=1e-6, err_msg=f"history: {name}",
        )
    np.testing.assert_allclose(actual["best_loss"], expected["best_loss"],
                               rtol=1e-6, atol=1e-6)
    assert len(actual["best_weights"]) == len(expected["best_weights"])
    for index, (a, b) in enumerate(zip(actual["best_weights"], expected["best_weights"])):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6,
                                   err_msg=f"best weights: {index}")
    print("PASS identical INEUS numerical histories and selected weights in both modes")


def main():
    results_root = ROOT / "results"
    results_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="check_linear_modes_", dir=results_root) as temporary:
        base = Path(temporary)
        run = RunConfiguration(
            experiment="linear", mode="ineus", seed=731, runs=1,
            output_dir=base / "ineus", parameters=tiny_config(),
        )
        validate(run)
        reset_runtime()
        with patch.object(
            experiment, "train_standard_pinn",
            side_effect=AssertionError("INEUS-only mode started PINN training"),
        ) as pinn, patch.object(
            experiment, "run_classical_local_bsde_to_global_surrogate",
            side_effect=AssertionError("INEUS-only mode started BSDE training"),
        ) as bsde:
            ineus_result = execute(run)
            pinn.assert_not_called()
            bsde.assert_not_called()
        assert ineus_result["mode"] == "ineus"
        print("PASS INEUS-only mode never calls PINN or BSDE trainers")
        check_written_outputs(run.output_dir, ineus_result)

        captured = {}
        train_function = experiment.train_bellman_ct

        def capture_ineus(*args, **kwargs):
            captured["ineus"] = train_function(*args, **kwargs)
            return captured["ineus"]

        comparison_run = replace(run, mode="comparison", output_dir=base / "comparison")
        validate(comparison_run)
        reset_runtime()
        with patch.object(experiment, "train_bellman_ct", side_effect=capture_ineus), \
                patch.object(experiment, "train_standard_pinn",
                             side_effect=ComparisonReachedPINN) as pinn, \
                patch.object(experiment, "run_classical_local_bsde_to_global_surrogate",
                             side_effect=AssertionError("The check unexpectedly reached BSDE")) as bsde:
            try:
                execute(comparison_run)
            except ComparisonReachedPINN:
                pass
            else:
                raise AssertionError("Comparison did not reach the PINN stage")
            pinn.assert_called_once()
            bsde.assert_not_called()
        compare_training_results(ineus_result["ineus"], captured["ineus"])
    print("All linear mode integration checks passed; temporary outputs removed.")


if __name__ == "__main__":
    main()
