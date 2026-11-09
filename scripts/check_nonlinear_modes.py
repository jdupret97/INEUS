"""Exercise INEUS-only nonlinear runs without training either baseline.

Run with the repository's TensorFlow environment:

    python scripts/check_nonlinear_modes.py

All generated outputs live in a temporary subdirectory of ignored results/.
This is an integration check with deliberately tiny, non-publication settings.
"""

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ineus.cli import execute
from ineus.configuration import RunConfiguration, read_yaml
from ineus.nonlinear import experiment
from ineus.nonlinear.config import default_config


class ComparisonReachedPINN(Exception):
    """Stop comparison immediately after capturing its INEUS stage."""


def tiny_config():
    config = default_config()
    return replace(
        config,
        runs=1,
        model=replace(config.model, hidden_dim=4),
        train=replace(
            config.train, epochs_ineus=1, epochs_pinn=1,
            batch_size=4, num_samples=8, buffer_size=32, n_inner=2,
            num_batches=1, averaging_block_size=1, distillation_steps=1,
            rad_period=1, rad_candidate_size=8, rad_eval_batch_size=4,
            jump_samples=3, test_jump_samples=3, rad_jump_samples=3,
            log_every=1, num_t0_each_epoch=2,
        ),
        evaluation=replace(
            config.evaluation, test_samples=4,
            reference_paths=23, reference_chunk_size=7,
        ),
    )


def assert_run_files(folder):
    history = pd.read_csv(folder / "ineus_history.csv")
    summary = pd.read_csv(folder / "ineus_summary.csv")
    assert history["epoch"].tolist() == [1]
    assert summary["final_logged_epoch"].tolist() == [1]
    assert np.isfinite(history["mae"]).all()
    assert (folder / "mae_vs_time_ineus.png").stat().st_size > 0
    assert not (folder / "three_method_history.csv").exists()
    assert not (folder / "classical_local_transformed_bsde").exists()


def main():
    config = tiny_config()
    output_root = ROOT / "results"
    output_root.mkdir(exist_ok=True)
    comparison_ineus = []
    train_function = experiment.train_bellman_ct_nl

    def capture_ineus(*args, **kwargs):
        result = train_function(*args, **kwargs)
        comparison_ineus.append(result)
        return result

    with TemporaryDirectory(prefix="check-nonlinear-modes-", dir=output_root) as folder:
        folder = Path(folder)
        with patch.object(experiment, "train_bellman_ct_nl", side_effect=capture_ineus), \
                patch.object(experiment, "train_standard_pinn_nl",
                             side_effect=ComparisonReachedPINN) as pinn, \
                patch.object(experiment, "run_classical_local_bsde_to_global_surrogate_nl",
                             side_effect=AssertionError("BSDE must not train")) as bsde:
            try:
                experiment.run_experiment(
                    config=config, runs=1, seed=29, mode="comparison",
                    output_dir=folder / "comparison-prefix",
                )
            except ComparisonReachedPINN:
                pass
            else:
                raise AssertionError("The comparison did not reach the PINN stage.")
            assert pinn.call_count == 1
            bsde.assert_not_called()
        assert len(comparison_ineus) == 1

        for runs in (1, 2):
            destination = folder / f"ineus-{runs}-runs"
            run = RunConfiguration(
                experiment="nonlinear", mode="ineus", seed=29, runs=runs,
                output_dir=destination, parameters=replace(config, runs=runs),
            )
            with patch.object(experiment, "train_standard_pinn_nl",
                              side_effect=AssertionError("PINN must not train")) as pinn, \
                    patch.object(experiment, "run_classical_local_bsde_to_global_surrogate_nl",
                                 side_effect=AssertionError("BSDE must not train")) as bsde:
                result = execute(run)
                pinn.assert_not_called()
                bsde.assert_not_called()
            assert result["mode"] == "ineus"
            assert len(result["runs"]) == runs
            assert len(result["summary_runs"]) == runs
            assert result["agg_ineus"] is None
            assert result["summary_runs"]["seed"].tolist() == list(range(29, 29 + runs))
            assert (destination / f"ineus_summary_over_{runs}_runs.csv").is_file()
            resolved = read_yaml(destination / "config_resolved.yaml")
            assert resolved["mode"] == "ineus"
            assert resolved["runs"] == runs
            for index in range(runs):
                assert_run_files(destination / f"run_{index + 1:02d}")

            actual = result["runs"][0]["ineus"]
            expected = comparison_ineus[0]
            for key in ("epoch", "mae", "train_loss", "test_residual"):
                np.testing.assert_allclose(
                    actual["history"][key], expected["history"][key],
                    rtol=1e-6, atol=1e-6, err_msg=f"INEUS mode parity: {key}",
                )
            np.testing.assert_allclose(actual["best_loss"], expected["best_loss"])
            for actual_weight, expected_weight in zip(
                actual["best_weights"], expected["best_weights"],
            ):
                np.testing.assert_allclose(
                    actual_weight, expected_weight, rtol=1e-6, atol=1e-6,
                )
            print(f"PASS INEUS-only {runs} run(s): output, seed, no baselines, trajectory")
    print("All nonlinear mode integration checks passed.")


if __name__ == "__main__":
    main()
