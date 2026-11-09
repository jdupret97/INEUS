"""Routing and reporting checks for independent linear PIDE repetitions."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None

if tf is not None:
    import pandas as pd

    from ineus.configuration import read_yaml
    from ineus.linear import experiment
    from ineus.linear.config import DEFAULT_CONFIG


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class LinearRepetitionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="ineus-linear-repetitions-")
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name) / "results"
        for target in ("builtins.print", "ineus.linear.experiment.gc.collect"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("ineus.linear.experiment.tf.keras.backend.clear_session")
        self.clear_session = patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_run_preserves_return_value_and_output_layout(self):
        expected = object()
        with patch.object(experiment, "_run_single_experiment", return_value=expected) as single:
            result = experiment.run_experiment(
                seed=17, output_dir=self.output_dir, mode="ineus", runs=1,
            )
        self.assertIs(result, expected)
        single.assert_called_once_with(
            seed=17, output_dir=self.output_dir, config=DEFAULT_CONFIG, mode="ineus",
        )
        self.assertFalse(self.output_dir.exists())
        self.clear_session.assert_not_called()

    def test_ineus_repetitions_use_distinct_seeds_and_save_scalar_statistics(self):
        def run_one(*, seed, output_dir, config, mode):
            self.assertIs(config, DEFAULT_CONFIG)
            self.assertEqual(mode, "ineus")
            resolved = read_yaml(output_dir / "config_resolved.yaml")
            self.assertEqual(resolved["seed"], seed)
            self.assertEqual(resolved["runs"], 1)
            self.assertEqual(resolved["experiment"], "linear")
            self.assertEqual(resolved["mode"], "ineus")
            self.assertEqual(Path(resolved["output_dir"]), output_dir)
            return {
                "summary": pd.DataFrame([{"final_mae_ineus": float(seed - 10)}]),
                "ineus": object(),
            }

        with patch.object(experiment, "_run_single_experiment", side_effect=run_one) as single:
            result = experiment.run_experiment(
                seed=11, output_dir=self.output_dir, mode="ineus", runs=3,
            )

        self.assertEqual([call.kwargs["seed"] for call in single.call_args_list], [11, 12, 13])
        self.assertEqual(
            [call.kwargs["output_dir"].name for call in single.call_args_list],
            ["run_01", "run_02", "run_03"],
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(result["summary_path"]), result["run_summaries"],
        )
        statistics = result["run_statistics"].set_index("metric")
        self.assertEqual(statistics.index.tolist(), ["final_mae_ineus"])
        self.assertEqual(statistics.loc["final_mae_ineus", "mean"], 2.0)
        self.assertEqual(statistics.loc["final_mae_ineus", "std"], 1.0)
        self.assertEqual(statistics.loc["final_mae_ineus", "n_runs"], 3)
        self.assertEqual(result["runs"], 3)
        self.assertNotIn("ineus", result)
        self.assertEqual(self.clear_session.call_count, 4)

    def test_comparison_reads_each_unchanged_three_method_summary(self):
        def run_one(*, seed, output_dir, **kwargs):
            pd.DataFrame([{
                "final_mae_bellman": float(seed),
                "final_mae_pinn": float(seed + 1),
                "final_mae_bsde": float(seed + 2),
            }]).to_csv(output_dir / "three_method_summary.csv", index=False)
            return (object(), object(), object(), object())

        with patch.object(experiment, "_run_single_experiment", side_effect=run_one):
            result = experiment.run_experiment(seed=4, output_dir=self.output_dir, runs=2)
        self.assertEqual(result["run_summaries"]["seed"].tolist(), [4, 5])
        statistics = pd.read_csv(result["statistics_path"]).set_index("metric")
        self.assertEqual(statistics.loc["final_mae_bellman", "mean"], 4.5)
        self.assertEqual(statistics.loc["final_mae_pinn", "mean"], 5.5)
        self.assertEqual(statistics.loc["final_mae_bsde", "mean"], 6.5)
        self.assertTrue((statistics["n_runs"] == 2).all())

    def test_default_is_ten_repetitions(self):
        summary = {"summary": pd.DataFrame([{"final_mae_ineus": 0.1}])}
        with patch.object(experiment, "_run_single_experiment", return_value=summary) as single:
            result = experiment.run_experiment(output_dir=self.output_dir, mode="ineus")
        self.assertEqual(single.call_count, 10)
        self.assertEqual(result["run_summaries"]["seed"].tolist(), list(range(29, 39)))

    def test_invalid_repetition_settings_fail_before_writing_or_training(self):
        invalid = (
            {"runs": 0}, {"runs": -1}, {"runs": True}, {"runs": 1.5},
            {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"seed": 1.5},
            {"seed": 2**32 - 1, "runs": 2}, {"mode": "unknown"},
        )
        with patch.object(experiment, "_run_single_experiment") as single:
            for settings in invalid:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    experiment.run_experiment(output_dir=self.output_dir, **settings)
                self.assertFalse(self.output_dir.exists())
            single.assert_not_called()

    def test_later_failure_keeps_completed_summaries_and_cleans_the_session(self):
        completed = {"summary": pd.DataFrame([{"final_mae_ineus": 0.1}])}
        with patch.object(
            experiment, "_run_single_experiment",
            side_effect=[completed, RuntimeError("interrupted")],
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                experiment.run_experiment(output_dir=self.output_dir, mode="ineus", runs=3)
        summaries = pd.read_csv(self.output_dir / "run_summaries.csv")
        self.assertEqual(summaries["run"].tolist(), [1])
        self.assertEqual(summaries["seed"].tolist(), [29])
        self.assertFalse((self.output_dir / "run_statistics.csv").exists())
        self.assertEqual(self.clear_session.call_count, 3)


if __name__ == "__main__":
    unittest.main()
