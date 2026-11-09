"""Repeated nonlinear Black–Scholes routing; training is replaced with a stub."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None

if tf is not None:
    import numpy as np
    import pandas as pd

    from ineus.black_scholes import experiment
    from ineus.black_scholes.config import default_config
    from ineus.configuration import from_document, read_yaml


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class BlackScholesRepetitionTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.clear_session = stack.enter_context(patch.object(experiment.tf.keras.backend, "clear_session"))
        self.collect = stack.enter_context(patch.object(experiment.gc, "collect"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_one_run_preserves_result_identity_arguments_and_layout(self):
        result = {"summary": {"seed": 8}, "model_weights": ["weights"], "history": ["history"]}
        with patch.object(experiment, "_run_single_experiment", return_value=result) as single, \
                patch.object(Path, "mkdir") as mkdir:
            actual = experiment.run_experiment(seed=8, config=self.config,
                                               output_dir="results/single", runs=1)
        self.assertIs(actual, result)
        single.assert_called_once_with(seed=8, mode="ineus", config=self.config,
                                       output_dir="results/single")
        mkdir.assert_not_called()
        self.clear_session.assert_not_called()

    def test_repeated_runs_use_consecutive_seeds_and_sample_statistics(self):
        arrays = []

        def single(**kwargs):
            seed = kwargs["seed"]
            weights = np.zeros((3, 2))
            arrays.append(weakref.ref(weights))
            return {
                "summary": {"seed": seed, "prediction": 2.0 * seed,
                            "abs_error": seed / 10.0, "epochs": 3,
                            "not_scalar": [1, 2]},
                "model_weights": [weights], "history": {"data": weights},
            }

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiment, "_run_single_experiment", side_effect=single) as runner:
            result = experiment.run_experiment(seed=9, config=self.config,
                                               output_dir=directory, runs=3)
            self.assertEqual([call.kwargs["seed"] for call in runner.call_args_list], [9, 10, 11])
            for index, call in enumerate(runner.call_args_list, 1):
                run_dir = Path(directory) / f"run_{index:02d}"
                self.assertEqual(call.kwargs["output_dir"], str(run_dir))
                self.assertIs(call.kwargs["config"], self.config)
                child = from_document(read_yaml(run_dir / "config_resolved.yaml"))
                self.assertEqual((child.experiment, child.seed, child.runs),
                                 ("black_scholes", 8 + index, 1))
                self.assertEqual(child.output_dir, run_dir)
                self.assertEqual(child.parameters, self.config)
            saved = pd.read_csv(Path(directory) / "run_summaries.csv")
            self.assertEqual(saved["run"].tolist(), [1, 2, 3])
            self.assertEqual(saved["seed"].tolist(), [9, 10, 11])
            self.assertNotIn("not_scalar", saved.columns)
            statistics = pd.read_csv(Path(directory) / "run_statistics.csv").set_index("metric")
            self.assertEqual(statistics.columns.tolist(), ["mean", "std", "n_runs"])
            self.assertAlmostEqual(statistics.loc["prediction", "mean"], 20.0)
            self.assertAlmostEqual(statistics.loc["prediction", "std"], 2.0)
            self.assertEqual(statistics.loc["prediction", "n_runs"], 3)
            self.assertNotIn("seed", statistics.index)
            self.assertNotIn("run", statistics.index)
            self.assertEqual(result["runs"], 3)
            self.assertEqual(result["summary_path"], str(Path(directory) / "run_summaries.csv"))
            self.assertEqual(result["statistics_path"], str(Path(directory) / "run_statistics.csv"))
            self.assertNotIn("model_weights", result)
            self.assertNotIn("history", result)
        self.assertTrue(all(reference() is None for reference in arrays))
        self.assertEqual(self.clear_session.call_count, 4)
        self.assertEqual(self.collect.call_count, 4)

    def test_default_run_count_is_five(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiment, "_run_single_experiment",
                             return_value={"summary": {"prediction": 55.0}}) as runner:
            result = experiment.run_experiment(output_dir=directory)
        self.assertEqual(result["runs"], 5)
        self.assertEqual([call.kwargs["seed"] for call in runner.call_args_list], [1, 2, 3, 4, 5])

    def test_completed_summaries_survive_a_later_run_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiment, "_run_single_experiment", side_effect=[
                    {"summary": {"prediction": 55.0}}, RuntimeError("training failed"),
                ]):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                experiment.run_experiment(output_dir=directory, runs=3)
            frame = pd.read_csv(Path(directory) / "run_summaries.csv")
            self.assertEqual(frame["run"].tolist(), [1])
            statistics = pd.read_csv(Path(directory) / "run_statistics.csv")
            self.assertEqual(statistics["n_runs"].tolist(), [1])
            self.assertTrue(statistics["std"].isna().all())
            self.assertFalse((Path(directory) / "run_03").exists())
        self.assertEqual(self.clear_session.call_count, 3)

    def test_invalid_counts_seeds_and_modes_fail_before_writes(self):
        for kwargs in (
            {"runs": 0}, {"runs": -1}, {"runs": True}, {"runs": 1.5},
            {"seed": -1}, {"seed": True}, {"seed": 2**32},
            {"seed": 2**32 - 1, "runs": 2}, {"mode": "comparison"},
        ):
            with self.subTest(kwargs=kwargs), patch.object(Path, "mkdir") as mkdir, \
                    patch.object(experiment, "_run_single_experiment") as single:
                with self.assertRaises(ValueError):
                    experiment.run_experiment(**kwargs)
                mkdir.assert_not_called()
                single.assert_not_called()
        self.clear_session.assert_not_called()

    def test_resume_is_supported_only_for_a_single_run(self):
        config = replace(self.config, train=replace(self.config.train, resume_path="trusted.pkl"))
        with patch.object(Path, "mkdir") as mkdir, \
                patch.object(experiment, "_run_single_experiment") as single:
            with self.assertRaisesRegex(ValueError, "resume_path"):
                experiment.run_experiment(config=config, runs=2)
            mkdir.assert_not_called()
            single.assert_not_called()
            result = experiment.run_experiment(config=config, runs=1)
        self.assertIs(result, single.return_value)
        single.assert_called_once_with(seed=1, mode="ineus", config=config, output_dir=None)
        self.clear_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
