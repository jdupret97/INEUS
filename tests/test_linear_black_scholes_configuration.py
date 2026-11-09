"""Linear basket-call dimension-sweep configuration without numerical imports."""

import contextlib
from dataclasses import FrozenInstanceError, replace
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ineus import cli
from ineus.configuration import (
    ConfigurationError, dump_yaml, from_document, override_epochs,
    read_yaml, to_document, validate,
)
from ineus.linear_black_scholes.config import default_config


class LinearBlackScholesConfigurationTests(unittest.TestCase):
    def resolve(self, document=None):
        run = from_document({"experiment": "linear_black_scholes", **(document or {})})
        validate(run)
        return run

    def test_defaults_match_full_yaml(self):
        run = self.resolve()
        self.assertEqual((run.mode, run.seed, run.runs), ("ineus", 23, 10))
        self.assertEqual(run.parameters, default_config())
        self.assertEqual(run.parameters.problem.dimensions, (10,))
        self.assertEqual(run.parameters.train.batch_size, 1024)
        self.assertEqual(run.parameters.model.n_layers, 3)
        self.assertEqual(run.output_dir, Path("rPINN/Black_Scholes/Black_Scholes_dimension_sweep"))
        full = from_document(read_yaml(ROOT / "configs/linear_black_scholes.yaml"))
        validate(full)
        self.assertEqual(full, run)

    def test_dimension_list_roundtrip_is_immutable_and_ordered(self):
        run = self.resolve({"parameters": {"problem": {"dimensions": [10, 2, 5]}}})
        self.assertEqual(run.parameters.problem.dimensions, (10, 2, 5))
        self.assertEqual(to_document(run)["parameters"]["problem"]["dimensions"], [10, 2, 5])
        self.assertEqual(from_document(to_document(run)), run)
        self.assertNotIn("!!python", dump_yaml(run))
        with self.assertRaises(FrozenInstanceError):
            run.parameters.problem.dimensions = (1,)

    def test_dimensions_reject_wrong_types_empty_duplicates_and_nonpositive(self):
        for dimensions in ([], [0], [-1], [2, 2], [True], [2.0], ["2"], "2,3", 2, None):
            with self.subTest(dimensions=dimensions), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {"problem": {"dimensions": dimensions}}})

    def test_derived_coefficients_are_not_serialized(self):
        run = self.resolve()
        self.assertEqual(run.parameters.problem.K, 30.0)
        self.assertAlmostEqual(run.parameters.train.alpha, 0.8)
        self.assertAlmostEqual(run.parameters.train.bet_scale, 0.5 / 19)
        document = to_document(run)
        self.assertNotIn("K", document["parameters"]["problem"])
        self.assertNotIn("alpha", document["parameters"]["train"])
        self.assertNotIn("bet_scale", document["parameters"]["train"])
        self.assertFalse(run.parameters.train.run_phase2)
        self.assertTrue(run.parameters.train.use_distill)
        self.assertTrue(run.parameters.train.use_rad)

    def test_epoch_override_changes_only_epoch_count(self):
        run = self.resolve()
        shorter = override_epochs(run, 20)
        validate(shorter)
        self.assertEqual(shorter.parameters, replace(
            run.parameters, train=replace(run.parameters.train, epochs=20)))
        self.assertEqual((shorter.seed, shorter.runs), (23, 10))
        self.assertEqual(run.parameters.train.epochs, 1000)

    def test_cli_selects_dimension_sweep_and_overrides_run_controls(self):
        args = cli.build_parser("linear_black_scholes").parse_args([
            "--config", str(ROOT / "configs/linear_black_scholes.yaml"),
            "--runs", "2", "--seed", "42", "--epochs", "20",
            "--output-dir", "results/basket_check",
        ])
        run = cli.resolve_arguments(args, "linear_black_scholes")
        self.assertEqual((run.runs, run.seed, run.parameters.train.epochs), (2, 42, 20))
        self.assertEqual(run.output_dir, Path("results/basket_check"))
        with patch.object(cli, "execute") as execute:
            cli.main(["--epochs", "20"], experiment="linear_black_scholes")
        self.assertEqual(execute.call_args.args[0].experiment, "linear_black_scholes")

    def test_execute_passes_runs_to_dimension_sweep(self):
        run = self.resolve({"runs": 2, "seed": 42})
        runner = Mock(return_value={"result": "stub"})
        with patch.object(cli.importlib, "import_module", return_value=SimpleNamespace(run_experiment=runner)) as importer, \
                patch.object(Path, "mkdir"), patch.object(Path, "write_text"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.execute(run), {"result": "stub"})
        importer.assert_called_once_with("ineus.linear_black_scholes.experiment")
        runner.assert_called_once_with(seed=42, mode="ineus", config=run.parameters,
                                       output_dir=str(run.output_dir), runs=2)

    def test_comparison_and_invalid_seed_ranges_are_rejected(self):
        for updates in ({"mode": "comparison"}, {"runs": 0}, {"seed": -1},
                        {"seed": 2**32 - 1, "runs": 2}):
            with self.subTest(updates=updates), self.assertRaises(ConfigurationError):
                self.resolve(updates)

    def test_invalid_numerical_budgets_are_rejected(self):
        for section, updates in (
            ("train", {"epochs": 0}), ("train", {"n_inner": 1}),
            ("train", {"relaxation": 0.6}), ("train", {"h_scale": 0}),
            ("train", {"rho1": 1.1}), ("train", {"batch_size": 10000}),
            ("train", {"rad_candidate_multiplier": 0}), ("model", {"n_layers": 1}),
            ("problem", {"max_S": 0.0001}), ("problem", {"sample_b": 0.001}),
            ("problem", {"correlation": 1.0}), ("problem", {"correlation": -0.2}),
            ("problem", {"jump_std": -0.1}), ("evaluation", {"mc_test_paths": 1}),
            ("evaluation", {"mc_curve_paths": 1}), ("evaluation", {"n_x_curve": 1}),
            ("evaluation", {"n_t_curve": 1}), ("evaluation", {"output_tag": "../run"}),
            ("evaluation", {"output_tag": ""}), ("evaluation", {"output_tag": ".."}),
        ):
            with self.subTest(section=section, updates=updates), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {section: updates}})

    def test_wrong_scalar_types_and_unknown_keys_are_rejected(self):
        for section, updates in (
            ("train", {"run_phase2": 1}), ("train", {"epochs": True}),
            ("train", {"lr1": "0.0005"}), ("problem", {"sigma": True}),
            ("problem", {"jump_mean": float("nan")}), ("problem", {"K": 30.0}),
            ("train", {"alpha": 0.8}), ("train", {"epohs": 10}),
        ):
            with self.subTest(section=section, updates=updates), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {section: updates}})

    def test_optional_phase_and_sampling_switches_remain_supported(self):
        run = self.resolve({"parameters": {"train": {
            "run_phase2": True, "use_rad": False, "use_distill": False,
            "replace_with_student": False,
        }}})
        self.assertTrue(run.parameters.train.run_phase2)
        self.assertFalse(run.parameters.train.replace_with_student)
        self.assertEqual(from_document(to_document(run)), run)

    def test_smoke_exercises_two_dimensions_rad_and_distillation(self):
        run = from_document(read_yaml(ROOT / "configs/linear_black_scholes_smoke.yaml"))
        validate(run)
        self.assertEqual(run.parameters.problem.dimensions, (2, 3))
        self.assertEqual((run.runs, run.parameters.train.epochs), (1, 2))
        self.assertEqual(run.parameters.train.rad_period, 1)
        self.assertEqual(run.parameters.train.distill_block_size, 1)
        self.assertEqual(run.parameters.train.distill_steps, 1)
        self.assertEqual(run.parameters.evaluation.mc_test_paths, 20)

    def test_dry_run_does_not_import_numpy_or_tensorflow(self):
        code = """
import builtins
real_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'tensorflow'):
        raise AssertionError('Unexpected numerical import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from ineus.cli import main
main(['--config', 'configs/linear_black_scholes.yaml', '--dry-run'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("experiment: linear_black_scholes", result.stdout)


if __name__ == "__main__":
    unittest.main()
