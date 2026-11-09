"""Black–Scholes YAML/CLI checks without the numerical runtime."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ineus import cli
from ineus.black_scholes.config import default_config
from ineus.configuration import (
    ConfigurationError, dump_yaml, from_document, override_epochs,
    read_yaml, to_document, validate,
)


class BlackScholesConfigurationTests(unittest.TestCase):
    def resolve(self, document=None):
        run = from_document({"experiment": "black_scholes", **(document or {})})
        validate(run)
        return run

    def test_defaults_select_five_runs_and_ineus(self):
        run = self.resolve()
        self.assertEqual((run.mode, run.seed, run.runs), ("ineus", 1, 5))
        self.assertEqual(run.output_dir, Path("results/black_scholes"))
        self.assertEqual(run.parameters, default_config())

    def test_full_yaml_matches_default_configuration(self):
        run = from_document(read_yaml(ROOT / "configs/black_scholes.yaml"))
        validate(run)
        self.assertEqual(run.parameters, default_config())
        self.assertEqual(run.parameters.train.epochs, 400)
        self.assertFalse(run.parameters.model.constrained)
        self.assertTrue(run.parameters.problem.use_jumps)
        self.assertTrue(run.parameters.train.use_path_sampling)
        self.assertFalse(run.parameters.train.use_rad)

    def test_derived_coefficients_are_not_serialized_or_configurable(self):
        run = self.resolve()
        train = run.parameters.train
        self.assertAlmostEqual(train.zeta, 0.8)
        self.assertAlmostEqual(train.alpha, 0.8)
        self.assertAlmostEqual(train.bet_scale, 1.0 / 19.0)
        self.assertEqual(train.rad_candidate_size, 16384)
        serialized = to_document(run)
        self.assertNotIn("Sigma", serialized["parameters"]["problem"])
        self.assertNotIn("x0_eval", serialized["parameters"]["evaluation"])
        for name in ("alpha", "bet_scale", "zeta", "save_dir"):
            self.assertNotIn(name, serialized["parameters"]["train"])
        with self.assertRaises(ConfigurationError):
            self.resolve({"parameters": {"train": {"alpha": 0.8}}})

    def test_default_reference_follows_jump_setting(self):
        for problem, expected in (
            ({}, 55.810),
            ({"use_jumps": False}, 57.300),
            ({"jump_intensity": 0.0}, 57.300),
        ):
            with self.subTest(problem=problem):
                run = self.resolve({"parameters": {"problem": problem}})
                self.assertEqual(run.parameters.evaluation.u_ref, expected)
                self.assertEqual(from_document(to_document(run)), run)

    def test_full_yaml_reference_follows_jump_switch(self):
        document = read_yaml(ROOT / "configs/black_scholes.yaml")
        for use_jumps, expected in ((True, 55.810), (False, 57.300)):
            with self.subTest(use_jumps=use_jumps):
                document["parameters"]["problem"]["use_jumps"] = use_jumps
                run = from_document(document)
                validate(run)
                self.assertEqual(run.parameters.evaluation.u_ref, expected)

    def test_explicit_reference_overrides_are_preserved(self):
        for use_jumps in (True, False):
            for reference in (None, 60.0, 57.300, 55.810):
                with self.subTest(use_jumps=use_jumps, reference=reference):
                    run = self.resolve({"parameters": {
                        "problem": {"use_jumps": use_jumps},
                        "evaluation": {"u_ref": reference},
                    }})
                    self.assertEqual(run.parameters.evaluation.u_ref, reference)

    def test_roundtrip_including_nullable_reference_and_checkpoint(self):
        run = self.resolve({"runs": 1, "parameters": {
            "evaluation": {"u_ref": None},
            "train": {"resume_path": "results/run/final.pkl"},
        }})
        self.assertEqual(from_document(to_document(run)), run)
        self.assertIn("u_ref: null", dump_yaml(run))
        self.assertEqual(run.parameters.train.resume_path, "results/run/final.pkl")

    def test_optional_fields_are_strictly_typed(self):
        for section, values in (
            ("train", {"resume_path": 42}),
            ("train", {"rad_candidate_size": True}),
            ("evaluation", {"u_ref": "57.300"}),
            ("evaluation", {"u_ref": True}),
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {section: values}})

    def test_comparison_and_nonpositive_runs_are_rejected(self):
        for updates in ({"mode": "comparison"}, {"runs": 0}):
            with self.subTest(updates=updates), self.assertRaises(ConfigurationError):
                self.resolve(updates)

    def test_resume_requires_a_single_run(self):
        with self.assertRaisesRegex(ConfigurationError, "runs: 1"):
            self.resolve({"parameters": {"train": {"resume_path": "final.pkl"}}})
        run = self.resolve({"runs": 1})
        self.assertEqual(run.output_dir, Path("results/black_scholes/seed_1"))

    def test_epoch_override_changes_only_train_epochs(self):
        run = self.resolve()
        shorter = override_epochs(run, 20)
        validate(shorter)
        self.assertEqual(shorter.parameters, replace(
            run.parameters, train=replace(run.parameters.train, epochs=20)))
        self.assertEqual(run.parameters.train.epochs, 400)
        with self.assertRaises(FrozenInstanceError):
            run.parameters.train.epochs = 20

    def test_cli_overrides_and_module_selection(self):
        args = cli.build_parser("black_scholes").parse_args([
            "--config", str(ROOT / "configs/black_scholes.yaml"),
            "--seed", "3", "--epochs", "20", "--output-dir", "results/bs_seed_3",
        ])
        run = cli.resolve_arguments(args, "black_scholes")
        self.assertEqual((run.seed, run.parameters.train.epochs), (3, 20))
        self.assertEqual(run.output_dir, Path("results/bs_seed_3"))
        with patch.object(cli, "execute") as execute:
            cli.main(["--epochs", "20"], experiment="black_scholes")
        self.assertEqual(execute.call_args.args[0].experiment, "black_scholes")

    def test_smoke_keeps_100_dimensions_and_exercises_a_block(self):
        run = from_document(read_yaml(ROOT / "configs/black_scholes_smoke.yaml"))
        validate(run)
        self.assertEqual(run.parameters.problem.dimen, 100)
        self.assertEqual(run.parameters.train.epochs, 2)
        self.assertEqual(run.parameters.train.block_size, 2)
        self.assertEqual(run.parameters.train.distill_steps, 1)
        self.assertIsNone(run.parameters.evaluation.u_ref)

    def test_pde_switch_and_consistent_dimension_are_supported(self):
        run = self.resolve({"parameters": {
            "problem": {"use_jumps": False, "dimen": 3},
            "evaluation": {"dimen": 3, "u_ref": None},
        }})
        self.assertFalse(run.parameters.problem.use_jumps)
        self.assertEqual(run.parameters.problem.dimen, 3)
        with self.assertRaises(ConfigurationError):
            self.resolve({"parameters": {"problem": {"dimen": 3}}})

    def test_path_rad_and_ignored_replacement_option_are_rejected(self):
        for fields in ({"use_rad": True}, {"replace_with_student": False}):
            with self.subTest(fields=fields), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {"train": fields}})
        self.resolve({"parameters": {"train": {"use_path_sampling": False, "use_rad": True}}})

    def test_invalid_budgets_and_equation_parameters_are_rejected(self):
        for section, fields in (
            ("train", {"n_inner": 1}), ("train", {"h_scale": 0}),
            ("train", {"epochs": 0}), ("train", {"distill_steps": 0}),
            ("train", {"num_samples": 3}), ("train", {"batch_size": 4000}),
            ("train", {"rho1": 2.0}), ("train", {"save_dir": "a"}),
            ("train", {"resume_path": "  "}), ("model", {"n_layers": 1}),
            ("problem", {"v_h": 80}), ("problem", {"jump_corr": 1.1}),
        ):
            with self.subTest(section=section, fields=fields), self.assertRaises(ConfigurationError):
                self.resolve({"parameters": {section: fields}})

    def test_hard_terminal_configuration_is_supported(self):
        self.resolve({"parameters": {
            "model": {"constrained": True}, "train": {"w_bc": 0.0, "n_inner": 25},
        }})
        with self.assertRaises(ConfigurationError):
            self.resolve({"parameters": {"train": {"w_bc": 0.0}}})

    def test_dry_run_never_imports_numpy_or_tensorflow(self):
        code = """
import builtins
real_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'tensorflow'):
        raise AssertionError('Unexpected numerical import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from ineus.cli import main
main(['--config', 'configs/black_scholes.yaml', '--dry-run'])
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("experiment: black_scholes", result.stdout)


if __name__ == "__main__":
    unittest.main()
