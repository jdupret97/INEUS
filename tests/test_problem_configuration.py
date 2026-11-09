"""Dimension/equation configuration tests without TensorFlow."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ineus import cli
from ineus.configuration import ConfigurationError, from_document, to_document, validate


class ProblemConfigurationTests(unittest.TestCase):
    def resolve(self, *arguments):
        return cli.resolve_arguments(cli.build_parser().parse_args(arguments))

    def test_single_dimension_and_no_jumps_override_full_yaml(self):
        for kind in ("linear", "nonlinear"):
            run = self.resolve("--config", str(ROOT / f"configs/{kind}.yaml"),
                               "--dimension", "100", "--jump-intensity", "0")
            self.assertEqual(run.parameters.problem.state_dim, 100)
            self.assertEqual(run.parameters.problem.dimensions, ())
            self.assertEqual(run.parameters.problem.jump_intensity, 0)
            if kind == "linear":
                self.assertEqual(run.parameters.problem.brownian_dim, 10)
            else:
                self.assertEqual(run.parameters.problem.jump_dim, 100)

    def test_partial_hjb_dimension_sets_jump_dimension(self):
        run = from_document({"experiment": "nonlinear", "parameters": {"problem": {"state_dim": 7}}})
        validate(run)
        self.assertEqual(run.parameters.problem.jump_dim, 7)

    def test_existing_equation_coefficients_are_editable(self):
        problems = {
            "linear": dict(state_dim=3, brownian_dim=2, horizon=0.7, drift=0.0,
                           discount=-0.1, diffusion_scale=0.28, diffusion_mode="constant",
                           jump_covariance_mode="isotropic", jump_covariance_scale=0.0,
                           jump_intensity=0.0, coefficient_seed=42),
            "nonlinear": dict(state_dim=3, horizon=0.7, eta=0.5,
                              driver_constant=-0.1, jump_intensity=0.0, jump_covariance_scale=0.0),
        }
        for kind, values in problems.items():
            with self.subTest(kind=kind):
                run = from_document({"experiment": kind, "parameters": {"problem": values}})
                validate(run)
                self.assertEqual(from_document(to_document(run)), run)
                for key, value in values.items():
                    self.assertEqual(getattr(run.parameters.problem, key), value)

    def test_invalid_problem_values_rejected(self):
        common = [dict(state_dim=0), dict(dimensions=[2, 2]), dict(dimensions=[-1]),
                  dict(dimensions=[True]), dict(dimensions="10,20"), dict(horizon=0),
                  dict(jump_intensity=-1), dict(jump_intensity=float("nan"))]
        specific = {
            "linear": [dict(brownian_dim=0), dict(diffusion_mode="typo"),
                       dict(jump_covariance_mode="typo"), dict(diffusion_scale=-1),
                       dict(jump_covariance_min=0.5), dict(coefficient_seed=-1)],
            "nonlinear": [dict(state_dim=3, jump_dim=2), dict(eta=0),
                          dict(jump_covariance_scale=-1)],
        }
        for kind in specific:
            for problem in common + specific[kind]:
                with self.subTest(kind=kind, problem=problem), self.assertRaises(ConfigurationError):
                    validate(from_document({"experiment": kind, "parameters": {"problem": problem}}))

    def test_sweep_profiles_and_smoke_comparisons_validate(self):
        for kind in ("linear", "nonlinear"):
            for suffix in ("sweep", "sweep_smoke"):
                run = self.resolve("--config", str(ROOT / f"configs/{kind}_{suffix}.yaml"))
                validate(run)
                self.assertTrue(run.parameters.problem.dimensions)
                self.assertEqual(from_document(to_document(run)), run)
                document = to_document(run)
                document["mode"] = "comparison"
                validate(from_document(document))

    def test_dimension_flag_disables_yaml_sweep(self):
        run = self.resolve("--config", str(ROOT / "configs/nonlinear_sweep_smoke.yaml"),
                           "--dimension", "100")
        self.assertEqual(run.parameters.problem.dimensions, ())
        self.assertEqual(run.parameters.problem.jump_dim, 100)

    def test_sweep_cli_override_and_dry_run_do_not_train(self):
        args = ["--experiment", "linear", "--dimensions", "10", "25", "100"]
        self.assertEqual(self.resolve(*args).parameters.problem.dimensions, (10, 25, 100))
        before = "tensorflow" in sys.modules
        with patch.object(cli, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            cli.main(args + ["--dry-run"])
        execute.assert_not_called()
        self.assertEqual("tensorflow" in sys.modules, before)

    def test_new_cli_flags_do_not_change_black_scholes_interfaces(self):
        for kind in ("black_scholes", "linear_black_scholes"):
            for arguments in (("--dimension", "3"), ("--jump-intensity", "0")):
                with self.subTest(kind=kind, arguments=arguments), self.assertRaises(ConfigurationError):
                    self.resolve("--experiment", kind, *arguments)


if __name__ == "__main__":
    unittest.main()
