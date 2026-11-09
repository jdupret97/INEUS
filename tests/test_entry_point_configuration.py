"""Configuration/CLI tests: no TensorFlow import or numerical training."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ineus import cli
from ineus.configuration import (
    ConfigurationError, dump_yaml, from_document, override_epochs,
    read_yaml, to_document, validate,
)
from ineus.linear.config import DEFAULT_CONFIG
from ineus.nonlinear.config import default_config


class EntryPointConfigurationTests(unittest.TestCase):
    def resolve(self, arguments, experiment=None):
        args = cli.build_parser(experiment).parse_args(arguments)
        return cli.resolve_arguments(args, experiment)

    def test_linear_no_arguments_uses_default_settings(self):
        run = self.resolve([], "linear")
        self.assertEqual(run.seed, 29)
        self.assertEqual(run.mode, "comparison")
        self.assertEqual(run.runs, 10)
        self.assertEqual(run.parameters, DEFAULT_CONFIG)
        self.assertEqual(run.output_dir, Path(DEFAULT_CONFIG.evaluation.default_output_dir))

    def test_nonlinear_no_arguments_uses_default_settings(self):
        run = self.resolve([], "nonlinear")
        self.assertEqual(run.seed, 29)
        self.assertEqual(run.runs, 10)
        self.assertEqual(run.parameters, default_config())

    def test_shipped_yaml_files_match_default_settings(self):
        for kind, defaults in (("linear", DEFAULT_CONFIG), ("nonlinear", default_config())):
            with self.subTest(experiment=kind):
                run = self.resolve(["--config", str(ROOT / "configs" / f"{kind}.yaml")])
                self.assertEqual(run.parameters, defaults)
                self.assertEqual(run.mode, "comparison")
                self.assertEqual(run, self.resolve([], kind))

    def test_twenty_epochs_changes_only_four_loop_counts(self):
        run = self.resolve(["--epochs", "20"], "linear")
        self.assertEqual(run.parameters, replace(
            DEFAULT_CONFIG,
            ineus=replace(DEFAULT_CONFIG.ineus, epochs=20),
            pinn=replace(DEFAULT_CONFIG.pinn, epochs=20),
            bsde=replace(DEFAULT_CONFIG.bsde, local_epochs=20, surrogate_epochs=20),
        ))
        defaults = default_config()
        run = self.resolve(["--epochs", "20"], "nonlinear")
        self.assertEqual(run.parameters, replace(
            defaults,
            train=replace(defaults.train, epochs_ineus=20, epochs_pinn=20),
            bsde=replace(defaults.bsde, local_epochs=20, surrogate_epochs=20),
        ))

    def test_command_line_overrides_yaml(self):
        run = self.resolve(["--config", str(ROOT / "configs/nonlinear.yaml"),
                            "--mode", "ineus", "--seed", "41", "--runs", "1",
                            "--output-dir", "results/override", "--epochs", "20"])
        self.assertEqual((run.mode, run.seed, run.runs), ("ineus", 41, 1))
        self.assertEqual(run.output_dir, Path("results/override"))
        self.assertEqual(run.parameters.base_seed, 41)
        self.assertEqual(run.parameters.runs, 1)
        self.assertEqual(run.parameters.evaluation, default_config().evaluation)

    def test_yaml_roundtrip_and_frozen_defaults(self):
        for kind in ("linear", "nonlinear"):
            with self.subTest(experiment=kind):
                yaml_defaults = self.resolve([
                    "--config", str(ROOT / "configs" / f"{kind}.yaml"),
                ]).parameters
                run = self.resolve(["--mode", "ineus", "--epochs", "20"], kind)
                self.assertEqual(from_document(to_document(run)), run)
                # CLI overrides must not mutate the defaults defined by the full YAML.
                self.assertEqual(self.resolve([], kind).parameters, yaml_defaults)

    def test_smoke_profiles_validate_and_are_explicitly_not_defaults(self):
        for kind in ("linear", "nonlinear"):
            run = self.resolve(["--config", str(ROOT / "configs" / f"{kind}_smoke.yaml")])
            validate(run)
            self.assertEqual(run.parameters.bsde.local_epochs, 20)
            self.assertEqual(run.parameters.bsde.num_local_models, 5)
            self.assertLess(run.parameters.evaluation.test_samples, 2000)

    def test_unknown_keys_and_wrong_scalar_types_rejected(self):
        documents = [
            {"experiment": "linear", "epohs": 20},
            {"experiment": "linear", "parameters": {"ineus": {"epohs": 20}}},
            {"experiment": "linear", "parameters": {"ineus": {"epochs": True}}},
            {"experiment": "linear", "parameters": {"ineus": {"learning_rate": "1e-4"}}},
            {"experiment": "nonlinear", "parameters": {"base_seed": 29}},
            {"experiment": "linear", "parameters": {"evaluation": {"default_output_dir": "a"}}},
        ]
        for document in documents:
            with self.subTest(document=document), self.assertRaises(ConfigurationError):
                from_document(document)

    def test_invalid_budgets_fail_before_training(self):
        for kind, args in (
            ("linear", ["--epochs", "0"]),
            ("linear", ["--runs", "0"]),
            ("nonlinear", ["--runs", "0"]),
            ("nonlinear", ["--epochs", "2"]),
            ("linear", ["--seed", "-1"]),
        ):
            with self.subTest(kind=kind, args=args), self.assertRaises(ConfigurationError):
                self.resolve(args, kind)

    def test_problem_changes_are_configurable(self):
        run = from_document({"experiment": "linear", "parameters": {"problem": {"horizon": 1.0}}})
        validate(run)
        self.assertEqual(run.parameters.problem.horizon, 1.0)
        self.assertEqual(DEFAULT_CONFIG.problem.horizon, 0.5)

    def test_shared_fields_required_only_for_comparison(self):
        document = {"experiment": "linear", "parameters": {"ineus": {"batch_size": 32}}}
        with self.assertRaisesRegex(ConfigurationError, "set both equally"):
            validate(from_document(document))
        validate(from_document({**document, "mode": "ineus"}))

    def test_wrong_experiment_launcher_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.resolve(["--config", str(ROOT / "configs/nonlinear.yaml")], "linear")

    def test_dry_run_does_not_execute_training_or_import_tensorflow(self):
        before = "tensorflow" in sys.modules
        with patch.object(cli, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(["--config", str(ROOT / "configs/linear.yaml"), "--dry-run"])
        execute.assert_not_called()
        self.assertIn("experiment: linear", output.getvalue())
        self.assertEqual("tensorflow" in sys.modules, before)

    def test_main_dispatches_selected_mode(self):
        with patch.object(cli, "execute") as execute:
            self.assertIsNone(cli.main(["--mode", "ineus"], experiment="linear"))
        self.assertEqual(execute.call_args.args[0].mode, "ineus")

    def test_yaml_safe_loader_rejects_duplicates_and_object_tags(self):
        outputs = ROOT / "results"
        outputs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=outputs) as directory:
            path = Path(directory) / "invalid.yaml"
            for content in (
                "experiment: linear\nexperiment: nonlinear\n",
                "!!python/object/apply:builtins.print ['unsafe']",
                "- linear\n- nonlinear\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ConfigurationError):
                    read_yaml(path)

    def test_execute_records_resolved_yaml_and_calls_only_selected_runner(self):
        outputs = ROOT / "results"
        outputs.mkdir(exist_ok=True)
        for kind in ("linear", "nonlinear", "black_scholes", "linear_black_scholes"):
            with tempfile.TemporaryDirectory(dir=outputs) as directory:
                run = self.resolve(["--mode", "ineus", "--output-dir", directory], kind)
                captured = []

                def runner(**kwargs):
                    captured.append(kwargs)
                    return {"result": "stub"}

                with patch.object(cli.importlib, "import_module", return_value=SimpleNamespace(run_experiment=runner)), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.execute(run), {"result": "stub"})
                saved = from_document(read_yaml(Path(directory) / "config_resolved.yaml"))
                self.assertEqual(saved, run)
                self.assertEqual(captured[0]["mode"], "ineus")
                self.assertEqual(captured[0]["config"], run.parameters)
                self.assertEqual(captured[0]["runs"], run.runs)

    def test_repeat_count_can_be_overridden_for_every_experiment(self):
        for kind in ("linear", "nonlinear", "black_scholes", "linear_black_scholes"):
            for runs in (1, 5, 10):
                with self.subTest(kind=kind, runs=runs):
                    run = self.resolve(["--runs", str(runs)], kind)
                    self.assertEqual(run.runs, runs)


if __name__ == "__main__":
    unittest.main()
