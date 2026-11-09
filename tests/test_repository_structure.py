"""TensorFlow-free repository checks."""

from __future__ import annotations

import importlib.util
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_repository.py"

SPEC = importlib.util.spec_from_file_location("validate_repository", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load validator from {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


class RepositoryStructureTests(unittest.TestCase):
    def test_required_repository_files_exist(self):
        self.assertEqual([], VALIDATOR.check_required_paths())

    def test_python_sources_parse_without_importing_tensorflow(self):
        self.assertEqual([], VALIDATOR.check_python_syntax())

    def test_local_package_imports_resolve(self):
        self.assertEqual([], VALIDATOR.check_local_imports())

    def test_public_modules_expose_architectures_and_training(self):
        self.assertEqual([], VALIDATOR.check_public_exports())

    def test_supported_entry_points_are_guarded(self):
        self.assertEqual([], VALIDATOR.check_entry_points())

    def test_default_configuration_matches_expected_parameters(self):
        self.assertEqual([], VALIDATOR.check_default_configuration())

    def test_training_diagnostics_are_available(self):
        self.assertEqual([], VALIDATOR.check_diagnostics())

    def test_diagnostics_check_rejects_missing_trainer_definitions(self):
        with patch.object(VALIDATOR, "parse_python", return_value=ast.parse("")):
            issues = VALIDATOR.check_diagnostics()
        self.assertEqual(4, len(issues))
        self.assertTrue(all("missing" in issue.detail for issue in issues))

    def test_public_export_check_rejects_absent_reexports(self):
        with patch.object(VALIDATOR, "parse_python", return_value=ast.parse("")):
            issues = VALIDATOR.check_public_exports()
        self.assertTrue(issues)
        self.assertTrue(any("DGMCell_V" in issue.detail for issue in issues))
        self.assertTrue(any("train_bellman_ct_nl" in issue.detail for issue in issues))

    def test_readme_documents_commands_and_links_experiment_guides(self):
        self.assertEqual([], VALIDATOR.check_readme())


if __name__ == "__main__":
    unittest.main()
