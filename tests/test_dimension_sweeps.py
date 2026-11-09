"""Sequential sweep orchestration with tiny scalar-only stub runners."""

import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ineus.configuration import dump_yaml, from_document, read_yaml
from ineus.sweeps import run_dimension_sweep


class DimensionSweepTests(unittest.TestCase):
    def make_run(self, directory, kind, mode, runs):
        return from_document(dict(experiment=kind, mode=mode, runs=runs, output_dir=directory,
                                  parameters={"problem": {"dimensions": [2, 3]}}))

    def write_summary(self, child):
        child.output_dir.mkdir(parents=True, exist_ok=True)
        (child.output_dir / "config_resolved.yaml").write_text(dump_yaml(child), encoding="utf-8")
        if child.experiment == "linear":
            name = ("run_summaries.csv" if child.runs > 1 else
                    "ineus_summary.csv" if child.mode == "ineus" else "three_method_summary.csv")
        else:
            prefix = "ineus_summary" if child.mode == "ineus" else "summary"
            name = f"{prefix}_over_{child.runs}_runs.csv"
        with (child.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["MAE", "mode"])
            writer.writeheader()
            for index in range(child.runs):
                writer.writerow({"MAE": child.parameters.problem.state_dim + index, "mode": child.mode})

    def test_modes_runs_and_dimensions_have_isolated_outputs(self):
        for kind in ("linear", "nonlinear"):
            for mode in ("ineus", "comparison"):
                for runs in (1, 2):
                    with self.subTest(kind=kind, mode=mode, runs=runs), tempfile.TemporaryDirectory() as directory:
                        run = self.make_run(directory, kind, mode, runs)
                        children = []

                        def execute(child):
                            children.append(child)
                            self.write_summary(child)

                        with patch("ineus.sweeps._clear_runtime"), contextlib.redirect_stdout(io.StringIO()):
                            result = run_dimension_sweep(run, execute)
                        self.assertEqual(run.parameters.problem.state_dim, 10)
                        self.assertEqual(run.parameters.problem.dimensions, (2, 3))
                        for dimension, child in zip((2, 3), children):
                            self.assertEqual(child.parameters.problem.state_dim, dimension)
                            self.assertEqual(child.parameters.problem.dimensions, ())
                            self.assertEqual(child.seed, run.seed)
                            self.assertEqual(child.runs, runs)
                            other_dim = (child.parameters.problem.brownian_dim if kind == "linear"
                                         else child.parameters.problem.jump_dim)
                            self.assertEqual(other_dim, 10 if kind == "linear" else dimension)
                            self.assertEqual(child.output_dir.name, f"dimen_{dimension:03d}")
                            self.assertEqual(from_document(read_yaml(child.output_dir / "config_resolved.yaml")), child)
                        with Path(result["summary_path"]).open() as stream:
                            summaries = list(csv.DictReader(stream))
                        self.assertEqual(len(summaries), 2 * runs)
                        self.assertEqual([int(row["seed"]) for row in summaries], list(range(29, 29 + runs)) * 2)
                        with Path(result["statistics_path"]).open() as stream:
                            stats = list(csv.DictReader(stream))
                        self.assertEqual(len(stats), 2)
                        self.assertEqual(float(stats[0]["mean"]), 2.0 + (runs - 1) / 2)
                        self.assertEqual(int(stats[0]["n_runs"]), runs)
                        if runs == 1:
                            self.assertEqual(stats[0]["std"], "")

    def test_completed_dimension_summary_survives_next_dimension_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory, "linear", "ineus", 1)

            def execute(child):
                if child.parameters.problem.state_dim == 3:
                    raise RuntimeError("test failure")
                self.write_summary(child)

            with patch("ineus.sweeps._clear_runtime"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    run_dimension_sweep(run, execute)
            with (Path(directory) / "dimension_run_summaries.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dimension"], "2")


if __name__ == "__main__":
    unittest.main()
