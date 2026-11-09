"""Sequential dimension sweeps for the quadratic linear and HJB experiments."""

import csv
from dataclasses import replace
import gc
import math
from pathlib import Path
import statistics
import sys

from .configuration import dump_yaml, validate


def _write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_run_summaries(run):
    if run.experiment == "linear":
        filename = ("run_summaries.csv" if run.runs > 1 else
                    "ineus_summary.csv" if run.mode == "ineus" else "three_method_summary.csv")
    else:
        prefix = "ineus_summary" if run.mode == "ineus" else "summary"
        filename = f"{prefix}_over_{run.runs}_runs.csv"
    with (run.output_dir / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        row.setdefault("run", index + 1)
        row.setdefault("seed", run.seed + index)
    return rows


def _statistics(rows):
    groups = {}
    metadata = {"dimension", "brownian_dim", "jump_dim", "run", "seed", "experiment"}
    for row in rows:
        for name, value in row.items():
            if name in metadata:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                groups.setdefault((row["dimension"], name), []).append(value)
    return [dict(dimension=dimension, metric=name, mean=statistics.mean(values),
                 std=statistics.stdev(values) if len(values) > 1 else "", n_runs=len(values))
            for (dimension, name), values in groups.items()]


def _clear_runtime():
    # Do not import TensorFlow for configuration tests or dry runs.
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        tensorflow.keras.backend.clear_session()
    gc.collect()


def run_dimension_sweep(run, execute):
    """Run each dimension with isolated settings and incrementally save summaries."""
    validate(run)
    children = []
    for dimension in run.parameters.problem.dimensions:
        changes = dict(state_dim=dimension, dimensions=())
        if run.experiment == "nonlinear":
            changes["jump_dim"] = dimension
        parameters = replace(run.parameters, problem=replace(run.parameters.problem, **changes))
        child = replace(run, parameters=parameters,
                        output_dir=run.output_dir / f"dimen_{dimension:03d}")
        validate(child)
        children.append(child)
    root = Path(run.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_resolved.yaml").write_text(dump_yaml(run), encoding="utf-8")
    summaries_path = root / "dimension_run_summaries.csv"
    statistics_path = root / "dimension_statistics.csv"
    rows = []
    for child in children:
        dimension = child.parameters.problem.state_dim
        print(f"\nDimension {dimension} | {child.runs} run(s) | {child.mode}")
        _clear_runtime()
        try:
            # Consume the saved scalar summaries; do not retain trained models.
            result = execute(child)
            del result
            problem = child.parameters.problem
            metadata = {"dimension": dimension,
                        "brownian_dim" if run.experiment == "linear" else "jump_dim":
                        problem.brownian_dim if run.experiment == "linear" else problem.jump_dim}
            rows.extend({**metadata, **row} for row in _read_run_summaries(child))
            _write_rows(summaries_path, rows)
            _write_rows(statistics_path, _statistics(rows))
        finally:
            _clear_runtime()
    return dict(mode=run.mode, dimensions=list(run.parameters.problem.dimensions),
                runs=run.runs, output_dir=str(root), summary_path=str(summaries_path),
                statistics_path=str(statistics_path))
