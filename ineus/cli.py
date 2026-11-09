"""Shared CLI: YAML settings, explicit mode selection, and reproducible runs."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .configuration import (
    ConfigurationError, dump_yaml, from_document, override_epochs, read_yaml, validate,
)


def build_parser(experiment=None):
    parser = argparse.ArgumentParser(
        description="Run INEUS and baseline comparisons from YAML configuration."
    )
    parser.add_argument("--config", type=Path, help="YAML file (e.g. configs/linear.yaml).")
    parser.add_argument("--experiment", choices=("linear", "nonlinear", "black_scholes", "linear_black_scholes"), default=experiment,
                        help="Experiment, if no YAML file is supplied.")
    parser.add_argument("--mode", choices=("ineus", "comparison"),
                        help="Train INEUS only, or INEUS + PINN + local deep BSDE. "
                             "Both Black-Scholes experiments support INEUS only.")
    parser.add_argument("--seed", type=int, help="Training seed / base seed for repeated runs.")
    parser.add_argument("--runs", type=int, help="Independent seeded runs per selected dimension.")
    dimensions = parser.add_mutually_exclusive_group()
    dimensions.add_argument("--dimension", type=int,
                            help="State dimension for quadratic linear/HJB; disables a configured sweep.")
    dimensions.add_argument("--dimensions", type=int, nargs="+",
                            help="Dimension sweep for quadratic linear/HJB, e.g. 10 25 50 100.")
    parser.add_argument("--jump-intensity", type=float,
                        help="Quadratic linear/HJB jump intensity; 0 selects the no-jump PDE.")
    parser.add_argument("--epochs", type=int,
                        help="Override training epochs (INEUS only for both Black-Scholes cases; otherwise "
                             "INEUS/PINN, each local BSDE, and surrogate loop counts). "
                             "Does not reduce reference paths or distillation steps.")
    parser.add_argument("--output-dir", type=Path, help="Result directory, relative to current directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print resolved YAML; do not load TensorFlow or train.")
    return parser


def resolve_arguments(args, experiment=None):
    document = read_yaml(args.config) if args.config is not None else {}
    if args.experiment is not None:
        selected = document.get("experiment", args.experiment)
        if selected != args.experiment:
            raise ConfigurationError("--experiment conflicts with the YAML experiment.")
        document["experiment"] = args.experiment
    for name in ("mode", "seed", "runs", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            document[name] = str(value) if name == "output_dir" else value
    if any(getattr(args, name) is not None for name in ("dimension", "dimensions", "jump_intensity")):
        selected = document.get("experiment", experiment)
        if selected not in ("linear", "nonlinear"):
            raise ConfigurationError("--dimension(s) and --jump-intensity apply to quadratic linear and HJB only.")
        sections = document.setdefault("parameters", {})
        if not isinstance(sections, dict):
            raise ConfigurationError("parameters must be a mapping.")
        problem = sections.setdefault("problem", {})
        if not isinstance(problem, dict):
            raise ConfigurationError("parameters.problem must be a mapping.")
        if args.dimension is not None:
            problem.update(state_dim=args.dimension, dimensions=[])
            if selected == "nonlinear":
                problem["jump_dim"] = args.dimension
        elif args.dimensions is not None:
            problem["dimensions"] = args.dimensions
        if args.jump_intensity is not None:
            problem["jump_intensity"] = args.jump_intensity
    run = from_document(document, experiment=experiment)
    if args.epochs is not None:
        run = override_epochs(run, args.epochs)
    validate(run)
    return run


def parse_args(argv=None, *, experiment=None):
    """Parse command-line options for the selected experiment."""
    return build_parser(experiment).parse_args(argv)


def execute(run):
    """Import TensorFlow only after validation; preserve numerical runner APIs."""
    validate(run)
    if run.experiment in ("linear", "nonlinear") and run.parameters.problem.dimensions:
        from .sweeps import run_dimension_sweep

        return run_dimension_sweep(run, execute)
    module = importlib.import_module(f"ineus.{run.experiment}.experiment")
    run.output_dir.mkdir(parents=True, exist_ok=True)
    (run.output_dir / "config_resolved.yaml").write_text(dump_yaml(run), encoding="utf-8")
    print(f"Experiment: {run.experiment} | mode: {run.mode} | seed: {run.seed}")
    print(f"Results: {run.output_dir.resolve()}")
    kwargs = dict(seed=run.seed, mode=run.mode, config=run.parameters,
                  output_dir=str(run.output_dir))
    kwargs["runs"] = run.runs
    return module.run_experiment(**kwargs)


def main(argv=None, *, experiment=None):
    parser = build_parser(experiment)
    args = parser.parse_args(argv)
    try:
        run = resolve_arguments(args, experiment)
    except ConfigurationError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(dump_yaml(run), end="")
        return None
    execute(run)
    return None
