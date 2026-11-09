#!/usr/bin/env python3
"""Validate repository structure without importing TensorFlow.

The checks in this script deliberately use only the Python standard library.
They are suitable for a login node, a pre-commit check, or a CI job that has
not installed the numerical runtime yet.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ".gitignore",
    "README.md",
    "environment_hpc.yml",
    "scripts/run_linear.slurm",
    "scripts/run_nonlinear.slurm",
    "scripts/run_black_scholes.slurm",
    "tests/test_entry_point_configuration.py",
    "configs/linear.yaml",
    "configs/nonlinear.yaml",
    "configs/linear_sweep.yaml",
    "configs/nonlinear_sweep.yaml",
    "configs/linear_sweep_smoke.yaml",
    "configs/nonlinear_sweep_smoke.yaml",
    "configs/black_scholes.yaml",
    "configs/black_scholes_smoke.yaml",
    "configs/linear_black_scholes.yaml",
    "configs/linear_black_scholes_smoke.yaml",
    "docs/configuration.md",
    "docs/linear.md",
    "docs/nonlinear_hjb.md",
    "docs/linear_black_scholes.md",
    "docs/nonlinear_black_scholes.md",
    "tests/test_black_scholes_configuration.py",
    "tests/test_linear_black_scholes_configuration.py",
    "ineus/linear_black_scholes/__init__.py",
    "ineus/linear_black_scholes/__main__.py",
    "ineus/linear_black_scholes/config.py",
    "ineus/__init__.py",
    "ineus/__main__.py",
    "ineus/configuration.py",
    "ineus/cli.py",
    "ineus/sweeps.py",
    "tests/test_problem_configuration.py",
    "tests/test_dimension_sweeps.py",
    "tests/test_linear_problem_runtime.py",
    "tests/test_nonlinear_problem_runtime.py",
    "ineus/reporting.py",
    "ineus/linear/__init__.py",
    "ineus/linear/__main__.py",
    "ineus/linear/config.py",
    "ineus/linear/problem.py",
    "ineus/linear/models.py",
    "ineus/linear/sampling.py",
    "ineus/linear/pde.py",
    "ineus/linear/averaging.py",
    "ineus/linear/trainer.py",
    "ineus/linear/bsde.py",
    "ineus/linear/plotting.py",
    "ineus/linear/experiment.py",
    "ineus/nonlinear/__init__.py",
    "ineus/nonlinear/__main__.py",
    "ineus/nonlinear/config.py",
    "ineus/nonlinear/problem.py",
    "ineus/nonlinear/models.py",
    "ineus/nonlinear/sampling.py",
    "ineus/nonlinear/pde.py",
    "ineus/nonlinear/averaging.py",
    "ineus/nonlinear/trainer.py",
    "ineus/nonlinear/bsde.py",
    "ineus/nonlinear/plotting.py",
    "ineus/nonlinear/experiment.py",
) + tuple(
    f"ineus/{experiment}/{relative}"
    for experiment in ("linear", "nonlinear")
    for relative in (
        "architectures/__init__.py",
        "architectures/value.py",
        "architectures/bsde.py",
        "architectures/surrogate.py",
        "training/__init__.py",
        "training/common.py",
        "training/ineus.py",
        "training/pinn.py",
        "training/local_bsde.py",
        "training/surrogate.py",
    )
) + tuple(
    f"ineus/black_scholes/{relative}"
    for relative in (
        "__init__.py", "__main__.py", "config.py", "problem.py", "payoff.py",
        "models.py", "trainer.py", "replay_buffer.py", "sampling.py", "utils.py",
        "averaging.py", "checkpointing.py", "experiment.py", "references.py",
        "plotting.py", "aggregation.py", "architectures/__init__.py",
        "architectures/value.py", "training/__init__.py", "training/ineus.py", "LICENSE",
    )
) + tuple(
    f"ineus/linear_black_scholes/{relative}"
    for relative in (
        "models.py", "trainer.py", "experiment.py", "averaging.py", "evaluation.py",
        "references.py", "reporting.py", "problem.py", "pde.py", "sampling.py",
        "replay_buffer.py", "utils.py", "architectures/__init__.py",
        "architectures/value.py", "training/__init__.py", "training/ineus.py",
    )
)

# Public modules expose the implementations in the architecture and training
# packages without duplicating their definitions.
PUBLIC_EXPORTS = {
    "linear_black_scholes": {
        "models": {
            "architectures": ("BSPINN_V", "ConstrainedValue", "DGMCell_V", "InputNormalizer", "make_model"),
        },
        "trainer": {
            "training.ineus": ("train2",),
        },
    },
    "black_scholes": {
        "models": {
            "architectures.value": ("DGMCell", "InputNormalizer", "BSPINN", "ConstrainedValue", "build_model"),
            "payoff": ("phi_tf",),
        },
        "trainer": {
            "training.ineus": ("train",),
            "checkpointing": ("save_checkpoint", "load_checkpoint", "save_history_json"),
        },
    },
    "linear": {
        "models": {
            "architectures.value": ("DGMCell_V", "ConstrainedValue", "make_model"),
        },
        "trainer": {
            "training.common": ("evaluate_test_mae_callable", "build_optimizer"),
            "training.ineus": ("train_bellman_ct",),
            "training.pinn": ("train_standard_pinn",),
        },
        "bsde": {
            "architectures.bsde": ("DirectZNet", "DirectJumpUNet", "ClassicalLocalDeepBSDEJump"),
            "architectures.surrogate": ("GlobalValueSurrogateFromLocalBSDE",),
            "training.local_bsde": ("train_one_classical_local_bsde_jump", "make_local_bsde_label_dataset"),
            "training.surrogate": ("train_global_surrogate_from_local_bsde_labels", "predict_u_local_bsde_surrogate"),
        },
    },
    "nonlinear": {
        "models": {
            "architectures.value": ("InputNormalizer", "BSPINN_V", "DGMCell_V", "ConstrainedValue", "make_model", "strategy"),
        },
        "trainer": {
            "training.common": ("evaluate_test_mae",),
            "training.ineus": ("train_bellman_ct_nl",),
            "training.pinn": ("train_standard_pinn_nl",),
        },
        "bsde": {
            "architectures.bsde": ("DirectZNetLocalTransformed", "DirectJumpUNetLocalTransformed", "ClassicalLocalDeepBSDETransformedJump"),
            "architectures.surrogate": ("GlobalUSurrogateFromLocalTransformedBSDE",),
            "training.local_bsde": ("train_one_classical_local_bsde_transformed_nl", "make_local_bsde_label_dataset_nl"),
            "training.surrogate": ("train_global_u_surrogate_from_local_bsde_labels_nl", "predict_u_local_bsde_surrogate_nl"),
        },
    },
}

README_SNIPPETS = (
    "python -u -m ineus --config",
    "--mode ineus",
    "--mode comparison",
    "configs/linear.yaml",
    "configs/nonlinear.yaml",
    "configs/linear_black_scholes.yaml",
    "configs/black_scholes.yaml",
    "docs/configuration.md",
    "docs/linear.md",
    "docs/nonlinear_hjb.md",
    "docs/linear_black_scholes.md",
    "docs/nonlinear_black_scholes.md",
)

EXPECTED_DEFAULTS = {
    "ineus/linear_black_scholes/config.py": {
        "ProblemConfig": {
            "dimensions": (10,), "T": 1.0, "r": 0.05,
            "max_S": 100.0, "K_multiplier": 0.3, "sample_b": 120.0,
            "jump_intensity": 0.5, "correlation": 0.10, "sigma": 0.15,
            "jump_mean": 0.20, "jump_std": 0.30,
        },
        "ModelConfig": {"hidden_dim": 50, "n_layers": 3},
        "TrainConfig": {
            "epochs": 1000, "n_inner": 20, "num_batches": 32,
            "batch_size": 1024, "num_samples": 2048, "nc_samples": 2048,
            "buffer_size": 4096, "rho1": 0.0, "h_scale": 0.5, "relaxation": 0.4,
            "w_bc": 0.0, "w_int": 1.0, "lr1": 0.0005, "lr2": 0.001,
            "run_phase2": False, "use_rad": True, "rad_period": 5,
            "rad_k": 1.0, "rad_c": 1.0, "rad_candidate_multiplier": 8,
            "rad_eval_batch_size": 2048, "use_distill": True,
            "distill_block_size": 10, "distill_steps": 200,
            "distill_student_lr": 0.001, "replace_with_student": True,
        },
        "EvalConfig": {
            "test_samples": 2000, "mc_test_paths": 20000, "mc_curve_paths": 20000,
            "n_x_curve": 401, "n_t_curve": 401, "output_tag": "bs_dimension_sweep",
        },
    },
    "ineus/black_scholes/config.py": {
        "PDEConfig": {
            "dimen": 100, "T": 1.0, "R_rate": 0.02, "mu_bar": 0.02,
            "sigma_bar": 0.20, "v_h": 50.0, "v_l": 70.0,
            "gamma_h": 0.20, "gamma_l": 0.02,
            "jump_intensity": 0.1, "jump_mean": -0.2, "jump_std": 0.15,
            "jump_corr": 0.5, "use_jumps": True,
        },
        "ModelConfig": {"hidden_dim": 100, "n_layers": 3, "constrained": False},
        "TrainConfig": {
            "epochs": 400, "n_inner": 20, "num_batches": 32, "h_scale": 1.0,
            "zeta_scale": 1.25, "lr": 1e-3, "w_bc": 1.0, "w_int": 1.0,
            "batch_size": 1024, "num_samples": 2048, "buffer_size": 16384,
            "use_path_sampling": True, "use_rad": False, "block_size": 10,
            "distill_steps": 200, "student_lr": 1e-4, "use_cosine_lr": True,
            "grad_clip": 1.0, "test_samples": 500,
        },
        "EvalConfig": {"x0_value": 100.0, "u_ref": 55.810, "dimen": 100},
    },
    "ineus/linear/config.py": {
        "ProblemConfig": {
            "state_dim": 10,
            "dimensions": (),
            "brownian_dim": 10,
            "horizon": 0.5,
            "jump_intensity": 0.25,
            "drift": 1.0,
            "discount": 2.0,
            "coefficient_seed": 23,
            "diffusion_scale": 0.3,
            "diffusion_mode": "random",
            "jump_covariance_mode": "random_diagonal",
            "jump_covariance_min": 0.01,
            "jump_covariance_scale": 0.4,
        },
        "ModelConfig": {"hidden_dim": 50, "n_layers": 4},
        "INEUSTrainConfig": {
            "epochs": 1000,
            "num_samples": 2048,
            "buffer_size": 4096,
            "batch_size": 1024,
            "learning_rate": 5e-4,
            "n_inner": 20,
            "num_batches": 32,
            "h_scale": 0.5,
            "relaxation": 0.4,
            "block_size": 10,
            "distill_steps": 200,
            "student_learning_rate": 1e-3,
            "jump_samples": 25,
            "test_jump_samples": 50,
            "log_every": 2,
        },
        "PINNTrainConfig": {
            "epochs": 1000,
            "batch_size": 1024,
            "learning_rate": 5e-4,
            "jump_samples": 64,
            "rad_jump_samples": 25,
            "test_jump_samples": 50,
        },
        "BSDEConfig": {
            "num_local_models": 100,
            "local_epochs": 10000,
            "local_batch_size": 1024,
            "local_learning_rate": 3e-4,
            "time_steps": 25,
            "jump_expectation_samples": 64,
            "clipnorm": 1.0,
            "surrogate_epochs": 5000,
        },
        "EvaluationConfig": {"test_samples": 2000},
    },
    "ineus/nonlinear/config.py": {
        "ProblemConfig": {
            "state_dim": 10,
            "dimensions": (),
            "jump_dim": 10,
            "horizon": 1.0,
            "eta": 1.0,
            "jump_intensity": 0.5,
            "driver_constant": 2.0,
            "jump_covariance_scale": 0.2,
        },
        "ModelConfig": {"hidden_dim": 50, "n_layers": 4},
        "TrainConfig": {
            "batch_size": 1024,
            "num_samples": 2048,
            "buffer_size": 4096,
            "epochs_ineus": 1000,
            "epochs_pinn": 1000,
            "learning_rate_ineus": 5e-4,
            "learning_rate_pinn": 5e-4,
            "n_inner": 20,
            "num_batches": 32,
            "h": 0.5,
            "alpha": 0.4,
            "averaging_block_size": 10,
            "distillation_steps": 200,
            "distillation_learning_rate": 1e-3,
            "jump_samples": 64,
            "test_jump_samples": 50,
            "rad_jump_samples": 25,
            "learning_rate_decay_steps": 1000,
            "learning_rate_decay_ineus": 1.0,
            "learning_rate_decay_pinn": 1.0,
            "learning_rate_staircase": True,
        },
        "BSDEConfig": {
            "num_local_models": 100,
            "local_epochs": 10000,
            "local_batch_size": 1024,
            "local_learning_rate": 3e-4,
            "n_time_steps": 25,
            "jump_expectation_samples": 64,
            "gradient_clip_norm": 1.0,
            "surrogate_epochs": 5000,
        },
        "EvalConfig": {"test_seed": 2, "reference_seed": 124, "test_samples": 2000,
                       "reference_paths": 2000000, "reference_chunk_size": 10000},
        "ExperimentConfig": {"runs": 10, "base_seed": 29},
    },
}


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    detail: str


def python_files() -> list[Path]:
    """Return maintained Python files that must parse successfully."""
    paths = sorted((ROOT / "ineus").rglob("*.py"))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").rglob("*.py")))
    return [path for path in paths if path.is_file()]


def parse_python(path: Path) -> ast.Module:
    """Parse *path* without importing or executing it."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_required_paths() -> list[ValidationIssue]:
    issues = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            issues.append(ValidationIssue("repository structure", f"missing {relative}"))
    return issues


def check_python_syntax() -> list[ValidationIssue]:
    issues = []
    for path in python_files():
        try:
            parse_python(path)
        except (SyntaxError, UnicodeError) as exc:
            issues.append(
                ValidationIssue(
                    "Python syntax",
                    f"{path.relative_to(ROOT)}: {exc}",
                )
            )
    return issues


def _module_path(module_name: str) -> Path:
    parts = module_name.split(".")
    package_init = ROOT.joinpath(*parts, "__init__.py")
    if package_init.is_file():
        return package_init
    return ROOT.joinpath(*parts).with_suffix(".py")


def _resolve_relative_import(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    relative_module = path.relative_to(ROOT).with_suffix("")
    package_parts = list(relative_module.parts[:-1])
    climb = node.level - 1
    if climb > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - climb]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def check_local_imports() -> list[ValidationIssue]:
    """Check that every explicit ``ineus`` import resolves to a local module."""
    issues = []
    for path in python_files():
        try:
            tree = parse_python(path)
        except (SyntaxError, UnicodeError):
            continue

        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_relative_import(path, node)
                if resolved:
                    candidates.append(resolved)

            for module_name in candidates:
                if not module_name.startswith("ineus"):
                    continue
                if not _module_path(module_name).is_file():
                    issues.append(
                        ValidationIssue(
                            "local imports",
                            (
                                f"{path.relative_to(ROOT)} imports unresolved "
                                f"module {module_name!r}"
                            ),
                        )
                    )
    return issues


def check_public_exports() -> list[ValidationIssue]:
    """Ensure public modules expose their architecture and training definitions."""
    issues = []
    for experiment, modules in PUBLIC_EXPORTS.items():
        for module_name, expected_imports in modules.items():
            relative = f"ineus/{experiment}/{module_name}.py"
            path = ROOT / relative
            if not path.is_file():
                issues.append(ValidationIssue("public exports", f"missing {relative}"))
                continue
            try:
                tree = parse_python(path)
            except (SyntaxError, UnicodeError) as exc:
                issues.append(ValidationIssue("public exports", f"{relative}: {exc}"))
                continue
            imports = {}
            local_names = set()
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    module = _resolve_relative_import(path, node)
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (module, alias.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    local_names.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        local_names.update(
                            name.id for name in ast.walk(target) if isinstance(name, ast.Name)
                        )
            for module, names in expected_imports.items():
                expected_module = f"ineus.{experiment}.{module}"
                for name in names:
                    if imports.get(name) != (expected_module, name) or name in local_names:
                        issues.append(ValidationIssue(
                            "public exports",
                            f"{relative} must directly re-export {expected_module}.{name}",
                        ))
    return issues


def _main_guard_present(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def check_entry_points() -> list[ValidationIssue]:
    issues = []
    entry_points = (
        "ineus/__main__.py",
        "ineus/linear/__main__.py",
        "ineus/nonlinear/__main__.py",
        "ineus/black_scholes/__main__.py",
        "ineus/linear_black_scholes/__main__.py",
    )
    for relative in entry_points:
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            tree = parse_python(path)
        except (SyntaxError, UnicodeError):
            continue
        if not _main_guard_present(tree):
            issues.append(
                ValidationIssue("entry points", f"{relative} has no __main__ guard")
            )
        shared_cli_import = any(
            isinstance(node, ast.ImportFrom)
            and _resolve_relative_import(path, node) == "ineus.cli"
            and any(alias.name == "main" and alias.asname is None for alias in node.names)
            for node in tree.body
        )
        if not shared_cli_import:
            issues.append(ValidationIssue(
                "entry points",
                f"{relative} must import ineus.cli.main directly so --dry-run stays TensorFlow-free",
            ))
    return issues


def _class_defaults(tree: ast.Module) -> dict[str, dict[str, object]]:
    classes: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        defaults: dict[str, object] = {}
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                try:
                    defaults[statement.target.id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    continue
        classes[node.name] = defaults
    return classes


def check_default_configuration() -> list[ValidationIssue]:
    """Verify the standard experiment parameters declared in configuration."""
    issues = []
    for relative, expected_classes in EXPECTED_DEFAULTS.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            actual_classes = _class_defaults(parse_python(path))
        except (SyntaxError, UnicodeError) as exc:
            issues.append(ValidationIssue("default configuration", f"{relative}: {exc}"))
            continue

        for class_name, expected_fields in expected_classes.items():
            if class_name not in actual_classes:
                issues.append(
                    ValidationIssue(
                        "default configuration",
                        f"{relative} has no {class_name}",
                    )
                )
                continue
            actual_fields = actual_classes[class_name]
            for field_name, expected in expected_fields.items():
                actual = actual_fields.get(field_name, object())
                if actual != expected:
                    issues.append(
                        ValidationIssue(
                            "default configuration",
                            (
                                f"{relative}:{class_name}.{field_name} "
                                f"is {actual!r}, expected {expected!r}"
                            ),
                        )
                    )
    return issues


def check_diagnostics() -> list[ValidationIssue]:
    """Verify evaluation and best-weight bookkeeping in the training loops."""
    issues = []
    expected_trainers = (
        ("linear", "ineus", "train_bellman_ct", "pinn_residual_pointwise"),
        ("linear", "pinn", "train_standard_pinn", "pinn_residual_pointwise"),
        ("nonlinear", "ineus", "train_bellman_ct_nl", "pinn_residual_pointwise_nl"),
        ("nonlinear", "pinn", "train_standard_pinn_nl", "pinn_residual_pointwise_nl"),
    )
    for experiment, module, name, residual_function in expected_trainers:
        relative = f"ineus/{experiment}/training/{module}.py"
        path = ROOT / relative
        if not path.is_file():
            issues.append(ValidationIssue("training diagnostics", f"missing {relative}"))
            continue
        try:
            tree = parse_python(path)
        except (SyntaxError, UnicodeError) as exc:
            issues.append(ValidationIssue("training diagnostics", f"{relative}: {exc}"))
            continue
        trainer = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == name), None)
        if trainer is None:
            issues.append(ValidationIssue(
                "training diagnostics", f"{relative} is missing {name}",
            ))
            continue
        strings = {item.value for item in ast.walk(trainer)
                   if isinstance(item, ast.Constant) and isinstance(item.value, str)}
        calls = {item.func.id for item in ast.walk(trainer)
                 if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
        if not {"test_residual", "best_loss", "best_weights"}.issubset(strings):
            issues.append(ValidationIssue(
                "training diagnostics",
                f"{relative}:{name} lost residual history/best-weight bookkeeping",
            ))
        if residual_function not in calls:
            issues.append(ValidationIssue(
                "training diagnostics",
                f"{relative}:{name} does not evaluate the test residual",
            ))
    return issues


def check_readme() -> list[ValidationIssue]:
    path = ROOT / "README.md"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    issues = []
    for snippet in README_SNIPPETS:
        if snippet not in text:
            issues.append(
                ValidationIssue("README", f"missing required documentation: {snippet!r}")
            )
    return issues


def validate() -> list[ValidationIssue]:
    """Run every TensorFlow-free validation check."""
    issues = []
    checks = (
        check_required_paths,
        check_python_syntax,
        check_local_imports,
        check_public_exports,
        check_entry_points,
        check_default_configuration,
        check_diagnostics,
        check_readme,
    )
    for check in checks:
        issues.extend(check())
    return issues


def main() -> int:
    issues = validate()
    if issues:
        print(f"Repository validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - [{issue.check}] {issue.detail}")
        return 1

    print(f"Repository validation passed ({len(python_files())} Python files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
