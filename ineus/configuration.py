"""Strict, TensorFlow-free loading of experiment YAML configuration.

CLI overrides are applied after YAML values; omitted values use the experiment
defaults. Numerical configuration is represented by frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, get_args, get_type_hints

from .linear.config import DEFAULT_CONFIG
from .nonlinear.config import default_config
from .black_scholes.config import default_config as black_scholes_config
from .linear_black_scholes.config import default_config as linear_black_scholes_config


class ConfigurationError(ValueError):
    """An invalid or unsupported experiment setting."""


@dataclass(frozen=True)
class RunConfiguration:
    experiment: str
    mode: str
    seed: int
    runs: int
    output_dir: Path
    parameters: Any


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "YAML configuration requires PyYAML. Install the supplied environment "
            "or run: python -m pip install PyYAML"
        ) from exc
    return yaml


def read_yaml(path: str | Path) -> dict:
    """Use a safe loader, rejecting duplicate keys instead of hiding typos."""
    yaml = _yaml_module()

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError("All YAML mapping keys must be strings.")
            if key in result:
                raise ConfigurationError(f"Duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        document = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("The YAML document must be a mapping, not an empty file or list.")
    return document


def _typed_value(value, default, location, annotation=None):
    optional_types = get_args(annotation)
    if type(None) in optional_types:
        if value is None:
            return None
        if default is None:
            concrete = [kind for kind in optional_types if kind is not type(None)]
            if len(concrete) == 1 and concrete[0] in (str, int, float, bool):
                default = concrete[0]()
    if isinstance(default, tuple):
        if not isinstance(value, (tuple, list)) or any(type(item) is not int for item in value):
            raise ConfigurationError(f"{location}: expected a list of integers, got {value!r}.")
        return tuple(value)
    if isinstance(default, bool):
        valid = type(value) is bool
    elif isinstance(default, int):
        valid = type(value) is int
    elif isinstance(default, float):
        valid = type(value) in (int, float) and math.isfinite(value)
        if valid:
            value = float(value)
    else:
        valid = isinstance(value, type(default))
    if not valid:
        raise ConfigurationError(
            f"{location}: expected {type(default).__name__}, got {value!r}. "
            "Write learning rates as decimals, e.g. 0.0001."
        )
    return value


def _merge_dataclass(default, updates, location="parameters"):
    if not isinstance(updates, dict):
        raise ConfigurationError(f"{location} must be a mapping.")
    known = {field.name for field in fields(default) if field.init}
    unknown = set(updates) - known
    if unknown:
        raise ConfigurationError(f"Unknown key(s) in {location}: {', '.join(sorted(unknown))}")
    changes = {}
    annotations = get_type_hints(type(default))
    for name, value in updates.items():
        default_value = getattr(default, name)
        key = f"{location}.{name}"
        changes[name] = (
            _merge_dataclass(default_value, value, key) if is_dataclass(default_value)
            else _typed_value(value, default_value, key, annotations.get(name))
        )
    try:
        return replace(default, **changes)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{location}: {exc}") from exc


def from_document(document: dict, experiment: str | None = None) -> RunConfiguration:
    """Resolve a YAML mapping without importing numerical runtime modules."""
    if not isinstance(document, dict):
        raise ConfigurationError("Configuration must be a mapping.")
    known = {"experiment", "mode", "seed", "runs", "output_dir", "parameters"}
    unknown = set(document) - known
    if unknown:
        raise ConfigurationError(f"Unknown top-level key(s): {', '.join(sorted(unknown))}")
    selected = document.get("experiment", experiment)
    if selected not in ("linear", "nonlinear", "black_scholes", "linear_black_scholes"):
        raise ConfigurationError("Choose experiment: linear, nonlinear, black_scholes or linear_black_scholes "
                                 "(in YAML or --experiment).")
    if experiment is not None and selected != experiment:
        raise ConfigurationError(f"The {experiment} launcher cannot run a {selected} configuration.")
    defaults = (DEFAULT_CONFIG if selected == "linear" else
                linear_black_scholes_config() if selected == "linear_black_scholes" else
                black_scholes_config() if selected == "black_scholes" else default_config())
    updates = document.get("parameters", {})
    if not isinstance(updates, dict):
        raise ConfigurationError("parameters must be a mapping.")
    if "runs" in updates or "base_seed" in updates:
        raise ConfigurationError("Set runs and seed at the YAML top level, not inside parameters.")
    evaluation = updates.get("evaluation", {})
    if isinstance(evaluation, dict) and "default_output_dir" in evaluation:
        raise ConfigurationError("Set output_dir at the YAML top level, not inside evaluation.")
    training = updates.get("train", {})
    if selected == "black_scholes" and isinstance(training, dict) and "save_dir" in training:
        raise ConfigurationError("Set output_dir at the YAML top level, not train.save_dir.")
    parameters = _merge_dataclass(defaults, updates)
    problem_updates = updates.get("problem", {})
    if selected == "nonlinear" and "state_dim" in problem_updates and "jump_dim" not in problem_updates:
        parameters = replace(parameters, problem=replace(
            parameters.problem, jump_dim=parameters.problem.state_dim))
    if selected == "black_scholes" and "u_ref" not in evaluation:
        # Default point benchmarks; explicit numeric/null overrides take precedence.
        jumps_active = parameters.problem.use_jumps and parameters.problem.jump_intensity > 0
        parameters = replace(parameters, evaluation=replace(
            parameters.evaluation, u_ref=55.810 if jumps_active else 57.300))
    ineus_only = selected in ("black_scholes", "linear_black_scholes")
    mode = document.get("mode", "ineus" if ineus_only else "comparison")
    default_seed = 1 if selected == "black_scholes" else 23 if selected == "linear_black_scholes" else 29
    seed = _typed_value(document.get("seed", default_seed), 29, "seed")
    default_runs = 5 if selected == "black_scholes" else 10
    runs = _typed_value(document.get("runs", default_runs), 1, "runs")
    default_output = (
        DEFAULT_CONFIG.evaluation.default_output_dir if selected == "linear"
        else "rPINN/nonlinear/nonlinear_three_methods"
    )
    if selected == "black_scholes":
        default_output = f"results/black_scholes/seed_{seed}" if runs == 1 else "results/black_scholes"
    elif selected == "linear_black_scholes":
        default_output = "rPINN/Black_Scholes/Black_Scholes_dimension_sweep"
    elif mode == "ineus":
        default_output = str(Path(default_output) / "ineus_only")
    output = document.get("output_dir", default_output)
    if not isinstance(output, str) or not output.strip():
        raise ConfigurationError("output_dir must be a nonempty path string.")
    if selected == "nonlinear":
        parameters = replace(parameters, runs=runs, base_seed=seed)
    return RunConfiguration(selected, mode, seed, runs, Path(output), parameters)


def override_epochs(run: RunConfiguration, epochs: int) -> RunConfiguration:
    if type(epochs) is not int or epochs < 1:
        raise ConfigurationError("--epochs must be a positive integer.")
    config = run.parameters
    if run.experiment in ("black_scholes", "linear_black_scholes"):
        return replace(run, parameters=replace(config, train=replace(config.train, epochs=epochs)))
    bsde = replace(config.bsde, local_epochs=epochs, surrogate_epochs=epochs)
    if run.experiment == "linear":
        config = replace(config, ineus=replace(config.ineus, epochs=epochs),
                         pinn=replace(config.pinn, epochs=epochs), bsde=bsde)
    else:
        config = replace(config, train=replace(config.train, epochs_ineus=epochs,
                                               epochs_pinn=epochs), bsde=bsde)
    return replace(run, parameters=config)


def validate(run: RunConfiguration) -> None:
    """Fail before TensorFlow initialization for unsupported/invalid settings."""
    if run.mode not in ("ineus", "comparison"):
        raise ConfigurationError("mode must be ineus or comparison.")
    if not 0 <= run.seed < 2**32 or run.runs < 1:
        raise ConfigurationError("seed must be in [0, 2**32); runs must be positive.")
    if run.seed + run.runs - 1 >= 2**32:
        raise ConfigurationError("The last run seed exceeds the NumPy seed range.")
    if run.experiment == "black_scholes":
        _validate_black_scholes(run)
        return
    if run.experiment == "linear_black_scholes":
        _validate_linear_black_scholes(run)
        return
    config = run.parameters
    _validate_pide_problem(run)
    if config.model.n_layers < 2:
        raise ConfigurationError("model.n_layers must be at least 2 for the DGM gates.")

    # All count/rate fields must be positive except these zero-valued
    # settings, optional block size 0 (no distillation), and RNG seeds.
    nonnegative = {"target_weight", "target_mixing", "block_size", "averaging_block_size",
                   "num_t0_each_epoch", "rad_exponent", "rad_offset", "test_seed", "reference_seed"}
    for section_name in ("model", "ineus", "pinn", "train", "bsde", "evaluation"):
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for field in fields(section):
            value = getattr(section, field.name)
            if type(value) not in (int, float):
                continue
            minimum_ok = value >= 0 if field.name in nonnegative else value > 0
            if not minimum_ok or not math.isfinite(value):
                raise ConfigurationError(f"parameters.{section_name}.{field.name} has invalid value {value!r}.")
    if run.experiment == "linear":
        shared = ("num_samples", "buffer_size", "batch_size", "n_inner", "num_batches",
                  "use_rad", "rad_period", "rad_exponent", "rad_offset",
                  "rad_candidate_multiplier", "rad_eval_batch_size", "log_every")
        if run.mode == "comparison":
            conflicting = [name for name in shared
                           if getattr(config.ineus, name) != getattr(config.pinn, name)]
            if conflicting:
                raise ConfigurationError("Linear comparison shares these INEUS/PINN fields; "
                                         "set both equally: " + ", ".join(conflicting))
        training = [config.ineus] + ([config.pinn] if run.mode == "comparison" else [])
        if config.model.output_dim != 1:
            raise ConfigurationError("The scalar value problem requires model.output_dim: 1.")
    else:
        training = [config.train]
        if run.mode == "comparison" and min(config.train.epochs_ineus, config.train.epochs_pinn) <= config.train.log_every:
            raise ConfigurationError("Nonlinear comparison needs more epochs than log_every "
                                     "for runtime interpolation (e.g. --epochs 20).")
    for item in training:
        if item.n_inner < 2:
            raise ConfigurationError("n_inner must be at least 2 (the inner loop starts at 1).")
        if min(item.num_samples, item.buffer_size) < item.batch_size:
            raise ConfigurationError("num_samples and buffer_size must both be at least batch_size.")
    for name in ("test_seed", "reference_seed"):
        value = getattr(config.evaluation, name, 0)
        if value >= 2**32:
            raise ConfigurationError(f"evaluation.{name} must be less than 2**32.")


def _validate_pide_problem(run: RunConfiguration) -> None:
    """Validate parameters of the quadratic linear and HJB equation families."""
    problem = run.parameters.problem
    if type(problem.state_dim) is not int or problem.state_dim < 1:
        raise ConfigurationError("problem.state_dim must be a positive integer.")
    dimensions = problem.dimensions
    if any(type(d) is not int or d < 1 for d in dimensions) or len(set(dimensions)) != len(dimensions):
        raise ConfigurationError("problem.dimensions must contain distinct positive integers (or be empty).")
    for item in fields(problem):
        value = getattr(problem, item.name)
        if type(value) in (int, float) and not math.isfinite(value):
            raise ConfigurationError(f"problem.{item.name} must be finite.")
    if problem.horizon <= 0 or problem.jump_intensity < 0:
        raise ConfigurationError("Require problem.horizon > 0 and jump_intensity >= 0 (zero means no jumps).")
    if run.experiment == "linear":
        if type(problem.brownian_dim) is not int or problem.brownian_dim < 1:
            raise ConfigurationError("problem.brownian_dim must be a positive integer.")
        if min(problem.sample_box, problem.test_box) <= 0:
            raise ConfigurationError("problem.sample_box and test_box must be positive.")
        if not 0 <= problem.coefficient_seed < 2**32:
            raise ConfigurationError("problem.coefficient_seed must be in [0, 2**32).")
        if problem.diffusion_mode not in ("random", "constant"):
            raise ConfigurationError("problem.diffusion_mode must be random or constant.")
        if problem.jump_covariance_mode not in ("random_diagonal", "isotropic"):
            raise ConfigurationError("problem.jump_covariance_mode must be random_diagonal or isotropic.")
        if min(problem.diffusion_scale, problem.jump_covariance_min, problem.jump_covariance_scale) < 0:
            raise ConfigurationError("Diffusion scale and jump covariance parameters must be nonnegative.")
        if (problem.jump_covariance_mode == "random_diagonal"
                and problem.jump_covariance_min > problem.jump_covariance_scale):
            raise ConfigurationError("Random jump covariance requires jump_covariance_min <= jump_covariance_scale.")
    else:
        if type(problem.jump_dim) is not int or problem.jump_dim != problem.state_dim:
            raise ConfigurationError("HJB requires problem.jump_dim == problem.state_dim.")
        if problem.eta <= 0 or problem.jump_covariance_scale < 0:
            raise ConfigurationError("HJB requires problem.eta > 0 and jump_covariance_scale >= 0.")


def _validate_black_scholes(run: RunConfiguration) -> None:
    """Validate the dynamically configured default-risk equation and INEUS run."""
    if run.mode != "ineus":
        raise ConfigurationError("Black-Scholes supports mode: ineus only. "
                                 "Its classical BSDE reference solvers are not training baselines.")
    config = run.parameters
    problem, train = config.problem, config.train
    if run.runs > 1 and train.resume_path is not None:
        raise ConfigurationError("Set runs: 1 when resuming a checkpoint; repeated runs must start independently.")
    nonnegative = {"rho1", "w_bc", "w_int", "rad_k", "rad_c"}
    for section_name in ("model", "train", "evaluation"):
        section = getattr(config, section_name)
        for item in fields(section):
            value = getattr(section, item.name)
            if type(value) not in (int, float):
                continue
            minimum_ok = (True if item.name == "u_ref" else
                          value >= 0 if item.name in nonnegative else value > 0)
            if not math.isfinite(value) or not minimum_ok:
                raise ConfigurationError(f"parameters.{section_name}.{item.name} has invalid value {value!r}.")
    if config.model.n_layers < 2:
        raise ConfigurationError("model.n_layers must be at least 2 for the DGM gates.")
    if problem.dimen < 1 or problem.T <= 0 or problem.sigma_bar <= 0:
        raise ConfigurationError("problem.dimen, T and sigma_bar must be positive.")
    if problem.v_h >= problem.v_l:
        raise ConfigurationError("problem.v_h must be less than problem.v_l.")
    if not 0 <= problem.delta_recovery <= 1 or not 0 <= problem.jump_corr <= 1:
        raise ConfigurationError("problem.delta_recovery and jump_corr must be in [0, 1].")
    if min(problem.gamma_h, problem.gamma_l, problem.jump_intensity, problem.jump_std) < 0:
        raise ConfigurationError("Default intensities and jump intensity/std must be nonnegative.")
    if not 0 <= train.rho1 <= 1:
        raise ConfigurationError("train.rho1 must be in [0, 1].")
    if train.num_samples < 4 or min(train.num_samples, train.buffer_size) < train.batch_size:
        raise ConfigurationError("num_samples must be at least 4; num_samples and buffer_size "
                                 "must both be at least batch_size.")
    if train.w_bc == 0 and not config.model.constrained:
        raise ConfigurationError("Set model.constrained: true when train.w_bc is zero "
                                 "so the terminal condition is enforced.")
    if train.use_rad and train.rad_candidate_size < train.num_samples:
        raise ConfigurationError("RAD needs rad_candidate_size >= num_samples "
                                 "because resampling is without replacement.")
    if train.resume_path is not None and not train.resume_path.strip():
        raise ConfigurationError("train.resume_path must be null or a nonempty checkpoint path.")


def _validate_linear_black_scholes(run: RunConfiguration) -> None:
    """Validate the basket-call dimension sweep without numerical imports."""
    if run.mode != "ineus":
        raise ConfigurationError("Linear Black-Scholes supports mode: ineus only; "
                                 "its Monte Carlo reference is not a training baseline.")
    config = run.parameters
    problem, train, evaluation = config.problem, config.train, config.evaluation
    dimensions = problem.dimensions
    if not dimensions or any(type(d) is not int or d < 1 for d in dimensions):
        raise ConfigurationError("problem.dimensions must be a nonempty list of positive integers.")
    if len(set(dimensions)) != len(dimensions):
        raise ConfigurationError("problem.dimensions must not contain duplicates (run directories are per dimension).")
    nonnegative = {"jump_intensity", "jump_std", "rho1", "w_bc", "w_int", "rad_k", "rad_c"}
    signed = {"r", "correlation", "jump_mean"}
    for section_name in ("problem", "model", "train", "evaluation"):
        for item in fields(getattr(config, section_name)):
            value = getattr(getattr(config, section_name), item.name)
            if type(value) not in (int, float):
                continue
            minimum_ok = (True if item.name in signed else
                          value >= 0 if item.name in nonnegative else value > 0)
            if not math.isfinite(value) or not minimum_ok:
                raise ConfigurationError(f"parameters.{section_name}.{item.name} has invalid value {value!r}.")
    if problem.max_S <= 0.001 or problem.sample_b <= 0.001:
        raise ConfigurationError("problem.max_S and sample_b must exceed the sampling lower bound 0.001.")
    if any(d > 1 and not -1.0 / (d - 1) < problem.correlation < 1.0 for d in dimensions):
        raise ConfigurationError("problem.correlation must make the equicorrelation matrix "
                                 "positive definite in every selected dimension.")
    if config.model.n_layers < 2:
        raise ConfigurationError("model.n_layers must be at least 2 for the DGM gates.")
    if train.n_inner < 2 or not 0 < train.relaxation <= train.h_scale:
        raise ConfigurationError("Require train.n_inner >= 2 and 0 < relaxation <= h_scale.")
    if not 0 <= train.rho1 <= 1:
        raise ConfigurationError("train.rho1 must be in [0, 1].")
    if min(train.num_samples, train.buffer_size) < train.batch_size:
        raise ConfigurationError("num_samples and buffer_size must both be at least batch_size.")
    if min(evaluation.mc_test_paths, evaluation.mc_curve_paths) < 2:
        raise ConfigurationError("Monte Carlo path counts must be at least 2 for antithetic sampling.")
    if min(evaluation.n_x_curve, evaluation.n_t_curve) < 2:
        raise ConfigurationError("Reference curves require at least 2 grid points.")
    if (not evaluation.output_tag.strip() or evaluation.output_tag in (".", "..")
            or any(char in evaluation.output_tag for char in ("/", "\\"))):
        raise ConfigurationError("evaluation.output_tag must be a nonempty filename tag, without path separators.")


def to_document(run: RunConfiguration) -> dict:
    """An editable, fully resolved YAML document with no duplicate run settings."""
    parameters = asdict(run.parameters)
    parameters.pop("runs", None)
    parameters.pop("base_seed", None)
    parameters["evaluation"].pop("default_output_dir", None)
    if run.experiment == "black_scholes":
        parameters["train"].pop("save_dir", None)
    if run.experiment in ("linear", "nonlinear", "linear_black_scholes"):
        parameters["problem"]["dimensions"] = list(parameters["problem"]["dimensions"])
    return dict(experiment=run.experiment, mode=run.mode, seed=run.seed,
                runs=run.runs, output_dir=run.output_dir.as_posix(), parameters=parameters)


def dump_yaml(run: RunConfiguration) -> str:
    return _yaml_module().safe_dump(to_document(run), sort_keys=False, allow_unicode=True)
