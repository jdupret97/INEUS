"""Configuration for the nonlinear INEUS and baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemConfig:
    """Coefficients of the nonlinear PIDE."""

    state_dim: int = 10
    jump_dim: int = 10
    horizon: float = 1.0
    eta: float = 1.0
    jump_intensity: float = 0.5
    driver_constant: float = 2.0
    jump_covariance_scale: float = 0.2
    dimensions: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    """Shared INEUS/PINN value-network architecture."""

    hidden_dim: int = 50
    n_layers: int = 4
    constrained_terminal_value: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Training settings for INEUS and PINN."""

    batch_size: int = 1024
    num_samples: int = 2048
    buffer_size: int = 4096
    epochs_ineus: int = 1000
    epochs_pinn: int = 1000
    learning_rate_ineus: float = 5e-4
    learning_rate_pinn: float = 5e-4
    learning_rate_decay_steps: int = 1000
    learning_rate_decay_ineus: float = 1.0
    learning_rate_decay_pinn: float = 1.0
    learning_rate_staircase: bool = True
    target_mixing: float = 0.0
    n_inner: int = 20
    num_batches: int = 32
    h: float = 0.5
    alpha: float = 0.4
    averaging_block_size: int = 10
    distillation_steps: int = 200
    distillation_learning_rate: float = 1e-3
    replace_with_student: bool = True
    use_rad: bool = True
    rad_period: int = 5
    rad_exponent: float = 1.0
    rad_offset: float = 1.0
    rad_candidate_size: int = 16384
    rad_eval_batch_size: int = 2048
    jump_samples: int = 64
    test_jump_samples: int = 50
    rad_jump_samples: int = 25
    log_every: int = 2
    num_t0_each_epoch: int = 5


@dataclass(frozen=True)
class BSDEConfig:
    """Local transformed deep-BSDE baseline settings."""

    num_local_models: int = 100
    local_epochs: int = 10000
    local_batch_size: int = 1024
    local_learning_rate: float = 3e-4
    local_eval_every: int = 500
    n_time_steps: int = 25
    z_hidden_dim: int = 128
    z_hidden_layers: int = 3
    jump_hidden_dim: int = 128
    jump_hidden_layers: int = 3
    jump_expectation_samples: int = 64
    gradient_clip_norm: float = 1.0
    surrogate_epochs: int = 5000
    surrogate_batch_size: int = 256
    surrogate_learning_rate: float = 1e-3
    surrogate_hidden_dim: int = 128
    surrogate_hidden_layers: int = 3
    surrogate_eval_every: int = 500


@dataclass(frozen=True)
class EvalConfig:
    """Fixed reference/test-set settings."""

    test_seed: int = 2
    reference_seed: int = 124
    test_samples: int = 2000
    test_space_bound: float = 1.5
    reference_paths: int = 2_000_000
    reference_chunk_size: int = 10_000


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete immutable configuration for the nonlinear experiment."""

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bsde: BSDEConfig = field(default_factory=BSDEConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    runs: int = 10
    base_seed: int = 29


def default_config() -> ExperimentConfig:
    """Return the default nonlinear experiment configuration."""

    return ExperimentConfig()


def paper_config() -> ExperimentConfig:
    """Alias for the default nonlinear experiment configuration."""

    return default_config()
