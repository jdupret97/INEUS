"""Default configuration for the linear three-method comparison.

Includes seeded problem coefficients, network architectures, training
schedules, and evaluation settings for the 10-dimensional experiment.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemConfig:
    """Coefficients of a configurable-dimensional quadratic linear PIDE."""

    state_dim: int = 10
    brownian_dim: int = 10
    horizon: float = 0.5
    jump_intensity: float = 0.25
    drift: float = 1.0
    discount: float = 2.0
    sample_box: float = 1.75
    test_box: float = 1.5
    coefficient_seed: int = 23
    diffusion_scale: float = 0.3
    jump_covariance_min: float = 0.01
    jump_covariance_scale: float = 0.4
    diffusion_mode: str = "random"
    jump_covariance_mode: str = "random_diagonal"
    dimensions: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    """Shared DGM architecture used by INEUS and PINN."""

    hidden_dim: int = 50
    n_layers: int = 4
    output_dim: int = 1
    constrained: bool = True


@dataclass(frozen=True)
class INEUSTrainConfig:
    """Training settings for the iterative neural update."""

    epochs: int = 1000
    num_samples: int = 2048
    buffer_size: int = 4096
    batch_size: int = 1024
    learning_rate: float = 5e-4
    target_weight: float = 0.0
    n_inner: int = 20
    num_batches: int = 32
    h_scale: float = 0.5
    relaxation: float = 0.4
    block_size: int = 10
    distill_steps: int = 200
    student_learning_rate: float = 1e-3
    replace_with_student: bool = True
    use_rad: bool = True
    rad_period: int = 5
    rad_exponent: float = 1.0
    rad_offset: float = 1.0
    rad_candidate_multiplier: int = 8
    rad_eval_batch_size: int = 2048
    jump_samples: int = 25
    test_jump_samples: int = 50
    log_every: int = 2


@dataclass(frozen=True)
class PINNTrainConfig:
    """Training settings for the strong-residual PINN baseline."""

    epochs: int = 1000
    num_samples: int = 2048
    buffer_size: int = 4096
    batch_size: int = 1024
    learning_rate: float = 5e-4
    n_inner: int = 20
    num_batches: int = 32
    use_rad: bool = True
    rad_period: int = 5
    rad_exponent: float = 1.0
    rad_offset: float = 1.0
    rad_candidate_multiplier: int = 8
    rad_eval_batch_size: int = 2048
    jump_samples: int = 64
    rad_jump_samples: int = 25
    test_jump_samples: int = 50
    log_every: int = 2


@dataclass(frozen=True)
class BSDEConfig:
    """Local Deep-BSDE-to-global-surrogate baseline settings."""

    num_local_models: int = 100
    local_epochs: int = 10000
    local_batch_size: int = 1024
    local_learning_rate: float = 3e-4
    local_eval_every: int = 500
    time_steps: int = 25
    hidden_dim_z: int = 128
    n_hidden_z: int = 3
    hidden_dim_u: int = 128
    n_hidden_u: int = 3
    jump_expectation_samples: int = 64
    clipnorm: float = 1.0
    surrogate_epochs: int = 5000
    surrogate_batch_size: int = 256
    surrogate_learning_rate: float = 1e-3
    surrogate_hidden_dim: int = 128
    surrogate_n_hidden: int = 3
    surrogate_eval_every: int = 500


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation sample drawn from the seeded training NumPy stream."""

    test_samples: int = 2000
    default_output_dir: str = "./rPINN/comparison_three_methods_local_bsde"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete immutable configuration for the linear comparison."""

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ineus: INEUSTrainConfig = field(default_factory=INEUSTrainConfig)
    pinn: PINNTrainConfig = field(default_factory=PINNTrainConfig)
    bsde: BSDEConfig = field(default_factory=BSDEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


DEFAULT_CONFIG = ExperimentConfig()
