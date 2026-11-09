"""TensorFlow-free configuration for the linear Black–Scholes experiment."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemConfig:
    """Basket-call Merton jump-diffusion equation and sampling domain."""

    dimensions: tuple[int, ...] = (10,)
    T: float = 1.0
    r: float = 0.05
    max_S: float = 100.0
    K_multiplier: float = 0.3
    sample_b: float = 120.0
    jump_intensity: float = 0.5
    correlation: float = 0.10
    sigma: float = 0.15
    jump_mean: float = 0.20
    jump_std: float = 0.30

    @property
    def K(self):
        """Strike of the equally weighted basket call."""
        return self.K_multiplier * self.max_S


@dataclass(frozen=True)
class ModelConfig:
    """DGM with independent stage gates and a hard terminal-condition wrapper."""

    hidden_dim: int = 50
    n_layers: int = 3


@dataclass(frozen=True)
class TrainConfig:
    """INEUS iteration, replay sampling, and optional second training phase."""

    batch_size: int = 1024
    num_samples: int = 2048
    # Retained experiment metadata; the training loop does not use this count.
    nc_samples: int = 2048
    buffer_size: int = 4096
    rho1: float = 0.0
    h_scale: float = 0.5
    relaxation: float = 0.4
    w_bc: float = 0.0
    w_int: float = 1.0
    epochs: int = 1000
    lr1: float = 0.0005
    lr2: float = 0.001
    n_inner: int = 20
    num_batches: int = 32
    run_phase2: bool = False
    use_rad: bool = True
    rad_period: int = 5
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_candidate_multiplier: int = 8
    rad_eval_batch_size: int = 2048
    use_distill: bool = True
    distill_block_size: int = 10
    distill_steps: int = 200
    distill_student_lr: float = 0.001
    replace_with_student: bool = True

    @property
    def alpha(self):
        """Weight of the new iterate in the outer average."""
        return self.relaxation / self.h_scale

    @property
    def bet_scale(self):
        """Time increment used by the inner stochastic iteration."""
        return self.h_scale / (self.n_inner - 1)


@dataclass(frozen=True)
class EvalConfig:
    """Fixed Monte Carlo test set and one-dimensional reference curves."""

    test_samples: int = 2000
    mc_test_paths: int = 20000
    mc_curve_paths: int = 20000
    n_x_curve: int = 401
    n_t_curve: int = 401
    output_tag: str = "bs_dimension_sweep"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete numerical configuration; seed and repeats belong to the runner."""

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def default_config():
    """Return the full-size basket-call configuration at dimension 10."""
    return ExperimentConfig()


DEFAULT_CONFIG = default_config()
