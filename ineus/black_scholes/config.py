"""Configuration of the nonlinear Black–Scholes default-risk experiment.

Configuration can be inspected without importing NumPy or TensorFlow.
Array-valued coefficients are constructed lazily for numerical execution.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PDEConfig:
    """Default-risk equation with optional multiplicative lognormal jumps."""

    dimen: int = 100
    T: float = 1.0
    delta_recovery: float = 2.0 / 3.0
    R_rate: float = 0.02
    mu_bar: float = 0.02
    sigma_bar: float = 0.20
    v_h: float = 50.0
    v_l: float = 70.0
    gamma_h: float = 0.20
    gamma_l: float = 0.02
    jump_intensity: float = 0.1
    jump_mean: float = -0.2
    jump_std: float = 0.15
    jump_corr: float = 0.5
    use_jumps: bool = True

    @property
    def sigma_diag(self):
        """Per-coordinate diffusion volatilities, in float32."""
        import numpy as np

        return self.sigma_bar * np.ones(self.dimen, dtype=np.float32)

    @property
    def Sigma(self):
        """Identity diffusion correlation matrix."""
        import numpy as np

        return np.eye(self.dimen, dtype=np.float32)

    @property
    def mu_bar_adjusted(self):
        """Drift used by the operator; no jump compensation is applied."""
        return self.mu_bar


@dataclass(frozen=True)
class ModelConfig:
    """DGM value network with an optional hard terminal-condition wrapper."""

    hidden_dim: int = 100
    n_layers: int = 3
    constrained: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Iterative updates, sampling, block distillation and checkpoint settings."""

    sample_b: float = 200.0
    h_scale: float = 1.0
    zeta_scale: float = 1.25
    n_inner: int = 20
    num_batches: int = 32
    lr: float = 1e-3
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.995
    epochs: int = 400
    batch_size: int = 1024
    num_samples: int = 2048
    rho1: float = 0.0
    w_bc: float = 1.0
    w_int: float = 1.0
    buffer_size: int = 16384
    use_path_sampling: bool = True
    block_size: int = 10
    distill_steps: int = 200
    student_lr: float = 1e-4
    replace_with_student: bool = True
    use_rad: bool = False
    rad_period: int = 5
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_candidate_size: Optional[int] = None
    rad_eval_batch_size: int = 2048
    test_samples: int = 500
    grad_clip: float = 1.0
    use_cosine_lr: bool = True
    save_dir: str = "results/black_scholes"
    resume_path: Optional[str] = None

    def __post_init__(self):
        if self.n_inner < 2:
            raise ValueError("train.n_inner must be at least 2.")
        if self.h_scale <= 0 or self.zeta_scale <= 0:
            raise ValueError("train.h_scale and train.zeta_scale must be positive.")
        if self.rad_candidate_size is None:
            object.__setattr__(self, "rad_candidate_size", 8 * self.num_samples)
        if self.use_rad and self.use_path_sampling:
            raise ValueError("RAD is supported only with uniform sampling; set use_path_sampling: false.")
        if not self.replace_with_student:
            raise ValueError("Block distillation requires replace_with_student: true.")

    @property
    def zeta(self):
        return self.h_scale / self.zeta_scale

    @property
    def alpha(self):
        return self.zeta / self.h_scale

    @property
    def bet_scale(self):
        return self.h_scale / (self.n_inner - 1)


@dataclass(frozen=True)
class EvalConfig:
    """Reference-point evaluation; set u_ref to null to omit reference errors.

    A reference value must be appropriate for the chosen dimension and jump
    setting. It is an external benchmark, not an exact solution formula.
    """

    x0_value: float = 100.0
    u_ref: Optional[float] = 55.810
    dimen: int = 100

    @property
    def x0_eval(self):
        import numpy as np

        return self.x0_value * np.ones((1, self.dimen), dtype=np.float32)

    @property
    def t0_eval(self):
        import numpy as np

        return np.array([[0.0]], dtype=np.float32)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for one seeded INEUS experiment."""

    problem: PDEConfig = field(default_factory=PDEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        if self.evaluation.dimen != self.problem.dimen:
            raise ValueError("evaluation.dimen must match problem.dimen.")


def default_config():
    """Return the full-size default experiment configuration."""
    return ExperimentConfig()


DEFAULT_CONFIG = default_config()
