"""Linear PIDE coefficients, TensorFlow constants, and exact solution."""

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .config import DEFAULT_CONFIG, ProblemConfig


@dataclass(frozen=True, eq=False)
class LinearProblem:
    """Per-run coefficients; construction never consumes the training RNG.

    Pass this object explicitly to numerical routines with ``problem=...``.
    The legacy module constants below always describe the default problem;
    they are never reassigned when another experiment runs.
    """

    config: ProblemConfig = field(default_factory=ProblemConfig)
    sigma: np.ndarray = field(init=False, repr=False)
    mu_j: np.ndarray = field(init=False, repr=False)
    cov_j: np.ndarray = field(init=False, repr=False)
    cov_j_chol: np.ndarray = field(init=False, repr=False)
    sigma_tf: tf.Tensor = field(init=False, repr=False)
    mu_j_tf: tf.Tensor = field(init=False, repr=False)
    cov_j_chol_tf: tf.Tensor = field(init=False, repr=False)
    lam_tf: tf.Tensor = field(init=False, repr=False)
    beta_tf: tf.Tensor = field(init=False, repr=False)
    c_tf: tf.Tensor = field(init=False, repr=False)
    T_tf: tf.Tensor = field(init=False, repr=False)
    sample_b_tf: tf.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        cfg = self.config
        rng = np.random.RandomState(cfg.coefficient_seed)
        # Preserve the original draw order and float32 conversion exactly.
        if cfg.diffusion_mode == "random":
            sigma = rng.normal(
                loc=0.0, scale=cfg.diffusion_scale,
                size=(self.dimen, self.dimend),
            ).astype(np.float32)
        elif cfg.diffusion_mode == "constant":
            sigma = np.full(
                (self.dimen, self.dimend), cfg.diffusion_scale, dtype=np.float32,
            )
        else:
            raise ValueError("diffusion_mode must be 'random' or 'constant'")
        if cfg.jump_covariance_mode == "random_diagonal":
            cov_j = np.diag(rng.uniform(
                low=cfg.jump_covariance_min,
                high=cfg.jump_covariance_scale, size=self.dimen,
            )).astype(np.float32)
        elif cfg.jump_covariance_mode == "isotropic":
            cov_j = np.eye(self.dimen, dtype=np.float32) * cfg.jump_covariance_scale
        else:
            raise ValueError(
                "jump_covariance_mode must be 'random_diagonal' or 'isotropic'"
            )
        diagonal = np.diag(cov_j)
        if np.any(diagonal < 0):
            raise ValueError("jump covariance must be nonnegative")
        # A zero covariance is valid, including in the no-jump limit.
        chol = (np.linalg.cholesky(cov_j) if np.all(diagonal > 0)
                else np.diag(np.sqrt(diagonal))).astype(np.float32)
        arrays = {
            "sigma": sigma, "mu_j": np.zeros(self.dimen, dtype=np.float32),
            "cov_j": cov_j, "cov_j_chol": chol,
        }
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("sigma", "mu_j", "cov_j_chol", "lam", "beta", "c", "T", "sample_b"):
            object.__setattr__(self, name + "_tf", tf.constant(getattr(self, name), tf.float32))

    @property
    def dimen(self):
        return self.config.state_dim

    @property
    def dimend(self):
        return self.config.brownian_dim

    @property
    def T(self):
        return self.config.horizon

    @property
    def lam(self):
        return self.config.jump_intensity

    @property
    def beta(self):
        return self.config.drift

    @property
    def c(self):
        return self.config.discount

    @property
    def sample_b(self):
        return self.config.sample_box

    @property
    def test_box(self):
        return self.config.test_box

    def exact_value(self, t, x):
        return value1(self.T, t, x, self.beta, self.c, self.lam, self.sigma, self.cov_j)


def make_problem(config=DEFAULT_CONFIG.problem):
    """Build an isolated problem from an immutable ``ProblemConfig``."""
    return LinearProblem(config)


DEFAULT_PROBLEM = make_problem()


def resolve_problem(problem=None):
    return DEFAULT_PROBLEM if problem is None else problem


# Backward-compatible, read-only aliases for callers of the original benchmark.
PROBLEM = DEFAULT_PROBLEM.config
dimen, dimend = DEFAULT_PROBLEM.dimen, DEFAULT_PROBLEM.dimend
T, lam, beta, c = DEFAULT_PROBLEM.T, DEFAULT_PROBLEM.lam, DEFAULT_PROBLEM.beta, DEFAULT_PROBLEM.c
sample_b, test_box = DEFAULT_PROBLEM.sample_b, DEFAULT_PROBLEM.test_box
sigma, mu_j = DEFAULT_PROBLEM.sigma, DEFAULT_PROBLEM.mu_j
cov_j, cov_j_chol = DEFAULT_PROBLEM.cov_j, DEFAULT_PROBLEM.cov_j_chol
sigma_tf, mu_j_tf = DEFAULT_PROBLEM.sigma_tf, DEFAULT_PROBLEM.mu_j_tf
cov_j_chol_tf = DEFAULT_PROBLEM.cov_j_chol_tf
lam_tf, beta_tf = DEFAULT_PROBLEM.lam_tf, DEFAULT_PROBLEM.beta_tf
c_tf, T_tf, sample_b_tf = DEFAULT_PROBLEM.c_tf, DEFAULT_PROBLEM.T_tf, DEFAULT_PROBLEM.sample_b_tf


def value1(T, t, x, beta, c, lam, sigma, cov_j):
    """
    Exact solution for terminal condition g(x)=||x||^2.
    """
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    x = np.asarray(x, dtype=np.float32)
    tau = T - t

    norm_x2 = np.sum(x**2, axis=1, keepdims=True)
    diff_trace = float(np.sum(sigma * sigma))
    jump_trace = float(np.trace(cov_j))
    a = diff_trace + lam * jump_trace

    if np.isclose(2.0 * beta, 0.0):
        u = np.exp(-c * tau) * (norm_x2 + a * tau)
    else:
        u = (
            np.exp((2.0 * beta - c) * tau) * norm_x2
            + np.exp(-c * tau) * a * (np.exp(2.0 * beta * tau) - 1.0) / (2.0 * beta)
        )

    return u.reshape(-1).astype(np.float32)


def phi_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
