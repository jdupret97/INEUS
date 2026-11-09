"""Nonlinear PIDE coefficients, terminal condition, and reference solution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import tensorflow as tf

from .config import ProblemConfig


@dataclass(frozen=True, eq=False)
class NonlinearProblem(ProblemConfig):
    """Immutable per-run coefficients; defaults remain the original 10D HJB."""

    mu_j: np.ndarray = field(init=False, repr=False)
    cov_j: np.ndarray = field(init=False, repr=False)
    cov_j_chol: tf.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if self.state_dim < 1 or self.jump_dim != self.state_dim:
            raise ValueError("Nonlinear HJB requires positive state_dim and jump_dim == state_dim.")
        coefficients = (self.horizon, self.eta, self.jump_intensity,
                        self.driver_constant, self.jump_covariance_scale)
        if not all(np.isfinite(value) for value in coefficients):
            raise ValueError("Nonlinear HJB coefficients must be finite.")
        if self.horizon <= 0 or self.eta <= 0:
            raise ValueError("horizon and eta must be positive.")
        if self.jump_intensity < 0 or self.jump_covariance_scale < 0:
            raise ValueError("jump_intensity and jump_covariance_scale must be nonnegative.")
        mu = np.zeros(self.state_dim, dtype=np.float32)
        covariance = self.jump_covariance_scale * np.eye(self.state_dim, dtype=np.float32)
        # Keep the original Cholesky calculation for positive covariance; the
        # zero-covariance case is a valid deterministic mark distribution.
        chol = np.linalg.cholesky(covariance) if self.jump_covariance_scale > 0 else np.zeros_like(covariance)
        mu.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mu_j", mu)
        object.__setattr__(self, "cov_j", covariance)
        object.__setattr__(self, "cov_j_chol", tf.cast(chol, tf.float32))

    @property
    def has_jumps(self):
        return self.jump_intensity > 0 and self.jump_covariance_scale > 0


def make_problem(config: ProblemConfig | None = None) -> NonlinearProblem:
    """Construct independent runtime constants without changing module state."""
    if isinstance(config, NonlinearProblem):
        return config
    return NonlinearProblem(**asdict(config or ProblemConfig()))


PROBLEM = ProblemConfig()
DEFAULT_PROBLEM = make_problem(PROBLEM)


def resolve_problem(problem=None) -> NonlinearProblem:
    return DEFAULT_PROBLEM if problem is None else make_problem(problem)


# Backwards-compatible, read-only aliases for default calls and older scripts.
DIMENSION = DEFAULT_PROBLEM.state_dim
JUMP_DIMENSION = DEFAULT_PROBLEM.jump_dim
T = DEFAULT_PROBLEM.horizon
ETA = DEFAULT_PROBLEM.eta
JUMP_INTENSITY = DEFAULT_PROBLEM.jump_intensity
DRIVER_CONSTANT = DEFAULT_PROBLEM.driver_constant
MU_J = DEFAULT_PROBLEM.mu_j
COV_J = DEFAULT_PROBLEM.cov_j
COV_J_CHOL = DEFAULT_PROBLEM.cov_j_chol


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float32)
    amax = np.max(a, axis=axis, keepdims=True)
    return (amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))).squeeze(axis)


def g_fin(x, *, problem=None):
    x = np.asarray(x, dtype=np.float32)
    return np.sum(x**2, axis=-1)


def g_fin_tf(x, *, problem=None):
    x = tf.cast(x, tf.float32)
    return tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)


def u_pairs(
    x_pts, t_pts, T=None, eta=None, lam=None,
    muJ=None, SigmaJ=None,
    g=None,
    n_paths=200_000,
    paths_chunk=20_000,
    f_const=None,
    seed=123,
    *,
    problem=None,
):
    """
    Monte Carlo reference solution:
      Z ~ N(muJ, SigmaJ), K ~ Poisson(lam*(T-t_i))
    """
    problem = resolve_problem(problem)
    T = problem.horizon if T is None else T
    eta = problem.eta if eta is None else eta
    lam = problem.jump_intensity if lam is None else lam
    muJ = problem.mu_j if muJ is None else muJ
    SigmaJ = problem.cov_j if SigmaJ is None else SigmaJ
    f_const = problem.driver_constant if f_const is None else f_const
    g = g_fin if g is None else g
    rng = np.random.default_rng(seed)

    x_pts = np.asarray(x_pts, dtype=np.float32)
    t_pts = np.asarray(t_pts, dtype=np.float32).reshape(-1)

    N, d = x_pts.shape
    if t_pts.shape[0] != N:
        raise ValueError("x_pts and t_pts must have same length N.")

    dt = (T - t_pts).astype(np.float32)
    if np.any(dt < 0):
        raise ValueError("All t_pts must satisfy t <= T.")

    muJ = np.asarray(muJ, dtype=np.float32).reshape(d)
    SigmaJ = np.asarray(SigmaJ, dtype=np.float32).reshape(d, d)

    L = np.linalg.cholesky(SigmaJ) if np.any(SigmaJ) else np.zeros_like(SigmaJ)
    has_jumps = lam > 0 and (np.any(SigmaJ) or np.any(muJ))

    u_out = np.empty(N, dtype=np.float32)

    is_T = (dt == 0.0)
    if np.any(is_T):
        gx = np.asarray(g(x_pts[is_T]), dtype=np.float32)
        if gx.ndim == 2:
            gx = gx[:, 0]
        u_out[is_T] = gx

    idx = np.where(~is_T)[0]
    if idx.size == 0:
        return u_out

    x = x_pts[idx]
    dtM = dt[idx]
    M = idx.size

    logS = np.full(M, -np.inf, dtype=np.float32)
    brown_scale = np.sqrt(2.0 * dtM)

    done = 0
    while done < n_paths:
        P = min(paths_chunk, n_paths - done)
        done += P

        ZW = rng.standard_normal(size=(M, P, d), dtype=np.float32)
        dW = ZW * brown_scale[:, None, None]

        if has_jumps:
            K = rng.poisson(lam * dtM[:, None], size=(M, P)).astype(np.float32)
            eps = rng.standard_normal(size=(M, P, d), dtype=np.float32)
            jump_noise = eps @ L.T
            jumps = K[:, :, None] * muJ[None, None, :] + np.sqrt(K)[:, :, None] * jump_noise
        else:
            jumps = 0.0

        XT = x[:, None, :] + dW + jumps

        gvals = np.asarray(g(XT), dtype=np.float32)
        if gvals.ndim == 3:
            gvals = gvals[..., 0]

        a = -eta * gvals
        chunk_logsum = logsumexp(a, axis=1)
        logS = np.logaddexp(logS, chunk_logsum)

    logE = logS - np.log(float(n_paths))
    u_mc = f_const * dtM - (1.0 / eta) * logE
    u_out[idx] = u_mc
    return u_out
