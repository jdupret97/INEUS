"""Basket-call payoff and fixed coefficients for one problem dimension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from .config import ProblemConfig


@dataclass(frozen=True)
class BSProblem:
    """Dimension-specific Black-Scholes/Merton coefficients and sampling data."""

    dimen: int
    dimend: int
    T: float
    r: float
    K: float
    max_S: float
    sample_b: float
    lam_vec: np.ndarray
    Sigma: np.ndarray
    sigma_diag: np.ndarray
    mu_j: np.ndarray
    sig_j: np.ndarray
    kappa_vec: np.ndarray
    w: np.ndarray
    S0_train: np.ndarray


def make_problem(dimen_value: int, config: ProblemConfig) -> BSProblem:
    """Construct the equicorrelated basket-call problem without drawing randomness."""
    lam_vec = np.full(dimen_value, config.jump_intensity, dtype=np.float32)
    rho_corr = config.correlation
    Sigma = (
        rho_corr * np.ones((dimen_value, dimen_value))
        + (1.0 - rho_corr) * np.eye(dimen_value)
    ).astype(np.float32)
    sigma_diag = np.full(dimen_value, config.sigma, dtype=np.float32)
    mu_j = np.full(dimen_value, config.jump_mean, dtype=np.float32)
    sig_j = np.full(dimen_value, config.jump_std, dtype=np.float32)
    kappa_vec = np.exp(mu_j + 0.5 * (sig_j**2)) - 1.0
    w = (np.ones(dimen_value) / dimen_value).astype(np.float32)
    K = config.K_multiplier * config.max_S
    S0_train = (K * np.ones(dimen_value)).astype(np.float32)
    return BSProblem(
        dimen=dimen_value,
        dimend=dimen_value,
        T=config.T,
        r=config.r,
        K=K,
        max_S=config.max_S,
        sample_b=config.sample_b,
        lam_vec=lam_vec,
        Sigma=Sigma,
        sigma_diag=sigma_diag,
        mu_j=mu_j,
        sig_j=sig_j,
        kappa_vec=kappa_vec,
        w=w,
        S0_train=S0_train,
    )


def phicontrol(S, w_val, K_val):
    S = np.asarray(S, dtype=float)
    w_val = np.asarray(w_val, dtype=float)
    basket = S @ w_val if S.ndim == 2 else float(np.dot(w_val, S))
    return np.maximum(basket - K_val, 0.0)


def phi_tf(S, w_val, K_val):
    S = tf.convert_to_tensor(S, dtype=tf.float32)
    w_tf = tf.convert_to_tensor(w_val, dtype=tf.float32)
    K_tf = tf.cast(K_val, tf.float32)

    basket = tf.linalg.matvec(S, w_tf) if len(S.shape) == 2 else tf.tensordot(S, w_tf, axes=1)
    return tf.reshape(tf.maximum(basket - K_tf, 0.0), (-1, 1))
