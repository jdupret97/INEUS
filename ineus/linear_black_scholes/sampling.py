"""Exact jump-diffusion marginals and collocation-point sampling."""

import numpy as np
import tensorflow as tf


def _sample_jumpdiff_marginal(t_vec, S0, r_val, sigma_diag_val, Sigma_val, lam_vec_val, mu_j_val, sig_j_val, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random

    t_vec = np.asarray(t_vec, dtype=np.float32).reshape(-1, 1)
    N = t_vec.shape[0]
    d = len(S0)

    S0 = np.asarray(S0, dtype=np.float32).reshape(d,)
    sigma_diag_val = np.asarray(sigma_diag_val, dtype=np.float32).reshape(d,)
    lam_vec_val = np.asarray(lam_vec_val, dtype=np.float32).reshape(d,)
    mu_j_val = np.asarray(mu_j_val, dtype=np.float32).reshape(d,)
    sig_j_val = np.asarray(sig_j_val, dtype=np.float32).reshape(d,)
    Sigma_val = np.asarray(Sigma_val, dtype=np.float32)

    kappa_val = np.exp(mu_j_val + 0.5 * sig_j_val**2) - 1.0

    L = np.linalg.cholesky(Sigma_val).astype(np.float32)
    Z = rng.standard_normal((N, d)).astype(np.float32)
    Z_corr = Z @ L.T

    sqrt_t = np.sqrt(t_vec).astype(np.float32)
    diff = sqrt_t * (Z_corr * sigma_diag_val[None, :])

    Nt = rng.poisson(lam=(t_vec * lam_vec_val[None, :])).astype(np.float32)
    G = rng.standard_normal((N, d)).astype(np.float32)
    sumZ = Nt * mu_j_val[None, :] + np.sqrt(Nt) * sig_j_val[None, :] * G

    drift = ((r_val - lam_vec_val * kappa_val) - 0.5 * sigma_diag_val**2)[None, :] * t_vec

    logS = np.log(S0)[None, :] + drift + diff + sumZ
    x = np.exp(logS).astype(np.float32)
    return x


def sample_xt_paths(num_points, T_val, S0, r_val, sigma_diag_val, Sigma_val, lam_vec_val, mu_j_val, sig_j_val, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random
    t = rng.uniform(0.0, T_val, size=(num_points, 1)).astype(np.float32)
    x = _sample_jumpdiff_marginal(
        t_vec=t,
        S0=S0,
        r_val=r_val,
        sigma_diag_val=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam_vec_val=lam_vec_val,
        mu_j_val=mu_j_val,
        sig_j_val=sig_j_val,
        seed=seed,
    )
    return x, t


def sample_x_terminal_paths(num_points, T_val, S0, r_val, sigma_diag_val, Sigma_val, lam_vec_val, mu_j_val, sig_j_val, seed=None):
    t = np.full((num_points, 1), T_val, dtype=np.float32)
    return _sample_jumpdiff_marginal(
        t_vec=t,
        S0=S0,
        r_val=r_val,
        sigma_diag_val=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam_vec_val=lam_vec_val,
        mu_j_val=mu_j_val,
        sig_j_val=sig_j_val,
        seed=seed,
    )


def sample_xt_paths_tf(num_points, T_val, S0, r_val, sigma_diag_val, Sigma_val, lam_vec_val, mu_j_val, sig_j_val):
    x_np, t_np = sample_xt_paths(
        num_points=num_points,
        T_val=T_val,
        S0=S0,
        r_val=r_val,
        sigma_diag_val=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam_vec_val=lam_vec_val,
        mu_j_val=mu_j_val,
        sig_j_val=sig_j_val,
    )
    return tf.convert_to_tensor(x_np, dtype=tf.float32), tf.convert_to_tensor(t_np, dtype=tf.float32)


def sample_x_terminal_paths_tf(num_points, T_val, S0, r_val, sigma_diag_val, Sigma_val, lam_vec_val, mu_j_val, sig_j_val):
    x_np = sample_x_terminal_paths(
        num_points=num_points,
        T_val=T_val,
        S0=S0,
        r_val=r_val,
        sigma_diag_val=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam_vec_val=lam_vec_val,
        mu_j_val=mu_j_val,
        sig_j_val=sig_j_val,
    )
    return tf.convert_to_tensor(x_np, dtype=tf.float32)


def sample_xt(num_points, T_val, mode="uniform", S0=None, *, problem):
    if mode == "uniform":
        t = np.random.uniform(0.0, T_val, num_points).reshape(-1, 1).astype(np.float32)
        x = np.random.uniform(0.001, problem.sample_b, (num_points, problem.dimen)).astype(np.float32)
        return x, t

    if mode == "paths":
        if S0 is None:
            raise ValueError("S0 must be provided when mode='paths'.")
        return sample_xt_paths(
            num_points=num_points,
            T_val=T_val,
            S0=S0,
            r_val=problem.r,
            sigma_diag_val=problem.sigma_diag,
            Sigma_val=problem.Sigma,
            lam_vec_val=problem.lam_vec,
            mu_j_val=problem.mu_j,
            sig_j_val=problem.sig_j,
        )

    raise ValueError(f"Unknown sampling mode: {mode}")


def sample_xt_tf(num_points, T_val, mode="uniform", S0=None, *, problem):
    if mode == "uniform":
        t = tf.random.uniform((num_points, 1), 0.0, T_val, dtype=tf.float32)
        x = tf.random.uniform((num_points, problem.dimen), 0.001, problem.sample_b, dtype=tf.float32)
        return x, t

    if mode == "paths":
        if S0 is None:
            raise ValueError("S0 must be provided when mode='paths'.")
        return sample_xt_paths_tf(
            num_points=num_points,
            T_val=T_val,
            S0=S0,
            r_val=problem.r,
            sigma_diag_val=problem.sigma_diag,
            Sigma_val=problem.Sigma,
            lam_vec_val=problem.lam_vec,
            mu_j_val=problem.mu_j,
            sig_j_val=problem.sig_j,
        )

    raise ValueError(f"Unknown sampling mode: {mode}")
