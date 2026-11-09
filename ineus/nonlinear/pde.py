"""Nonlinear PIDE operators and residual-adaptive sampling."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .problem import (
    COV_J_CHOL as cov_j_chol,
    DIMENSION as dimen,
    DRIVER_CONSTANT as f_const,
    ETA as eta,
    JUMP_DIMENSION as dimend,
    JUMP_INTENSITY as lam,
    MU_J as mu_j,
)
from .sampling import sample_xt
from .problem import resolve_problem


def fprime(h, model1, x_tf, t_tf, dimend=None, eta=None, *, problem=None):
    problem = resolve_problem(problem)
    dimend = problem.jump_dim if dimend is None else dimend
    eta = problem.eta if eta is None else eta
    dtype = model1.dtype_policy.compute_dtype

    x = tf.cast(x_tf, dtype)
    t = tf.cast(t_tf, dtype)
    h = tf.cast(h, dtype)

    B = tf.shape(x)[0]
    d = tf.shape(x)[1]
    m = dimend

    denom = tf.cast(2.0 * dimend, dtype)
    t2 = t + tf.square(h) / denom

    I = tf.eye(d, dtype=dtype)
    shift = h[:, :, None] * I[None, :, :]       # [B,d,d]
    x2 = x[:, None, :] + shift                  # [B,d,d]
    t2 = tf.broadcast_to(t2[:, None, :], [B, m, 1])

    x2_flat = tf.reshape(x2, [-1, d])
    t2_flat = tf.reshape(t2, [-1, 1])

    V = model1([x2_flat, t2_flat], training=False)
    V = tf.reshape(V, [B, m, 1])

    u = tf.exp(-tf.cast(eta,dtype) * V)
    return tf.reduce_sum(u, axis=1)


def compute_second_derivative_term(modela, x_tf, t_tf, h_tf, *, problem=None):
    problem = resolve_problem(problem)
    with tf.GradientTape() as tape2:
        tape2.watch(h_tf)
        with tf.GradientTape() as tape1:
            tape1.watch(h_tf)
            f_val = fprime(h_tf, modela, x_tf, t_tf, problem=problem)
        df_dh = tape1.gradient(f_val, h_tf)
    d2f_dh2 = tape2.gradient(df_dh, h_tf)
    return d2f_dh2


def pinn_loss1bis(model1, model1targ, x, t, e, scale, T=None, *, problem=None):
    problem = resolve_problem(problem)
    eta, lam, f_const = problem.eta, problem.jump_intensity, problem.driver_constant
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    e_tf = tf.convert_to_tensor(e, dtype=tf.float32)

    batch_n = tf.shape(x_tf)[0]
    h_tf = tf.zeros((batch_n, 1), dtype=tf.float32)

    d2f_dh2 = compute_second_derivative_term(model1targ, x_tf, t_tf, h_tf, problem=problem)

    u_targ = model1targ([x_tf, t_tf], training=False)
    diff_term = -tf.exp(eta * u_targ) * d2f_dh2 / eta
    if problem.has_jumps:
        u_targ_e = model1targ([x_tf + e_tf, t_tf], training=False)
        jump_term = -lam * (tf.exp(-eta * (u_targ_e - u_targ)) - 1.0) / eta
    else:
        jump_term = tf.zeros_like(u_targ)

    lhs = tf.reshape(model1([x_tf, t_tf], training=True), [batch_n])
    rhs = tf.reshape(u_targ + scale * (diff_term + f_const + jump_term), [batch_n])

    residual = lhs - rhs
    loss = tf.reduce_mean(tf.square(residual))
    return loss


def pinn_residual_pointwise_nl(model1, x, t, T=None, K=20, training=False, *, problem=None):
    """
    Nonlinear residual:
        diff_term + f_const - lam/eta * ( E[e^{-eta(u(x+e)-u(x))}] - 1 )

    Returns
    -------
    rhs        : pointwise residual vector
    rhs_mean   : mean absolute residual
    rhs_square : mean squared residual
    """
    problem = resolve_problem(problem)
    eta, lam, f_const = problem.eta, problem.jump_intensity, problem.driver_constant
    dimen, mu_j, cov_j_chol = problem.state_dim, problem.mu_j, problem.cov_j_chol
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)

    batch_n = tf.shape(x_tf)[0]
    h_tf = tf.zeros((batch_n, 1), dtype=tf.float32)

    d2f_dh2 = compute_second_derivative_term(model1, x_tf, t_tf, h_tf, problem=problem)
    V = model1([x_tf, t_tf], training=training)   # [B,1]

    diff_term = -tf.exp(eta * V) * d2f_dh2 / eta

    if not problem.has_jumps:
        rhs = tf.reshape(diff_term + f_const, [batch_n])
        return rhs, tf.reduce_mean(tf.abs(rhs)), tf.reduce_mean(tf.square(rhs))

    # Monte Carlo jump estimate
    z = tf.random.normal(
        shape=tf.stack([
            batch_n,
            tf.constant(K, dtype=tf.int32),
            tf.constant(dimen, dtype=tf.int32)
        ]),
        dtype=tf.float32
    )

    mu_j_tf = tf.cast(mu_j, tf.float32)
    e_samples = tf.linalg.matmul(z, cov_j_chol, transpose_b=True) + mu_j_tf[None, None, :]  # [B,K,d]

    x_shifted = x_tf[:, None, :] + e_samples
    t_rep = tf.repeat(t_tf[:, None, :], repeats=K, axis=1)

    x_flat = tf.reshape(x_shifted, [-1, dimen])
    t_flat = tf.reshape(t_rep, [-1, 1])

    V_e_flat = model1([x_flat, t_flat], training=training)
    V_e = tf.reshape(
        V_e_flat,
        tf.stack([batch_n, tf.constant(K, dtype=tf.int32), tf.constant(1, dtype=tf.int32)])
    )

    exp_term = tf.exp(-eta * (V_e - V[:, None, :]))
    jump_mc = -lam * (tf.reduce_mean(exp_term, axis=1) - 1.0) / eta

    rhs = tf.reshape(diff_term + f_const + jump_mc, [batch_n])
    rhs_mean = tf.reduce_mean(tf.abs(rhs))
    rhs_square = tf.reduce_mean(tf.square(rhs))

    return rhs, rhs_mean, rhs_square


def pinn_rhs_square_loss_nl(model1, x, t, T=None, K=50, *, problem=None):
    return pinn_residual_pointwise_nl(model1, x, t, T=T, K=K, training=True, problem=problem)[2]


def rad_resample_dataset_nl(
    model1,
    T,
    num_points,
    candidate_size,
    K_jump=25,
    k=1.0,
    c_rad=1.0,
    eval_batch_size=2048,
    replace=False,
    eps_floor=1e-12,
    *,
    problem=None,
):
    problem = resolve_problem(problem)
    x_cand, t_cand = sample_xt(candidate_size, T, problem=problem)

    residual_abs = np.empty(candidate_size, dtype=np.float32)

    for start in range(0, candidate_size, eval_batch_size):
        end = min(start + eval_batch_size, candidate_size)
        r, _, _ = pinn_residual_pointwise_nl(
            model1=model1,
            x=x_cand[start:end],
            t=t_cand[start:end],
            T=T,
            K=K_jump,
            training=False,
            problem=problem,
        )
        residual_abs[start:end] = np.abs(r.numpy())

    resid_k = np.power(np.maximum(residual_abs, eps_floor), k)
    denom = np.mean(resid_k) + eps_floor
    scores = resid_k / denom + c_rad
    probs = scores / np.sum(scores)

    use_replace = replace or (num_points > candidate_size)
    idx = np.random.choice(candidate_size, size=num_points, replace=use_replace, p=probs)

    stats = {
        "mean_residual": float(np.mean(residual_abs)),
        "max_residual": float(np.max(residual_abs)),
        "min_residual": float(np.min(residual_abs)),
        "mean_score": float(np.mean(scores)),
    }

    return x_cand[idx], t_cand[idx], stats
