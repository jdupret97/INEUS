"""Bellman target, strong PIDE residual, and RAD sampling.

The comparison uses the strong residual for PINN training,
RAD sampling, and diagnostic checkpoint selection for both INEUS and PINN.
INEUS itself trains against the iterative target defined by pinn_loss1bis.
"""

import numpy as np
import tensorflow as tf

from .problem import resolve_problem
from .sampling import sample_xt


def fprime(h, model1, x_tf, t_tf, dimend, sigma, beta):
    dtype = tf.float32

    x = tf.cast(x_tf, dtype)
    t = tf.cast(t_tf, dtype)
    h = tf.cast(h, dtype)

    sigma_used = tf.cast(sigma, dtype)
    sigma_T = tf.transpose(sigma_used)

    B = tf.shape(x)[0]
    d = tf.shape(x)[1]
    m = dimend

    denom = tf.cast(2.0 * dimend, dtype)
    t2 = t + tf.square(h) / denom
    common = x + tf.square(h) * tf.cast(beta, dtype) * x / denom

    shift = (h / tf.sqrt(tf.cast(2.0, dtype)))[:, :, None] * sigma_T[None, :, :]
    x2 = common[:, None, :] + shift
    t2 = tf.broadcast_to(t2[:, None, :], tf.stack([B, m, 1]))

    x2_flat = tf.reshape(x2, tf.stack([B * m, d]))
    t2_flat = tf.reshape(t2, tf.stack([B * m, 1]))

    V = model1([x2_flat, t2_flat], training=False)
    V = tf.reshape(V, tf.stack([B, m, 1]))

    return tf.reduce_sum(V, axis=1)


def compute_second_derivative(modela, x_tf, t_tf, h, *, problem=None):
    problem = resolve_problem(problem)
    with tf.GradientTape() as tape2:
        tape2.watch(h)
        with tf.GradientTape() as tape1:
            tape1.watch(h)
            f_val = fprime(
                h, modela, x_tf, t_tf, problem.dimend, problem.sigma_tf, problem.beta,
            )
        df_dh = tape1.gradient(f_val, h)
    d2f_dh2 = tape2.gradient(df_dh, h)
    return d2f_dh2


def pinn_loss1bis(model1, model1targ, x, t, e, scale, T, h, *, problem=None):
    problem = resolve_problem(problem)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    e_tf = tf.convert_to_tensor(e, dtype=tf.float32)
    h_tf = tf.convert_to_tensor(h, dtype=tf.float32)

    df_dh2 = compute_second_derivative(model1targ, x_tf, t_tf, h_tf, problem=problem)
    V_targ = model1targ([x_tf, t_tf], training=False)
    if problem.lam == 0:
        jump_term = tf.zeros_like(V_targ)
    else:
        V_targ_e = model1targ([x_tf + e_tf, t_tf], training=False)
        jump_term = problem.lam_tf * (V_targ_e - V_targ)

    lhs = tf.reshape(model1([x_tf, t_tf], training=True), [-1])
    rhs = tf.reshape(
        V_targ + scale * (df_dh2 - problem.c_tf * V_targ + jump_term),
        [-1]
    )

    residual = lhs - rhs
    loss = tf.reduce_mean(tf.square(residual))
    return loss


def pinn_residual_pointwise(model1, x, t, K=20, training=False, *, problem=None):
    """
    Standard PINN residual for the PIDE.
    Returns:
        rhs        : pointwise residual
        rhs_mean   : mean absolute residual
        rhs_square : mean squared residual
    """
    problem = resolve_problem(problem)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)

    batch_n = tf.shape(x_tf)[0]
    h_local = tf.zeros(tf.stack([batch_n, 1]), dtype=tf.float32)

    df_dh2 = compute_second_derivative(model1, x_tf, t_tf, h_local, problem=problem)
    V = model1([x_tf, t_tf], training=training)

    if problem.lam == 0:
        jump_term = tf.zeros_like(V)
    else:
        z = tf.random.normal(tf.stack([batch_n * K, problem.dimen]), dtype=tf.float32)
        e_flat = (tf.linalg.matmul(z, problem.cov_j_chol_tf, transpose_b=True)
                  + problem.mu_j_tf[None, :])
        e_samples = tf.reshape(e_flat, tf.stack([batch_n, K, problem.dimen]))

        x_shifted = x_tf[:, None, :] + e_samples
        t_rep = tf.repeat(t_tf[:, None, :], repeats=K, axis=1)

        x_flat = tf.reshape(x_shifted, tf.stack([batch_n * K, problem.dimen]))
        t_flat = tf.reshape(t_rep, tf.stack([batch_n * K, 1]))

        V_e_flat = model1([x_flat, t_flat], training=training)
        V_e = tf.reshape(V_e_flat, tf.stack([batch_n, K, 1]))
        EV_e = tf.reduce_mean(V_e, axis=1)
        jump_term = problem.lam_tf * (EV_e - V)

    rhs = tf.reshape(df_dh2 - problem.c_tf * V + jump_term, [-1])
    rhs_mean = tf.reduce_mean(tf.abs(rhs))
    rhs_square = tf.reduce_mean(tf.square(rhs))

    return rhs, rhs_mean, rhs_square


def pinn_rhs_square_loss(model1, x, t, K=20, *, problem=None):
    return pinn_residual_pointwise(model1, x, t, K=K, training=True, problem=problem)[2]


def rad_resample_dataset(
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
        r, _, _ = pinn_residual_pointwise(
            model1=model1,
            x=x_cand[start:end],
            t=t_cand[start:end],
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
    }

    return x_cand[idx], t_cand[idx], stats
