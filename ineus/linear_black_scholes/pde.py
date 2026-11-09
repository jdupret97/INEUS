"""Auxiliary-direction derivatives, INEUS losses, and RAD scores."""

import numpy as np
import tensorflow as tf

from .problem import phi_tf
from .sampling import sample_xt


def _ensure_2d(x):
    if len(x.shape) == 1:
        return tf.expand_dims(x, axis=0)
    return x


def diffusion_columns_B(S_tf, sigma_diag_val, cholSigma):
    """
    Build B(S) columns for correlated BS diffusion:
        B(S) = diag(S) * diag(sigma_diag) * cholSigma

    Returns tensor of shape [B, d, d], where [:, :, i] is the i-th column.
    """
    S_tf = _ensure_2d(S_tf)
    sigma_diag_tf = tf.reshape(sigma_diag_val, (1, -1))
    base = S_tf * sigma_diag_tf
    B = base[:, :, None] * cholSigma[None, :, :]
    return B


def fprime_bs(h, model, S_tf, t_tf, r_val, lam_vec_val, kappa_vec_val, sigma_diag_val, Sigma_val, training=False, *, problem):
    """
    Proposition-based psi(h) for the Black-Scholes/Merton operator.
    """
    S_tf = tf.convert_to_tensor(S_tf, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t_tf, dtype=tf.float32)
    S_tf = _ensure_2d(S_tf)
    t_tf = _ensure_2d(t_tf)

    d = problem.dimen
    d_f = tf.cast(d, tf.float32)

    cholSigma = tf.linalg.cholesky(tf.convert_to_tensor(Sigma_val, dtype=tf.float32))

    lam_vec_tf = tf.convert_to_tensor(lam_vec_val, dtype=tf.float32)
    kappa_vec_tf = tf.convert_to_tensor(kappa_vec_val, dtype=tf.float32)
    drift_rate = r_val - (kappa_vec_tf * lam_vec_tf)
    beta_val = S_tf * tf.reshape(drift_rate, (1, -1))

    t_tf2 = t_tf + (h**2) / (2.0 * d_f)
    common = S_tf + (h**2) * beta_val / (2.0 * d_f)

    sigma_diag_tf = tf.convert_to_tensor(sigma_diag_val, dtype=tf.float32)
    B = diffusion_columns_B(S_tf, sigma_diag_tf, cholSigma)

    B_cols = tf.transpose(B, perm=[2, 0, 1])
    S_shift = common[None, :, :] + (h / tf.sqrt(2.0)) * B_cols

    t_rep = tf.repeat(t_tf2[None, :, :], repeats=d, axis=0)

    out = model(
        [
            tf.reshape(S_shift, (-1, tf.shape(S_tf)[1])),
            tf.reshape(t_rep, (-1, 1)),
        ],
        training=training,
    )

    out = tf.reshape(out, (d, tf.shape(S_tf)[0], -1))
    psi = tf.reduce_sum(out, axis=0)
    return psi


def second_derivative_wrt_h(f_of_h, h):
    with tf.GradientTape() as tape2:
        tape2.watch(h)
        with tf.GradientTape() as tape1:
            tape1.watch(h)
            fval = f_of_h(h)
        df = tape1.gradient(fval, h)
    d2f = tape2.gradient(df, h)
    return d2f


def pinn_loss1bis(model1, model1targ, x, t, e, scale, T_val, h, *, problem):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    e_tf = tf.convert_to_tensor(e, dtype=tf.float32)

    def psi_of_h(h_):
        return fprime_bs(
            h_,
            model1targ,
            x_tf,
            t_tf,
            problem.r,
            problem.lam_vec,
            problem.kappa_vec,
            problem.sigma_diag,
            problem.Sigma,
            training=False,
            problem=problem,
        )

    d2psi = second_derivative_wrt_h(psi_of_h, h)
    d2psi = tf.reshape(d2psi, (tf.shape(x_tf)[0],))

    B = tf.shape(x_tf)[0]
    V_mod = tf.reshape(model1([x_tf, t_tf], training=True), [B])
    V_targ = tf.reshape(model1targ([x_tf, t_tf], training=False), [B])
    jump = tf.zeros([B], tf.float32)

    for i in range(problem.dimen):
        updates = x_tf[:, i] * tf.exp(e_tf[:, i])
        idx = tf.stack([tf.range(B), tf.fill([B], i)], axis=1)
        x_mod = tf.tensor_scatter_nd_update(x_tf, idx, updates)

        jump += tf.cast(problem.lam_vec[i], tf.float32) * (
            tf.reshape(model1targ([x_mod, t_tf], training=False), [B]) - V_targ
        )

    residual = V_mod - (V_targ + scale * (d2psi + jump - problem.r * V_targ))
    loss = tf.reduce_mean(tf.square(residual))
    return loss


def terminal_loss(model, x_T, T_val, training=False, *, problem):
    x_T_tf = tf.convert_to_tensor(x_T, dtype=tf.float32)
    B = tf.shape(x_T_tf)[0]
    t_T_tf = tf.ones((B, 1), dtype=tf.float32) * tf.cast(T_val, tf.float32)

    V_T = model([x_T_tf, t_T_tf], training=training)
    phi_T = phi_tf(x_T_tf, problem.w, problem.K)
    return tf.reduce_mean(tf.square(V_T - phi_T))


def pinn_residual_pointwise(model1, x, t, T_val, K_mc=20, *, problem):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)

    batch_n = tf.shape(x_tf)[0]
    h_local2 = tf.Variable(tf.zeros((batch_n, 1), dtype=tf.float32))
    B = tf.shape(x_tf)[0]

    def psi_of_h(h_):
        return fprime_bs(
            h_,
            model1,
            x_tf,
            t_tf,
            problem.r,
            problem.lam_vec,
            problem.kappa_vec,
            problem.sigma_diag,
            problem.Sigma,
            training=False,
            problem=problem,
        )

    d2psi = second_derivative_wrt_h(psi_of_h, h_local2)
    d2psi = tf.reshape(d2psi, [B])

    V = tf.reshape(model1([x_tf, t_tf], training=False), [B])

    z = tf.random.normal([B, K_mc, problem.dimen], dtype=tf.float32)
    mu_tf = tf.constant(problem.mu_j, dtype=tf.float32)[None, None, :]
    sig_tf = tf.constant(problem.sig_j, dtype=tf.float32)[None, None, :]
    e_samples = mu_tf + sig_tf * z

    K_eff = tf.shape(e_samples)[1]

    x_rep = tf.repeat(x_tf[:, None, :], repeats=K_eff, axis=1)
    t_rep = tf.repeat(t_tf[:, None, :], repeats=K_eff, axis=1)

    jump_mc = tf.zeros([B], dtype=tf.float32)

    for i in range(problem.dimen):
        new_xi = x_tf[:, None, i] * tf.exp(e_samples[:, :, i])
        mask = tf.reshape(tf.one_hot(i, depth=problem.dimen, dtype=tf.float32), [1, 1, problem.dimen])
        x_mod_i = x_rep * (1.0 - mask) + mask * new_xi[:, :, None]

        x_flat = tf.reshape(x_mod_i, [-1, problem.dimen])
        t_flat = tf.reshape(t_rep, [-1, 1])

        V_jump_flat = model1([x_flat, t_flat], training=False)
        V_jump = tf.reshape(V_jump_flat, [B, K_eff])
        EV_jump = tf.reduce_mean(V_jump, axis=1)

        jump_mc += tf.cast(problem.lam_vec[i], tf.float32) * (EV_jump - V)

    pde_residual = d2psi + jump_mc - problem.r * V
    rhs_mean = tf.reduce_mean(tf.abs(pde_residual))
    return pde_residual, rhs_mean


def rad_resample_dataset(
    model1,
    T_val,
    num_points,
    candidate_size,
    K_jump=25,
    k=1.0,
    c_rad=1.0,
    eval_batch_size=2048,
    replace=False,
    eps_floor=1e-12,
    sample_mode="uniform",
    S0=None,
    *,
    problem,
):
    x_cand, t_cand = sample_xt(candidate_size, T_val, mode=sample_mode, S0=S0, problem=problem)
    residual_abs = np.empty(candidate_size, dtype=np.float32)

    for start in range(0, candidate_size, eval_batch_size):
        end = min(start + eval_batch_size, candidate_size)
        r_batch, _ = pinn_residual_pointwise(
            model1=model1,
            x=x_cand[start:end],
            t=t_cand[start:end],
            T_val=T_val,
            K_mc=K_jump,
            problem=problem,
        )
        residual_abs[start:end] = np.abs(r_batch.numpy())

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
