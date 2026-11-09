"""PDE operators and physics-informed losses for the default-risk problem.

Key ideas
---------
1. The Black–Scholes operator  A₀u = ∂_t u + μx·∇u + ½ Tr(BB^T Hess u)
   is evaluated via the *proposition-based* ψ(h) trick, which replaces
   the full d×d Hessian by a single scalar second derivative ψ''(0).

2. The INEUS loss regresses  u_θ  onto the Euler target
       u_targ + β · [A₀ u_targ + f(u_targ)]
   treating u_targ as a frozen constant.
"""

import tensorflow as tf
import numpy as np
from .payoff import phi_tf



# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _ensure_2d(x):
    return tf.expand_dims(x, axis=0) if len(x.shape) == 1 else x


def second_derivative_wrt_h(f_of_h, h):
    """d²f/dh² via nested gradient tapes."""
    with tf.GradientTape() as t2:
        t2.watch(h)
        with tf.GradientTape() as t1:
            t1.watch(h)
            val = f_of_h(h)
        dv = t1.gradient(val, h)
    return t2.gradient(dv, h)


# ------------------------------------------------------------------
# Nonlinear reaction  f(u) = −[(1−δ)Q(u) + R] u
# ------------------------------------------------------------------

def Q_tf(u, cfg):
    """Piecewise-linear default intensity.

    Q(u) = γ_h                             if u < v_h
    Q(u) = γ_l                             if u ≥ v_l
    Q(u) = γ_h + slope·(u − v_h)           otherwise
    """
    u = tf.cast(u, tf.float32)
    slope = (cfg.gamma_h - cfg.gamma_l) / (cfg.v_h - cfg.v_l)
    mid = slope * (u - cfg.v_h) + cfg.gamma_h
    return tf.where(u < cfg.v_h, cfg.gamma_h,
                    tf.where(u >= cfg.v_l, cfg.gamma_l, mid))


def default_reaction(u, cfg):
    """f(u) = −[(1−δ)Q(u) + R] u."""
    u = tf.cast(u, tf.float32)
    return -(((1.0 - cfg.delta_recovery) * Q_tf(u, cfg)) + cfg.R_rate) * u


# ------------------------------------------------------------------
# Diffusion matrix  B(S) = diag(S) diag(σ) chol(Σ)
# ------------------------------------------------------------------

def _diffusion_columns(S, sigma_diag, cholSigma):
    """Returns [B, d, d] tensor of B(S)."""
    S = _ensure_2d(S)
    base = S * tf.reshape(sigma_diag, (1, -1))
    return base[:, :, None] * cholSigma[None, :, :]


# ------------------------------------------------------------------
# ψ(h) construction  (Proposition-based PDE operator)
# ------------------------------------------------------------------

def _psi(h, model, x, t, cfg, training=False):
    """Build ψ(h) so that ψ''(0) ≈ A₀ u(t, x).

    ψ(h) = Σ_{i=1}^d  u( t + h²/2d,  x + h²μx/2d + (h/√2) B_{·,i} )
    """
    x = _ensure_2d(tf.convert_to_tensor(x, dtype=tf.float32))
    t = _ensure_2d(tf.convert_to_tensor(t, dtype=tf.float32))
    d = cfg.dimen
    d_f = tf.cast(d, tf.float32)

    cholSigma = tf.linalg.cholesky(
        tf.convert_to_tensor(cfg.Sigma, dtype=tf.float32))
    sigma_tf = tf.convert_to_tensor(cfg.sigma_diag, dtype=tf.float32)

    t_shift = t + (h ** 2) / (2.0 * d_f)
    x_common = x + (h ** 2) * cfg.mu_bar_adjusted * x / (2.0 * d_f)

    B_cols = tf.transpose(                           # [d, B, d]
        _diffusion_columns(x, sigma_tf, cholSigma),  # [B, d, d]
        perm=[2, 0, 1])

    x_shift = x_common[None, :, :] + (h / tf.sqrt(2.0)) * B_cols   # [d, B, d]
    t_rep = tf.repeat(t_shift[None, :, :], repeats=d, axis=0)      # [d, B, 1]

    out = model([
        tf.reshape(x_shift, (-1, tf.shape(x)[1])),
        tf.reshape(t_rep,   (-1, 1)),
    ], training=training)

    out = tf.reshape(out, (d, tf.shape(x)[0], -1))
    return tf.reduce_sum(out, axis=0)                               # [B, 1]


# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------

def pinn_loss(model, model_targ, x, t, scale, cfg, h_var):
    """INEUS regression loss.

    L = E[ (u_θ(Y) − G_β(Y, u_targ))² ]

    where  G_β = u_targ + β·(ψ''_targ(0) + f(u_targ))
    and ψ''(0) ≈ A₀ u_targ.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    B = tf.shape(x)[0]

    d2psi = second_derivative_wrt_h(
        lambda h_: _psi(h_, model_targ, x, t, cfg, training=False), h_var)
    d2psi = tf.reshape(d2psi, [B])

    V_targ   = tf.reshape(model_targ([x, t], training=False), [B])
    reaction = tf.reshape(default_reaction(V_targ, cfg), [B])

    rhs = d2psi + reaction                             # A₀u + f(u)
    
    if getattr(cfg, 'use_jumps', False):
        J = jump_nonlocal(model_targ, x, t, cfg, training=False)
        rhs = rhs + tf.reshape(J, [B])

    target = V_targ + scale * rhs                      # Euler step

    V = tf.reshape(model([x, t], training=True), [B])
    return tf.reduce_mean(tf.square(V - target))


def residual_pointwise(model, x, t, cfg):
    """Evaluate the signed operator diagnostic at each collocation point.

    Returns (residual [B], mean |residual|).
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    B = tf.shape(x)[0]

    h_var = tf.Variable(tf.zeros((B, 1), dtype=tf.float32))

    d2psi = second_derivative_wrt_h(
        lambda h_: _psi(h_, model, x, t, cfg, training=False), h_var)
    d2psi = tf.reshape(d2psi, [B])

    V = tf.reshape(model([x, t], training=False), [B])
    r = d2psi + tf.reshape(default_reaction(V, cfg), [B])
    # ── Jump ──
    if getattr(cfg, 'use_jumps', False):
        J = jump_nonlocal(model, x, t, cfg, training=False)
        r = r + tf.reshape(J, [B])

    return r, tf.reduce_mean(tf.abs(r))


def terminal_loss(model, x_term, T, training=False):
    """Soft terminal penalty: E[|u(T,x) - φ(x)|²]."""
    x_term = tf.convert_to_tensor(x_term, dtype=tf.float32)
    B = tf.shape(x_term)[0]
    t_term = tf.ones((B, 1), dtype=tf.float32) * tf.cast(T, tf.float32)
    u_term = model([x_term, t_term], training=training)
    phi_term = phi_tf(x_term)
    return tf.reduce_mean(tf.square(u_term - phi_term))


def jump_nonlocal(model, x, t, cfg, training=False):
    """One-sample Monte Carlo estimate of the nonlocal jump term.

    J[u](t,x) = λ · [u(t, x ⊙ eᶻ) - u(t, x)]

    Z_i = μ_J + σ_J · (√ρ · ξ_common + √(1-ρ) · ξ_i)

    One independent jump is drawn for each collocation point. Individual
    outputs are stochastic estimates, not exact conditional expectations.
    """
    x = _ensure_2d(tf.convert_to_tensor(x, dtype=tf.float32))
    t = _ensure_2d(tf.convert_to_tensor(t, dtype=tf.float32))

    B = tf.shape(x)[0]
    d = cfg.dimen

    # Correlated Gaussian log-jumps
    rho_j = cfg.jump_corr
    xi_common = tf.random.normal((B, 1), dtype=tf.float32)
    xi_idio   = tf.random.normal((B, d), dtype=tf.float32)

    z = cfg.jump_mean + cfg.jump_std * (
        tf.sqrt(rho_j) * xi_common
        + tf.sqrt(1.0 - rho_j) * xi_idio
    )

    # Multiplicative jump: x ⊙ eᶻ
    x_jumped = x * tf.exp(z)

    u_jumped = model([x_jumped, t], training=training)
    u_base   = model([x, t],       training=training)

    return cfg.jump_intensity * (u_jumped - u_base)
