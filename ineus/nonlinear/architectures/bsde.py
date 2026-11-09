"""Direct Z/U networks and the local transformed deep-BSDE model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..problem import g_fin_tf, resolve_problem
from ..sampling import sample_jump_marks_mc_local_nl


class DirectZNetLocalTransformed(tf.keras.Model):
    """
    Direct parameterization of the Brownian BSDE integrand

        Z_n = Z_theta(t_n, X_n)

    for the transformed martingale w(t,X_t).
    """

    def __init__(
        self,
        dimen,
        T,
        x_scale,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)
        if problem is not None and dimen != self.problem.state_dim:
            raise ValueError("dimen must match problem.state_dim")
        self.dimen = dimen
        self.T = tf.cast(T, tf.float32)
        self.x_scale = tf.cast(x_scale, tf.float32)

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation, dtype=tf.float32)
            for _ in range(n_hidden)
        ]
        self.out = tf.keras.layers.Dense(dimen, activation=None, dtype=tf.float32)

        dummy_t = tf.zeros((1, 1), dtype=tf.float32)
        dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
        _ = self((dummy_t, dummy_x), training=False)

    def call(self, inputs, training=False):
        t, x = inputs
        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)
        t = tf.reshape(t, [-1, 1])

        t_scaled = 2.0 * t / tf.maximum(self.T, 1e-8) - 1.0
        x_scaled = x / tf.maximum(self.x_scale, 1e-8)
        z = tf.concat([t_scaled, x_scaled], axis=1)

        for layer in self.hidden:
            z = layer(z, training=training)

        return self.out(z, training=training)


class DirectJumpUNetLocalTransformed(tf.keras.Model):
    """
    Direct parameterization of the jump BSDE integrand

        U_n(e) = U_theta(t_n, X_n, e)

    for the transformed martingale w(t,X_t).
    """

    def __init__(
        self,
        dimen,
        T,
        x_scale,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)
        if problem is not None and dimen != self.problem.state_dim:
            raise ValueError("dimen must match problem.state_dim")
        self.dimen = dimen
        self.T = tf.cast(T, tf.float32)
        self.x_scale = tf.cast(x_scale, tf.float32)

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation, dtype=tf.float32)
            for _ in range(n_hidden)
        ]
        self.out = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32)

        dummy_t = tf.zeros((1, 1), dtype=tf.float32)
        dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
        dummy_e = tf.zeros((1, dimen), dtype=tf.float32)
        _ = self((dummy_t, dummy_x, dummy_e), training=False)

    def call(self, inputs, training=False):
        t, x, e = inputs
        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)
        e = tf.cast(e, tf.float32)
        t = tf.reshape(t, [-1, 1])

        t_scaled = 2.0 * t / tf.maximum(self.T, 1e-8) - 1.0
        x_scaled = x / tf.maximum(self.x_scale, 1e-8)
        e_scaled = e / tf.maximum(self.x_scale, 1e-8)

        z = tf.concat([t_scaled, x_scaled, e_scaled], axis=1)

        for layer in self.hidden:
            z = layer(z, training=training)

        return self.out(z, training=training)


class ClassicalLocalDeepBSDETransformedJump(tf.keras.Model):
    """
    One genuinely local Deep BSDE solve for the transformed quantity w.

    For one fixed starting point (t0,x0), it learns the scalar

        W0 ~= w(t0,x0),

    together with direct parameterizations of Z and U.

    Transformed forward process:
        dX_t = sqrt(2) dW_t + dJ_t.

    Transformed BSDE:
        dW_t = Z_t dB_t + int U_t(e) \tilde N(dt,de),

    with terminal condition
        W_T = exp(-eta g(X_T)).
    """

    def __init__(
        self,
        t0,
        x0,
        dimen,
        T,
        x_scale,
        N_time=50,
        hidden_dim_z=128,
        n_hidden_z=3,
        hidden_dim_u=128,
        n_hidden_u=3,
        K_jump_expect=64,
        w0_init=None,
        max_jumps_per_step=8,
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)
        if problem is not None and dimen != self.problem.state_dim:
            raise ValueError("dimen must match problem.state_dim")
        self.dimen = dimen
        self.N_time = N_time
        self.K_jump_expect = K_jump_expect
        self.max_jumps_per_step = int(max_jumps_per_step)

        self.T = tf.cast(T, tf.float32)
        self.t0 = tf.constant(float(t0), dtype=tf.float32)
        self.x_scale = tf.cast(x_scale, tf.float32)

        self.x0 = tf.constant(
            np.asarray(x0, dtype=np.float32).reshape(1, dimen),
            dtype=tf.float32,
        )

        if w0_init is None:
            # Terminal transformed payoff at x0 is a stable initialization.
            g0 = float(np.sum(np.asarray(x0, dtype=np.float32) ** 2))
            w0_init = float(np.exp(-float(self.problem.eta) * g0))

        w0_init = float(np.clip(w0_init, 1e-6, 1e6))
        raw_init = float(np.log(np.expm1(w0_init))) if w0_init < 20.0 else w0_init

        self.raw_w0 = self.add_weight(
            name="raw_w0",
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(raw_init),
            trainable=True,
            dtype=tf.float32,
        )

        self.z_net = DirectZNetLocalTransformed(
            dimen=dimen,
            T=T,
            x_scale=x_scale,
            hidden_dim=hidden_dim_z,
            n_hidden=n_hidden_z,
            activation="tanh",
            problem=problem,
        )

        self.u_jump_net = DirectJumpUNetLocalTransformed(
            dimen=dimen,
            T=T,
            x_scale=x_scale,
            hidden_dim=hidden_dim_u,
            n_hidden=n_hidden_u,
            activation="tanh",
            problem=problem,
        )

    @property
    def w0(self):
        return tf.nn.softplus(self.raw_w0) + 1e-12

    def payoff_w(self, x):
        return tf.exp(
            -tf.cast(self.problem.eta, tf.float32)
            * g_fin_tf(x, problem=self.problem)
        )

    def expected_jump_integrand(self, t, x, training=True):
        """
        Monte Carlo approximation of E_E[U_theta(t,x,E)].
        """
        B = tf.shape(x)[0]
        if not self.problem.has_jumps:
            return tf.zeros([B, 1], dtype=tf.float32)
        K = self.K_jump_expect

        e_mc = sample_jump_marks_mc_local_nl(B, K, problem=self.problem)

        x_rep = tf.repeat(x[:, None, :], repeats=K, axis=1)
        t_rep = tf.repeat(t[:, None, :], repeats=K, axis=1)

        x_flat = tf.reshape(x_rep, [-1, self.dimen])
        t_flat = tf.reshape(t_rep, [-1, 1])
        e_flat = tf.reshape(e_mc, [-1, self.dimen])

        u_flat = self.u_jump_net(
            (t_flat, x_flat, e_flat),
            training=training,
        )
        u_vals = tf.reshape(u_flat, [B, K, 1])

        return tf.reduce_mean(u_vals, axis=1)

    def bsde_loss(self, batch_size, training=True):
        """
        Simulate paths from fixed (t0,x0), propagate the transformed BSDE
        forward, and minimize terminal mismatch.

        We retain the compound-Poisson structure by allowing several jump
        marks in one time step (clipped only at a very conservative static
        upper bound max_jumps_per_step).
        """
        B = int(batch_size)

        horizon = self.T - self.t0
        dt = horizon / tf.cast(self.N_time, tf.float32)
        dt = tf.maximum(dt, 0.0)
        sqrt_dt = tf.sqrt(tf.maximum(dt, 1e-12))

        x = tf.repeat(self.x0, repeats=B, axis=0)
        y = tf.ones((B, 1), dtype=tf.float32) * self.w0

        sqrt2_tf = tf.cast(np.sqrt(2.0), tf.float32)
        lam_tf = tf.cast(self.problem.jump_intensity, tf.float32)
        mu_j_tf = tf.convert_to_tensor(self.problem.mu_j, dtype=tf.float32)
        chol_tf = tf.cast(self.problem.cov_j_chol, tf.float32)
        K_MAX = self.max_jumps_per_step

        for n in range(self.N_time):
            t_n_scalar = self.t0 + tf.cast(n, tf.float32) * dt
            t_n = tf.ones((B, 1), dtype=tf.float32) * t_n_scalar

            z_n = self.z_net((t_n, x), training=training)

            # Brownian increment for the driving d-dimensional Brownian motion.
            dW = sqrt_dt * tf.random.normal((B, self.dimen), dtype=tf.float32)
            dX_diff = sqrt2_tf * dW

            if not self.problem.has_jumps:
                y = y + tf.reduce_sum(z_n * dW, axis=1, keepdims=True)
                x = x + dX_diff
                continue

            # Compound-Poisson count over this time step.
            k_jump = tf.random.poisson(
                shape=tf.stack([tf.constant(B, dtype=tf.int32)]),
                lam=lam_tf * dt,
                dtype=tf.float32,
            )
            k_jump = tf.cast(tf.round(k_jump), tf.int32)
            k_jump = tf.minimum(k_jump, K_MAX)

            # Draw a static maximum number of candidate marks, then mask.
            z_marks = tf.random.normal((B, K_MAX, self.dimen), dtype=tf.float32)
            marks = (
                tf.einsum("bkd,ed->bke", z_marks, chol_tf)
                + mu_j_tf[None, None, :]
            )

            k_range = tf.range(K_MAX, dtype=tf.int32)[None, :]
            mask = tf.cast(k_range < k_jump[:, None], tf.float32)[:, :, None]

            jump_sum = tf.reduce_sum(marks * mask, axis=1)

            # Sum the direct U_theta(t,x,E_j) contributions for realized jumps.
            x_rep = tf.repeat(x[:, None, :], repeats=K_MAX, axis=1)
            t_rep = tf.repeat(t_n[:, None, :], repeats=K_MAX, axis=1)

            u_actual_flat = self.u_jump_net(
                (
                    tf.reshape(t_rep, [-1, 1]),
                    tf.reshape(x_rep, [-1, self.dimen]),
                    tf.reshape(marks, [-1, self.dimen]),
                ),
                training=training,
            )
            u_actual = tf.reshape(u_actual_flat, [B, K_MAX, 1])
            jump_bsde_term = tf.reduce_sum(u_actual * mask, axis=1)

            eu = self.expected_jump_integrand(
                t=t_n,
                x=x,
                training=training,
            )

            brownian_term = tf.reduce_sum(z_n * dW, axis=1, keepdims=True)

            # Transformed BSDE has zero driver.
            y = (
                y
                + brownian_term
                + jump_bsde_term
                - lam_tf * dt * eu
            )

            x = x + dX_diff + jump_sum

        terminal = self.payoff_w(x)
        terminal_loss = tf.reduce_mean(tf.square(y - terminal))
        return terminal_loss
