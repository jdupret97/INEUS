"""Direct Brownian and jump networks and local BSDE model."""

import numpy as np
import tensorflow as tf

from ..problem import phi_tf, resolve_problem
from ..sampling import sample_jump_marks, sample_jump_marks_mc


class DirectZNet(tf.keras.Model):
    """
    Direct parameterization of the Brownian BSDE integrand:

        Z_n = Z_theta(t_n, X_n).

    Output shape: [B, dimend].
    """

    def __init__(
        self,
        dimen,
        dimend,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)

        self.dimen = dimen
        self.dimend = dimend

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation)
            for _ in range(n_hidden)
        ]

        self.out = tf.keras.layers.Dense(dimend, activation=None)

        # Build variables.
        dummy_t = tf.zeros((1, 1), dtype=tf.float32)
        dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
        _ = self((dummy_t, dummy_x), training=False)

    def make_features(self, t, x):
        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)

        t = tf.reshape(t, [-1, 1])

        t_scaled = 2.0 * t / self.problem.T_tf - 1.0
        x_scaled = x / self.problem.sample_b_tf

        feats = tf.concat(
            [
                t_scaled,
                x_scaled,
            ],
            axis=1,
        )

        return feats

    def call(self, inputs, training=False):
        t, x = inputs

        z = self.make_features(t, x)

        for layer in self.hidden:
            z = layer(z, training=training)

        return self.out(z, training=training)


class DirectJumpUNet(tf.keras.Model):
    """
    Direct parameterization of the jump BSDE integrand:

        U_n(e) = U_theta(t_n, X_n, e).

    For a jump mark E, U_theta(t,x,E) approximates the jump in Y.

    Output shape: [B, 1].
    """

    def __init__(
        self,
        dimen,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)

        self.dimen = dimen

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation)
            for _ in range(n_hidden)
        ]

        self.out = tf.keras.layers.Dense(1, activation=None)

        # Build variables.
        dummy_t = tf.zeros((1, 1), dtype=tf.float32)
        dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
        dummy_e = tf.zeros((1, dimen), dtype=tf.float32)
        _ = self((dummy_t, dummy_x, dummy_e), training=False)

    def make_features(self, t, x, e):
        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)
        e = tf.cast(e, tf.float32)

        t = tf.reshape(t, [-1, 1])

        t_scaled = 2.0 * t / self.problem.T_tf - 1.0
        x_scaled = x / self.problem.sample_b_tf
        e_scaled = e / self.problem.sample_b_tf

        feats = tf.concat(
            [
                t_scaled,
                x_scaled,
                e_scaled,
            ],
            axis=1,
        )

        return feats

    def call(self, inputs, training=False):
        t, x, e = inputs

        z = self.make_features(t, x, e)

        for layer in self.hidden:
            z = layer(z, training=training)

        return self.out(z, training=training)


class ClassicalLocalDeepBSDEJump(tf.keras.Model):
    """
    Classical local Deep BSDE solver with jumps.

    For one fixed initial condition (t0, x0), this model learns

        Y0 ≈ u(t0, x0),

    together with direct parameterizations

        Z_n = Z_theta(t_n, X_n),
        U_n(e) = U_theta(t_n, X_n, e).

    This model is local. One trained model gives one scalar value
    u(t0, x0). To obtain a global approximation, many independent
    local models must be trained and then distilled into a global surrogate.
    """

    def __init__(
        self,
        t0,
        x0,
        dimen,
        dimend,
        N_time=20,
        hidden_dim_z=128,
        n_hidden_z=3,
        hidden_dim_u=128,
        n_hidden_u=3,
        K_jump_expect=64,
        y0_init=None,
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)
        if dimen != self.problem.dimen or dimend != self.problem.dimend:
            raise ValueError("BSDE dimensions must match the supplied problem")

        self.dimen = dimen
        self.dimend = dimend
        self.N_time = N_time
        self.K_jump_expect = K_jump_expect

        self.t0 = tf.constant(float(t0), dtype=tf.float32)

        self.x0 = tf.constant(
            np.asarray(x0, dtype=np.float32).reshape(1, dimen),
            dtype=tf.float32,
        )

        if y0_init is None:
            # This is only an initialization, not a label.
            y0_init = float(np.sum(np.asarray(x0, dtype=np.float32) ** 2))

        self.y0 = self.add_weight(
            name="y0",
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(y0_init),
            trainable=True,
            dtype=tf.float32,
        )

        self.z_net = DirectZNet(
            dimen=dimen,
            dimend=dimend,
            hidden_dim=hidden_dim_z,
            n_hidden=n_hidden_z,
            activation="tanh",
            problem=self.problem,
        )

        self.u_jump_net = DirectJumpUNet(
            dimen=dimen,
            hidden_dim=hidden_dim_u,
            n_hidden=n_hidden_u,
            activation="tanh",
            problem=self.problem,
        )

    def payoff(self, x):
        return phi_tf(x)

    def expected_jump_integrand(self, t, x, training=True):
        """
        Monte Carlo approximation of

            E_E[U_theta(t, x, E)].

        Shape: [B, 1].
        """
        B = tf.shape(x)[0]
        K = self.K_jump_expect

        e_mc = sample_jump_marks_mc(B, K, problem=self.problem)  # [B,K,dimen]

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
        Simulate paths from fixed (t0, x0), propagate Y forward,
        and penalize terminal mismatch:

            E[|Y_T - phi(X_T)|^2].

        The time grid is from t0 to T, so the local model solves from
        the sampled starting time t0 rather than always from 0.
        """
        B = int(batch_size)

        horizon = self.problem.T_tf - self.t0
        dt = horizon / tf.cast(self.N_time, tf.float32)
        sqrt_dt = tf.sqrt(tf.maximum(dt, 1e-12))

        # Repeat the fixed initial point across Monte Carlo paths.
        x = tf.repeat(self.x0, repeats=B, axis=0)  # [B,dimen]

        # Y0 is a trainable scalar associated with this one local solve.
        y = tf.ones((B, 1), dtype=tf.float32) * self.y0

        for n in range(self.N_time):
            t_n_scalar = self.t0 + tf.cast(n, tf.float32) * dt
            t_n = tf.ones((B, 1), dtype=tf.float32) * t_n_scalar

            # Direct Z-parameterization.
            z_n = self.z_net((t_n, x), training=training)  # [B,dimend]

            # Brownian increment.
            dW = sqrt_dt * tf.random.normal((B, self.dimend), dtype=tf.float32)
            dX_diff = tf.linalg.matmul(dW, self.problem.sigma_tf, transpose_b=True)

            # Bernoulli approximation of Poisson jumps over one time step.
            if self.problem.lam == 0:
                jump_occurs = tf.zeros((B, 1), dtype=tf.float32)
                e_actual = tf.zeros_like(x)
                u_actual = tf.zeros_like(y)
                eu = tf.zeros_like(y)
            else:
                p_jump = tf.minimum(self.problem.lam_tf * dt, 1.0)
                jump_occurs = tf.cast(
                    tf.random.uniform((B, 1), dtype=tf.float32) < p_jump,
                    tf.float32,
                )

                # Actual jump mark for realized jump.
                e_actual = sample_jump_marks(B, problem=self.problem)

                # Direct U-parameterization.
                u_actual = self.u_jump_net(
                    (t_n, x, e_actual), training=training,
                )  # [B,1]

                # Compensator term.
                eu = self.expected_jump_integrand(
                    t=t_n, x=x, training=training,
                )  # [B,1]

            brownian_term = tf.reduce_sum(z_n * dW, axis=1, keepdims=True)

            # BSDE forward Euler update:
            #
            # Y_{n+1}
            #   = Y_n + c Y_n dt + Z_n · dW
            #     + 1_jump U_n(E_n)
            #     - lambda dt E[U_n(E)].
            y = (
                y
                + self.problem.c_tf * y * dt
                + brownian_term
                + jump_occurs * u_actual
                - self.problem.lam_tf * dt * eu
            )

            # Forward jump-diffusion update:
            #
            # X_{n+1}
            #   = X_n + beta X_n dt + sigma dW + 1_jump E_n.
            x = (
                x
                + self.problem.beta_tf * x * dt
                + dX_diff
                + jump_occurs * e_actual
            )

        terminal = self.payoff(x)
        terminal_loss = tf.reduce_mean(tf.square(y - terminal))

        return terminal_loss
