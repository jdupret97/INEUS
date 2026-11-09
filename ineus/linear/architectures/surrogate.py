"""Constrained global value surrogate for local BSDE labels."""

import tensorflow as tf

from ..problem import phi_tf, resolve_problem


class GlobalValueSurrogateFromLocalBSDE(tf.keras.Model):
    """
    Global supervised surrogate trained on local BSDE labels:

        (t0_i, x0_i) -> y0_i ≈ u(t0_i, x0_i).

    The constrained form enforces

        u(T,x) = phi(x)

    exactly:

        u_psi(t,x) = phi(x) + (T - t) r_psi(t,x).
    """

    def __init__(
        self,
        dimen,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        constrained=True,
        *,
        problem=None,
    ):
        super().__init__()
        self.problem = resolve_problem(problem)

        self.dimen = dimen
        self.constrained = constrained

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation)
            for _ in range(n_hidden)
        ]

        self.out = tf.keras.layers.Dense(1, activation=None)

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
                tf.square(x_scaled),
            ],
            axis=1,
        )

        return feats

    def call(self, inputs, training=False):
        t, x = inputs

        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)

        z = self.make_features(t, x)

        for layer in self.hidden:
            z = layer(z, training=training)

        raw = self.out(z, training=training)

        if self.constrained:
            t = tf.reshape(t, [-1, 1])
            return phi_tf(x) + (self.problem.T_tf - t) * raw

        return raw
