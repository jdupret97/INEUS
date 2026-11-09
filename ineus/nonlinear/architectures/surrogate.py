"""Global value surrogate architecture for transformed BSDE labels."""

from __future__ import annotations

import tensorflow as tf

from ..problem import g_fin_tf, resolve_problem


class GlobalUSurrogateFromLocalTransformedBSDE(tf.keras.Model):
    """
    Global supervised surrogate for the untransformed nonlinear value u(t,x),
    trained from the local transformed Deep BSDE labels.

    Terminal condition is enforced exactly:
        u_psi(T,x) = g(x).
    """

    def __init__(
        self,
        dimen,
        T,
        x_scale,
        hidden_dim=128,
        n_hidden=3,
        activation="tanh",
        constrained=True,
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
        self.constrained = constrained

        self.hidden = [
            tf.keras.layers.Dense(hidden_dim, activation=activation, dtype=tf.float32)
            for _ in range(n_hidden)
        ]
        self.out = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32)

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

        z = tf.concat(
            [
                t_scaled,
                x_scaled,
                tf.square(x_scaled),
            ],
            axis=1,
        )

        for layer in self.hidden:
            z = layer(z, training=training)

        raw = self.out(z, training=training)

        if self.constrained:
            return g_fin_tf(x, problem=self.problem) + (self.T - t) * raw

        return raw
