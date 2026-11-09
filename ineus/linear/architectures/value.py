"""Shared DGM value architecture for INEUS and PINN."""

import tensorflow as tf

from ..problem import phi_tf


class DGMCell_V(tf.keras.Model):
    """DGM stages with independent gate weights and an unactivated candidate H."""

    def __init__(self, hidden_dim, n_layers=5, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)

        self.Sw = tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
        # Each gated stage owns its input/state projections, including biases.
        self.Uz = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsz = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Ug = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsg = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Ur = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsr = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Uh = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsh = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Wf = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)

    def build(self, input_shape):
        x_shape, t_shape = input_shape
        xt_shape = (x_shape[0], x_shape[1] + t_shape[1])

        self.Sw.build(xt_shape)
        for stage in range(self.n - 1):
            self.Uz[stage].build(xt_shape)
            self.Wsz[stage].build((None, self.hidden_dim))
            self.Ug[stage].build(xt_shape)
            self.Wsg[stage].build((None, self.hidden_dim))
            self.Ur[stage].build(xt_shape)
            self.Wsr[stage].build((None, self.hidden_dim))
            self.Uh[stage].build(xt_shape)
            self.Wsh[stage].build((None, self.hidden_dim))
        self.Wf.build((None, self.hidden_dim))

        super().build(input_shape)

    def call(self, inputs):
        x, t = inputs
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)

        xt = tf.concat([x, t], axis=1)
        S1 = self.sig_act(self.Sw(xt))
        out = S1

        for stage in range(self.n - 1):
            S = out
            Z = self.sig_act(self.Uz[stage](xt) + self.Wsz[stage](S))
            G = self.sig_act(self.Ug[stage](xt) + self.Wsg[stage](S))
            R = self.sig_act(self.Ur[stage](xt) + self.Wsr[stage](S))
            H = self.Uh[stage](xt) + self.Wsh[stage](S * R)
            out = (1.0 - G) * H + Z * S

        out = self.Wf(out)
        return out


class ConstrainedValue(tf.keras.Model):
    def __init__(self, base_net, T):
        super().__init__()
        self.base_net = base_net
        self.T = tf.cast(T, tf.float32)

    def call(self, inputs):
        x, t = inputs
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        N = self.base_net([x, t])
        return phi_tf(x) + (self.T - t) * N


def make_model(hidden_dim, n_layers, T, dimen, constrained=True, output_dim=1):
    base_net = DGMCell_V(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=output_dim
    )

    model = ConstrainedValue(base_net, T) if constrained else base_net

    dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
    dummy_t = tf.zeros((1, 1), dtype=tf.float32)
    _ = model([dummy_x, dummy_t])

    return model
