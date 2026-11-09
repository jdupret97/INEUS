"""Normalized DGM value networks with a hard basket-call terminal condition."""

import tensorflow as tf

from ..problem import phi_tf


class InputNormalizer(tf.keras.layers.Layer):
    def __init__(self, x_min, x_max, t_min, t_max, eps=1e-8):
        super().__init__()
        self.x_min = tf.constant(x_min, dtype=tf.float32)
        self.x_max = tf.constant(x_max, dtype=tf.float32)
        self.t_min = tf.constant(t_min, dtype=tf.float32)
        self.t_max = tf.constant(t_max, dtype=tf.float32)
        self.eps = tf.constant(eps, dtype=tf.float32)

    def call(self, x, t):
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)

        x_den = tf.maximum(self.x_max - self.x_min, self.eps)
        t_den = tf.maximum(self.t_max - self.t_min, self.eps)

        x_norm = 2.0 * (x - self.x_min) / x_den - 1.0
        t_norm = 2.0 * (t - self.t_min) / t_den - 1.0
        return x_norm, t_norm


class BSPINN_V(tf.keras.Model):
    def __init__(self, num_neurons, n_layers, x_min, x_max, t_min, t_max):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.normalizer = InputNormalizer(x_min, x_max, t_min, t_max)
        self.hidden_layers = [tf.keras.layers.Dense(num_neurons, activation="tanh") for _ in range(n_layers)]
        self.out_layer = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=False):
        x, t = inputs
        x, t = self.normalizer(x, t)
        xt = tf.concat([x, t], axis=1)
        for dense in self.hidden_layers:
            xt = dense(xt)
        return self.out_layer(xt)


class DGMCell_V(tf.keras.Model):
    """DGM stages with independent gate parameters and tanh candidate H."""

    def __init__(self, hidden_dim, n_layers=5, output_dim=1, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers

        self.normalizer = InputNormalizer(x_min, x_max, t_min, t_max)
        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)

        self.Sw = tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
        # Each gated stage has its own input and state projections.
        self.Uz = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsz = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]
        self.Ug = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsg = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]
        self.Ur = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsr = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]
        self.Uh = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                   for _ in range(self.n - 1)]
        self.Wsh = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
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

    def call(self, inputs, training=False):
        x, t = inputs
        x, t = self.normalizer(x, t)
        xt = tf.concat([x, t], axis=1)

        S1 = self.sig_act(self.Sw(xt))
        out = S1

        for stage in range(self.n - 1):
            S = out
            Z = self.sig_act(self.Uz[stage](xt) + self.Wsz[stage](S))
            G = self.sig_act(self.Ug[stage](xt) + self.Wsg[stage](S))
            R = self.sig_act(self.Ur[stage](xt) + self.Wsr[stage](S))
            H = self.sig_act(self.Uh[stage](xt) + self.Wsh[stage](S * R))
            out = (1.0 - G) * H + Z * S

        return self.Wf(out)


class ConstrainedValue(tf.keras.Model):
    def __init__(self, base_net, T_val, *, problem):
        super().__init__()
        self.base_net = base_net
        self.T = tf.cast(T_val, tf.float32)
        self.payoff_weights = problem.w
        self.strike = problem.K

    def call(self, inputs, training=False):
        x, t = inputs
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        N = self.base_net([x, t], training=training)
        return phi_tf(x, self.payoff_weights, self.strike) + (self.T - t) * N


def make_model(hidden_dim_val, n_layers_val, T_val, dimen_val, constrained=True, output_dim=1, x_min=0.0, x_max=1.0, t_min=0.0, t_max=None, *, problem):
    if t_max is None:
        t_max = T_val

    base_net = DGMCell_V(
        hidden_dim=hidden_dim_val,
        n_layers=n_layers_val,
        output_dim=output_dim,
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
    )

    # To use the plain BSPINN instead, replace base_net above with:
    # base_net = BSPINN_V(hidden_dim_val, n_layers_val, x_min, x_max, t_min, t_max)

    model = ConstrainedValue(base_net, T_val, problem=problem) if constrained else base_net

    dummy_x = tf.zeros((1, dimen_val), dtype=tf.float32)
    dummy_t = tf.zeros((1, 1), dtype=tf.float32)
    _ = model([dummy_x, dummy_t], training=False)
    return model
