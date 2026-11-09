"""Value-network architectures used by INEUS and the PINN baseline."""

from __future__ import annotations

import tensorflow as tf

from ..problem import g_fin_tf, resolve_problem


tf.keras.backend.set_floatx("float32")
tf.config.experimental.enable_tensor_float_32_execution(False)
strategy = tf.distribute.MirroredStrategy()


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

        # normalize to [-1, 1]
        x_norm = 2.0 * (x - self.x_min) / x_den - 1.0
        t_norm = 2.0 * (t - self.t_min) / t_den - 1.0

        return x_norm, t_norm


class BSPINN_V(tf.keras.Model):
    def __init__(self, num_neurons):
        super(BSPINN_V, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_neurons, activation='tanh', dtype=tf.float32)
        self.bn1 = tf.keras.layers.BatchNormalization(dtype=tf.float32)
        self.dense2 = tf.keras.layers.Dense(num_neurons, activation='tanh', dtype=tf.float32)
        self.bn2 = tf.keras.layers.BatchNormalization(dtype=tf.float32)
        self.dense3 = tf.keras.layers.Dense(num_neurons, activation='tanh', dtype=tf.float32)
        self.bn3 = tf.keras.layers.BatchNormalization(dtype=tf.float32)
        self.dense4 = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32)

    def call(self, inputs):
        x, t = inputs
        xt = tf.concat([x, t], axis=1)
        xt = self.dense1(xt)
        xt = self.bn1(xt)
        xt = self.dense2(xt)
        xt = self.bn2(xt)
        xt = self.dense3(xt)
        xt = self.bn3(xt)
        xt = self.dense4(xt)
        return xt


class DGMCell_V(tf.keras.Model):
    """Normalized DGM with independent gate parameters at every stage."""

    def __init__(self, hidden_dim, n_layers=5, output_dim=1,
                 x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        super(DGMCell_V, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers

        self.normalizer = InputNormalizer(x_min, x_max, t_min, t_max)

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)
        self.sig_out = tf.keras.layers.Activation(tf.nn.softplus)

        self.Sw  = tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
        self.Uz  = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Wsz = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]

        self.Ug  = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Wsg = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]

        self.Ur  = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Wsr = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]

        self.Uh  = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32)
                    for _ in range(self.n - 1)]
        self.Wsh = [tf.keras.layers.Dense(self.hidden_dim, dtype=tf.float32, use_bias=False)
                    for _ in range(self.n - 1)]

        self.Wf  = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)

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

        super(DGMCell_V, self).build(input_shape)

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

        out = self.Wf(out)
        return out


class ConstrainedValue(tf.keras.Model):
    def __init__(self, base_net, T, *, problem=None):
        super().__init__()
        self.base_net = base_net
        self.T = tf.cast(T, tf.float32)
        self.problem = resolve_problem(problem)

    def call(self, inputs, training=False):
        x, t = inputs
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        N = self.base_net([x, t], training=training)
        return g_fin_tf(x, problem=self.problem) + (self.T - t) * N


def make_model(hidden_dim, n_layers, T, dimen, constrained=True, output_dim=1,
               x_min=None, x_max=None, t_min=0.0, t_max=None, *, problem=None):
    if problem is not None:
        problem = resolve_problem(problem)
        if dimen != problem.state_dim:
            raise ValueError("dimen must equal problem.state_dim.")
    if x_min is None or x_max is None:
        raise ValueError("x_min and x_max must be provided for input normalization.")

    if t_max is None:
        t_max = T

    base_net = DGMCell_V(hidden_dim=hidden_dim, n_layers=n_layers,output_dim=output_dim,x_min=x_min,x_max=x_max,t_min=t_min,t_max=t_max)

    # If you want BSPINN instead:
    # base_net = BSPINN_V(num_neurons=hidden_dim,x_min=x_min,x_max=x_max,t_min=t_min,t_max=t_max)#

    model = ConstrainedValue(base_net, T, problem=problem) if constrained else base_net

    dummy_x = tf.zeros((1, dimen), dtype=tf.float32)
    dummy_t = tf.zeros((1, 1), dtype=tf.float32)
    _ = model([dummy_x, dummy_t], training=False)

    return model
