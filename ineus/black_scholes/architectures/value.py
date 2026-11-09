"""Neural-network architectures for the INEUS value function.

Provides
--------
- DGMCell   : Deep Galerkin Method cell with GRU-like gating.
- BSPINN    : Plain fully-connected residual network (alternative).
- ConstrainedValue : hard terminal-condition wrapper  u(T,x) = φ(x).
- build_model : factory that assembles and initialises a model.
"""

import tensorflow as tf

from ..payoff import phi_tf


# ------------------------------------------------------------------
# Terminal payoff  φ(x) = min(x₁, …, x_d)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Input normalisation  [x_min, x_max] × [t_min, t_max] → [−1, 1]²
# ------------------------------------------------------------------

class InputNormalizer(tf.keras.layers.Layer):

    def __init__(self, x_min, x_max, t_min, t_max, eps=1e-8):
        super().__init__()
        self.x_min = tf.constant(x_min, dtype=tf.float32)
        self.x_max = tf.constant(x_max, dtype=tf.float32)
        self.t_min = tf.constant(t_min, dtype=tf.float32)
        self.t_max = tf.constant(t_max, dtype=tf.float32)
        self.eps   = tf.constant(eps,   dtype=tf.float32)

    def call(self, x, t):
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        x_norm = 2.0 * (x - self.x_min) / tf.maximum(self.x_max - self.x_min, self.eps) - 1.0
        t_norm = 2.0 * (t - self.t_min) / tf.maximum(self.t_max - self.t_min, self.eps) - 1.0
        return x_norm, t_norm


# ------------------------------------------------------------------
# DGM cell  (Sirignano & Spiliopoulos, 2018)
# ------------------------------------------------------------------

class DGMCell(tf.keras.Model):
    """Deep Galerkin Method cell with GRU-like gating.

    At each layer ℓ the raw input [x, t] is re-injected:
        Z = σ(U_z [x,t] + W_z S)
        G = σ(U_g [x,t] + W_g S)
        R = σ(U_r [x,t] + W_r S)
        H = σ(U_h [x,t] + W_h (S⊙R))
        S ← (1−G)⊙H + Z⊙S
    """

    def __init__(self, hidden_dim, n_layers=5, output_dim=1,
                 x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n_layers
        self.normalizer = InputNormalizer(x_min, x_max, t_min, t_max)
        self.act = tf.keras.layers.Activation(tf.nn.tanh)

        # First layer
        self.Sw  = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32)
        # Gating layers
        self.Uz  = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32)
        self.Wsz = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32, use_bias=False)
        self.Ug  = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32)
        self.Wsg = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32, use_bias=False)
        self.Ur  = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32)
        self.Wsr = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32, use_bias=False)
        self.Uh  = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32)
        self.Wsh = tf.keras.layers.Dense(hidden_dim, dtype=tf.float32, use_bias=False)
        # Output
        self.Wf  = tf.keras.layers.Dense(output_dim, dtype=tf.float32)

    def build(self, input_shape):
        x_shape, t_shape = input_shape
        xt_shape = (x_shape[0], x_shape[1] + t_shape[1])
        h = self.hidden_dim
        self.Sw.build(xt_shape);  self.Uz.build(xt_shape)
        self.Wsz.build((None, h)); self.Ug.build(xt_shape)
        self.Wsg.build((None, h)); self.Ur.build(xt_shape)
        self.Wsr.build((None, h)); self.Uh.build(xt_shape)
        self.Wsh.build((None, h)); self.Wf.build((None, h))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, t = inputs
        x, t = self.normalizer(x, t)
        xt = tf.concat([x, t], axis=1)

        S = self.act(self.Sw(xt))
        for _ in range(1, self.n):
            Z = self.act(self.Uz(xt) + self.Wsz(S))
            G = self.act(self.Ug(xt) + self.Wsg(S))
            R = self.act(self.Ur(xt) + self.Wsr(S))
            H = self.act(self.Uh(xt) + self.Wsh(S * R))
            S = (1.0 - G) * H + Z * S

        return self.Wf(S)


# ------------------------------------------------------------------
# Plain feed-forward alternative
# ------------------------------------------------------------------

class BSPINN(tf.keras.Model):
    """Simple fully-connected network (tanh activations)."""

    def __init__(self, num_neurons, n_layers):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
        self.hidden_layers = [
            tf.keras.layers.Dense(num_neurons, activation='tanh')
            for _ in range(n_layers)
        ]
        self.out_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x, t = inputs
        h = tf.concat([x, t], axis=1)
        for layer in self.hidden_layers:
            h = layer(h)
        return self.out_layer(h)


# ------------------------------------------------------------------
# Hard terminal-condition wrapper
# ------------------------------------------------------------------

class ConstrainedValue(tf.keras.Model):
    """Enforces u(T, x) = φ(x) exactly via the ansatz

        u(t, x) = φ(x) + (T − t) · N(t, x)

    where N is an unconstrained base network.
    """

    def __init__(self, base_net, T):
        super().__init__()
        self.base_net = base_net
        self.T = tf.cast(T, tf.float32)

    def call(self, inputs, training=False):
        x, t = inputs
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        N = self.base_net([x, t], training=training)
        return phi_tf(x) + (self.T - t) * N


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_model(model_cfg, pde_cfg, train_cfg):
    """Construct, wrap (optionally), and initialise the value network.

    Parameters
    ----------
    model_cfg  : ModelConfig
    pde_cfg    : PDEConfig
    train_cfg  : TrainConfig  (supplies spatial bounds for normalisation)

    Returns
    -------
    model : tf.keras.Model   (built and ready for training)
    """
    base_net = DGMCell(
        hidden_dim=model_cfg.hidden_dim,
        n_layers=model_cfg.n_layers,
        output_dim=1,
        x_min=0.0, x_max=train_cfg.sample_b,
        t_min=0.0, t_max=pde_cfg.T,
    )

    model = ConstrainedValue(base_net, pde_cfg.T) if model_cfg.constrained else base_net

    # Trigger build with a dummy forward pass
    dummy_x = tf.zeros((1, pde_cfg.dimen), dtype=tf.float32)
    dummy_t = tf.zeros((1, 1), dtype=tf.float32)
    _ = model([dummy_x, dummy_t], training=False)

    return model
