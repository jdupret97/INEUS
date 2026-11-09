"""Terminal rainbow-min payoff."""

import tensorflow as tf


def phi_tf(S):
    """Vectorised rainbow-min payoff."""
    S = tf.convert_to_tensor(S, dtype=tf.float32)
    return tf.reduce_min(S, axis=1, keepdims=True)
