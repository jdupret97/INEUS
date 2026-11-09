"""Polyak block averaging with knowledge distillation.

Every *m* outer epochs the code:
  1. Unrolls the Polyak recursion  ū = Σ cⱼ vⱼ  over the saved
     weight snapshots  (function-space weighted average).
  2. Trains a single *student* network to match the ensemble teacher.
  3. Replaces the current model with the student.
"""

import numpy as np
import tensorflow as tf

from .utils import copy_weights
from .sampling import sample_xt, sample_xt_paths


# ------------------------------------------------------------------

def block_coeffs(alpha, q):
    """Coefficients of the unrolled Polyak recursion.

    ū_q  =  c₀ u₀  +  Σⱼ₌₁..q  cⱼ vⱼ

    where  c₀ = (1−α)^q,  cⱼ = α(1−α)^{q−j}.
    """
    c = [(1.0 - alpha) ** q]
    c += [alpha * (1.0 - alpha) ** (q - j) for j in range(1, q + 1)]
    return c


def _build_frozen(make_model_fn, weights, strategy):
    """Create a frozen copy with given weights."""
    with strategy.scope():
        m = make_model_fn()
    m.set_weights(weights)
    m.trainable = False
    return m


# ------------------------------------------------------------------

def distill_block(make_model_fn, block_start_weights, block_epoch_weights,
                  alpha, T, sample_b, dimen, strategy,
                  distill_steps=200, batch_size=256, student_lr=1e-3,
                  sample_fn=None):
    """Distil the Polyak-averaged ensemble into a single student.

    Parameters
    ----------
    block_start_weights  :  weights of u₀   (block start)
    block_epoch_weights  :  [v₁, …, v_m]    (raw inner-loop outputs)
    alpha                :  ζ / h            (Polyak coefficient)

    Returns
    -------
    student, coeffs, last_distillation_loss
    """
    q = len(block_epoch_weights)
    coeffs = block_coeffs(alpha, q)

    # --- build ensemble teacher (frozen) ---
    all_w    = [block_start_weights] + block_epoch_weights
    teachers = [_build_frozen(make_model_fn, w, strategy) for w in all_w]

    # --- build student (initialised with weight-space averaging) ---
    with strategy.scope():
        student = make_model_fn()
        # initialize student from weight-space EMA
        # This is already close to the function-space average
        ema_weights = []
        for i in range(len(block_start_weights)):
            w = coeffs[0] * block_start_weights[i]
            for j, epoch_w in enumerate(block_epoch_weights):
                w = w + coeffs[j + 1] * epoch_w[i]
            ema_weights.append(w)
        student.set_weights(ema_weights)
        opt = tf.keras.optimizers.Adam(learning_rate=student_lr)

    @tf.function
    def _step(xb, tb):
        xb = tf.cast(xb, tf.float32)
        tb = tf.cast(tb, tf.float32)
        with tf.GradientTape() as tape:
            teacher_out = tf.add_n([
                tf.cast(c, tf.float32) * m([xb, tb], training=False)
                for c, m in zip(coeffs, teachers)
            ])
            loss = tf.reduce_mean(
                tf.square(student([xb, tb], training=True)
                          - tf.stop_gradient(teacher_out)))
        opt.apply_gradients(zip(tape.gradient(loss, student.trainable_variables),
                                student.trainable_variables))
        return loss
    
    if sample_fn is None:
        sample_fn = lambda n: sample_xt(n, T, sample_b, dimen)

    last = None
    for _ in range(distill_steps):
        xb, tb = sample_fn(batch_size)
        per_rep = strategy.run(_step, args=(xb, tb))
        last = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_rep, axis=None)

    return student, coeffs, last
