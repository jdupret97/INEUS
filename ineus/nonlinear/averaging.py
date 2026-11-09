"""Optimizer, weight-averaging, and block-distillation helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .architectures.value import strategy
from .sampling import sample_xt


def build_optimizer(optimizer, learning_rate, optimizer_kwargs=None):
    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_kwargs["learning_rate"] = learning_rate

    if isinstance(optimizer, str):
        opt_map = {
            "adam": tf.keras.optimizers.Adam,
            "adagrad": tf.keras.optimizers.Adagrad,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
        }
        key = optimizer.lower()
        if key not in opt_map:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return opt_map[key](**optimizer_kwargs)

    if isinstance(optimizer, type) and issubclass(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer(**optimizer_kwargs)

    raise ValueError("optimizer must be a string or a tf.keras optimizer class.")


def copy_weights(model):
    return [np.array(w, copy=True) for w in model.get_weights()]


def set_weights(model, weights):
    model.set_weights(weights)


def block_coeffs(alpha, q):
    coeffs = [(1.0 - alpha) ** q]
    coeffs += [alpha * (1.0 - alpha) ** (q - j) for j in range(1, q + 1)]
    return coeffs


def build_frozen_model(make_model, weights):
    with strategy.scope():
        m = make_model()
    m.set_weights(weights)
    m.trainable = False
    return m


def distill_block_teacher(
    make_model,
    block_start_weights,
    block_epoch_weights,
    alpha,
    T,
    distill_steps=200,
    distill_batch_size=256,
    student_lr=1e-3,
    *,
    problem=None,
):
    q = len(block_epoch_weights)
    coeffs = block_coeffs(alpha, q)

    teacher_weights = [block_start_weights] + block_epoch_weights
    teacher_models = [build_frozen_model(make_model, w) for w in teacher_weights]

    with strategy.scope():
        student = make_model()
        student.set_weights(block_epoch_weights[-1])
        student_opt = tf.keras.optimizers.Adam(learning_rate=student_lr)

    @tf.function
    def distill_step(x_batch, t_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)

        with tf.GradientTape() as tape:
            teacher_out = tf.add_n([
                tf.cast(c, tf.float32) * tm([x_batch, t_batch], training=False)
                for c, tm in zip(coeffs, teacher_models)
            ])
            student_out = student([x_batch, t_batch], training=True)
            loss = tf.reduce_mean(tf.square(student_out - tf.stop_gradient(teacher_out)))

        grads = tape.gradient(loss, student.trainable_variables)
        student_opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    last_loss = None
    for _ in range(distill_steps):
        x_batch, t_batch = sample_xt(distill_batch_size, T, problem=problem)
        per_replica_loss = strategy.run(distill_step, args=(x_batch, t_batch))
        last_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    return student, coeffs, last_loss
