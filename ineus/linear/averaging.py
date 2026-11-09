"""Target-network and block-distillation helpers."""

import numpy as np
import tensorflow as tf

from .problem import resolve_problem
from .sampling import sample_xt


def copy_weights(model):
    return [np.array(w, copy=True) for w in model.get_weights()]


def set_weights(model, weights):
    model.set_weights(weights)


def block_coeffs(alpha, q):
    coeffs = [(1.0 - alpha) ** q]
    coeffs += [alpha * (1.0 - alpha) ** (q - j) for j in range(1, q + 1)]
    return coeffs


def build_frozen_model(make_model_fn, weights):
    m = make_model_fn()
    m.set_weights(weights)
    m.trainable = False
    return m


def distill_block_teacher(
    make_model_fn,
    block_start_weights,
    block_epoch_weights,
    alpha,
    distill_steps=200,
    distill_batch_size=256,
    student_lr=1e-3,
    *,
    problem=None,
):
    problem = resolve_problem(problem)
    q = len(block_epoch_weights)
    coeffs = block_coeffs(alpha, q)

    teacher_weights = [block_start_weights] + block_epoch_weights
    teacher_models = [build_frozen_model(make_model_fn, w) for w in teacher_weights]

    student = make_model_fn()
    student.set_weights(block_epoch_weights[-1])
    student_opt = tf.keras.optimizers.Adam(learning_rate=student_lr)

    @tf.function
    def distill_step(x_batch, t_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)

        with tf.GradientTape() as tape:
            teacher_out = tf.add_n([
                tf.cast(cff, tf.float32) * tm([x_batch, t_batch], training=False)
                for cff, tm in zip(coeffs, teacher_models)
            ])
            student_out = student([x_batch, t_batch], training=True)
            loss = tf.reduce_mean(tf.square(student_out - tf.stop_gradient(teacher_out)))

        grads = tape.gradient(loss, student.trainable_variables)
        student_opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    last_loss = None
    for _ in range(distill_steps):
        x_batch, t_batch = sample_xt(distill_batch_size, problem.T, problem=problem)
        last_loss = distill_step(x_batch, t_batch)

    return student, coeffs, last_loss
