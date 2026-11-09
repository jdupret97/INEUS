"""Nonlinear INEUS iteration, replay, RAD, and distillation."""

from __future__ import annotations

import timeit

import numpy as np
import tensorflow as tf

from ..averaging import build_optimizer, copy_weights, distill_block_teacher
from ..architectures.value import strategy
from ..pde import pinn_loss1bis, pinn_residual_pointwise_nl, rad_resample_dataset_nl
from ..problem import COV_J_CHOL as cov_j_chol, DIMENSION as dimen, MU_J as mu_j
from ..problem import resolve_problem
from ..sampling import sample_x_t0
from .common import evaluate_test_mae


def train_bellman_ct_nl(
    model1,
    model1targ,
    make_model,
    T,
    epochs,
    num_samples,
    lr,
    rho1,
    n_inner,
    num_batches_N1,
    h_scale,
    zeta,
    replay_buffer,
    xtest,
    ttest,
    values,
    batch_size=256,
    m=10,
    distill_steps=200,
    student_lr=1e-3,
    replace_with_student=True,
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    optimizer_name="adam",
    optimizer_kwargs=None,
    log_every=10,
    K_test=50,
    num_t0_each_epoch=0,
    K_rad=25,
    decay_steps=1000,
    decay_rate=0.995,
    staircase=True,
    *,
    problem=None,
):
    problem = resolve_problem(problem)
    dimen, mu_j, cov_j_chol = problem.state_dim, problem.mu_j, problem.cov_j_chol
    history = {
        "epoch": [],
        "mae": [],
        "time": [],
        "train_loss": [],
        "test_residual": [],
    }

    if rad_candidate_size is None:
        rad_candidate_size = 8 * num_samples

    bet_scale = h_scale / (n_inner - 1)
    alpha = zeta / h_scale

    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )

    with strategy.scope():
        optimizer1 = build_optimizer(
            optimizer=optimizer_name,
            learning_rate=schedule,
            optimizer_kwargs=optimizer_kwargs,
        )

    best_loss = float("inf")
    best_weights = copy_weights(model1)

    saved_blocks = []
    distilled_block_weights = []

    block_start_weights = copy_weights(model1)
    block_epoch_weights = []

    @tf.function
    def train_step1(x_batch, t_batch, scale):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)

        local_bs = tf.shape(x_batch)[0]
        if problem.has_jumps:
            z = tf.random.normal(
                shape=tf.stack([local_bs, tf.constant(dimen, dtype=tf.int32)]),
                dtype=tf.float32,
            )
            mu_j_tf = tf.cast(mu_j, tf.float32)
            e_fresh = tf.linalg.matmul(z, cov_j_chol, transpose_b=True) + mu_j_tf[None, :]
        else:
            e_fresh = tf.zeros_like(x_batch)

        with tf.GradientTape() as tape:
            loss1 = pinn_loss1bis(model1, model1targ, x_batch, t_batch, e_fresh, scale, T=T, problem=problem)

        gradients = tape.gradient(loss1, model1.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, model1.trainable_variables))
        return loss1

    start_time = timeit.default_timer()

    for epoch in range(epochs):
        if num_t0_each_epoch > 0:
            x_t0, t_t0 = sample_x_t0(num_t0_each_epoch, problem=problem)
            replay_buffer.extend(x_t0, t_t0)

        # target update
        if epoch == 0:
            model1targ.set_weights(model1.get_weights())
        else:
            new_weights = [
                rho1 * w_targ + (1.0 - rho1) * w_cur
                for w_targ, w_cur in zip(model1targ.get_weights(), model1.get_weights())
            ]
            model1targ.set_weights(new_weights)

        total_loss = 0.0

        for _ in range(n_inner - 1):
            for _ in range(num_batches_N1):
                x_batch, t_batch = replay_buffer.sample_batch(batch_size)

                per_replica_loss = strategy.run(train_step1, args=(x_batch, t_batch, bet_scale))
                loss1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
                total_loss += float(loss1.numpy())

            model1targ.set_weights(model1.get_weights())

        avg_loss = total_loss / ((n_inner - 1) * num_batches_N1)

        block_epoch_weights.append(copy_weights(model1))

        if (m is not None) and (m > 0) and (len(block_epoch_weights) == m):
            student_model, coeffs, distill_loss = distill_block_teacher(
                make_model=make_model,
                block_start_weights=block_start_weights,
                block_epoch_weights=block_epoch_weights,
                alpha=alpha,
                T=T,
                distill_steps=distill_steps,
                distill_batch_size=batch_size,
                student_lr=student_lr,
                problem=problem,
            )

            saved_blocks.append({
                "k0": epoch + 1 - m,
                "start_weights": block_start_weights,
                "epoch_weights": block_epoch_weights.copy(),
                "coeffs": coeffs,
            })
            distilled_block_weights.append(copy_weights(student_model))

            print(f"[Bellman block finished at epoch {epoch+1}] distill loss = {float(distill_loss.numpy()):.6e}")

            if replace_with_student:
                model1.set_weights(student_model.get_weights())
                model1targ.set_weights(student_model.get_weights())

            block_start_weights = copy_weights(model1)
            block_epoch_weights = []

        if use_rad and ((epoch + 1) % rad_period == 0):
            x_rad, t_rad, rad_stats = rad_resample_dataset_nl(
                model1=model1,
                T=T,
                num_points=num_samples,
                candidate_size=rad_candidate_size,
                K_jump=K_rad,
                k=rad_k,
                c_rad=rad_c,
                eval_batch_size=rad_eval_batch_size,
                replace=False,
                problem=problem,
            )
            replay_buffer.extend(x_rad, t_rad)

            print(
                f"[Bellman RAD @ epoch {epoch+1}] "
                f"mean|r|={rad_stats['mean_residual']:.6e}, "
                f"max|r|={rad_stats['max_residual']:.6e}, "
                f"min|r|={rad_stats['min_residual']:.6e}"
            )

        if ((epoch + 1) % log_every == 0) or (epoch == epochs - 1):
            test_residual = float(
                pinn_residual_pointwise_nl(model1, xtest, ttest, T=T, K=K_test, training=False, problem=problem)[2].numpy()
            )
            mae = evaluate_test_mae(model1, xtest, ttest, values)
            elapsed = timeit.default_timer() - start_time

            history["epoch"].append(epoch + 1)
            history["mae"].append(mae)
            history["time"].append(elapsed)
            history["train_loss"].append(avg_loss)
            history["test_residual"].append(test_residual)

            if test_residual < best_loss:
                best_loss = test_residual
                best_weights = copy_weights(model1)

            print(
                f"[Bellman-nonlinear] Epoch {epoch+1}/{epochs} | "
                f"train={avg_loss:.6e} | test rhs_square={test_residual:.6e} | "
                f"MAE={mae:.6e} | time={elapsed:.2f}s"
            )

    return {
        "history": history,
        "best_weights": best_weights,
        "best_loss": best_loss,
        "saved_blocks": saved_blocks,
        "distilled_block_weights": distilled_block_weights,
    }
