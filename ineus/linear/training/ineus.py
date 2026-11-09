"""Iterative neural update, block distillation, and RAD schedule."""

import timeit

import numpy as np
import tensorflow as tf

from ..averaging import copy_weights, distill_block_teacher, set_weights
from ..pde import pinn_loss1bis, pinn_residual_pointwise, rad_resample_dataset
from ..problem import resolve_problem
from .common import build_optimizer


def train_bellman_ct(
    model1,
    model1targ,
    make_model_fn,
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
    K_rad=25,
    *,
    problem=None,
):
    """Run the INEUS loop, including residual checkpoint selection.

    Test residuals are evaluated for logging and best-weight restoration.
    They are not the INEUS training loss.
    """
    problem = resolve_problem(problem)
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

    optimizer1 = build_optimizer(optimizer_name, lr, optimizer_kwargs)

    best_loss = float("inf")
    best_weights = copy_weights(model1)

    block_start_weights = copy_weights(model1)
    block_epoch_weights = []

    @tf.function
    def train_step1(x_batch, t_batch, scale):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)

        batch_n = tf.shape(x_batch)[0]
        if problem.lam == 0:
            e_fresh = tf.zeros_like(x_batch)
        else:
            z = tf.random.normal(tf.stack([batch_n, problem.dimen]), dtype=tf.float32)
            e_fresh = (tf.linalg.matmul(z, problem.cov_j_chol_tf, transpose_b=True)
                       + problem.mu_j_tf[None, :])
        h_local = tf.zeros(tf.stack([batch_n, 1]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss1 = pinn_loss1bis(
                model1=model1,
                model1targ=model1targ,
                x=x_batch,
                t=t_batch,
                e=e_fresh,
                scale=scale,
                T=T,
                h=h_local,
                problem=problem,
            )

        gradients = tape.gradient(loss1, model1.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, model1.trainable_variables))
        return loss1

    start_time = timeit.default_timer()

    for epoch in range(epochs):
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
                loss1 = train_step1(x_batch, t_batch, bet_scale)
                total_loss += float(loss1.numpy())

            model1targ.set_weights(model1.get_weights())

        avg_loss = total_loss / ((n_inner - 1) * num_batches_N1)
        block_epoch_weights.append(copy_weights(model1))

        if (m is not None) and (m > 0) and (len(block_epoch_weights) == m):
            student_model, _, distill_loss = distill_block_teacher(
                make_model_fn=make_model_fn,
                block_start_weights=block_start_weights,
                block_epoch_weights=block_epoch_weights,
                alpha=alpha,
                distill_steps=distill_steps,
                distill_batch_size=batch_size,
                student_lr=student_lr,
                problem=problem,
            )

            print(f"[Bellman block finished at epoch {epoch+1}] distill loss = {float(distill_loss.numpy()):.6e}")

            if replace_with_student:
                model1.set_weights(student_model.get_weights())
                model1targ.set_weights(student_model.get_weights())

            block_start_weights = copy_weights(model1)
            block_epoch_weights = []

        if use_rad and ((epoch + 1) % rad_period == 0):
            x_rad, t_rad, rad_stats = rad_resample_dataset(
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
                pinn_residual_pointwise(
                    model1, xtest, ttest, K=K_test, training=False, problem=problem,
                )[2].numpy()
            )
            mae = float(np.mean(np.abs(model1([xtest, ttest], training=False).numpy() - values)))
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
                f"[Bellman] Epoch {epoch+1}/{epochs} | "
                f"train={avg_loss:.6e} | "
                f"test rhs_square={test_residual:.6e} | "
                f"MAE={mae:.6e} | "
                f"time={elapsed:.2f}s"
            )

    set_weights(model1, best_weights)

    return {
        "history": history,
        "best_weights": best_weights,
        "best_loss": best_loss,
    }
