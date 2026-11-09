"""Nonlinear strong-PIDE-residual PINN training loop."""

from __future__ import annotations

import timeit

import tensorflow as tf

from ..averaging import build_optimizer, copy_weights
from ..architectures.value import strategy
from ..pde import pinn_residual_pointwise_nl, pinn_rhs_square_loss_nl, rad_resample_dataset_nl
from ..sampling import sample_x_t0
from ..problem import resolve_problem
from .common import evaluate_test_mae


def train_standard_pinn_nl(
    model1,
    T,
    epochs,
    num_samples,
    lr,
    n_inner,
    num_batches_N1,
    replay_buffer,
    xtest,
    ttest,
    values,
    batch_size=256,
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    optimizer_name="adam",
    optimizer_kwargs=None,
    log_every=10,
    K_train=100,
    K_test=50,
    num_t0_each_epoch=0,
    K_rad=25,
    decay_steps=1000,
    decay_rate=0.991,
    staircase=True,
    *,
    problem=None,
):
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

    @tf.function
    def train_step1(x_batch, t_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)

        with tf.GradientTape() as tape:
            loss1 = pinn_rhs_square_loss_nl(model1, x_batch, t_batch, T=T, K=K_train, problem=problem)

        gradients = tape.gradient(loss1, model1.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, model1.trainable_variables))
        return loss1

    start_time = timeit.default_timer()

    for epoch in range(epochs):
        if num_t0_each_epoch > 0:
            x_t0, t_t0 = sample_x_t0(num_t0_each_epoch, problem=problem)
            replay_buffer.extend(x_t0, t_t0)

        total_loss = 0.0

        for _ in range(n_inner - 1):
            for _ in range(num_batches_N1):
                x_batch, t_batch = replay_buffer.sample_batch(batch_size)

                per_replica_loss = strategy.run(train_step1, args=(x_batch, t_batch))
                loss1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
                total_loss += float(loss1.numpy())

        avg_loss = total_loss / ((n_inner - 1) * num_batches_N1)

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
                f"[PINN RAD @ epoch {epoch+1}] "
                f"mean|r|={rad_stats['mean_residual']:.6e}, "
                f"max|r|={rad_stats['max_residual']:.6e}, "
                f"min|r|={rad_stats['min_residual']:.6e}"
            )

        if ((epoch) % log_every == 0) or (epoch == epochs - 1):
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
                f"[PINN-nonlinear] Epoch {epoch+1}/{epochs} | "
                f"train rhs_square={avg_loss:.6e} | test rhs_square={test_residual:.6e} | "
                f"MAE={mae:.6e} | time={elapsed:.2f}s"
            )

    return {
        "history": history,
        "best_weights": best_weights,
        "best_loss": best_loss,
    }
