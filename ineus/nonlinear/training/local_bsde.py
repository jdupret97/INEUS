"""Training independent local transformed deep-BSDE models."""

from __future__ import annotations

import timeit

import numpy as np
import tensorflow as tf

from ..architectures.bsde import ClassicalLocalDeepBSDETransformedJump
from ..averaging import build_optimizer, copy_weights, set_weights
from ..problem import resolve_problem
from ..sampling import sample_local_t0_x0_nl


def train_one_classical_local_bsde_transformed_nl(
    t0,
    x0,
    T,
    x_scale,
    local_epochs=5000,
    batch_size=1024,
    lr=3e-4,
    optimizer_name="adam",
    N_time=50,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    clipnorm=1.0,
    eval_every=500,
    *,
    problem=None,
):
    """
    Train one independent local Deep BSDE model and return W0 and its
    corresponding nonlinear value U0.
    """
    problem = resolve_problem(problem)
    model = ClassicalLocalDeepBSDETransformedJump(
        t0=t0,
        x0=x0,
        dimen=problem.state_dim,
        T=T,
        x_scale=x_scale,
        N_time=N_time,
        hidden_dim_z=hidden_dim_z,
        n_hidden_z=n_hidden_z,
        hidden_dim_u=hidden_dim_u,
        n_hidden_u=n_hidden_u,
        K_jump_expect=K_jump_expect,
        problem=problem,
    )

    optimizer = build_optimizer(
        optimizer_name,
        lr,
        optimizer_kwargs={"clipnorm": clipnorm},
    )

    best_loss = float("inf")
    best_weights = copy_weights(model)

    history = {
        "epoch": [],
        "loss": [],
        "w0": [],
        "u0": [],
    }

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = model.bsde_loss(
                batch_size=batch_size,
                training=True,
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads_vars = [
            (g, v)
            for g, v in zip(grads, model.trainable_variables)
            if g is not None
        ]
        optimizer.apply_gradients(grads_vars)
        return loss

    for epoch in range(1, local_epochs + 1):
        loss = train_step()
        loss_val = float(loss.numpy())

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = copy_weights(model)

        if (epoch % eval_every == 0) or (epoch == 1) or (epoch == local_epochs):
            w0_val = float(model.w0.numpy().reshape(-1)[0])
            w0_safe = max(w0_val, 1e-12)
            u0_val = (
                float(problem.driver_constant) * (float(T) - float(t0))
                - np.log(w0_safe) / float(problem.eta)
            )

            history["epoch"].append(epoch)
            history["loss"].append(loss_val)
            history["w0"].append(w0_val)
            history["u0"].append(float(u0_val))

            print(
                f"[Local transformed BSDE] epoch {epoch}/{local_epochs} | "
                f"t0={float(t0):.4f} | "
                f"loss={loss_val:.6e} | "
                f"w0={w0_val:.6e} | "
                f"u0={u0_val:.6e}"
            )

    set_weights(model, best_weights)

    w0_hat = float(model.w0.numpy().reshape(-1)[0])
    u0_hat = (
        float(problem.driver_constant) * (float(T) - float(t0))
        - np.log(max(w0_hat, 1e-12)) / float(problem.eta)
    )

    return {
        "model": model,
        "w0": w0_hat,
        "u0": float(u0_hat),
        "best_loss": best_loss,
        "history": history,
    }


def make_local_bsde_label_dataset_nl(
    num_local_models,
    T,
    x_scale,
    local_epochs=5000,
    local_batch_size=1024,
    lr=3e-4,
    optimizer_name="adam",
    N_time=50,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    clipnorm=1.0,
    eval_every=500,
    t_eps=1e-5,
    *,
    problem=None,
):
    """
    Train independent local transformed Deep BSDE solvers.

    Returns local transformed labels W0 and the corresponding nonlinear labels
    U0 = f_const*(T-t0) - log(W0)/eta.
    """
    problem = resolve_problem(problem)
    t0_labels, x0_labels = sample_local_t0_x0_nl(
        num_points=num_local_models,
        T=T,
        t_eps=t_eps,
        problem=problem,
    )

    w0_labels = np.zeros((num_local_models, 1), dtype=np.float32)
    u0_labels = np.zeros((num_local_models, 1), dtype=np.float32)
    local_losses = np.zeros((num_local_models, 1), dtype=np.float32)

    tic = timeit.default_timer()

    for i in range(num_local_models):
        print("\n" + "-" * 80)
        print(f"Training local transformed BSDE model {i + 1}/{num_local_models}")
        print("-" * 80)

        res_i = train_one_classical_local_bsde_transformed_nl(
            t0=float(t0_labels[i, 0]),
            x0=x0_labels[i],
            T=T,
            x_scale=x_scale,
            local_epochs=local_epochs,
            batch_size=local_batch_size,
            lr=lr,
            optimizer_name=optimizer_name,
            N_time=N_time,
            hidden_dim_z=hidden_dim_z,
            n_hidden_z=n_hidden_z,
            hidden_dim_u=hidden_dim_u,
            n_hidden_u=n_hidden_u,
            K_jump_expect=K_jump_expect,
            clipnorm=clipnorm,
            eval_every=eval_every,
            problem=problem,
        )

        w0_labels[i, 0] = res_i["w0"]
        u0_labels[i, 0] = res_i["u0"]
        local_losses[i, 0] = res_i["best_loss"]

        elapsed = timeit.default_timer() - tic
        print(
            f"[Local label {i + 1}/{num_local_models}] "
            f"w0_hat={w0_labels[i, 0]:.6e} | "
            f"u0_hat={u0_labels[i, 0]:.6e} | "
            f"best_loss={local_losses[i, 0]:.6e} | "
            f"elapsed={elapsed:.2f}s"
        )

    return {
        "x0": x0_labels,
        "t0": t0_labels,
        "w0": w0_labels,
        "u0": u0_labels,
        "local_losses": local_losses,
    }
