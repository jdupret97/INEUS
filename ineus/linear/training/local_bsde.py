"""Local Deep BSDE training and sampled label generation."""

import timeit

import numpy as np
import tensorflow as tf

from ..architectures.bsde import ClassicalLocalDeepBSDEJump
from ..averaging import copy_weights, set_weights
from ..problem import resolve_problem
from ..sampling import sample_t0_x0
from .common import build_optimizer


def train_one_classical_local_bsde_jump(
    t0,
    x0,
    local_epochs=10000,
    batch_size=1024,
    lr=3e-4,
    optimizer_name="adam",
    N_time=20,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    clipnorm=1.0,
    eval_every=1000,
    *,
    problem=None,
):
    """
    Train one independent local Deep BSDE model for one fixed (t0, x0).

    Returns:
        y0 ≈ u(t0, x0)
    """
    problem = resolve_problem(problem)
    model = ClassicalLocalDeepBSDEJump(
        t0=t0,
        x0=x0,
        dimen=problem.dimen,
        dimend=problem.dimend,
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
        "y0": [],
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
            y0_val = float(model.y0.numpy().reshape(-1)[0])

            history["epoch"].append(epoch)
            history["loss"].append(loss_val)
            history["y0"].append(y0_val)

            print(
                f"[Local BSDE] epoch {epoch}/{local_epochs} | "
                f"t0={float(t0):.4f} | "
                f"loss={loss_val:.6e} | "
                f"y0={y0_val:.6e}"
            )

    set_weights(model, best_weights)

    y0_hat = float(model.y0.numpy().reshape(-1)[0])

    return {
        "model": model,
        "y0": y0_hat,
        "best_loss": best_loss,
        "history": history,
    }


def make_local_bsde_label_dataset(
    num_local_models,
    local_epochs=10000,
    local_batch_size=1024,
    lr=3e-4,
    optimizer_name="adam",
    N_time=20,
    hidden_dim_z=128,
    n_hidden_z=3,
    hidden_dim_u=128,
    n_hidden_u=3,
    K_jump_expect=64,
    clipnorm=1.0,
    eval_every=1000,
    t_eps=1e-5,
    *,
    problem=None,
):
    """
    Train num_local_models independent classical local BSDE solvers.

    Each solver is trained for a sampled point

        (t0, x0) ~ rho,

    where rho is uniform over [0,T) x [-sample_b, sample_b]^d.

    Returns:
        x0_labels : [M,dimen]
        t0_labels : [M,1]
        y0_labels : [M,1]
    """
    problem = resolve_problem(problem)
    t0_labels, x0_labels = sample_t0_x0(
        num_points=num_local_models,
        t_eps=t_eps,
        problem=problem,
    )

    y0_labels = np.zeros((num_local_models, 1), dtype=np.float32)
    local_losses = np.zeros((num_local_models, 1), dtype=np.float32)
    exact_y0 = problem.exact_value(t0_labels, x0_labels).reshape(-1, 1)

    tic = timeit.default_timer()

    for i in range(num_local_models):
        print("\n" + "-" * 80)
        print(f"Training local BSDE model {i + 1}/{num_local_models}")
        print("-" * 80)

        res_i = train_one_classical_local_bsde_jump(
            t0=float(t0_labels[i, 0]),
            x0=x0_labels[i],
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

        y0_labels[i, 0] = res_i["y0"]
        local_losses[i, 0] = res_i["best_loss"]

        elapsed = timeit.default_timer() - tic

        print(
            f"[Local label {i + 1}/{num_local_models}] "
            f"y0_hat={y0_labels[i, 0]:.6e} | "
            f"exact={exact_y0[i, 0]:.6e} | "
            f"abs_err={abs(y0_labels[i, 0] - exact_y0[i, 0]):.6e} | "
            f"elapsed={elapsed:.2f}s"
        )

    return {
        "x0": x0_labels,
        "t0": t0_labels,
        "y0": y0_labels,
        "exact_y0": exact_y0.astype(np.float32),
        "local_losses": local_losses,
    }
