"""Global surrogate optimization on local Deep BSDE labels."""

import numpy as np
import tensorflow as tf

from ..architectures.surrogate import GlobalValueSurrogateFromLocalBSDE
from ..averaging import copy_weights, set_weights
from ..problem import resolve_problem
from .common import build_optimizer


def train_global_surrogate_from_local_bsde_labels(
    x_labels,
    t_labels,
    y_labels,
    epochs=5000,
    batch_size=256,
    lr=1e-3,
    optimizer_name="adam",
    hidden_dim=128,
    n_hidden=3,
    constrained=True,
    eval_every=500,
    validation_fraction=0.2,
    *,
    problem=None,
):
    """
    Train global u_psi(t,x) from local BSDE labels.
    """
    problem = resolve_problem(problem)
    x_labels = np.asarray(x_labels, dtype=np.float32)
    t_labels = np.asarray(t_labels, dtype=np.float32).reshape(-1, 1)
    y_labels = np.asarray(y_labels, dtype=np.float32).reshape(-1, 1)

    n_data = x_labels.shape[0]
    idx_all = np.arange(n_data)
    np.random.shuffle(idx_all)

    n_val = int(validation_fraction * n_data)
    n_val = max(1, n_val) if n_data >= 5 else 0

    if n_val > 0:
        val_idx = idx_all[:n_val]
        train_idx = idx_all[n_val:]
    else:
        val_idx = np.array([], dtype=np.int64)
        train_idx = idx_all

    model = GlobalValueSurrogateFromLocalBSDE(
        dimen=problem.dimen,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation="tanh",
        constrained=constrained,
        problem=problem,
    )

    optimizer = build_optimizer(
        optimizer_name,
        lr,
        optimizer_kwargs=None,
    )

    best_loss = float("inf")
    best_weights = copy_weights(model)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    @tf.function
    def train_step(x_batch, t_batch, y_batch):
        with tf.GradientTape() as tape:
            pred = model((t_batch, x_batch), training=True)
            loss = tf.reduce_mean(tf.square(pred - y_batch))

        grads = tape.gradient(loss, model.trainable_variables)

        grads_vars = [
            (g, v)
            for g, v in zip(grads, model.trainable_variables)
            if g is not None
        ]

        optimizer.apply_gradients(grads_vars)

        return loss

    def eval_loss(indices):
        if len(indices) == 0:
            return np.nan

        x_eval = tf.convert_to_tensor(x_labels[indices], dtype=tf.float32)
        t_eval = tf.convert_to_tensor(t_labels[indices], dtype=tf.float32)
        y_eval = tf.convert_to_tensor(y_labels[indices], dtype=tf.float32)

        pred = model((t_eval, x_eval), training=False)
        return float(tf.reduce_mean(tf.square(pred - y_eval)).numpy())

    for epoch in range(1, epochs + 1):
        idx = np.random.choice(
            train_idx,
            size=batch_size,
            replace=(batch_size > len(train_idx)),
        )

        x_batch = tf.convert_to_tensor(x_labels[idx], dtype=tf.float32)
        t_batch = tf.convert_to_tensor(t_labels[idx], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_labels[idx], dtype=tf.float32)

        loss = train_step(x_batch, t_batch, y_batch)
        train_loss_val = float(loss.numpy())

        monitor_loss = train_loss_val

        if n_val > 0:
            val_loss_val = eval_loss(val_idx)
            monitor_loss = val_loss_val
        else:
            val_loss_val = np.nan

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_weights = copy_weights(model)

        if (epoch % eval_every == 0) or (epoch == 1) or (epoch == epochs):
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss_val)
            history["val_loss"].append(val_loss_val)

            print(
                f"[Global surrogate] epoch {epoch}/{epochs} | "
                f"train={train_loss_val:.6e} | "
                f"val={val_loss_val:.6e}"
            )

    set_weights(model, best_weights)

    return {
        "model": model,
        "best_loss": best_loss,
        "history": history,
    }


def predict_u_local_bsde_surrogate(global_model, x_np, t_np):
    x_tf = tf.convert_to_tensor(
        np.asarray(x_np, dtype=np.float32),
        dtype=tf.float32,
    )

    t_tf = tf.convert_to_tensor(
        np.asarray(t_np, dtype=np.float32).reshape(-1, 1),
        dtype=tf.float32,
    )

    return global_model((t_tf, x_tf), training=False).numpy()
