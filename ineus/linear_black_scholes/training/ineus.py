"""Iterative target regression, replay sampling and block distillation."""

import timeit

import numpy as np
import tensorflow as tf

from ..averaging import distill_block_teacher
from ..pde import pinn_loss1bis, terminal_loss, pinn_residual_pointwise, rad_resample_dataset
from ..sampling import sample_xt, sample_xt_tf, sample_x_terminal_paths_tf
from ..utils import build_optimizer, copy_weights


def train2(
    model1,
    model1targ,
    make_model_fn,
    T_val,
    epochs,
    num_samples_val,
    lr,
    rho1_val,
    n_inner_val,
    num_batches_N1_val,
    h_scale_val,
    zeta_val,
    replay_buffer,
    w_bc_val=1.0,
    w_int_val=1.0,
    batch_size_val=256,
    m=10,
    distill_steps=200,
    student_lr=1e-3,
    replace_with_student=True,
    use_distill=True,
    use_rad=True,
    rad_period=5,
    rad_k=1.0,
    rad_c=1.0,
    rad_candidate_size=None,
    rad_eval_batch_size=2048,
    sample_mode="uniform",
    S0=None,
    optimizer_name="adam",
    optimizer_kwargs=None,
    *,
    problem,
    evaluation,
    rad_candidate_multiplier=8,
):
    test_loss_hist = []
    train_loss_hist = []
    train_pde_loss_hist = []
    train_bc_loss_hist = []
    mae_hist = []
    time_hist = []

    bet_scale = h_scale_val / (n_inner_val - 1)
    alpha = zeta_val / h_scale_val

    if rad_candidate_size is None:
        rad_candidate_size = rad_candidate_multiplier * num_samples_val

    optimizer1 = build_optimizer(optimizer_name, lr, optimizer_kwargs)

    h_local = tf.Variable(tf.zeros((batch_size_val, 1), dtype=tf.float32))
    w_bc_tf = tf.constant(w_bc_val, dtype=tf.float32)
    w_int_tf = tf.constant(w_int_val, dtype=tf.float32)

    block_start_weights = copy_weights(model1)
    block_epoch_weights = []
    saved_blocks = []
    distilled_block_weights = []

    last_epoch_loss = None
    last_epoch_train_loss = None
    last_epoch_pde_loss = None
    last_epoch_bc_loss = None
    last_epoch_mae = None

    @tf.function
    def train_step1(x_batch, t_batch, x_T_batch, model1_arg, model2_arg, scale):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        x_T_batch = tf.cast(x_T_batch, tf.float32)

        B1 = tf.shape(x_batch)[0]
        z = tf.random.normal([B1, problem.dimen], dtype=tf.float32)
        mu_j_tf = tf.constant(problem.mu_j, dtype=tf.float32)[None, :]
        sig_j_tf = tf.constant(problem.sig_j, dtype=tf.float32)[None, :]
        e_fresh = mu_j_tf + sig_j_tf * z

        with tf.GradientTape() as tape:
            loss_pde = pinn_loss1bis(
                model1_arg,
                model2_arg,
                x_batch,
                t_batch,
                e_fresh,
                scale,
                T_val=T_val,
                h=h_local,
                problem=problem,
            )
            # The hPINN imposes u(T,x)=phi(x) exactly; no terminal penalty is needed.
            loss_T = (terminal_loss(model1_arg, x_T_batch, T_val, training=True, problem=problem)
                      if w_bc_val else tf.constant(0.0, dtype=tf.float32))
            loss_total = w_int_tf * loss_pde + w_bc_tf * loss_T

        gradients = tape.gradient(loss_total, model1_arg.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, model1_arg.trainable_variables))
        return loss_total, loss_pde, loss_T

    replay_buffer.clear()
    if use_rad:
        x_init, t_init = sample_xt(num_samples_val, T_val, mode=sample_mode, S0=S0, problem=problem)
        replay_buffer.extend(x_init, t_init)

    tic_train = timeit.default_timer()

    for epoch in range(epochs):
        if epoch == 0:
            model1targ.set_weights(model1.get_weights())
        else:
            new_weights = [
                rho1_val * w_targ + (1.0 - rho1_val) * w_cur
                for w_targ, w_cur in zip(model1targ.get_weights(), model1.get_weights())
            ]
            model1targ.set_weights(new_weights)

        total_loss = 0.0
        total_pde_loss = 0.0
        total_bc_loss = 0.0
        bsize = replay_buffer.size()
        num_batches = num_batches_N1_val

        for _ in range(n_inner_val - 1):
            for _ in range(num_batches):
                if use_rad:
                    x_batch, t_batch = replay_buffer.sample_batch(batch_size_val)
                else:
                    x_batch, t_batch = sample_xt_tf(batch_size_val, T_val, mode=sample_mode, S0=S0, problem=problem)

                if w_bc_val and sample_mode == "paths":
                    x_T_batch = sample_x_terminal_paths_tf(
                        num_points=batch_size_val, T_val=T_val, S0=S0,
                        r_val=problem.r, sigma_diag_val=problem.sigma_diag, Sigma_val=problem.Sigma,
                        lam_vec_val=problem.lam_vec, mu_j_val=problem.mu_j, sig_j_val=problem.sig_j,
                    )
                elif w_bc_val:
                    x_T_batch, _ = sample_xt_tf(batch_size_val, T_val, mode="uniform", problem=problem)
                else:
                    x_T_batch = x_batch

                loss_total, loss_pde, loss_bc = train_step1(
                    x_batch,
                    t_batch,
                    x_T_batch,
                    model1,
                    model1targ,
                    bet_scale,
                )

                total_loss += float(loss_total.numpy())
                total_pde_loss += float(loss_pde.numpy())
                total_bc_loss += float(loss_bc.numpy())

            model1targ.set_weights(model1.get_weights())

        denom = (n_inner_val - 1) * num_batches
        avg_loss = total_loss / denom
        avg_pde_loss = total_pde_loss / denom
        avg_bc_loss = total_bc_loss / denom

        block_epoch_weights.append(copy_weights(model1))

        if use_distill and len(block_epoch_weights) == m:
            student_model, coeffs, distill_loss = distill_block_teacher(
                make_model_fn=make_model_fn,
                block_start_weights=block_start_weights,
                block_epoch_weights=block_epoch_weights,
                alpha=alpha,
                T_val=T_val,
                distill_steps=distill_steps,
                distill_batch_size=batch_size_val,
                student_lr=student_lr,
                sample_mode=sample_mode,
                S0=S0,
                problem=problem,
            )

            saved_blocks.append(
                {
                    "k0": epoch + 1 - m,
                    "start_weights": block_start_weights,
                    "epoch_weights": block_epoch_weights.copy(),
                    "coeffs": coeffs,
                }
            )
            distilled_block_weights.append(copy_weights(student_model))

            print(f"[Block finished at epoch {epoch+1}] Distillation loss = {float(distill_loss.numpy()):.6e}")

            if replace_with_student:
                model1.set_weights(student_model.get_weights())
                model1targ.set_weights(student_model.get_weights())

            block_start_weights = copy_weights(model1)
            block_epoch_weights = []

        loss1t = pinn_residual_pointwise(model1, evaluation.xtest, evaluation.ttest, T_val=T_val, K_mc=50, problem=problem)[1]
        tot_loss = float(loss1t.numpy())

        preds = model1([evaluation.xtest, evaluation.ttest], training=False).numpy()
        mae = float(np.mean(np.abs(preds - evaluation.values)))

        last_epoch_loss = tot_loss
        last_epoch_train_loss = avg_loss
        last_epoch_pde_loss = avg_pde_loss
        last_epoch_bc_loss = avg_bc_loss
        last_epoch_mae = mae

        if use_rad and ((epoch + 1) % rad_period == 0):
            x_rad, t_rad, rad_stats = rad_resample_dataset(
                model1=model1,
                T_val=T_val,
                num_points=num_samples_val,
                candidate_size=rad_candidate_size,
                K_jump=25,
                k=rad_k,
                c_rad=rad_c,
                eval_batch_size=rad_eval_batch_size,
                replace=False,
                sample_mode=sample_mode,
                S0=S0,
                problem=problem,
            )
            replay_buffer.extend(x_rad, t_rad)

            print(
                f"[RAD @ epoch {epoch+1}] "
                f"mean|r|={rad_stats['mean_residual']:.6e}, "
                f"max|r|={rad_stats['max_residual']:.6e}, "
                f"min|r|={rad_stats['min_residual']:.6e}"
            )

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            elapsed = timeit.default_timer() - tic_train
            test_loss_hist.append(tot_loss)
            train_loss_hist.append(avg_loss)
            train_pde_loss_hist.append(avg_pde_loss)
            train_bc_loss_hist.append(avg_bc_loss)
            mae_hist.append(mae)
            time_hist.append(elapsed)

            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Test Loss = {tot_loss:.6e}, "
                f"Train Loss = {avg_loss:.6e}, "
                f"PDE Loss = {avg_pde_loss:.6e}, "
                f"Terminal Loss = {avg_bc_loss:.6e}, "
                f"MAE = {mae:.6e}, "
                f"Buffer = {bsize}"
            )

    return {
        "last_epoch_loss": last_epoch_loss,
        "last_epoch_train_loss": last_epoch_train_loss,
        "last_epoch_pde_loss": last_epoch_pde_loss,
        "last_epoch_bc_loss": last_epoch_bc_loss,
        "last_epoch_mae": last_epoch_mae,
        "test_loss_history": test_loss_hist,
        "train_loss_history": train_loss_hist,
        "train_pde_loss_history": train_pde_loss_hist,
        "train_bc_loss_history": train_bc_loss_hist,
        "mae_history": mae_hist,
        "time_history": time_hist,
        "saved_blocks": saved_blocks,
        "distilled_block_weights": distilled_block_weights,
    }
