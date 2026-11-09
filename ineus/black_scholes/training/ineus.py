"""Iterative neural updates with clipping, resampling and distillation."""

import os
import timeit
import numpy as np
import tensorflow as tf

from ..utils import copy_weights, evaluate_point
from ..sampling import (
    sample_xt, sample_x_terminal, sample_xt_paths, sample_x_terminal_paths,
    sample_xt_with_jumps, sample_x_terminal_with_jumps, rad_resample,
)
from ..averaging import distill_block
from ..problem import pinn_loss, terminal_loss, residual_pointwise
from ..checkpointing import save_checkpoint, save_history_json


def train(model, model_targ, make_model_fn,
          pde_cfg, train_cfg, eval_cfg,
          replay_buffer, strategy,
          xtest, ttest,
          start_epoch=0):
    # --- unpack configs ---
    T          = pde_cfg.T
    dimen      = pde_cfg.dimen
    epochs     = train_cfg.epochs
    bs         = train_cfg.batch_size
    n_inner    = train_cfg.n_inner
    n_batches  = train_cfg.num_batches
    bet        = train_cfg.bet_scale
    rho        = train_cfg.rho1
    alpha      = train_cfg.alpha
    m          = train_cfg.block_size
    sample_b   = train_cfg.sample_b
    w_bc       = train_cfg.w_bc
    w_int      = train_cfg.w_int
    use_paths  = train_cfg.use_path_sampling

    grad_clip   = getattr(train_cfg, 'grad_clip',     1.0)
    use_cosine  = getattr(train_cfg, 'use_cosine_lr', True)
    save_dir    = getattr(train_cfg, 'save_dir',      'results')
    save_every  = train_cfg.block_size

    use_terminal = (w_bc > 0.0)
    print(f"Mode: {'terminal loss' if use_terminal else 'INEUS without terminal penalty'}")
    print(f"Sampling: {'GBM paths' if use_paths else 'uniform'}")
    print(f"grad_clip={grad_clip}  cosine_lr={use_cosine}")
    print(f"Checkpoints → {save_dir}/  (every {save_every} epochs)")

    # --- build sampling helpers ---
    if use_paths:
        _x0  = eval_cfg.x0_value
        _mu  = pde_cfg.mu_bar_adjusted
        _sig = pde_cfg.sigma_bar
        _jkw = dict(
            jump_mean=pde_cfg.jump_mean,
            jump_std=pde_cfg.jump_std,
            jump_corr=pde_cfg.jump_corr,
        ) if getattr(pde_cfg, 'use_jumps', False) else {}
    
        def _sample_xt(n):
            if _jkw:
                return sample_xt_with_jumps(n, T, _x0, _mu, _sig, dimen, **_jkw)
            return sample_xt_paths(n, T, _x0, _mu, _sig, dimen)
    
        def _sample_x_term(n):
            if _jkw:
                return sample_x_terminal_with_jumps(n, T, _x0, _mu, _sig, dimen, **_jkw)
            return sample_x_terminal_paths(n, T, _x0, _mu, _sig, dimen)
        
    else:
        def _sample_xt(n):
            return sample_xt(n, T, sample_b, dimen)

        def _sample_x_term(n):
            return sample_x_terminal(n, sample_b, dimen)

        _rad_sample_fn = None

    # --- optimiser ---
    total_steps = epochs * (n_inner - 1) * n_batches

    if use_cosine:
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=train_cfg.lr,
            decay_steps=total_steps,
            alpha=0.02,
        )
    else:
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=train_cfg.lr,
            decay_steps=train_cfg.lr_decay_steps,
            decay_rate=train_cfg.lr_decay_rate,
            staircase=True,
        )

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    # --- tf constants ---
    h_var = tf.Variable(
        tf.zeros((bs // strategy.num_replicas_in_sync, 1),
                 dtype=tf.float32))
    
    w_bc_tf  = tf.constant(w_bc,      dtype=tf.float32)
    w_int_tf = tf.constant(w_int,     dtype=tf.float32)
    clip_val = tf.constant(grad_clip, dtype=tf.float32)

    # --- tracking ---
    history = dict(
        train_loss=[], pde_loss=[], term_loss=[],
        student_pred=[], student_error=[],
        student_test_loss=[], student_pde_loss=[], student_term_loss=[],
        distill_loss=[], distill_epoch=[], distill_coeffs=[],
        time=[],
    )

    block_start_w = copy_weights(model)
    block_epoch_w = []

    tic = timeit.default_timer()

    # ==============================================================
    # Compiled train steps
    # ==============================================================

    @tf.function
    def _step_terminal(x_batch, t_batch, x_term_batch):
        x_batch      = tf.cast(x_batch,      tf.float32)
        t_batch      = tf.cast(t_batch,      tf.float32)
        x_term_batch = tf.cast(x_term_batch, tf.float32)
        with tf.GradientTape() as tape:
            loss_pde  = pinn_loss(model, model_targ, x_batch, t_batch,
                                  bet, pde_cfg, h_var)
            loss_term = terminal_loss(model, x_term_batch, T,
                                      training=True)
            loss_total = w_int_tf * loss_pde + w_bc_tf * loss_term
        grads = tape.gradient(loss_total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_val)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_total, loss_pde, loss_term

    @tf.function
    def _step_hpinn(x_batch, t_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        with tf.GradientTape() as tape:
            loss = pinn_loss(model, model_targ, x_batch, t_batch,
                             bet, pde_cfg, h_var)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_val)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # ==============================================================
    # Fixed terminal test set
    # ==============================================================
    if use_terminal:
        xbctest = _sample_x_term(train_cfg.test_samples)

    # ==============================================================
    # MAIN LOOP
    # ==============================================================
    for epoch in range(start_epoch, epochs):

        # ── 0. Fresh collocation points every epoch ──
        if epoch == 0:
            replay_buffer.clear()
            x_new, t_new = _sample_xt(train_cfg.num_samples)
            replay_buffer.extend(x_new, t_new)
        else:
            # refresh 25% each epoch, keep 75% from previous epochs
            n_fresh = train_cfg.num_samples // 4   # 512 new points
            x_new, t_new = _sample_xt(n_fresh)
            replay_buffer.extend(x_new, t_new)     # earlier points pushed out via FIFO

        # ── 0b. RAD augmentation ──
        if (train_cfg.use_rad
                and epoch > 0
                and epoch % train_cfg.rad_period == 0):
            xr, tr, stats = rad_resample(
                model, pde_cfg, train_cfg,
                train_cfg.rad_eval_batch_size,
                sample_fn=_rad_sample_fn)
            replay_buffer.extend(xr, tr)
            print(f"[RAD @ epoch {epoch+1}] "
                  f"mean|r|={stats['mean_residual']:.6e}, "
                  f"max|r|={stats['max_residual']:.6e}")

        # --- 1. Target update ---
        if epoch == 0:
            model_targ.set_weights(model.get_weights())
        else:
            model_targ.set_weights([
                rho * wt + (1.0 - rho) * wc
                for wt, wc in zip(model_targ.get_weights(),
                                  model.get_weights())
            ])

        # --- 2. Inner loop ---
        total_loss = 0.0
        total_pde  = 0.0
        total_term = 0.0

        for _inner in range(n_inner - 1):
            for _batch in range(n_batches):
                xb, tb = replay_buffer.sample(bs)

                if use_terminal:
                    xt = _sample_x_term(bs)
                    r_tot, r_pde, r_term = strategy.run(
                        _step_terminal, args=(xb, tb, xt))
                    total_loss += float(strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, r_tot,
                        axis=None).numpy())
                    total_pde += float(strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, r_pde,
                        axis=None).numpy())
                    total_term += float(strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, r_term,
                        axis=None).numpy())
                else:
                    r = strategy.run(_step_hpinn, args=(xb, tb))
                    step_loss = float(strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, r,
                        axis=None).numpy())
                    total_loss += step_loss
                    total_pde  += step_loss

            model_targ.set_weights(model.get_weights())

        denom    = (n_inner - 1) * n_batches
        avg_loss = total_loss / denom
        avg_pde  = total_pde  / denom
        avg_term = total_term / denom
        
        # --- 2b. Evaluate teacher every epoch ---
        pred_teacher, err_teacher = evaluate_point(
            model, eval_cfg.x0_eval, eval_cfg.t0_eval, eval_cfg.u_ref)
        elapsed = timeit.default_timer() - tic

        err_str = f"|err|={err_teacher:.6e}" if err_teacher is not None else ""
        print(f"[Teacher @ epoch {epoch+1}] "
              f"u={pred_teacher:.6f}  {err_str}  "
              f"train_loss={avg_loss:.6e}  "
              f"PDE={avg_pde:.6e}  "
              f"Term={avg_term:.6e}  "
              f"[{elapsed:.0f}s]")

        # --- 3. Block distillation every m epochs ---
        block_epoch_w.append(copy_weights(model))

        if len(block_epoch_w) == m:
            student, coeffs, d_loss = distill_block(
                make_model_fn=make_model_fn,
                block_start_weights=block_start_w,
                block_epoch_weights=block_epoch_w,
                alpha=alpha, T=T,
                sample_b=sample_b, dimen=dimen,
                strategy=strategy,
                distill_steps=train_cfg.distill_steps,
                batch_size=bs,
                student_lr=train_cfg.student_lr,
                sample_fn=_sample_xt,
            )

            # ── Evaluate student ──
            pred_student, err_student = evaluate_point(
                student, eval_cfg.x0_eval, eval_cfg.t0_eval,
                eval_cfg.u_ref)

            pde_student = float(
                residual_pointwise(student, xtest, ttest,
                                   pde_cfg)[1].numpy())
            if use_terminal:
                term_student = float(
                    terminal_loss(student, xbctest, T,
                                  training=False).numpy())
                tot_student = w_int * pde_student + w_bc * term_student
            else:
                term_student = 0.0
                tot_student  = pde_student

            elapsed = timeit.default_timer() - tic

            # ── Record history ──
            history['train_loss'].append(avg_loss)
            history['pde_loss'].append(avg_pde)
            history['term_loss'].append(avg_term)
            history['student_pred'].append(pred_student)
            history['student_error'].append(
                np.nan if err_student is None else err_student)
            history['student_test_loss'].append(tot_student)
            history['student_pde_loss'].append(pde_student)
            history['student_term_loss'].append(term_student)
            history['distill_loss'].append(float(d_loss.numpy()))
            history['distill_epoch'].append(epoch + 1)
            history['distill_coeffs'].append(coeffs)
            history['time'].append(elapsed)

            # ── Print ──
            err_str = f"|err|={err_student:.6e}" if err_student is not None else ""
            print(f"[Student @ epoch {epoch+1}] "
                  f"u={pred_student:.6f}  "
                  f"{err_str}  "
                  f"test_loss={tot_student:.6e}  "
                  f"PDE={pde_student:.6e}  "
                  f"Term={term_student:.6e}  "
                  f"distill_loss={float(d_loss.numpy()):.6e}  "
                  f"[{elapsed:.0f}s]")

            # ── Replace model with student ──
            model.set_weights(student.get_weights())
            model_targ.set_weights(student.get_weights())

            # ── Save checkpoint ──
            save_checkpoint(
                os.path.join(save_dir, f'epoch_{epoch+1:04d}.pkl'),
                model_weights=copy_weights(model),
                history=history,
                epoch=epoch + 1,
            )
            save_history_json(
                history,
                os.path.join(save_dir, 'history.json'),
            )

            block_start_w = copy_weights(model)
            block_epoch_w = []

    # ==============================================================
    # Final save
    # ==============================================================
    save_checkpoint(
        os.path.join(save_dir, 'final.pkl'),
        model_weights=copy_weights(model),
        history=history,
        epoch=epochs,
    )

    return dict(
        model_weights=copy_weights(model),
        history=history,
    )
