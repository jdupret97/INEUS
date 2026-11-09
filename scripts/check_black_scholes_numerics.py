"""Small CPU checks for the nonlinear Black--Scholes implementation.

Run ``python scripts/check_black_scholes_numerics.py`` from the repository root.
Checks the proposition-based operator against analytic derivatives, sampling,
the jump estimator, terminal constraints, and two tiny training runs including
distillation and checkpoints. This is not a convergence or benchmark test.
Temporary training outputs are removed when the checks finish.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf

from ineus.black_scholes.averaging import block_coeffs
from ineus.black_scholes.config import EvalConfig, ModelConfig, PDEConfig, TrainConfig
from ineus.black_scholes.models import build_model, phi_tf
from ineus.black_scholes.problem import (
    Q_tf, _psi, default_reaction, jump_nonlocal, pinn_loss,
    residual_pointwise, second_derivative_wrt_h, terminal_loss,
)
from ineus.black_scholes.replay_buffer import ReplayBuffer
from ineus.black_scholes.references import bender_denk_picard, glw_bsde
from ineus.black_scholes.sampling import (
    sample_x_terminal, sample_x_terminal_paths,
    sample_x_terminal_with_jumps, sample_xt, sample_xt_paths, sample_xt_with_jumps,
)
from ineus.black_scholes.trainer import load_checkpoint, train
from ineus.black_scholes.utils import evaluate_point, set_seeds


class PolynomialValue(tf.keras.Model):
    """Smooth test function with nonzero time, diagonal and mixed derivatives."""

    def call(self, inputs, training=False):
        x, t = inputs
        return (tf.square(t) + tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
                + 0.3 * t * tf.reduce_sum(x, axis=1, keepdims=True)
                + 0.4 * x[:, :1] * x[:, 1:2])


class CorrelatedPDEConfig(PDEConfig):
    """Operator-test coefficients with mixed diffusion covariance."""

    @property
    def Sigma(self):
        return np.array([[1.0, 0.2, -0.1], [0.2, 1.5, 0.3],
                         [-0.1, 0.3, 2.0]], dtype=np.float32)


def check_operator():
    cfg = CorrelatedPDEConfig(dimen=3, use_jumps=False)
    x = np.array([[2.0, 3.0, 4.0], [4.0, 1.0, 2.0],
                  [1.0, 1.5, 2.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    t = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    model = PolynomialValue()
    h = tf.Variable(tf.zeros((len(x), 1), dtype=tf.float32))
    computed = second_derivative_wrt_h(lambda h_: _psi(h_, model, x, t, cfg), h)
    grad = 2.0 * x + 0.3 * t
    grad[:, 0] += 0.4 * x[:, 1]
    grad[:, 1] += 0.4 * x[:, 0]
    expected = (2.0 * t[:, 0] + 0.3 * x.sum(axis=1)
                + cfg.mu_bar_adjusted * np.sum(x * grad, axis=1)
                + np.sum(np.square(x * cfg.sigma_diag) * np.diag(cfg.Sigma), axis=1)
                + 0.4 * x[:, 0] * x[:, 1] * cfg.sigma_diag[0]
                * cfg.sigma_diag[1] * cfg.Sigma[0, 1])
    np.testing.assert_allclose(computed.numpy().ravel(), expected, rtol=2e-6, atol=2e-6)
    values = np.array([40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
    np.testing.assert_allclose(Q_tf(values, cfg), [0.2, 0.2, 0.11, 0.02, 0.02],
                               rtol=2e-6, atol=2e-6)
    reaction = default_reaction(model([x, t]), cfg).numpy().ravel()
    residual, mae = residual_pointwise(model, x, t, cfg)
    np.testing.assert_allclose(residual, expected + reaction, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(mae, np.abs(expected + reaction).mean(), rtol=2e-6)
    beta = 0.05
    loss = pinn_loss(model, model, x, t, beta, cfg, h)
    np.testing.assert_allclose(loss, np.mean(np.square(beta * (expected + reaction))),
                               rtol=2e-5, atol=2e-6)
    print("PASS analytic Black--Scholes operator, reaction, residual and Euler regression loss")


def check_jump_estimator():
    cfg = PDEConfig(dimen=3, use_jumps=True)
    model = PolynomialValue()
    x = tf.constant([[2.0, 3.0, 4.0], [4.0, 2.0, 1.0]], dtype=tf.float32)
    t = tf.constant([[0.2], [0.4]], dtype=tf.float32)
    set_seeds(271)
    common = tf.random.normal((2, 1), dtype=tf.float32)
    independent = tf.random.normal((2, 3), dtype=tf.float32)
    z = cfg.jump_mean + cfg.jump_std * (
        tf.sqrt(cfg.jump_corr) * common + tf.sqrt(1.0 - cfg.jump_corr) * independent)
    expected = cfg.jump_intensity * (model([x * tf.exp(z), t]) - model([x, t]))
    set_seeds(271)
    np.testing.assert_allclose(jump_nonlocal(model, x, t, cfg), expected, rtol=1e-6, atol=1e-6)
    print("PASS correlated single-draw nonlocal jump estimator")


def check_models_and_sampling():
    cfg = PDEConfig(dimen=3, use_jumps=False)
    training = TrainConfig(batch_size=4, num_samples=8, n_inner=2)
    set_seeds(173)
    x, t = sample_xt(4, cfg.T, training.sample_b, cfg.dimen)
    np.random.seed(173)
    expected_t = np.random.uniform(0.0, cfg.T, 4).reshape(-1, 1).astype(np.float32)
    expected_x = np.random.uniform(0.0, training.sample_b, (4, 3)).astype(np.float32)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(t, expected_t)
    set_seeds(174)
    xp, tp = sample_xt_paths(4, cfg.T, 100.0, cfg.mu_bar_adjusted, cfg.sigma_bar, 3)
    np.random.seed(174)
    expected_tp = np.random.uniform(0.0, cfg.T, 4).reshape(-1, 1).astype(np.float32)
    noise = np.random.randn(4, 3).astype(np.float32)
    expected_xp = 100.0 * np.exp((cfg.mu_bar_adjusted - 0.5 * cfg.sigma_bar**2)
                                * expected_tp + cfg.sigma_bar * np.sqrt(expected_tp) * noise)
    np.testing.assert_array_equal(tp, expected_tp)
    np.testing.assert_array_equal(xp, expected_xp)
    xj, tj = sample_xt_with_jumps(8, cfg.T, 100.0, cfg.mu_bar_adjusted, cfg.sigma_bar, 3)
    assert xj.shape == (8, 3) and tj.shape == (8, 1) and np.all(xj > 0)
    for terminal_x in (
        sample_x_terminal(8, 200.0, 3),
        sample_x_terminal_paths(8, cfg.T, 100.0, cfg.mu_bar_adjusted, cfg.sigma_bar, 3),
        sample_x_terminal_with_jumps(8, cfg.T, 100.0, cfg.mu_bar_adjusted, cfg.sigma_bar, 3),
    ):
        assert terminal_x.shape == (8, 3) and np.isfinite(terminal_x).all()
    model = build_model(ModelConfig(hidden_dim=4, n_layers=2, constrained=True), cfg, training)
    np.testing.assert_array_equal(model([x, np.ones_like(t) * cfg.T]), phi_tf(x))
    assert float(terminal_loss(model, x, cfg.T)) == 0.0
    _, error = evaluate_point(model, x[:1], t[:1], u_ref=None)
    assert error is None
    coeffs = block_coeffs(0.8, 2)
    np.testing.assert_allclose(coeffs, [0.04, 0.16, 0.8], atol=1e-14)
    assert abs(sum(coeffs) - 1.0) < 1e-14
    replay = ReplayBuffer(4)
    replay.extend(x[:3], t[:3])
    replay.extend(x[2:], t[2:])
    assert len(replay) == 4
    assert replay.sample(3)[0].shape == (3, 3)
    replay.clear()
    assert len(replay) == 0
    print("PASS seeded samplers, terminal constraint, Polyak coefficients and replay buffer")


def check_reference_solvers():
    """Test both regression routines on a deterministic, constant-rate equation."""
    cfg = PDEConfig(dimen=3, use_jumps=False, sigma_bar=0.0,
                    gamma_h=0.02, gamma_l=0.02)
    rate = (1.0 - cfg.delta_recovery) * cfg.gamma_l + cfg.R_rate
    payoff = 100.0 * np.exp(cfg.mu_bar_adjusted * cfg.T)
    settings = dict(x0=100.0, n_paths=32, n_steps=3, n_reg_per_step=32, seed=731)
    glw_value, glw_spread = glw_bsde(cfg, **settings)
    picard_value, picard_spread = bender_denk_picard(cfg, n_picard=2, **settings)
    np.testing.assert_allclose(glw_value, payoff * (1.0 - rate * cfg.T / 3.0)**3,
                               rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(picard_value, payoff * np.exp(-rate * cfg.T),
                               rtol=5e-5, atol=5e-5)
    assert np.isfinite([glw_spread, picard_spread]).all()
    print("PASS regression solvers against deterministic constant-rate values")


def check_training(output_dir: Path, *, constrained: bool):
    tf.keras.backend.clear_session()
    set_seeds(731)
    pde = PDEConfig(dimen=3, use_jumps=not constrained)
    model_cfg = ModelConfig(hidden_dim=4, n_layers=2, constrained=constrained)
    train_cfg = TrainConfig(
        epochs=2, batch_size=4, num_samples=8, buffer_size=16,
        n_inner=2, num_batches=1, block_size=2, distill_steps=1,
        test_samples=4, w_bc=0.0 if constrained else 1.0,
        use_path_sampling=not constrained, use_rad=constrained,
        rad_period=1, rad_candidate_size=16, rad_eval_batch_size=4,
        save_dir=str(output_dir),
    )
    evaluation = EvalConfig(dimen=3, u_ref=57.3)
    strategy = tf.distribute.OneDeviceStrategy("/CPU:0")

    def factory():
        return build_model(model_cfg, pde, train_cfg)

    with strategy.scope():
        model = factory()
        target = factory()
    x, t = sample_xt_paths(4, pde.T, 100.0, pde.mu_bar_adjusted, pde.sigma_bar, 3)
    initial = [np.array(weight, copy=True) for weight in model.get_weights()]
    result = train(model, target, factory, pde, train_cfg, evaluation,
                   ReplayBuffer(train_cfg.buffer_size), strategy, x, t)
    assert result["history"]["distill_epoch"] == [2]
    for name, values in result["history"].items():
        assert np.isfinite(np.asarray(values, dtype=float)).all(), name
    assert any(not np.array_equal(a, b) for a, b in zip(initial, result["model_weights"]))
    assert {item.name for item in output_dir.iterdir()} == {
        "epoch_0002.pkl", "final.pkl", "history.json"}
    for filename in ("epoch_0002.pkl", "final.pkl"):
        checkpoint = load_checkpoint(str(output_dir / filename))
        assert checkpoint["epoch"] == 2
        for actual, expected in zip(checkpoint["model_weights"], model.get_weights()):
            np.testing.assert_array_equal(actual, expected)
    with (output_dir / "history.json").open(encoding="utf-8") as handle:
        saved_history = json.load(handle)
    assert saved_history["distill_epoch"] == [2.0]
    for actual, expected in zip(target.get_weights(), model.get_weights()):
        np.testing.assert_array_equal(actual, expected)
    if constrained:
        assert float(terminal_loss(model, x, pde.T)) == 0.0
    variant = "hard terminal + uniform + RAD" if constrained else "soft terminal + paths + jumps"
    print(f"PASS two-epoch training, distillation and checkpoints ({variant})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true",
                        help="Run operator, model and sampling checks only.")
    args = parser.parse_args()
    check_operator()
    check_jump_estimator()
    check_models_and_sampling()
    check_reference_solvers()
    if not args.skip_training:
        results_root = ROOT / "results"
        results_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix="check_black_scholes_", dir=results_root) as temporary:
            base = Path(temporary)
            check_training(base / "soft_terminal", constrained=False)
            check_training(base / "hard_terminal", constrained=True)
    print("All Black--Scholes numerical checks passed.")


if __name__ == "__main__":
    main()
