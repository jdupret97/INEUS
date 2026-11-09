"""Small CPU checks of configurable nonlinear HJB coefficients and dimensions."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None

if tf is not None:
    import numpy as np

    from ineus.nonlinear.config import ProblemConfig, default_config
    from ineus.nonlinear.experiment import build_fixed_test_set
    from ineus.nonlinear.pde import (
        compute_second_derivative_term, pinn_loss1bis, pinn_residual_pointwise_nl,
    )
    from ineus.nonlinear.problem import (
        COV_J, COV_J_CHOL, DEFAULT_PROBLEM, DIMENSION, MU_J,
        g_fin, make_problem, u_pairs,
    )
    from ineus.nonlinear.sampling import sample_xt, sample_jump_marks_mc_local_nl


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class NonlinearProblemRuntimeTests(unittest.TestCase):
    def setUp(self):
        cpu = tf.device("/CPU:0")
        cpu.__enter__()
        self.addCleanup(cpu.__exit__, None, None, None)

    @staticmethod
    def problem(d, **kwargs):
        return make_problem(ProblemConfig(state_dim=d, jump_dim=d, **kwargs))

    @staticmethod
    def polynomial():
        class Polynomial(tf.keras.Model):
            def call(self, inputs, training=False):
                x, t = inputs
                return 0.17 * tf.reduce_sum(x * x, axis=1, keepdims=True) + 0.31 * t
        return Polynomial()

    def points(self, d):
        x = tf.constant(np.linspace(-0.3, 0.4, 2 * d).reshape(2, d), tf.float32)
        return x, tf.constant([[0.1], [0.2]], tf.float32)

    def expected_local(self, x, problem):
        return 0.31 + 0.34 * problem.state_dim - problem.eta * 0.34**2 * np.sum(x.numpy()**2, axis=1) + problem.driver_constant

    def test_no_jump_operator_matches_polynomial_in_3d_and_7d(self):
        for d in (3, 7):
            for eta in (0.6, 1.8):
                with self.subTest(d=d, eta=eta):
                    problem = self.problem(d, eta=eta, jump_intensity=0.0, driver_constant=-0.4)
                    x, t = self.points(d)
                    model = self.polynomial()
                    with patch.object(tf.random, "normal", side_effect=AssertionError("no jump draws")):
                        actual, _, _ = pinn_residual_pointwise_nl(model, x, t, problem=problem)
                    np.testing.assert_allclose(actual.numpy(), self.expected_local(x, problem), rtol=2e-6)
                    loss = pinn_loss1bis(model, model, x, t, tf.zeros_like(x), 0.2, problem=problem)
                    expected_loss = np.mean((0.2 * self.expected_local(x, problem))**2)
                    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=3e-6)

    def test_sampled_jump_operator_matches_direct_polynomial_formula(self):
        for d in (3, 7):
            with self.subTest(d=d):
                problem = self.problem(d, eta=0.7, jump_intensity=1.2, jump_covariance_scale=0.11, driver_constant=0.9)
                x, t = self.points(d)
                model = self.polynomial()
                tf.random.set_seed(71)
                z = tf.random.normal((2, 9, d), dtype=tf.float32)
                marks = tf.linalg.matmul(z, problem.cov_j_chol, transpose_b=True)
                delta = 0.17 * (tf.reduce_sum((x[:, None, :] + marks)**2, axis=2) - tf.reduce_sum(x*x, axis=1)[:, None])
                jump = -problem.jump_intensity / problem.eta * (tf.reduce_mean(tf.exp(-problem.eta * delta), axis=1) - 1.0)
                tf.random.set_seed(71)
                actual = pinn_residual_pointwise_nl(model, x, t, K=9, problem=problem)[0]
                np.testing.assert_allclose(actual.numpy(), self.expected_local(x, problem) + jump.numpy(), rtol=3e-6)

    def test_default_context_preserves_legacy_calls_and_random_draws(self):
        x, t = self.points(10)
        model = self.polynomial()
        context = make_problem()
        tf.random.set_seed(43)
        legacy = pinn_residual_pointwise_nl(model, x, t, T=1.0, K=7)[0]
        tf.random.set_seed(43)
        explicit = pinn_residual_pointwise_nl(model, x, t, T=1.0, K=7, problem=context)[0]
        np.testing.assert_array_equal(legacy.numpy(), explicit.numpy())
        np.random.seed(67)
        legacy_x, legacy_t = sample_xt(4, 1.0)
        np.random.seed(67)
        active_x, active_t = sample_xt(4, problem=context)
        np.testing.assert_array_equal(legacy_x, active_x)
        np.testing.assert_array_equal(legacy_t, active_t)
        legacy_ref = u_pairs(x, t, 1.0, 1.0, 0.5, MU_J, COV_J, g_fin, n_paths=80, paths_chunk=20, seed=88)
        active_ref = u_pairs(x, t, n_paths=80, paths_chunk=20, seed=88, problem=context)
        np.testing.assert_array_equal(legacy_ref, active_ref)

    def test_interleaved_contexts_do_not_change_default_constants(self):
        p3 = self.problem(3, eta=0.4, driver_constant=0.0)
        p7 = self.problem(7, eta=1.4, jump_covariance_scale=0.0)
        model = self.polynomial()
        x, t = self.points(3)
        h = tf.zeros((2, 1), tf.float32)
        first = compute_second_derivative_term(model, x, t, h, problem=p3)
        sample_xt(2, problem=p7)
        second = compute_second_derivative_term(model, x, t, h, problem=p3)
        np.testing.assert_array_equal(first.numpy(), second.numpy())
        self.assertEqual(DIMENSION, 10)
        self.assertEqual(MU_J.shape, (10,))
        self.assertEqual(COV_J_CHOL.shape, (10, 10))
        self.assertEqual(DEFAULT_PROBLEM.eta, 1.0)
        self.assertFalse(p3.cov_j.flags.writeable)
        np.testing.assert_array_equal(sample_jump_marks_mc_local_nl(2, 3, problem=p7).numpy(), np.zeros((2, 3, 7)))

    def test_reference_uses_context_and_preserves_test_sampling_seeds(self):
        config = default_config()
        config = replace(config, problem=ProblemConfig(state_dim=3, jump_dim=3, horizon=0.4, eta=0.8, jump_intensity=0.0, jump_covariance_scale=0.0, driver_constant=1.7), evaluation=replace(config.evaluation, test_samples=3, reference_paths=30000, reference_chunk_size=3000))
        x, t, values = build_fixed_test_set(config)
        repeated = build_fixed_test_set(config)
        for first, second in zip((x, t, values), repeated):
            np.testing.assert_array_equal(first, second)
        dt = config.problem.horizon - t[:, 0]
        eta = config.problem.eta
        denominator = 1.0 + 4.0 * eta * dt
        exact = config.problem.driver_constant * dt + 3.0 / (2.0 * eta) * np.log(denominator) + np.sum(x*x, axis=1) / denominator
        np.testing.assert_allclose(values[:, 0], exact, atol=0.035, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
