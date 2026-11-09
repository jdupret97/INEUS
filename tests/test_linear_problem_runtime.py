"""Small CPU regression checks for isolated quadratic-linear problem contexts."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None

if tf is not None:
    import numpy as np

    from ineus.linear.architectures.bsde import ClassicalLocalDeepBSDEJump
    from ineus.linear.architectures.surrogate import GlobalValueSurrogateFromLocalBSDE
    from ineus.linear.architectures.value import make_model
    from ineus.linear.config import DEFAULT_CONFIG, ProblemConfig
    from ineus.linear.experiment import make_fixed_test_set
    from ineus.linear.pde import compute_second_derivative, pinn_residual_pointwise
    from ineus.linear.problem import DEFAULT_PROBLEM, make_problem
    from ineus.linear.sampling import ReplayBuffer, sample_jump_marks, sample_jump_marks_mc, sample_xt
    from ineus.linear.training.ineus import train_bellman_ct
    from ineus.linear.training.pinn import train_standard_pinn


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class LinearProblemRuntimeTests(unittest.TestCase):
    def setUp(self):
        cpu = tf.device("/CPU:0")
        cpu.__enter__()
        self.addCleanup(cpu.__exit__, None, None, None)

    def test_default_coefficients_match_original_draw_order_exactly(self):
        cfg = ProblemConfig()
        rng = np.random.RandomState(cfg.coefficient_seed)
        sigma = rng.normal(0.0, cfg.diffusion_scale, (10, 10)).astype(np.float32)
        covariance = np.diag(rng.uniform(
            cfg.jump_covariance_min, cfg.jump_covariance_scale, 10,
        )).astype(np.float32)
        problem = make_problem(cfg)
        np.testing.assert_array_equal(problem.sigma, sigma)
        np.testing.assert_array_equal(problem.cov_j, covariance)
        np.testing.assert_array_equal(problem.cov_j_chol, np.linalg.cholesky(covariance))
        np.testing.assert_array_equal(DEFAULT_PROBLEM.sigma, sigma)
        self.assertFalse(problem.sigma.flags.writeable)

    def test_problem_creation_does_not_consume_numpy_or_tensorflow_rng(self):
        np.random.seed(71)
        tf.random.set_seed(71)
        expected_np = np.random.normal(size=8)
        expected_tf = tf.random.normal((8,))
        np.random.seed(71)
        tf.random.set_seed(71)
        make_problem(replace(ProblemConfig(), state_dim=3, brownian_dim=2))
        np.testing.assert_array_equal(np.random.normal(size=8), expected_np)
        tf.debugging.assert_equal(tf.random.normal((8,)), expected_tf)

    def test_explicit_paper_matrix_modes(self):
        problem = make_problem(replace(
            ProblemConfig(), state_dim=3, brownian_dim=2,
            diffusion_mode="constant", diffusion_scale=0.4,
            jump_covariance_mode="isotropic", jump_covariance_scale=0.7,
        ))
        np.testing.assert_array_equal(problem.sigma, np.full((3, 2), 0.4, np.float32))
        np.testing.assert_array_equal(problem.cov_j, 0.7 * np.eye(3, dtype=np.float32))

    def test_sampling_and_test_references_use_active_context(self):
        cfg = replace(
            DEFAULT_CONFIG, problem=replace(
                ProblemConfig(), state_dim=3, brownian_dim=2, horizon=1.2,
                sample_box=0.3, test_box=0.2, drift=0.0, discount=0.4,
                jump_intensity=0.0,
            ), evaluation=replace(DEFAULT_CONFIG.evaluation, test_samples=7),
        )
        problem = make_problem(cfg.problem)
        x, t = sample_xt(9, problem=problem)
        self.assertEqual(x.shape, (9, 3))
        self.assertTrue(np.all(np.abs(x) <= 0.3))
        self.assertTrue(np.all((t >= 0) & (t <= 1.2)))
        self.assertEqual(sample_jump_marks(4, problem=problem).shape, (4, 3))
        self.assertEqual(sample_jump_marks_mc(4, 5, problem=problem).shape, (4, 5, 3))
        x, t, values = make_fixed_test_set(cfg, problem=problem)
        self.assertEqual(x.shape, (7, 3))
        tau = problem.T - t
        expected = np.exp(-problem.c * tau) * (
            np.sum(x * x, axis=1, keepdims=True) + np.sum(problem.sigma**2) * tau
        )
        np.testing.assert_allclose(values, expected, rtol=2e-6)

    def test_default_test_set_retains_seeded_sampling_sequence(self):
        cfg = replace(DEFAULT_CONFIG, evaluation=replace(DEFAULT_CONFIG.evaluation, test_samples=5))
        np.random.seed(123)
        t = np.random.uniform(0.0, 0.5, (5, 1)).astype(np.float32)
        x = np.random.uniform(-1.5, 1.5, (5, 10)).astype(np.float32)
        next_draw = np.random.normal(size=4)
        np.random.seed(123)
        actual_x, actual_t, _ = make_fixed_test_set(cfg)
        np.testing.assert_array_equal(actual_x, x)
        np.testing.assert_array_equal(actual_t, t)
        np.testing.assert_array_equal(np.random.normal(size=4), next_draw)

    def test_polynomial_directional_operator_for_unequal_dimensions(self):
        for d, q, beta in ((3, 2, 0.7), (2, 4, 0.0), (1, 3, -0.3)):
            with self.subTest(d=d, q=q, beta=beta):
                problem = make_problem(replace(
                    ProblemConfig(), state_dim=d, brownian_dim=q, drift=beta,
                ))
                weights = tf.constant(np.arange(1, d + 1, dtype=np.float32)[None, :])
                x = tf.constant(np.linspace(-0.7, 0.9, 4 * d).reshape(4, d), tf.float32)
                t = tf.constant([[0.1], [0.2], [0.3], [0.4]])

                def polynomial(inputs, training=False):
                    x_val, t_val = inputs
                    return tf.reduce_sum(weights * x_val**2, axis=1, keepdims=True) + t_val**2

                expected = (
                    2.0 * t + 2.0 * beta * tf.reduce_sum(weights * x**2, axis=1, keepdims=True)
                    + tf.reduce_sum(weights * tf.reduce_sum(problem.sigma_tf**2, axis=1))
                )
                actual = compute_second_derivative(
                    polynomial, x, t, tf.zeros_like(t), problem=problem,
                )
                tf.debugging.assert_near(actual, expected, atol=2e-6, rtol=2e-6)

    def test_exact_no_jump_solution_has_zero_residual_without_jump_draws(self):
        problem = make_problem(replace(
            ProblemConfig(), state_dim=2, brownian_dim=3, horizon=0.8,
            drift=0.0, discount=0.9, jump_intensity=0.0,
            jump_covariance_min=0.0, jump_covariance_scale=0.0,
        ))

        def exact(inputs, training=False):
            x, t = inputs
            tau = problem.T_tf - t
            return tf.exp(-problem.c_tf * tau) * (
                tf.reduce_sum(x * x, axis=1, keepdims=True)
                + tf.reduce_sum(problem.sigma_tf**2) * tau
            )

        x = tf.constant([[0.2, -0.3], [0.4, 0.1]])
        t = tf.constant([[0.1], [0.6]])
        with patch("tensorflow.random.normal", side_effect=AssertionError("no jump RNG")):
            residual, _, _ = pinn_residual_pointwise(exact, x, t, K=0, problem=problem)
        np.testing.assert_allclose(residual.numpy(), 0.0, atol=2e-6)
        np.testing.assert_allclose(
            problem.exact_value(t, x), exact([x, t]).numpy().reshape(-1), rtol=2e-6,
        )

    def test_jump_residual_uses_active_covariance_intensity_and_dimension(self):
        problem = make_problem(replace(
            ProblemConfig(), state_dim=2, brownian_dim=3, drift=0.3,
            discount=0.6, jump_intensity=0.8, jump_covariance_mode="isotropic",
            jump_covariance_scale=0.25,
        ))

        def quadratic(inputs, training=False):
            x, t = inputs
            return tf.reduce_sum(x * x, axis=1, keepdims=True) + t**2

        x = tf.constant([[0.2, -0.3], [0.4, 0.1]])
        t = tf.constant([[0.1], [0.2]])
        # Equal positive/negative marks have mean zero and covariance .25 I.
        marks = tf.constant([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
        with patch("tensorflow.random.normal", return_value=marks):
            actual, _, _ = pinn_residual_pointwise(quadratic, x, t, K=2, problem=problem)
        expected = (
            2 * t + 2 * problem.beta * tf.reduce_sum(x*x, axis=1, keepdims=True)
            + tf.reduce_sum(problem.sigma_tf**2) - problem.c * quadratic([x, t])
            + problem.lam * tf.linalg.trace(tf.constant(problem.cov_j))
        )
        np.testing.assert_allclose(actual.numpy(), expected.numpy().reshape(-1), rtol=2e-6)

    def test_bsde_and_surrogate_use_nondefault_problem(self):
        for intensity in (0.0, 0.4):
            with self.subTest(intensity=intensity):
                problem = make_problem(replace(
                    ProblemConfig(), state_dim=2, brownian_dim=3,
                    horizon=1.3, sample_box=0.6, jump_intensity=intensity,
                ))
                local = ClassicalLocalDeepBSDEJump(
                    0.2, [0.1, -0.2], 2, 3, N_time=1, hidden_dim_z=2,
                    n_hidden_z=1, hidden_dim_u=2, n_hidden_u=1,
                    K_jump_expect=2, problem=problem,
                )
                loss = local.bsde_loss(2)
                self.assertTrue(np.isfinite(float(loss.numpy())))
                self.assertIs(local.z_net.problem, problem)
                self.assertIs(local.u_jump_net.problem, problem)
                model = GlobalValueSurrogateFromLocalBSDE(
                    2, hidden_dim=2, n_hidden=1, problem=problem,
                )
                x = tf.constant([[0.2, 0.3]])
                tf.debugging.assert_equal(
                    model((tf.constant([[1.3]]), x)),
                    tf.reduce_sum(x*x, axis=1, keepdims=True),
                )
        self.assertEqual(DEFAULT_PROBLEM.dimen, 10)
        self.assertEqual(DEFAULT_PROBLEM.T, 0.5)

    def test_one_step_trainers_distillation_and_rad_keep_the_active_problem(self):
        problem = make_problem(replace(
            ProblemConfig(), state_dim=2, brownian_dim=3, horizon=0.8,
            jump_intensity=0.0, drift=0.0,
        ))

        def factory():
            return make_model(hidden_dim=2, n_layers=1, T=problem.T, dimen=problem.dimen)

        x, t = sample_xt(2, problem=problem)
        values = problem.exact_value(t, x).reshape(-1, 1)
        replay = ReplayBuffer(8)
        replay.extend(x, t)
        common = dict(
            T=problem.T, epochs=1, num_samples=2, lr=1e-3,
            n_inner=2, num_batches_N1=1, replay_buffer=replay,
            xtest=x, ttest=t, values=values, batch_size=2,
            use_rad=True, rad_period=1, rad_candidate_size=4,
            rad_eval_batch_size=2, log_every=1, K_test=2, K_rad=2,
            problem=problem,
        )
        with patch("builtins.print"):
            ineus = train_bellman_ct(
                factory(), factory(), factory, rho1=0.0, h_scale=0.5,
                zeta=0.4, m=1, distill_steps=1, **common,
            )
            pinn = train_standard_pinn(factory(), K_train=2, **common)
        self.assertTrue(np.isfinite(ineus["best_loss"]))
        self.assertTrue(np.isfinite(pinn["best_loss"]))
        self.assertEqual(ineus["history"]["epoch"], [1])
        self.assertEqual(pinn["history"]["epoch"], [1])
        self.assertEqual(replay.sample_batch(2)[0].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
