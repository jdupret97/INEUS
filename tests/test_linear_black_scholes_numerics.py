"""Small CPU checks for the basket-call equation, architecture, and samplers."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

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

    from ineus.linear_black_scholes.config import ProblemConfig
    from ineus.linear_black_scholes.models import make_model
    from ineus.linear_black_scholes.pde import fprime_bs, second_derivative_wrt_h
    from ineus.linear_black_scholes.problem import make_problem
    from ineus.linear_black_scholes.references import (
        mc_basket_call_jumpdiff,
        mc_prices_on_grid,
    )
    from ineus.linear_black_scholes.sampling import sample_xt_paths


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class LinearBlackScholesNumericsTests(unittest.TestCase):
    def setUp(self):
        cpu = tf.device("/CPU:0")
        cpu.__enter__()
        self.addCleanup(cpu.__exit__, None, None, None)
        self.config = ProblemConfig()
        self.problem = make_problem(2, self.config)

    def make_network(self, problem=None, n_layers=3, constrained=True):
        problem = self.problem if problem is None else problem
        return make_model(
            hidden_dim_val=4,
            n_layers_val=n_layers,
            T_val=problem.T,
            dimen_val=problem.dimen,
            constrained=constrained,
            x_min=0.0,
            x_max=problem.max_S,
            problem=problem,
        )

    def reference_kwargs(self):
        problem = self.problem
        return dict(
            T_val=problem.T,
            w_val=problem.w,
            K_val=problem.K,
            r_val=problem.r,
            sigma=problem.sigma_diag,
            Sigma_val=problem.Sigma,
            lam=problem.lam_vec,
            muZ=problem.mu_j,
            sigZ=problem.sig_j,
        )

    def test_problem_coefficients_and_contexts_are_dimension_local(self):
        problem = self.problem
        model = self.make_network()
        x = tf.constant([[25.0, 45.0]])
        t = tf.constant([[0.2]])
        prediction = model([x, t]).numpy()
        another = make_problem(5, replace(self.config, K_multiplier=0.5))
        self.make_network(another)

        self.assertEqual(problem.dimen, 2)
        self.assertEqual(another.dimen, 5)
        self.assertEqual(problem.K, 30.0)
        self.assertEqual(another.K, 50.0)
        self.assertEqual(problem.Sigma.shape, (2, 2))
        np.testing.assert_array_equal(problem.Sigma, np.array([[1.0, 0.1], [0.1, 1.0]], np.float32))
        np.testing.assert_array_equal(problem.lam_vec, np.full(2, 0.5, np.float32))
        np.testing.assert_array_equal(problem.w, np.full(2, 0.5, np.float32))
        np.testing.assert_array_equal(problem.S0_train, np.full(2, 30.0, np.float32))
        np.testing.assert_array_equal(model([x, t]).numpy(), prediction)
        tf.debugging.assert_equal(model([x, tf.ones_like(t)]), [[5.0]])

    def test_dgm_has_independent_stage_gates_with_expected_biases(self):
        shallow = self.make_network(n_layers=2, constrained=False)
        deep = self.make_network(n_layers=5, constrained=False)
        # Each added stage has four biased input projections and four
        # unbiased state projections: 4 * ((d + 1) * width + width + width**2).
        stage_params = 4 * (3 * 4 + 4 + 4**2)
        self.assertEqual(deep.count_params() - shallow.count_params(), 3 * stage_params)
        all_variables = []
        for name in ("Uz", "Ug", "Ur", "Uh"):
            gates = getattr(deep, name)
            self.assertEqual(len(gates), 4)
            self.assertEqual(len({id(gate) for gate in gates}), 4)
            for gate in gates:
                self.assertIsInstance(gate, tf.keras.layers.Dense)
                self.assertTrue(gate.use_bias)
                all_variables.extend(gate.trainable_variables)
        for name in ("Wsz", "Wsg", "Wsr", "Wsh"):
            gates = getattr(deep, name)
            self.assertEqual(len(gates), 4)
            self.assertEqual(len({id(gate) for gate in gates}), 4)
            for gate in gates:
                self.assertIsInstance(gate, tf.keras.layers.Dense)
                self.assertFalse(gate.use_bias)
                all_variables.extend(gate.trainable_variables)
        self.assertEqual(len({id(variable) for variable in all_variables}), len(all_variables))
        before = deep.Uz[1].kernel.numpy().copy()
        deep.Uz[0].kernel.assign_add(tf.ones_like(deep.Uz[0].kernel))
        np.testing.assert_array_equal(deep.Uz[1].kernel.numpy(), before)

    def test_all_independent_gate_variables_receive_gradients(self):
        model = self.make_network(constrained=False)
        x = tf.constant([[25.0, 75.0], [10.0, 40.0]])
        t = tf.constant([[0.2], [0.7]])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model([x, t]) ** 2)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertEqual(len(gradients), 4 + 12 * (model.n - 1))
        for variable, gradient in zip(model.trainable_variables, gradients):
            with self.subTest(variable=variable.name):
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))

    def test_weight_copy_preserves_prediction_without_aliasing_gates(self):
        source = self.make_network()
        clone = self.make_network()
        clone.set_weights(source.get_weights())
        x = tf.constant([[25.0, 75.0], [10.0, 40.0]])
        t = tf.constant([[0.2], [0.7]])
        tf.debugging.assert_near(source([x, t]), clone([x, t]))
        for original, copied in zip(source.trainable_variables, clone.trainable_variables):
            self.assertIsNot(original, copied)
        before = clone.base_net.Uz[0].kernel.numpy().copy()
        source.base_net.Uz[0].kernel.assign_add(tf.ones_like(source.base_net.Uz[0].kernel))
        np.testing.assert_array_equal(clone.base_net.Uz[0].kernel.numpy(), before)

    def test_dgm_forward_keeps_tanh_candidate_and_input_normalization(self):
        model = self.make_network(constrained=False)
        for variable in model.trainable_variables:
            variable.assign(tf.fill(variable.shape, 0.07))
        for gate in model.Uh:
            gate.bias.assign(tf.fill((model.hidden_dim,), 1.2))
        x = tf.constant([[25.0, 75.0], [10.0, 40.0]])
        t = tf.constant([[0.2], [0.7]])

        def dense(layer, value):
            result = tf.matmul(value, layer.kernel)
            return result + layer.bias if layer.use_bias else result

        def unroll(activate_candidate):
            xt = tf.concat([2.0 * x / 100.0 - 1.0, 2.0 * t - 1.0], axis=1)
            state = tf.tanh(dense(model.Sw, xt))
            for stage in range(model.n - 1):
                z = tf.tanh(dense(model.Uz[stage], xt) + dense(model.Wsz[stage], state))
                g = tf.tanh(dense(model.Ug[stage], xt) + dense(model.Wsg[stage], state))
                r = tf.tanh(dense(model.Ur[stage], xt) + dense(model.Wsr[stage], state))
                candidate = dense(model.Uh[stage], xt) + dense(model.Wsh[stage], state * r)
                if activate_candidate:
                    candidate = tf.tanh(candidate)
                state = (1.0 - g) * candidate + z * state
            return dense(model.Wf, state)

        actual = model([x, t])
        tf.debugging.assert_near(actual, unroll(True), atol=1e-6, rtol=1e-6)
        self.assertGreater(float(tf.reduce_max(tf.abs(actual - unroll(False)))), 0.05)

    def test_hard_terminal_condition_is_exact_for_all_payoff_branches(self):
        model = self.make_network()
        x = tf.constant([[10.0, 20.0], [25.0, 35.0], [40.0, 60.0]])
        t = tf.ones((3, 1), tf.float32)
        tf.debugging.assert_equal(model([x, t]), [[0.0], [0.0], [20.0]])

    def test_seeded_path_sampler_restarts_generator_for_marginal_draws(self):
        problem, count, seed = self.problem, 4, 73
        kwargs = dict(
            num_points=count, T_val=problem.T, S0=problem.S0_train,
            r_val=problem.r, sigma_diag_val=problem.sigma_diag,
            Sigma_val=problem.Sigma, lam_vec_val=problem.lam_vec,
            mu_j_val=problem.mu_j, sig_j_val=problem.sig_j, seed=seed,
        )
        actual_x, actual_t = sample_xt_paths(**kwargs)
        expected_t = np.random.default_rng(seed).uniform(0.0, problem.T, (count, 1)).astype(np.float32)
        # Time and marginal sampling intentionally start separate generators.
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(problem.Sigma).astype(np.float32)
        z = rng.standard_normal((count, 2)).astype(np.float32)
        diffusion = np.sqrt(expected_t).astype(np.float32) * ((z @ chol.T) * problem.sigma_diag[None, :])
        n = rng.poisson(expected_t * problem.lam_vec[None, :]).astype(np.float32)
        g = rng.standard_normal((count, 2)).astype(np.float32)
        jumps = n * problem.mu_j[None, :] + np.sqrt(n) * problem.sig_j[None, :] * g
        drift = ((problem.r - problem.lam_vec * problem.kappa_vec) - 0.5 * problem.sigma_diag**2)[None, :] * expected_t
        expected_x = np.exp(np.log(problem.S0_train)[None, :] + drift + diffusion + jumps).astype(np.float32)
        np.testing.assert_array_equal(actual_t, expected_t)
        np.testing.assert_array_equal(actual_x, expected_x)
        repeated_x, repeated_t = sample_xt_paths(**kwargs)
        np.testing.assert_array_equal(actual_x, repeated_x)
        np.testing.assert_array_equal(actual_t, repeated_t)

    def test_auxiliary_second_derivative_matches_polynomial_generator(self):
        problem = self.problem
        x = tf.constant([[2.0, 3.0], [4.0, 5.0]])
        t = tf.constant([[0.2], [0.7]])
        h = tf.Variable(tf.zeros((2, 1), tf.float32))

        def polynomial(inputs, training=False):
            x_val, t_val = inputs
            return tf.reduce_sum(x_val**2, axis=1, keepdims=True) + 3.0 * t_val

        def psi(h_val):
            return fprime_bs(
                h_val, polynomial, x, t, problem.r, problem.lam_vec,
                problem.kappa_vec, problem.sigma_diag, problem.Sigma,
                problem=problem,
            )

        derivative = second_derivative_wrt_h(psi, h)
        drift_rates = problem.r - problem.lam_vec * problem.kappa_vec
        expected = 3.0 + tf.reduce_sum(
            (2.0 * drift_rates + problem.sigma_diag**2) * x**2,
            axis=1, keepdims=True,
        )
        tf.debugging.assert_near(derivative, expected, atol=2e-5, rtol=2e-5)

    def test_monte_carlo_terminal_value_has_zero_sampling_error(self):
        value, stderr = mc_basket_call_jumpdiff(
            S_t=[40.0, 60.0], t=self.problem.T,
            n_paths=12, seed=9, **self.reference_kwargs(),
        )
        self.assertEqual(value, 20.0)
        self.assertEqual(stderr, 0.0)

    def test_monte_carlo_prices_and_grid_are_seed_reproducible(self):
        kwargs = self.reference_kwargs()
        first = mc_basket_call_jumpdiff([30.0, 35.0], 0.2, n_paths=32, seed=42, **kwargs)
        second = mc_basket_call_jumpdiff([30.0, 35.0], 0.2, n_paths=32, seed=42, **kwargs)
        self.assertEqual(first, second)
        self.assertGreater(first[1], 0.0)
        times = np.array([[0.2], [1.0]])
        points = np.array([[30.0, 35.0], [40.0, 60.0]])
        grid1 = mc_prices_on_grid(times, points, n_paths=32, seed=42, **kwargs)
        grid2 = mc_prices_on_grid(times, points, n_paths=32, seed=42, **kwargs)
        for first, second in zip(grid1, grid2):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(grid1[0][1], 20.0)
        self.assertEqual(grid1[1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
