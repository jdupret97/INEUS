"""Small CPU checks for the nonlinear experiment's stage-specific DGM weights."""

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
    from ineus.nonlinear.architectures.value import make_model


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class NonlinearArchitectureTests(unittest.TestCase):
    GATE_NAMES = ("Uz", "Wsz", "Ug", "Wsg", "Ur", "Wsr", "Uh", "Wsh")
    X_MIN = (-2.0, -1.0, 0.0)
    X_MAX = (2.0, 3.0, 4.0)

    def setUp(self):
        cpu = tf.device("/CPU:0")
        cpu.__enter__()
        self.addCleanup(cpu.__exit__, None, None, None)
        tf.random.set_seed(23)
        self.x = tf.constant([[-0.3, 0.2, 0.4], [0.5, -0.4, 0.1]])
        self.t = tf.constant([[0.1], [0.3]])

    def make_network(self, n_layers=4, constrained=False):
        return make_model(
            hidden_dim=4, n_layers=n_layers, T=0.5, dimen=3,
            constrained=constrained, x_min=self.X_MIN, x_max=self.X_MAX,
        )

    @staticmethod
    def dense(layer, value):
        result = tf.matmul(value, layer.kernel)
        return result + layer.bias if layer.use_bias else result

    def manual_forward(self, model, activate_candidate=True, shared=False):
        # Compute normalization independently of the model's normalizer.
        x_min = tf.constant(self.X_MIN)
        x_max = tf.constant(self.X_MAX)
        x = 2.0 * (self.x - x_min) / (x_max - x_min) - 1.0
        t = 2.0 * self.t / 0.5 - 1.0
        xt = tf.concat([x, t], axis=1)
        state = tf.tanh(self.dense(model.Sw, xt))
        for stage in range(model.n - 1):
            index = 0 if shared else stage
            z = tf.tanh(self.dense(model.Uz[index], xt)
                        + self.dense(model.Wsz[index], state))
            g = tf.tanh(self.dense(model.Ug[index], xt)
                        + self.dense(model.Wsg[index], state))
            r = tf.tanh(self.dense(model.Ur[index], xt)
                        + self.dense(model.Wsr[index], state))
            candidate = (self.dense(model.Uh[index], xt)
                         + self.dense(model.Wsh[index], state * r))
            if activate_candidate:
                candidate = tf.tanh(candidate)
            state = (1.0 - g) * candidate + z * state
        return self.dense(model.Wf, state)

    def test_each_stage_owns_distinct_layers_and_variables(self):
        model = self.make_network()
        layers = []
        variables = []
        for name in self.GATE_NAMES:
            stages = getattr(model, name)
            self.assertEqual(len(stages), 3, name)
            expected_bias = name.startswith("U")
            for layer in stages:
                self.assertEqual(layer.use_bias, expected_bias, name)
                self.assertEqual(len(layer.trainable_variables), 2 if expected_bias else 1)
                layers.append(layer)
                variables.extend(layer.trainable_variables)
        self.assertEqual(len({id(layer) for layer in layers}), len(layers))
        self.assertEqual(len({id(value) for value in variables}), len(variables))
        tracked_ids = {id(value) for value in model.trainable_variables}
        self.assertTrue(all(id(value) in tracked_ids for value in variables))

    def test_parameter_count_grows_by_one_independent_stage(self):
        input_dim, hidden_dim, output_dim = 4, 4, 1
        endpoints = (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim
        stage = 4 * (input_dim + 1) * hidden_dim + 4 * hidden_dim * hidden_dim
        for n_layers in (1, 2, 4):
            with self.subTest(n_layers=n_layers):
                model = self.make_network(n_layers=n_layers)
                self.assertEqual(model.count_params(), endpoints + (n_layers - 1) * stage)
                self.assertEqual(len(model.trainable_variables), 4 + 12 * (n_layers - 1))

    def test_normalization_preserves_bounds_and_network_inputs(self):
        model = self.make_network(n_layers=1)
        x = tf.constant([self.X_MIN, (0.0, 1.0, 2.0), self.X_MAX], dtype=tf.float64)
        t = tf.constant([[0.0], [0.25], [0.5]], dtype=tf.float64)
        x_norm, t_norm = model.normalizer(x, t)
        self.assertEqual(x_norm.dtype, tf.float32)
        self.assertEqual(t_norm.dtype, tf.float32)
        tf.debugging.assert_equal(x_norm, [[-1.0] * 3, [0.0] * 3, [1.0] * 3])
        tf.debugging.assert_equal(t_norm, [[-1.0], [0.0], [1.0]])
        tf.debugging.assert_near(model([self.x, self.t]), self.manual_forward(model))

    def test_forward_equations_keep_tanh_on_candidate_state(self):
        model = self.make_network()
        for variable in model.trainable_variables:
            variable.assign(tf.fill(variable.shape, tf.constant(0.07)))
        for stage in range(model.n - 1):
            # Keep H outside tanh's near-linear region so a missing tanh is detected.
            model.Uh[stage].bias.assign(tf.fill((model.hidden_dim,), 1.8))
        actual = model([self.x, self.t])
        tf.debugging.assert_near(actual, self.manual_forward(model), atol=1e-6, rtol=1e-6)
        without_tanh = self.manual_forward(model, activate_candidate=False)
        self.assertGreater(float(tf.reduce_max(tf.abs(actual - without_tanh))), 0.1)

    def test_equal_stage_weights_reproduce_original_shared_gate_equations(self):
        model = self.make_network()
        for name in self.GATE_NAMES:
            stages = getattr(model, name)
            shared_weights = stages[0].get_weights()
            for layer in stages[1:]:
                layer.set_weights(shared_weights)
        tf.debugging.assert_near(
            model([self.x, self.t]), self.manual_forward(model, shared=True),
            atol=1e-6, rtol=1e-6,
        )

    def test_every_stage_weight_has_a_finite_gradient(self):
        model = self.make_network()
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.square(model([self.x, self.t]) - 1.0))
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertEqual(len(gradients), len(model.trainable_variables))
        for variable, gradient in zip(model.trainable_variables, gradients):
            self.assertIsNotNone(gradient, variable.name)
            tf.debugging.assert_all_finite(gradient, variable.name)

    def test_traced_forward_matches_eager_predictions(self):
        model = self.make_network(constrained=True)

        @tf.function(input_signature=(
            tf.TensorSpec((None, 3), tf.float32),
            tf.TensorSpec((None, 1), tf.float32),
        ))
        def predict(x, t):
            return model([x, t])

        tf.debugging.assert_near(predict(self.x, self.t), model([self.x, self.t]))
        tf.debugging.assert_near(
            predict(self.x[:1], self.t[:1]), model([self.x[:1], self.t[:1]])
        )

    def test_constrained_model_preserves_quadratic_terminal_condition(self):
        model = self.make_network(constrained=True)
        terminal_t = tf.fill((2, 1), 0.5)
        expected = tf.reduce_sum(tf.square(self.x), axis=1, keepdims=True)
        tf.debugging.assert_equal(model([self.x, terminal_t]), expected)

    def test_weight_copy_preserves_predictions_without_sharing_variables(self):
        source = self.make_network(constrained=True)
        clone = self.make_network(constrained=True)
        clone.set_weights(source.get_weights())
        tf.debugging.assert_near(source([self.x, self.t]), clone([self.x, self.t]))
        self.assertEqual(len(source.trainable_variables), len(clone.trainable_variables))
        for original, copied in zip(source.trainable_variables, clone.trainable_variables):
            self.assertIsNot(original, copied)
        source.base_net.Uz[0].kernel.assign_add(tf.ones_like(source.base_net.Uz[0].kernel))
        difference = source.base_net.Uz[0].kernel - clone.base_net.Uz[0].kernel
        tf.debugging.assert_near(difference, tf.ones_like(difference))


if __name__ == "__main__":
    unittest.main()
