"""Small CPU checks for the linear experiment's stage-specific DGM weights."""

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
    from ineus.linear.architectures.value import make_model


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class LinearArchitectureTests(unittest.TestCase):
    GATE_NAMES = ("Uz", "Wsz", "Ug", "Wsg", "Ur", "Wsr", "Uh", "Wsh")

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
            constrained=constrained,
        )

    def test_each_stage_owns_distinct_layers_and_variables(self):
        model = self.make_network()
        layers = []
        variables = []
        for name in self.GATE_NAMES:
            stages = getattr(model, name)
            self.assertEqual(len(stages), 3, name)
            for layer in stages:
                self.assertTrue(layer.use_bias)
                self.assertEqual(len(layer.trainable_variables), 2)
                layers.append(layer)
                variables.extend(layer.trainable_variables)
        self.assertEqual(len({id(layer) for layer in layers}), len(layers))
        self.assertEqual(len({id(value) for value in variables}), len(variables))
        tracked_ids = {id(value) for value in model.trainable_variables}
        self.assertTrue(all(id(value) in tracked_ids for value in variables))

    def test_parameter_count_grows_by_one_independent_stage(self):
        input_dim, hidden_dim, output_dim = 4, 4, 1
        endpoints = (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim
        stage = 4 * (input_dim + 1) * hidden_dim + 4 * (hidden_dim + 1) * hidden_dim
        for n_layers in (1, 2, 4):
            with self.subTest(n_layers=n_layers):
                model = self.make_network(n_layers=n_layers)
                self.assertEqual(model.count_params(), endpoints + (n_layers - 1) * stage)
                self.assertEqual(len(model.trainable_variables), 4 + 16 * (n_layers - 1))

    def test_forward_equations_keep_candidate_state_unactivated(self):
        model = self.make_network()
        for variable in model.trainable_variables:
            variable.assign(tf.fill(variable.shape, tf.constant(0.07)))
        for stage in range(model.n - 1):
            # Keep H outside tanh's near-linear region so an added tanh is detected.
            model.Uh[stage].bias.assign(tf.fill((model.hidden_dim,), 1.2))
            model.Wsh[stage].bias.assign(tf.fill((model.hidden_dim,), 0.7))

        def dense(layer, value):
            return tf.matmul(value, layer.kernel) + layer.bias

        def unroll(activate_candidate):
            xt = tf.concat([self.x, self.t], axis=1)
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

        actual = model([self.x, self.t])
        tf.debugging.assert_near(actual, unroll(False), atol=1e-6, rtol=1e-6)
        self.assertGreater(float(tf.reduce_max(tf.abs(actual - unroll(True)))), 0.1)

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
        for original, copied in zip(source.trainable_variables, clone.trainable_variables):
            self.assertIsNot(original, copied)
        source.base_net.Uz[0].kernel.assign_add(tf.ones_like(source.base_net.Uz[0].kernel))
        difference = source.base_net.Uz[0].kernel - clone.base_net.Uz[0].kernel
        tf.debugging.assert_near(difference, tf.ones_like(difference))


if __name__ == "__main__":
    unittest.main()
