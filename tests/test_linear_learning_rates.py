"""Check that both quadratic-linear trainers use the configured constant rate."""

import inspect
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
    from ineus.linear.training import ineus, pinn
    from ineus.linear.training.common import build_optimizer


class OptimizerCaptured(Exception):
    """Stop before creating training graphs or updating model parameters."""


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class LinearLearningRateTests(unittest.TestCase):
    def check_rate(self, module, trainer, rate):
        parameters = inspect.signature(trainer).parameters
        kwargs = {
            name: None for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        settings = dict(num_samples=8, n_inner=20, h_scale=0.5, zeta=0.4, lr=rate)
        kwargs.update({name: value for name, value in settings.items() if name in parameters})
        with patch.object(module, "build_optimizer", side_effect=OptimizerCaptured) as builder:
            with self.assertRaises(OptimizerCaptured):
                trainer(**kwargs)
        builder.assert_called_once_with("adam", rate, None)
        passed_rate = builder.call_args.args[1]
        self.assertIsInstance(passed_rate, float)
        optimizer = build_optimizer("adam", passed_rate)
        for updates in (0, 1000, 2000, 608000):
            optimizer.iterations.assign(updates)
            self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), rate, places=10)

    def test_ineus_uses_constant_configured_rate(self):
        for rate in (0.0005, 0.0001):
            with self.subTest(rate=rate):
                self.check_rate(ineus, ineus.train_bellman_ct, rate)

    def test_pinn_uses_constant_configured_rate(self):
        for rate in (0.0005, 0.0001):
            with self.subTest(rate=rate):
                self.check_rate(pinn, pinn.train_standard_pinn, rate)


if __name__ == "__main__":
    unittest.main()
