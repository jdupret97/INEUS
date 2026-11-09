"""Directory compatibility checks for nonlinear Black–Scholes aggregation."""

import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    np = None

if np is not None:
    from ineus.black_scholes.aggregation import load_histories, load_logged_losses


@unittest.skipIf(np is None, "NumPy is not installed")
class BlackScholesAggregationTests(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def checkpoint(self, directory, prediction, name="final.pkl"):
        directory.mkdir(parents=True, exist_ok=True)
        history = {"distill_epoch": [10], "student_pred": [prediction]}
        (directory / name).write_bytes(pickle.dumps({"history": history}))
        return history

    def test_repeated_layout_uses_actual_seed_not_run_index(self):
        (self.root / "run_summaries.csv").write_text(
            "run,seed,prediction\n1,41,55.0\n2,42,56.0\n", encoding="utf-8")
        first = self.checkpoint(self.root / "run_01", 55.0)
        second = self.checkpoint(self.root / "run_02", 56.0)
        self.assertEqual(load_histories(self.root, [42, 41]), {42: second, 41: first})

    def test_standalone_seed_layout_and_epoch_fallback_remain_supported(self):
        first = self.checkpoint(self.root / "seed_7", 55.0)
        second = self.checkpoint(self.root / "seed_8", 56.0, "epoch_0020.pkl")
        self.assertEqual(load_histories(self.root, [7, 8]), {7: first, 8: second})

    def test_missing_mapping_or_run_directory_falls_back_to_seed_layout(self):
        (self.root / "run_summaries.csv").write_text("run,seed\n1,7\n", encoding="utf-8")
        first = self.checkpoint(self.root / "seed_7", 55.0)
        second = self.checkpoint(self.root / "seed_8", 56.0)
        self.assertEqual(load_histories(self.root, [7, 8]), {7: first, 8: second})

    def test_explicit_checkpoint_is_resolved_within_mapped_run(self):
        (self.root / "run_summaries.csv").write_text("run,seed\n2,42\n", encoding="utf-8")
        expected = self.checkpoint(self.root / "run_02", 54.0, "epoch_0010.pkl")
        self.checkpoint(self.root / "run_02", 55.0)
        self.assertEqual(load_histories(self.root, [42], "epoch_0010.pkl"), {42: expected})

    def test_loss_logs_follow_mapped_directories(self):
        (self.root / "run_summaries.csv").write_text("run,seed\n1,41\n2,42\n", encoding="utf-8")
        for index in (1, 2):
            directory = self.root / f"run_{index:02d}"
            directory.mkdir()
            (directory / "job.out").write_text(
                f"[Teacher @ epoch 10] train_loss={index}.0\n"
                f"[Student @ epoch 10] test_loss={index + 2}.0\n", encoding="utf-8")
        train_epochs, train_values, test_epochs, test_values = load_logged_losses(self.root, [41, 42])
        np.testing.assert_array_equal(train_epochs, [10])
        np.testing.assert_array_equal(test_epochs, [10])
        np.testing.assert_array_equal(train_values, [[1.0], [2.0]])
        np.testing.assert_array_equal(test_values, [[3.0], [4.0]])

    def test_malformed_run_metadata_is_rejected(self):
        for content in ("seed\n41\n", "run,seed\n../outside,41\n", "run,seed\n1,41\n2,41\n"):
            with self.subTest(content=content):
                (self.root / "run_summaries.csv").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_histories(self.root, [41])


if __name__ == "__main__":
    unittest.main()
