"""Small evaluation helpers shared by the two value-network trainers."""

import numpy as np


def evaluate_test_mae(model, xtest, ttest, values):
    preds = model([xtest, ttest], training=False).numpy().reshape(-1)
    return float(np.mean(np.abs(preds - values.reshape(-1))))

