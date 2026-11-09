
"""General utilities: seeding, weight management, evaluation."""

import random
import numpy as np
import tensorflow as tf


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def copy_weights(model: tf.keras.Model) -> list:
    """Return a deep copy of *model*'s weights."""
    return [np.array(w, copy=True) for w in model.get_weights()]


def evaluate_point(model, x_eval, t_eval, u_ref=None):
    """Evaluate the model at a single space-time point.

    Returns
    -------
    prediction : float
    abs_error  : float | None
    """
    pred = model([x_eval, t_eval], training=False).numpy().flatten()
    prediction = float(pred[0])
    abs_error = float(abs(prediction - u_ref)) if u_ref is not None else None
    return prediction, abs_error
