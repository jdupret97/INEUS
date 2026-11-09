"""Runtime, weight-copy, and optimizer helpers."""

from pathlib import Path
import random

import numpy as np
import tensorflow as tf


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def copy_weights(model):
    return [np.array(w_, copy=True) for w_ in model.get_weights()]


def build_optimizer(optimizer, learning_rate, optimizer_kwargs=None):
    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_kwargs["learning_rate"] = learning_rate

    if isinstance(optimizer, str):
        opt_map = {
            "adam": tf.keras.optimizers.Adam,
            "adagrad": tf.keras.optimizers.Adagrad,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
        }
        return opt_map[optimizer.lower()](**optimizer_kwargs)

    if isinstance(optimizer, type) and issubclass(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer(**optimizer_kwargs)

    raise ValueError("Unsupported optimizer specification.")
