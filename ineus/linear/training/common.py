"""Shared optimization and evaluation helpers; no training side effects."""

import numpy as np
import tensorflow as tf


def evaluate_test_mae_callable(predict_fn, xtest, ttest, values):
    preds = predict_fn(xtest, ttest)
    return float(np.mean(np.abs(preds - values)))


def build_optimizer(name, learning_rate, optimizer_kwargs=None):
    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_kwargs["learning_rate"] = learning_rate
    name = name.lower()

    if name == "adam":
        return tf.keras.optimizers.Adam(**optimizer_kwargs)
    if name == "adagrad":
        return tf.keras.optimizers.Adagrad(**optimizer_kwargs)
    if name == "sgd":
        return tf.keras.optimizers.SGD(**optimizer_kwargs)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(**optimizer_kwargs)

    raise ValueError(f"Unknown optimizer: {name}")
