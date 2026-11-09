"""Nonlinear INEUS experiment package.

The package initializer intentionally exposes only lightweight configuration
objects. Import :mod:`ineus.nonlinear.experiment` to run the TensorFlow code.
"""

from .config import (
    BSDEConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    ProblemConfig,
    TrainConfig,
    default_config,
    paper_config,
)

__all__ = [
    "BSDEConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ProblemConfig",
    "TrainConfig",
    "default_config",
    "paper_config",
]
