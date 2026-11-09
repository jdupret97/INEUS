"""Replay-buffer, collocation, and jump sampling utilities."""

import random
from collections import deque

import numpy as np
import tensorflow as tf

from .problem import resolve_problem


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, x, t):
        self.buffer.append((x, t))

    def extend(self, x_arr, t_arr):
        for x, t in zip(x_arr, t_arr):
            self.buffer.append((x, t))

    def clear(self):
        self.buffer.clear()

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x, t = zip(*batch)
        return np.array(x, dtype=np.float32), np.array(t, dtype=np.float32)

    def size(self):
        return len(self.buffer)


def sample_xt(num_points, T=None, *, problem=None):
    problem = resolve_problem(problem)
    T = problem.T if T is None else T
    t = np.random.uniform(0.0, T, size=(num_points, 1)).astype(np.float32)
    x = np.random.uniform(
        -problem.sample_b, problem.sample_b, size=(num_points, problem.dimen),
    ).astype(np.float32)
    return x, t


def sample_x0(num_points, *, problem=None):
    problem = resolve_problem(problem)
    return np.random.uniform(
        low=-problem.sample_b,
        high=problem.sample_b,
        size=(num_points, problem.dimen)
    ).astype(np.float32)


def sample_t0_x0(num_points, t_eps=1e-5, *, problem=None):
    """
    Sample initial points (t0, x0) ~ rho.

    rho is uniform over [0,T) x [-sample_b, sample_b]^d.
    """
    problem = resolve_problem(problem)
    t0 = np.random.uniform(
        low=0.0,
        high=max(problem.T - t_eps, 0.0),
        size=(num_points, 1),
    ).astype(np.float32)

    x0 = sample_x0(num_points, problem=problem)

    return t0, x0


def sample_jump_marks(batch_size, *, problem=None):
    """
    Draw E ~ N(mu_j, cov_j), shape [B, dimen].
    """
    problem = resolve_problem(problem)
    z = tf.random.normal([batch_size, problem.dimen], dtype=tf.float32)
    return (tf.linalg.matmul(z, problem.cov_j_chol_tf, transpose_b=True)
            + problem.mu_j_tf[None, :])


def sample_jump_marks_mc(batch_size, K, *, problem=None):
    """
    Draw E ~ N(mu_j, cov_j), shape [B, K, dimen].

    Uses antithetic sampling to reduce variance.
    """
    problem = resolve_problem(problem)
    if K < 1:
        raise ValueError("K must be positive")
    half = K // 2
    marks = []

    if half > 0:
        z = tf.random.normal([batch_size, half, problem.dimen], dtype=tf.float32)

        # Equivalent to z @ chol^T for each sample.
        e_pos = tf.einsum("bkd,ed->bke", z, problem.cov_j_chol_tf)
        e_neg = -e_pos

        marks.extend([e_pos, e_neg])

    if K % 2 == 1:
        z_extra = tf.random.normal([batch_size, 1, problem.dimen], dtype=tf.float32)
        e_extra = tf.einsum("bkd,ed->bke", z_extra, problem.cov_j_chol_tf)
        marks.append(e_extra)

    e = tf.concat(marks, axis=1)
    e = e[:, :K, :]

    return e + problem.mu_j_tf[None, None, :]
