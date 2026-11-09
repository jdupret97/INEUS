"""Sampling and replay-buffer utilities for the nonlinear experiment."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import tensorflow as tf

from .problem import COV_J_CHOL as cov_j_chol, DIMENSION as dimen, MU_J as mu_j
from .problem import resolve_problem


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
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


def set_all_seeds(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def sample_xt(num_points, T=None, *, problem=None):
    problem = resolve_problem(problem)
    T = problem.horizon if T is None else T
    dimen = problem.state_dim
    t = np.random.uniform(0.0, T, num_points).reshape(-1, 1).astype(np.float32)
    x = np.random.normal(0.0, 0.8, (num_points, dimen)).astype(np.float32)#np.random.uniform(-sample_b, sample_b, (num_points, dimen)).astype(np.float32)
    return x, t


def sample_x_t0(num_points, *, problem=None):
    dimen = resolve_problem(problem).state_dim
    x = np.random.normal(0.0, 0.35, (num_points, dimen)).astype(np.float32)
    t = np.zeros((num_points, 1), dtype=np.float32)
    return x, t


def sample_local_t0_x0_nl(num_points, T=None, t_eps=1e-5, *, problem=None):
    """
    Sample local Deep BSDE starting points.

    We use the same spatial law as sample_xt() in this nonlinear experiment:
        X ~ N(0, 0.8^2 I),
        t ~ Uniform(0,T).
    """
    problem = resolve_problem(problem)
    T = problem.horizon if T is None else T
    dimen = problem.state_dim
    t0 = np.random.uniform(
        low=0.0,
        high=max(float(T) - t_eps, 0.0),
        size=(num_points, 1),
    ).astype(np.float32)

    x0 = np.random.normal(
        loc=0.0,
        scale=0.8,
        size=(num_points, dimen),
    ).astype(np.float32)

    return t0, x0


def sample_jump_marks_mc_local_nl(batch_size, K, *, problem=None):
    """
    Antithetic Gaussian jump marks E ~ N(mu_j, cov_j), shape [B,K,d].
    """
    problem = resolve_problem(problem)
    dimen = problem.state_dim
    mu_j = problem.mu_j
    cov_j_chol = problem.cov_j_chol
    if not problem.has_jumps:
        return tf.zeros([batch_size, K, dimen], dtype=tf.float32)
    half = K // 2
    marks = []
    mu_j_tf = tf.convert_to_tensor(mu_j, dtype=tf.float32)
    chol_tf = tf.cast(cov_j_chol, tf.float32)

    if half > 0:
        z = tf.random.normal([batch_size, half, dimen], dtype=tf.float32)
        e_pos = tf.einsum("bkd,ed->bke", z, chol_tf)
        e_neg = -e_pos
        marks.extend([e_pos, e_neg])

    if K % 2 == 1:
        z_extra = tf.random.normal([batch_size, 1, dimen], dtype=tf.float32)
        e_extra = tf.einsum("bkd,ed->bke", z_extra, chol_tf)
        marks.append(e_extra)

    e = tf.concat(marks, axis=1)
    e = e[:, :K, :]
    return e + mu_j_tf[None, None, :]
