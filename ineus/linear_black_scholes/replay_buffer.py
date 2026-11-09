"""Replay buffer for sampled space-time points."""

from collections import deque
import random

import numpy as np


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
        return np.array(x), np.array(t)

    def size(self):
        return len(self.buffer)
