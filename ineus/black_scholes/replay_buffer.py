"""Fixed-capacity replay buffer for collocation points."""

import numpy as np


class ReplayBuffer:
    """Ring-buffer storing (x, t) collocation pairs as contiguous arrays."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.x = None       # lazily allocated on first extend
        self.t = None
        self.size = 0
        self.ptr = 0

    def _init_storage(self, x_sample, t_sample):
        """Allocate arrays matching the shape of incoming data."""
        x_shape = (self.capacity,) + x_sample.shape[1:]   # e.g. (16384, 100)
        t_shape = (self.capacity,) + t_sample.shape[1:]   # e.g. (16384,) or (16384, 1)
        self.x = np.empty(x_shape, dtype=np.float32)
        self.t = np.empty(t_shape, dtype=np.float32)

    def extend(self, x_arr: np.ndarray, t_arr: np.ndarray):
        if self.x is None:
            self._init_storage(x_arr, t_arr)

        n = len(x_arr)
        if n >= self.capacity:
            self.x[:] = x_arr[-self.capacity:]
            self.t[:] = t_arr[-self.capacity:]
            self.size = self.capacity
            self.ptr = 0
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.x[self.ptr:end] = x_arr
            self.t[self.ptr:end] = t_arr
        else:
            first = self.capacity - self.ptr
            self.x[self.ptr:] = x_arr[:first]
            self.x[:n - first] = x_arr[first:]
            self.t[self.ptr:] = t_arr[:first]
            self.t[:n - first] = t_arr[first:]

        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.x[idx], self.t[idx]

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        self.ptr = 0
