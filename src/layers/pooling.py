from __future__ import annotations

from typing import Tuple

import numpy as np

from base import Layer


class MaxPoolLayer(Layer):

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        self._pool_size = pool_size
        self._stride = stride
        self._a = None
        self._cache = {}

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a = np.array(a_prev, copy=True)
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        return output

    def _save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask
