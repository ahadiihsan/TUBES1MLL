from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from base import Layer


class DenseLayer(Layer):

    def __init__(self, w: np.array, b: np.array):
        self._w, self._b = w, b
        self._dw, self._db = None, None
        self._a_prev = None

    @classmethod
    def initialize(cls, units_prev: int, units_curr: int) -> DenseLayer:
        w = np.random.randn(units_curr, units_prev) * 0.1
        b = np.random.randn(1, units_curr) * 0.1
        return cls(w=w, b=b)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        return np.dot(a_prev, self._w.T) + self._b
