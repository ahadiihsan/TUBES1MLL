import numpy as np

from base import Layer


class ReluLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._z = np.maximum(0, a_prev)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz

class SoftmaxLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        self._z = e / np.sum(e, axis=1, keepdims=True)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        return da_curr

class SigmoidLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._z = (1 / (1 + np.exp(-a_prev)))
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        return da_curr
