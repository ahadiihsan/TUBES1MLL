import numpy as np

from base import Layer


class FlattenLayer(Layer):

    def __init__(self):
        self._shape = ()

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)
