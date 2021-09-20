from __future__ import annotations
from typing import List, Dict, Callable, Optional
import time

import numpy as np

from base import Layer
from utils.core import generate_batches, format_time
from utils.metrics import softmax_accuracy, softmax_cross_entropy


class SequentialModel:
    def __init__(self, layers: List[Layer]):
        self._layers = layers

    def predict(self, x: np.array) -> np.array:
        return self._forward(x, training=False)

    def _forward(self, x: np.array, training: bool) -> np.array:
        activation = x
        for idx, layer in enumerate(self._layers):
            activation = layer.forward_pass(a_prev=activation, training=training)
        return activation