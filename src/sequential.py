from __future__ import annotations
from typing import List, Dict, Callable, Optional, Tuple
import time

import numpy as np

from base import Layer
from utils.core import generate_batches, format_time
from utils.metrics import softmax_accuracy, softmax_cross_entropy

conv_layer = ["ConvLayer2D", "FastConvLayer2D", "SuperFastConvLayer2D"]

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

    def print_model(self, input):
        print("Model : \t\tSequential")
        print('─' * 80) 
        print("Layer (type) \t\tOutput Shape \t\tParam #")
        print("═" * 80)
        shape = input.shape
        prev = input
        tot = 0
        print("Input \t\t\t(%d, %d, %d)" % (shape[1], shape[2], shape[3]))
        print('─' * 80) 
        for layer in self._layers:
            prev = layer.forward_pass(prev, training=False)
            shape = prev.shape
            if type(layer).__name__ in conv_layer:
                weigth, _ = layer.weights
                hf, wf, _, ff = weigth.shape
                tmp = (shape[1] * shape[2]) * (hf * wf * ff + 1)
                tot += tmp
                print("%s \t\t(%d, %d, %d) \t\t%d" % (type(layer).__name__, shape[1], shape[2], shape[3], tmp))
            elif len(shape) == 2 :
                print("%s \t\t(%d) \t\t%d" % (type(layer).__name__, shape[1], 0))
            else :
                print("%s \t\t(%d, %d, %d) \t\t%d" % (type(layer).__name__, shape[1], shape[2], shape[3], 0))
            print('─' * 80) 
        print("Total params : \t\t %d" % (tot))