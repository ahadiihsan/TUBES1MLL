from __future__ import annotations
from typing import List, Dict, Callable, Optional, Tuple
import time

import numpy as np

from base import Layer
from utils.core import generate_batches, format_time
from utils.metrics import softmax_accuracy, softmax_cross_entropy

conv_layer = ["ConvLayer2D", "FastConvLayer2D", "SuperFastConvLayer2D"]

class SequentialModel:
    def __init__(self, layers: List[Layer], optimizer: Optimizer):
        self._layers = layers
        self._optimizer = optimizer

        self._train_acc = []
        self._test_acc = []
        self._train_loss = []
        self._test_loss = []

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
        epochs: int,
        bs: int = 64,
        verbose: bool = False,
        callback: Optional[Callable[[SequentialModel], None]] = None
        ) -> None:
        print("starting training")
        for epoch in range(epochs):
            print("in epoch")
            epoch_start = time.time()
            y_hat = np.zeros_like(y_train)
            for idx, (x_batch, y_batch) in enumerate(generate_batches(x_train, y_train, bs)):
                print("in batch")
                y_hat_batch = self._forward(x_batch, training=True)
                activation = y_hat_batch - y_batch
                self._backward(activation)
                self._update()
                n_start = idx * bs
                n_end = n_start + y_hat_batch.shape[0]
                y_hat[n_start:n_end, :] = y_hat_batch

            self._train_acc.append(softmax_accuracy(y_hat, y_train))
            self._train_loss.append(softmax_cross_entropy(y_hat, y_train))

            y_hat = self._forward(x_test, training=False)
            test_acc = softmax_accuracy(y_hat, y_test)
            self._test_acc.append(test_acc)
            test_loss = softmax_cross_entropy(y_hat, y_test)
            self._test_loss.append(test_loss)

            if verbose:
                epoch_time = format_time(start_time=epoch_start, end_time=time.time())
                print("iter: {:05} | test loss: {:.5f} | test accuracy: {:.5f} | time: {}"
                      .format(epoch+1, test_loss, test_acc, epoch_time))

    def predict(self, x: np.array) -> np.array:
        return self._forward(x, training=False)

    @property
    def history(self) -> Dict[str, List[float]]:
        return {
            "train_acc": self._train_acc,
            "test_acc": self._test_acc,
            "train_loss": self._train_loss,
            "test_loss": self._test_loss
        }

    def _forward(self, x: np.array, training: bool) -> np.array:
        activation = x
        for idx, layer in enumerate(self._layers):
            activation = layer.forward_pass(a_prev=activation, training=training)
        return activation

    def _backward(self, x: np.array) -> None:
        activation = x
        for layer in reversed(self._layers):
            activation = layer.backward_pass(da_curr=activation)

    def _update(self) -> None:
        self._optimizer.update(layers=self._layers)

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
