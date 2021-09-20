import numpy as np

from utils.core import convert_prob2one_hot


def softmax_accuracy(y_hat: np.array, y: np.array) -> float:
    y_hat = convert_prob2one_hot(y_hat)
    return (y_hat == y).all(axis=1).mean()


def softmax_cross_entropy(y_hat, y, eps=1e-20) -> float:
    n = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / n
