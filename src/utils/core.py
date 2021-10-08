import numpy as np
import time
import pickle

def convert_categorical2one_hot(y: np.array) -> np.array:
    one_hot_matrix = np.zeros((y.size, y.max() + 1))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix


def convert_prob2categorical(probs: np.array) -> np.array:
    return np.argmax(probs, axis=1)


def convert_prob2one_hot(probs: np.array) -> np.array:
    class_idx = convert_prob2categorical(probs)
    one_hot_matrix = np.zeros_like(probs)
    one_hot_matrix[np.arange(probs.shape[0]), class_idx] = 1
    return one_hot_matrix


def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def generate_batches(x: np.array, y: np.array, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )


def format_time(start_time: time.time, end_time: time.time) -> str:
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
