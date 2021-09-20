import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

import sys
sys.path.append("../")

from activation import ReluLayer
from activation import SigmoidLayer
from activation import SoftmaxLayer
from layers.dense import DenseLayer
from layers.convolutional import ConvLayer2D, SuperFastConvLayer2D
from sequential import SequentialModel
from utils.core import convert_categorical2one_hot, convert_prob2categorical
from utils.metrics import softmax_accuracy
from utils.plots import lines

# number of samples in the train data set
N_TRAIN_SAMPLES = 50000
# number of samples in the test data set
N_TEST_SAMPLES = 2500
# number of samples in the validation data set
N_VALID_SAMPLES = 250
# number of classes
N_CLASSES = 10
# image size
IMAGE_SIZE = 28

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)

X_train = trainX[:N_TRAIN_SAMPLES, :, :]
y_train = trainY[:N_TRAIN_SAMPLES]

X_test = trainX[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES, :, :]
y_test = trainY[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES]

X_valid = testX[:N_VALID_SAMPLES, :, :]
y_valid = testY[:N_VALID_SAMPLES]

X_train = X_train / 255
X_train = np.expand_dims(X_train, axis=3)
y_train = convert_categorical2one_hot(y_train)
X_test = X_test / 255
X_test = np.expand_dims(X_test, axis=3)
y_test = convert_categorical2one_hot(y_test)
X_valid = X_valid / 255
X_valid = np.expand_dims(X_valid, axis=3)
y_valid = convert_categorical2one_hot(y_valid)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)

layers = [
    # input (N, 28, 28, 1) out (N, 28, 28, 32)
    SuperFastConvLayer2D.initialize(filters=32, kernel_shape=(3, 3, 1), stride=1, padding="same"),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    ReluLayer(),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    SuperFastConvLayer2D.initialize(filters=32, kernel_shape=(3, 3, 32), stride=1, padding="same"),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    ReluLayer(),
    # input (N, 28, 28, 32) out (N, 14, 14, 32)
    MaxPoolLayer(pool_size=(2, 2), stride=2),
    # input (N, 14, 14, 32) out (N, 14, 14, 32)
    SuperFastConvLayer2D.initialize(filters=64, kernel_shape=(3, 3, 32), stride=1, padding="same"),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    ReluLayer(),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    SuperFastConvLayer2D.initialize(filters=64, kernel_shape=(3, 3, 64), stride=1, padding="same"),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    ReluLayer(),
    # input (N, 14, 14, 64) out (N, 7, 7, 64)
    MaxPoolLayer(pool_size=(2, 2), stride=2),
    # input (N, 7, 7, 64) out (N, 7 * 7 * 64)
    FlattenLayer(),
    # input (N, 7 * 7 * 64) out (N, 256)
    DenseLayer.initialize(units_prev=7 * 7 * 64, units_curr=256),
    # input (N, 256) out (N, 256)
    ReluLayer(),
     # input (N, 256) out (N, 32)
    DenseLayer.initialize(units_prev=256, units_curr=32),
     # input (N, 32) out (N, 32)
    ReluLayer(),
     # input (N, 32) out (N, 10)
    DenseLayer.initialize(units_prev=32, units_curr=N_CLASSES),
     # input (N, 10) out (N, 10)
    SoftmaxLayer()
]

model = SequentialModel(
    layers=layers,
)

y_hat = model.predict(X_valid)
acc = softmax_accuracy(y_hat, y_valid)
print("acc: ", acc)

y_hat = convert_prob2categorical(y_hat)
y_valid = convert_prob2categorical(y_valid)

df_cm = pd.DataFrame(
    confusion_matrix(y_valid, y_hat),
    range(10),
    range(10)
)
plt.figure(figsize = (16,16))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu", linewidths=.5, cbar=False)
plt.savefig("../viz/cm.png", dpi=100)
plt.show()
