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
from layers.pooling import MaxPoolLayer
from layers.dense import DenseLayer
from layers.flatten import FlattenLayer
from layers.convolutional import ConvLayer2D, SuperFastConvLayer2D
from sequential import SequentialModel
from optimizers.gradient_descent import GradientDescent
from utils.core import convert_categorical2one_hot, convert_prob2categorical, load_model, save_model
from utils.metrics import softmax_accuracy
from utils.plots import lines

temp = input("load file: (y/n): ")
while temp != "y" and temp != "n":
    print("enter y or n correctly\n")
    temp = input("load file: (y/n)")

from_file = False
if temp=="y":
    from_file = True

learning_rate = float(input("learning rate: "))
print(learning_rate)
epoch = int(input("epoch: "))
momentum = int(input("momentum: "))
batch_size = int(input("batch size: "))

# number of samples in the train data set
N_TRAIN_SAMPLES = 5000
# number of samples in the test data set
N_TEST_SAMPLES = 250
# number of samples in the validation data set
N_VALID_SAMPLES = 25
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

optimizer = GradientDescent(lr=learning_rate)

if from_file:
    try:
        model = load_model("learning_model")
        print("load success")
    except Exception as e:
        model = SequentialModel(
        layers=layers,
        optimizer=optimizer
        )
else:
    model = SequentialModel(
        layers=layers,
        optimizer=optimizer
    )

model.train(
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    epochs=epoch,
    bs=batch_size,
    verbose=True
)

save_model(model,"learning_model")

lines(
    y_1=np.array(model.history["train_acc"]),
    y_2=np.array(model.history["test_acc"]),
    label_1="train",
    label_2="test",
    title="ACCURACY",
    fig_size=(16,10),
    path="./viz/cnn_acc.png"
)

lines(
    y_1=np.array(model.history["train_loss"]),
    y_2=np.array(model.history["test_loss"]),
    label_1="train",
    label_2="test",
    title="LOSS",
    fig_size=(16,10),
    path="./viz/cnn_loss.png"
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
plt.savefig("./viz/cm.png", dpi=100)
plt.show()
model.print_model(X_valid)
