import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from optimizer import *

i, h, o = 4, 10, 3

A, a = np.random.random((i, h)), np.random.random(h)
B, b = np.random.random((h, o)), np.random.random(o)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(y):
    return y * (1 - y)


def relu(x):
    x[x < 0] = 0
    return x


def deriv_relu(y):
    y[y > 0] = 1
    return y


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def deriv_mse(y_pred, y_true):
    return y_pred - y_true


def net(x):
    hidden = relu(x @ A + a)
    output = relu(hidden @ B + b)
    return hidden, output


EPOCH = 1000
optimizer = Momentum(A, a, B, b, lr=.0001)

X, y = load_iris(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
y = y.reshape(-1, 1)
y = OneHotEncoder(sparse=False).fit(y).transform(y)

for epoch in range(EPOCH):
    hidden, output = net(X)  # 150, 5; 150, 3

    loss = mse(y, output)

    dl = deriv_mse(output, y)  # 150 * 3
    do = deriv_relu(output)  # 150 * 3

    temp = -dl * do  # (150, 3)
    db = temp.mean(0)  # 3
    dB = -hidden.T @ temp  # 5 * 3

    temp = deriv_relu(hidden) * (db @ B.T)
    da = temp.mean(0)
    dA = -X.T @ temp

    optimizer.step(dA, da, dB, db)

    if epoch % 100 == 99:
        print(epoch + 1, loss)

output = net(X)[1]
print((np.argmax(output, 1) == np.argmax(y, 1)).mean())