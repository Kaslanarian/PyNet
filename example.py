from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
import matplotlib.pyplot as plt

from net import Net
from dataloader import train_loader
from functional import *
from optimizers import *
from net_io import load_model, save_model, random_init


def train(
    net: Net,
    optim: BaseOptimizer,
    epoch: int,
    train_X: np.ndarray,
    train_y: np.ndarray,
    batch_size: int,
    test_X: np.ndarray = None,
    test_y: np.ndarray = None,
    verbose=False,
):
    '''
    train_y是one hot的，test_y是整数数组
    '''
    test = (test_X is not None) and (test_y is not None)
    output = net(train_X)
    train_loss_list = [net.criterion(output, train_y)]
    train_acc_list = [
        accuracy_score(
            normal_y := np.argmax(train_y, axis=1),
            np.argmax(output, axis=1),
        )
    ]
    if test:
        test_acc_list = [
            accuracy_score(
                test_y,
                np.argmax(net(test_X), axis=1),
            )
        ]

    data = train_loader(train_X, train_y, batch_size)
    for epoch in range(epoch):
        for batch_X, batch_y in data:
            net.forward(batch_X)
            net.backward(batch_y)
            optim.step(*net.grads)

        output = net(train_X)
        train_loss_list.append(net.criterion(output, train_y))
        train_acc_list.append(
            accuracy_score(
                normal_y,
                np.argmax(output, axis=1),
            ))
        if test:
            test_acc_list.append(
                accuracy_score(
                    test_y,
                    np.argmax(net(test_X), axis=1),
                ))
        if verbose:
            print("epoch {} over".format(epoch + 1))

    if not test:
        test_acc_list = None
    return train_loss_list, train_acc_list, test_acc_list


if __name__ == "__main__":
    std_scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder()

    X, y = load_digits(return_X_y=True)
    std_scaler.fit(X)
    X = std_scaler.transform(X)
    one_hot_encoder.fit(y.reshape(-1, 1))

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=42,
    )

    train_y = one_hot_encoder.transform(train_y.reshape(-1, 1)).toarray()
    i, o = train_X.shape[1], train_y.shape[1]

    net = random_init(
        Net(
            (i, "linear"),
            (i + o, "relu"),
            (o, "softmax"),
            he=True,
        ))
    optim = Adam(net.parameters(), 0.01)

    train_loss_list, train_acc, test_acc = train(
        net,
        optim,
        25,
        train_X,
        train_y,
        32,
        test_X,
        test_y,
    )

    save_model(net, "digits")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="training loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train acc", marker="o")
    plt.plot(test_acc, label="test acc", marker="o")
    plt.legend()
    plt.savefig("src/digits.png")

