import numpy as np
from numpy.core.fromnumeric import size
'''
损失函数与激活函数
'''


def mse(output, y):
    return np.sum(np.square(output - y)) / len(y)


def deriv_mse(output, y):
    return output - y


def cross_enretpy(output, y):
    return -np.sum(y * np.log(output))


def deriv_cross_enretpy(output, y):
    return -y / output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(y):
    grad = y * (1 - y)
    n, d = y.shape
    ret = np.zeros((n, d, d))
    id = (np.arange(0, n * d * d, d * d).reshape(-1, 1) +
          np.arange(0, d * d, d + 1)).reshape(-1)
    ret.flat[id] = grad.reshape(-1)
    return ret


def relu(x):
    x[x < 0] = 0
    return x


def deriv_relu(y):
    y[y > 0] = 1
    n, d = y.shape
    ret = np.zeros((n, d, d))
    id = (np.arange(0, n * d * d, d * d).reshape(-1, 1) +
          np.arange(0, d * d, d + 1)).reshape(-1)
    ret.flat[id] = y.reshape(-1)



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     sns.set()

#     n, d, N = 64, 64, 2000
#     x = np.random.normal(size=(n, d))
#     from time import time

#     time_list = []
#     t = time()
#     for i in range(N):
#         deriv_sigmoid(x)
#         time_list.append(time() - t)
#     plt.plot(time_list, label="new n=64, d=64", color="blue")

#     time_list = []
#     t = time()
#     for i in range(N):
#         X = np.zeros((n, d, d))
#         for j in range(n):
#             X[j] = np.diag(x[j])
#         time_list.append(time() - t)
#     plt.plot(time_list, "-.", label="old n=64, d=64", color="blue")

#     n, d = 64, 32
#     x = np.random.normal(size=(n, d))
#     time_list = []
#     t = time()
#     for i in range(N):
#         deriv_sigmoid(x)
#         time_list.append(time() - t)
#     plt.plot(time_list, label="new n=64, d=32", color="red")

#     time_list = []
#     t = time()
#     for i in range(N):
#         X = np.zeros((n, d, d))
#         for j in range(n):
#             X[j] = np.diag(x[j])
#         time_list.append(time() - t)
#     plt.plot(time_list, "-.", label="old n=64, d=32", color="red")

#     n, d = 32, 64
#     x = np.random.normal(size=(n, d))
#     time_list = []
#     t = time()
#     for i in range(N):
#         deriv_sigmoid(x)
#         time_list.append(time() - t)
#     plt.plot(time_list, label="new n=32, d=64", color="yellowgreen")

#     time_list = []
#     t = time()
#     for i in range(N):
#         X = np.zeros((n, d, d))
#         for j in range(n):
#             X[j] = np.diag(x[j])
#         time_list.append(time() - t)
#     plt.plot(time_list, "-.", label="old n=32, d=64", color="yellowgreen")

#     plt.legend()
#     plt.show()
