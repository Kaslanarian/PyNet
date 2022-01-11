import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.special import xlogy, expit


def tensor_diag(x, n, d):
    '''
    将n*d个元素放置到n*d*d张量的n个对角线上
    ref : https://welts.xyz/2021/12/08/diag/
    '''
    ret = np.zeros((n, d, d))
    id = (np.arange(0, n * d * d, d * d).reshape(-1, 1) +
          np.arange(0, d * d, d + 1)).reshape(-1)
    if type(x) == np.ndarray:
        ret.flat[id] = x.reshape(-1)
    else:
        ret.flat[id] = x
    return ret


def linear(x: np.array):
    '''线性激活函数'''
    return x


def sigmoid(x: np.array):
    '''sigmoid函数'''
    return expit(x)


def tanh(x):
    '''tanh函数'''
    return np.tanh(x)


def relu(x):
    '''relu函数'''
    x[x < 0] = 0
    return x


def elu(x):
    '''elu函数'''
    x[x < 0] = np.exp(x[x < 0]) - 1
    return x


def leaky_relu(x):
    '''leaky relu函数'''
    x[x < 0] = 0.1 * x[x < 0]
    return x


def softmax(x):
    '''softmax函数'''
    return scipy_softmax(x, axis=-1)


def deriv_linear(y: np.array) -> np.array:
    '''线性激活函数的梯度'''
    n, _, d = y.shape
    return tensor_diag(1, n, d)


def deriv_sigmoid(y):
    '''sigmoid函数的梯度'''
    grad = y * (1 - y)
    n, _, d = y.shape
    return tensor_diag(grad, n, d)


def deriv_tanh(y):
    '''tanh函数的梯度'''
    grad = 1 - y**2
    n, _, d = y.shape
    return tensor_diag(grad, n, d)


def deriv_relu(y):
    '''relu函数的梯度'''
    y[y > 0] = 1
    y[y < 0] = 0
    n, _, d = y.shape
    return tensor_diag(y, n, d)


def deriv_elu(y):
    '''elu函数的梯度'''
    y[y > 0] = 1
    y[y <= 0] += 1
    n, _, d = y.shape
    return tensor_diag(y, n, d)


def deriv_leaky(y):
    y[y > 0] = 1
    y[y < 0] = 0.1
    n, _, d = y.shape
    return tensor_diag(y, n, d)


def deriv_softmax(y):
    '''softmax函数的梯度'''
    base = -y.swapaxes(1, 2) @ y
    n, _, d = y.shape
    return base + tensor_diag(y, n, d)


def mse_loss(pred, y):
    '''均方误差损失'''
    return ((y - pred)**2).sum() / len(y)


def deriv_mse(pred, y):
    '''MSE函数的梯度'''
    return pred - y


def ce_loss(pred, y):
    '''交叉熵损失函数'''
    return -np.sum(xlogy(y / len(y), pred))


def deriv_ce(pred, y):
    '''交叉熵损失函数的梯度'''
    return -y / pred


def function_state(func_name: str):
    '''函数注册'''
    return {
        "linear": (linear, deriv_linear),
        "sigmoid": (sigmoid, deriv_sigmoid),
        "relu": (relu, deriv_relu),
        "leaky": (leaky_relu, deriv_leaky),
        "softmax": (softmax, deriv_softmax),
        "elu": (elu, deriv_elu),
        "tanh": (tanh, deriv_tanh),
        "mse": (mse_loss, deriv_mse),
        "ce": (ce_loss, deriv_ce),
    }[func_name]
