import numpy as np
from f import *


class Net:
    def __init__(self, *params) -> None:
        # 参数注册
        self.layer_info = params
        self.n_layer = len(self.layer_info)
        self.Ws = [None] * (self.n_layer - 1)  # 权重
        self.bs = [None] * (self.n_layer - 1)  # 偏置
        self.gW = [None] * (self.n_layer - 1)  # 权重梯度
        self.gb = [None] * (self.n_layer - 1)  # 偏置梯度
        self.layer_value = [None] * self.n_layer  # 网络层值

        # 初始化
        for i in range(self.n_layer - 1):
            

class Net:
    def __init__(self, *params) -> None:
        # 参数注册
        self.layer_info = params
        self.n_layer = len(self.layer_info)
        self.params = []
        self.grads = []
        self.layer_value = [None] * self.n_layer

        # 网络参数初始化, 梯度参数初始化
        # 形式：[权重矩阵, 偏置向量, 权重矩阵, 偏置向量, ...]
        for i in range(self.n_layer - 1):
            input, output = self.layer_info[i], self.layer_info[i + 1]
            self.params.extend((
                np.random.normal(size=(input, output)),
                np.random.normal(size=(output)),
            ))
            self.grads.extend((
                np.zeros(shape=(input, output)),
                np.zeros(shape=(output)),
            ))

    def backward(self, y_true):
        '''
        给定输入，求梯度
        '''
        lr = 0.1
        # loss = cross_enretpy(self.layer_value[-1, y_true])
        output = self.layer_value[-1]
        grad_loss = np.expand_dims(deriv_mse(output, y_true), -1)  # l * n * 1
        grad_outp = deriv_sigmoid(output)  # l * n * n

        temp = (grad_outp @ grad_loss).squeeze()  # l * n
        grad_b = -temp.sum(0)  # n2
        grad_B = -self.layer_value[-2].T @ grad_b  # n1 * n2

        grad_a = grad_b @ self.params[-2].T @ deriv_sigmoid(
            self.layer_value[-2])
        grad_A = -self.layer_value[-3].T @ grad_a

    def parameterss(self):
        return self.params

    def forward(self, x):
        self.layer_value[0] = x
        for i in range(self.n_layer - 1):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            x = x @ W + b
            self.layer_value[i + 1] = x @ W + b

    def __call__(self, x):
        self.forward(x)
        return self.layer_value[-1]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    net = Net(4, 5, 3)
    print(iris.data.shape)
    net(iris.data)