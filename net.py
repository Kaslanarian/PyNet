from math import ceil
import numpy as np
from functional import function_state, ce_loss, relu, sigmoid, softmax, tanh


class Net:
    '''
    神经网络类

    parameters
    ----------
    *args : 注册神经网络的各层神经元数和激活函数
    criterion : 损失函数，可选mse和ce
    regularize : 正则化选项，可选None, L1和L2
    alpha : 正则化项的参数
    xavier : 是否对sigmoid层和tanh层的权重参数进行Xavier初始化
    he : 是否对relu层权重参数进行He初始化

    example
    -------
    ```python
    from net import Net

    # 一个10输入3输出，两个32神经元隐藏层的神经网络
    # 激活函数依次为relu,relu和softmax，交叉熵作为损失函数
    # 采取L2正则化，并进行He初始化
    net = Net(
        (10, "linear"),
        (32, "relu"),
        (32, "relu"),
        (3, "softmax"),
        criterion="ce",
        regualarize="L2",
        he=True
    )
    ```

    '''
    def __init__(
        self,
        *args,
        criterion="ce",
        regularize=None,
        alpha=0.01,
        xavier=False,
        he=False,
    ) -> None:
        self.args = args
        self.nr_layer = len(self.args)
        self.ct_layer = [args[0][0]]
        self.fu_layer = []
        self.df_layer = []
        self.criterion, self.dcriterion = function_state(criterion)
        self.grads = [None] * (self.nr_layer * 2 - 2)
        self.layer_value = [None] * self.nr_layer
        self.xavier = xavier
        self.he = he

        for arg in args[1:]:
            n = arg[0]
            f, df = function_state(arg[1])
            self.ct_layer.append(n)
            self.fu_layer.append(f)
            self.df_layer.append(df)

        self.reset_net(None, xavier, he)

        # 注册正则化
        self.regularize = regularize if regularize in {"L1", "L2"} else None
        self.alpha = alpha

    def forward(self, X):
        '''前向传播'''
        self.layer_value[0] = X
        for i in range(self.nr_layer - 1):
            self.layer_value[i + 1] = self.fu_layer[i](
                self.layer_value[i] @ self.__weight[i] - self.__bias[i])
        return self.layer_value[-1]

    def backward(self, y):
        '''反向传播'''
        if self.criterion == ce_loss and self.fu_layer[-1] == softmax:
            # 特殊公式
            dbias = y - self.layer_value[-1]
        else:
            dy = self.dcriterion(self.layer_value[-1], y)
            dbias = -dy @ self.df_layer[-1](self.layer_value[-1])

        dweight = -self.layer_value[-2].swapaxes(1, 2) * dbias

        if self.regularize == "L1":
            bias_reg_term = np.sign(self.__bias[-1])
            weight_reg_term = np.sign(self.__weight[-1])
        elif self.regularize == "L2":
            bias_reg_term = self.__bias[-1]
            weight_reg_term = self.__weight[-1]
        else:
            bias_reg_term, weight_reg_term = 0, 0

        self.grads[-1] = dbias.mean(0) + self.alpha * bias_reg_term
        self.grads[-2] = dweight.mean(0) + self.alpha * weight_reg_term

        for i in range(2, self.nr_layer):
            dbias = dbias @ self.__weight[-i + 1].T @ self.df_layer[-i](
                self.layer_value[-i])
            dweight = -self.layer_value[-i - 1].swapaxes(1, 2) * dbias

            if self.regularize == "L1":
                bias_reg_term = np.sign(self.__bias[-i])
                weight_reg_term = np.sign(self.__weight[-i])
            elif self.regularize == "L2":
                bias_reg_term = self.__bias[-i]
                weight_reg_term = self.__weight[-i]
            else:
                bias_reg_term, weight_reg_term = 0, 0

            self.grads[-2 * i + 1] = dbias.mean(0) + self.alpha * bias_reg_term
            self.grads[-2 * i] = dweight.mean(0) + self.alpha * weight_reg_term

    def reset_net(self, parameters=None, xavier=False, he=False):
        '''随机初始化参数，或指定参数对网络赋值，可指定是否进行Xavier初始化、He初始化'''
        if parameters == None:
            self.__weight = [
                np.random.randn(self.ct_layer[i], self.ct_layer[i + 1])
                for i in range(self.nr_layer - 1)
            ]
            self.__bias = [
                np.zeros((1, self.ct_layer[i]))
                for i in range(1, self.nr_layer)
            ]
        else:
            for i in range(self.nr_layer - 1):
                self.__weight[i] = parameters[2 * i]
                self.__bias[i] = parameters[2 * i + 1]

        for i in range(self.nr_layer - 1):
            if xavier:
                if self.fu_layer[i] == sigmoid:
                    self.__weight[i] *= 4 * np.sqrt(
                        2 / (self.ct_layer[i] + self.ct_layer[i + 1]))
                elif self.fu_layer[i] == tanh:
                    self.__weight[i] *= np.sqrt(
                        2 / (self.ct_layer[i] + self.ct_layer[i + 1]))
            if he and self.fu_layer[i] == relu:
                self.__weight[i] *= np.sqrt(2 / self.ct_layer[i])

    def predict(self, x):
        '''预测函数，输出指定类别'''
        output = self.forward(np.squeeze(x))
        return np.argmax(output, axis=1)

    def __call__(self, x):
        '''类可作为函数调用，参数为输入，返回是网络输出'''
        return self.forward(x)

    def __str__(self) -> str:
        '''print函数重载，可以直接看到网络参数'''
        ret = "{\n"
        for arg in self.args:
            ret += "  %2d %s\n" % (arg[0], arg[1])
        ret += "}"
        return ret

    def parameters(self):
        '''返回网络参数，作为网络与优化器的接口'''
        paras = [None] * (2 * self.nr_layer - 2)
        for i in range(self.nr_layer - 1):
            paras[2 * i] = self.__weight[i]
            paras[2 * i + 1] = self.__bias[i]
        return paras