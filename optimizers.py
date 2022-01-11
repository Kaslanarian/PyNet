import numpy as np


class BaseOptimizer:
    '''优化器基类'''
    def __init__(self, params_list: list) -> None:
        self.params = params_list
        self.n_params = len(params_list)

    def step(self, *grad_params):
        assert len(grad_params) == self.n_params

    def get_params(self):
        return self.params


class BGD(BaseOptimizer):
    '''普通BGD'''
    def __init__(self, params_list: list, lr=0.1) -> None:
        super().__init__(params_list)
        self.lr = lr

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.params[param_id] -= self.lr * grad_params[param_id]

    def get_params(self):
        return super().get_params()


class FIX_DECAY_BGD(BaseOptimizer):
    '''
    学习率固定衰减方法下的BGD
    可选的方法(method)有
    - invt : 逆时衰减
    - exp  : 指数衰减
    - nexp:自然指数衰减
    '''
    def __init__(self,
                 params_list: list,
                 lr=1,
                 method="exp",
                 beta=0.9) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.method = method
        self.epoch = 0
        self.beta = beta

    def step(self, *grad_params):
        super().step(*grad_params)

        self.lr *= {
            "invt": (1 - 1 / (1 + self.beta * self.epoch + self.beta)),
            "exp": self.beta,
            "nexp": np.exp(-self.beta),
        }[self.method]

        for param_id in range(self.n_params):
            self.params[param_id] -= self.lr * grad_params[param_id]
        self.epoch += 1

    def get_params(self):
        return super().get_params()


class WARM_UP_BGD(BaseOptimizer):
    '''预热学习率BGD，但不能作为整个训练过程的BGD'''
    def __init__(self, params_list: list, lr=0.5, T=10) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.t = 1
        self.T = 10

    def step(self, *grad_params):
        super().step(*grad_params)
        assert self.t <= self.T  # 在指定轮数过后，预热结束
        for param_id in range(self.n_params):
            self.params[param_id] -= (self.t /
                                      self.T) * self.lr * grad_params[param_id]
        self.t += 1

    def get_params(self):
        return super().get_params()


class Momentum(BaseOptimizer):
    '''动量机制的梯度下降'''
    def __init__(self, params_list: list, lr=0.1, momentum=0.9) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in params_list]

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.v[param_id] *= self.momentum
            self.v[param_id] += self.lr * grad_params[param_id]
            self.params[param_id] -= self.v[param_id]

    def get_params(self):
        return super().get_params()


class Adagrad(BaseOptimizer):
    '''Adagrad下降'''
    def __init__(self, params_list: list, lr=0.01) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.G = [np.zeros(param.shape) for param in params_list]

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.G[param_id] += grad_params[param_id]**2
            self.params[param_id] -= self.lr * grad_params[param_id] / np.sqrt(
                1e-8 + self.G[param_id])

    def get_params(self):
        return super().get_params()


class Adadelta(BaseOptimizer):
    '''Adadelta下降'''
    def __init__(self, params_list: list, lr=0.001, gamma=0.9) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.gamma = gamma
        self.G = [np.zeros(param.shape) for param in params_list]

    def step(self, *grad_params):
        super().step(*grad_params)
        gamma, lr = self.gamma, self.lr
        for param_id in range(self.n_params):
            self.G[param_id] = gamma * self.G[param_id] + (
                1 - gamma) * grad_params[param_id]**2
            self.params[param_id] -= lr * grad_params[param_id] / np.sqrt(
                self.G[param_id] + 1e-8)

    def get_params(self):
        return super().get_params()


class Adam(BaseOptimizer):
    '''Adam下降'''
    def __init__(self,
                 params_list: list,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999) -> None:
        super().__init__(params_list)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros(param.shape) for param in params_list]
        self.v = [np.zeros(param.shape) for param in params_list]
        self.epoch = 0

    def step(self, *grad_params):
        super().step(*grad_params)
        lr, beta1, beta2 = self.lr, self.beta1, self.beta2
        self.epoch += 1
        for param_id in range(self.n_params):
            self.m[param_id] = beta1 * self.m[param_id] + (
                1 - beta1) * grad_params[param_id]
            self.v[param_id] = beta2 * self.v[param_id] + (
                1 - beta2) * grad_params[param_id]**2
            m_t = self.m[param_id] / (1 - beta1**self.epoch)
            v_t = self.v[param_id] / (1 - beta2**self.epoch)

            self.params[param_id] -= lr * m_t / (np.sqrt(v_t) + 1e-8)

    def get_params(self):
        return super().get_params()