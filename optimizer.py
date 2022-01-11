import numpy as np


class BaseOptimizer:
    def __init__(self, *params) -> None:
        self.params = list(params)
        self.n_params = len(params)

    def step(self, *grad_params):
        assert len(grad_params) == self.n_params

    def get_params(self):
        return self.params


class BGD(BaseOptimizer):
    def __init__(self, *params, lr=0.1) -> None:
        super().__init__(*params)
        self.lr = lr

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.params[param_id] -= self.lr * grad_params[param_id]

    def get_params(self):
        return super().get_params()


class NoisyBGD(BaseOptimizer):
    def __init__(self, *params, lr=0.1, gamma=0.9) -> None:
        super().__init__(*params)
        self.lr = lr
        self.gamma = gamma
        self.t = 0

    def step(self, *grad_params):
        super().step(*grad_params)
        sigma = (self.lr / (1 + self.t)**self.gamma)**0.5
        for param_id in range(self.n_params):
            self.params[param_id] -= self.lr * (
                grad_params[param_id] + np.random.normal(
                    size=grad_params[param_id].shape,
                    scale=sigma,
                ))
        self.t += 1

    def get_params(self):
        return super().get_params()


class Momentum(BaseOptimizer):
    def __init__(self, *params, lr=0.1, momentum=0.9) -> None:
        super().__init__(*params)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in params]

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.v[param_id] *= self.momentum
            self.v[param_id] += self.lr * grad_params[param_id]
            self.params[param_id] -= self.v[param_id]

    def get_params(self):
        return super().get_params()


class Nesterov(BaseOptimizer):
    def __init__(self, *params, lr=0.1, momentum=0.9) -> None:
        super().__init__(*params)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in params]
        self.real_params = [np.copy(param) for param in params]

    def step(self, *grad_params):
        '''
        在Nesterov中, grad_params是预测位置的梯度
        '''
        super().step(*grad_params)
        for param_id in range(self.n_params):
            # v_t = gamma * v_{t-1} + lr * grad(theta - gamma * v_{t-1})
            self.v[param_id] = self.momentum * self.v[
                param_id] + self.lr * grad_params[param_id]
            # theta = theta - v_t
            self.real_params[param_id] -= self.v[param_id]
            # theta - gamma * v_t，对于下一轮就是theta - gamma * v_{t-1}
            self.params[param_id] = self.real_params[
                param_id] - self.momentum * self.v[param_id]

    def get_params(self):
        return super().get_params()

    def get_real_params(self):
        return self.real_params


class Adagrad(BaseOptimizer):
    def __init__(self, *params, lr=0.01) -> None:
        super().__init__(*params)
        self.lr = lr
        self.G = [np.zeros(param.shape) for param in params]

    def step(self, *grad_params):
        super().step(*grad_params)
        for param_id in range(self.n_params):
            self.G[param_id] += grad_params[param_id]**2
            self.params[param_id] -= self.lr * grad_params[param_id] / np.sqrt(
                1e-8 + self.G[param_id])

    def get_params(self):
        return super().get_params()


class Adadelta(BaseOptimizer):
    def __init__(self, *params, lr=0.001, gamma=0.9) -> None:
        super().__init__(*params)
        self.lr = lr
        self.gamma = gamma
        self.G = [np.zeros(param.shape) for param in params]

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
    def __init__(self, *params, lr=0.001, beta1=0.9, beta2=0.999) -> None:
        super().__init__(*params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros(param.shape) for param in params]
        self.v = [np.zeros(param.shape) for param in params]
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
