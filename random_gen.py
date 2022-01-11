import numpy as np
'''
生成200000个正态分布随机数，用以初始化网络参数，控制变量进行网络优化状态的监视
'''
if __name__ == "__main__":

    N = 200000
    x = np.random.randn(N)
    np.save("./data/random.npy", x)
