import numpy as np
from net import Net
from functional import *
from os import remove

temp_path = "./model/param"


def save_model(net: Net, name: str):
    '''
    将网络信息保存

    parameters
    ----------
    net : 神经网络类
    name : 文件名，文件将被保存到model文件夹中的指定名称文件中

    return
    ------
    1 : 表示保存成功
    '''
    path = "./model/{}".format(name)

    args = net.args
    layer_info = "layer info:\n"
    for layer in args:
        layer_info += "{} {}\n".format(*layer)

    criterion = "criterion : {}\n".format("ce" if net.criterion ==
                                          ce_loss else "mse")

    regualarize = "regularize : " + ("{} with alpha={}\n".format(
        net.regularize, net.alpha) if net.regularize else "None\n")
    with open(path, "w") as f:
        f.write(layer_info)
        f.write(criterion)
        f.write(regualarize)

        for param in net.parameters():
            np.savetxt(temp_path, param)
            with open(temp_path, "r") as fa:
                f.write(fa.read())
    remove(temp_path)
    return 1


def load_model(name: str):
    '''
    指定文件名，函数将读取文件，生成文件中描述的神经网络模型

    return
    ------
    net : 模型文件所描述的网络
    '''
    path = "./model/{}".format(name)
    parameters = []

    with open(path, "r") as f:
        f.readline()  # 读掉第一行
        layer_info = []
        while True:
            s = f.readline()[:-1]
            if "criterion" in s:
                break
            n, act = s.split()
            layer_info.append((eval(n), act))
        criterion = s.split(" : ")[-1]
        s = f.readline()
        if "alpha" in s:  # 有正则化设置
            regualarize = s[:2]
            alpha = eval(s.split("=")[-1])
        else:
            regualarize = None
            alpha = 0.01

        net = Net(
            *layer_info,
            criterion=criterion,
            regularize=regualarize,
            alpha=alpha,
        )

        for l in range(len(layer_info) - 1):
            i, o = layer_info[l][0], layer_info[l + 1][0]

            str_W = "".join([f.readline() for l in range(i)])
            str_b = f.readline()
            with open(temp_path, "w") as fw:
                fw.writelines(str_W)
            W = np.loadtxt(temp_path).reshape(i, o)
            with open(temp_path, "w") as fb:
                fb.writelines(str_b)
            b = np.loadtxt(temp_path).reshape(1, o)

            parameters.extend((W, b))

    net.reset_net(parameters)
    remove(temp_path)
    return net


def random_init(net: Net, path="./data/random.npy"):
    '''用指定数组来初始化参数'''
    n_layer = net.ct_layer
    n_weight_list = [
        n_layer[i] * n_layer[i + 1] for i in range(len(n_layer) - 1)
    ]
    parameters = []
    x = np.load(path)[:sum(n_weight_list)]
    ptr = 0
    for i in range(len(n_layer) - 1):
        W = x[ptr:ptr + n_weight_list[i]].reshape((n_layer[i], n_layer[i + 1]))
        b = np.zeros((1, n_layer[i + 1]))
        parameters.extend((W, b))
        ptr += n_weight_list[i]

    net.reset_net(parameters, net.xavier, net.he)
    return net
