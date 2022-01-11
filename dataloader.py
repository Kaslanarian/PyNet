import numpy as np


def train_loader(X, y, batch_size, shuffle=False) -> list:
    '''
    对训练集数据进行划分，实现mini-batch机制

    parameters
    ----------
    X : 训练集特征数据
    y : 训练集标签数据
    batch_size : 划分数据的大小
    shuffle : 是否对训练数据进行打乱后再划分

    return
    ------
    以(特征数据, 标签数据)为单位的列表
    '''
    l = X.shape[0]

    if shuffle:
        order = np.random.choice(range(l), l, replace=False)
        X, y = X[order], y[order]

    X = np.expand_dims(np.squeeze(X).squeeze(), axis=1)
    y = np.expand_dims(np.squeeze(y).squeeze(), axis=1)

    X_split = np.split(X, range(0, l, batch_size)[1:])
    y_split = np.split(y, range(0, l, batch_size)[1:])
    return list(zip(X_split, y_split))
