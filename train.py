import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from net import Net
from optimizers import BGD, Momentum, Adagrad, Adadelta, Adam
from dataloader import train_loader
from net_io import save_model, random_init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-n",
        help="number of neurons in hidden layer, (default n_input+n_output)",
        type=int,
    )
    parser.add_argument(
        "-ha",
        help='''activation function of hidden layer(default 0)
        0 --- ReLU
        1 --- elu
        2 --- leaky relu with alpha=0.1
        3 --- sigmoid
        4 --- tanh
        5 --- softmax
        ''',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "-oa",
        help='''activation function of output layer(default 5)
        0 --- ReLU
        1 --- elu
        2 --- leaky relu with alpha=0.1
        3 --- sigmoid
        4 --- tanh
        5 --- softmax
        ''',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "-l",
        help='''loss function of network(default 0)
        0 --- cross entropy loss
        1 --- mean square loss
        ''',
        type=int,
        choices=[0, 1],
    )
    parser.add_argument(
        "-he",
        help="whether use He initialization",
        action="store_true",
    )
    parser.add_argument(
        "-xavier",
        help="whether use Xavier initialization",
        action="store_true",
    )
    parser.add_argument(
        "-reg",
        help='''whether use regularization(default 0)
        0 --- No regularize
        1 --- L1 regularize
        2 --- L2 regularize
        ''',
        type=int,
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "-coef",
        help="regularized term coefficient",
        type=float,
    )
    parser.add_argument(
        "-fix",
        help="Use fixed data to initialize net",
        action="store_true",
    )

    parser.add_argument("-e", help="training epoch(default 10)", type=int)
    parser.add_argument("-b", help="batch size(default 32)", type=int)
    parser.add_argument(
        "-opt",
        help='''choose net optimizer(default 0)
        0 --- BGD
        1 --- Momentum
        2 --- Adagrad
        3 --- Adadetla
        4 --- Adam
        ''',
        type=int,
        choices=[0, 1, 2, 3, 4],
    )

    parser.add_argument("-lr", help="learning rate(default=0.1)", type=float)
    parser.add_argument("data", help="path of training data")
    parser.add_argument(
        "-std",
        help="whether standard scale training data",
        action="store_true",
    )
    parser.add_argument("-name",
                        help="filename of model, default {data}.model")

    args = parser.parse_args()

    # process args
    if args.data == "heart_scale":
        from sklearn.datasets import load_svmlight_file
        X, y = load_svmlight_file("./data/heart_scale")
        X = X.toarray()
        y = y.reshape(-1, 1)
    else:
        data = np.loadtxt("./data/" + args.data, delimiter=",")
        X, y = data[:, 0:-1], data[:, -1].reshape(-1, 1)
    if args.std:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    one_hot_encoder = OneHotEncoder().fit(y)
    y = one_hot_encoder.transform(y).toarray()

    n_input, n_output = X.shape[1], y.shape[1]
    n_hidden = args.n if args.n else (n_input + n_output)

    function_list = ["relu", "elu", "leaky", "sigmoid", "tanh", "softmax"]
    loss_list = ["ce", "mse"]
    reg_list = [None, "L1", "L2"]
    optim_list = [BGD, Momentum, Adagrad, Adadelta, Adam]

    model = Net(
        (n_input, "linear"),
        (n_hidden, function_list[args.ha if args.ha else 0]),
        (n_output, function_list[args.oa if args.oa else 5]),
        criterion=loss_list[args.l if args.l else 0],
        regularize=reg_list[args.reg if args.reg else 0],
        alpha=(args.coef if args.coef else 0.01),
        he=args.he,
        xavier=args.xavier,
    )
    if args.fix:
        model = random_init(model)

    optim = optim_list[args.opt if args.opt else 0](
        model.parameters(),
        lr=args.lr if args.lr else 0.1,
    )

    data = train_loader(X, y, batch_size=args.b if args.b else 10)
    print("start training:")
    for epoch in range(args.e if args.e else 10):
        print("*", end='')
        for batch_X, batch_y in data:
            model.forward(batch_X)
            model.backward(batch_y)
            optim.step(*model.grads)
    print("\ntrain over")
    save_model(model, args.name if args.name else "{}.model".format(args.data))
