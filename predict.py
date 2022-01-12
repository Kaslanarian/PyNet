import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from net import Net
from net_io import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="net model filename",
    )
    parser.add_argument(
        "data",
        help="dataset filename",
    )
    parser.add_argument(
        "output",
        help="predict output filename",
    )
    parser.add_argument(
        "-std",
        help="whether standard scale training data",
        action="store_true",
    )
    args = parser.parse_args()

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

    model: Net = load_model(args.model)
    y_pred = model.predict(X)
    y_true = np.argmax(y, axis=1)
    print("Accuracy : {:.2f}%".format(100 * accuracy_score(y_true, y_pred)))
    np.savetxt("./data/{}".format(args.output), y_pred, "%d")
