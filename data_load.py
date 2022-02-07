import torch
import numpy as np


def data_load(path, train_size, test_size, transfer_to_device=True):
    data = np.load(path)
    X_train = torch.Tensor(data["X_train"])
    y_train = torch.Tensor(data["y_train"])

    X_test = torch.Tensor(data["X_test"])
    y_test = torch.Tensor(data["y_test"])

    if transfer_to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

    return X_train, y_train, X_test, y_test
