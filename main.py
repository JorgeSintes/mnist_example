from data_load import data_load
from train_test_model import train_test_model
from models import LNet, CLNet, CPLNet
import torch
import numpy as np


def main():
    train_size, test_size = 1000, 500
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 50

    model = CPLNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train, y_train, X_test, y_test = data_load("mnist.npz", train_size, test_size)

    train_acc, test_acc, losses = train_test_model(
        model,
        criterion,
        optimizer,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        num_epochs,
    )


if __name__ == "__main__":
    main()
