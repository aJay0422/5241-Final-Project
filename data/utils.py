import torchvision
from sklearn import preprocessing
import torch
from torch.utils import data
import pandas as pd


class my_dataset(data.Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.label = Y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def get_mnist(type = "dataloader"):
    """
    :param type: "array", "tensor", "dataset" or "dataloader"
    :return: nparray, tensor, dataset object or dataloader object
    """
    train_data = torchvision.datasets.MNIST(
        root = './.mnist/',
        train = True,
        download = True
    )

    test_data = torchvision.datasets.MNIST(
        root='./.mnist/',
        train=False,
        download=True
    )

    #change features to numpy
    X_train = train_data.data.numpy()
    X_test = test_data.data.numpy()
    #change labels to numpy
    Y_train = train_data.targets.numpy()
    Y_test = test_data.targets.numpy()

    #scale data (minmax)
    X_train_scaled = X_train.reshape(X_train.shape[0], -1)
    X_test_scaled = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = preprocessing.minmax_scale(X_train_scaled)
    X_test_scaled = preprocessing.minmax_scale(X_test_scaled)
    if type == "array":
        return X_train_scaled, Y_train, X_test_scaled, Y_test

    #get tensor
    X_train_tensor = torch.Tensor(X_train_scaled)
    X_test_tensor = torch.Tensor(X_test_scaled)
    Y_train_tensor = torch.LongTensor(Y_train)
    Y_test_tensor = torch.LongTensor(Y_test)
    if type == "tensor":
        return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor

    #get dataset
    trainset = my_dataset(X_train_tensor, Y_train_tensor)
    testset = my_dataset(X_test_tensor, Y_test_tensor)
    if type == "dataset":
        return trainset, testset

    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=20)
    if type == "dataloader":
        return trainloader, testloader

def get_mnist2(type = "dataloader"):
    X_train = pd.read_csv('train.txt', header=None).to_numpy()
    Y_train = X_train[:, -1]
    X_train = X_train[:, :-1]
    X_val = pd.read_csv('val.txt', header=None).to_numpy()
    Y_val = X_val[:, -1]
    X_val = X_val[:, :-1]
    X_test = pd.read_csv('test.txt', header=None).to_numpy()
    Y_test = X_test[:, -1]
    X_test = X_test[:, :-1]

    #minmax scale
    X_train_scaled = X_train.reshape(X_train.shape[0], -1)
    X_val_scaled = X_val.reshape(X_val.shape[0], -1)
    X_test_scaled = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = preprocessing.minmax_scale(X_train_scaled)
    X_val_scaled = preprocessing.minmax_scale(X_val_scaled)
    X_test_scaled = preprocessing.minmax_scale(X_test_scaled)
    if type == "array":
        return X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test

    #get tneosr
    X_train_tensor = torch.Tensor(X_train_scaled.reshape(-1, 28, 56))
    X_val_tensor = torch.Tensor(X_val_scaled.reshape(-1, 28, 56))
    X_test_tensor = torch.Tensor(X_test_scaled.reshape(-1, 28, 56))
    Y_train_tensor = torch.LongTensor(Y_train)
    Y_val_tensor = torch.LongTensor(Y_val)
    Y_test_tensor = torch.LongTensor(Y_test)
    if type == "tensor":
        return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor, X_test_tensor, Y_val_tensor

    #get dataset
    trainset = my_dataset(X_train_tensor, Y_train_tensor)
    valset = my_dataset(X_val_tensor, Y_val_tensor)
    testset = my_dataset(X_test_tensor, Y_test_tensor)
    if type == "dataset":
        return trainset, valset, testset

    #get dataloader
    train_loader = data.DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(valset, batch_size=20)
    test_loader = data.DataLoader(testset, batch_size=20)
    if type == "dataloader":
        return train_loader, val_loader, test_loader