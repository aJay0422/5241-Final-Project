import torch
import torch.nn as nn
from model.utils import get_loss_acc





def train(model, epochs, trainloader, testloader, optimizer, criterion):
    """

    :param model: the model you want to train
    :param epochs: number of epochs
    :param trainloader: training data
    :param testloader: testing data
    :param optimizer: optimizer
    :param criterion: loss function
    :return: (model, training_record, best_model_state)
            model: the trained model
            training_record: a record of train and test loss and accuracy
            best_model_state: best(loss) model evaluated on testloader
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_record = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    model.train()
    best_test_loss = 0
    best_model_state = model.state_dict()
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        for X_batch, Y_batch in trainloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            num_batches += 1
            # forward
            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
            training_record["train_loss"].append(train_loss)
            training_record["train_acc"].append(train_acc)

            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
            training_record["test_loss"].append(test_loss)
            training_record["test_acc"].append(test_acc)

        print("Epoch {}/{}  train_loss={} test_loss={} train_acc={} test_acc={}".format(
            epoch, epochs, training_record["train_loss"][-1], training_record["test_loss"][-1],
            training_record["train_acc"][-1], training_record["test_acc"][-1]))

        # save model state if it's the best
        if training_record["test_loss"][-1] < best_test_loss:
            best_model_state = model.state_dict()

    return model, training_record, best_model_state


def train7(model, epochs, trainloader, testloader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_record = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    model.train()
    best_test_acc = 0
    best_model_state = model.state_dict()
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        for X_batch, Y_batch in trainloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            num_batches += 1
            # forward
            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
            training_record["train_loss"].append(train_loss)
            training_record["train_acc"].append(train_acc)

            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
            training_record["test_loss"].append(test_loss)
            training_record["test_acc"].append(test_acc)

        print("Epoch {}/{}  train_loss={} val_loss={} train_acc={} val_acc={}".format(
            epoch, epochs, training_record["train_loss"][-1], training_record["test_loss"][-1],
            training_record["train_acc"][-1], training_record["test_acc"][-1]))

        # save model state if it's the best
        if training_record["test_acc"][-1] < best_test_acc:
            best_model_state = model.state_dict()

    return model, training_record, best_model_state


def train_ae(model, epochs, trainloader, testloader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_record = {
        "train_loss": []
    }
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        for X_batch, _ in trainloader:
            X_batch = X_batch.to(device)
            num_batches += 1
            # forward
            outputs = model(X_batch)

            loss = criterion(outputs, X_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print("Epoch {}/{}  train_loss={}".format(
            epoch, epochs, train_loss / num_batches))

    return model