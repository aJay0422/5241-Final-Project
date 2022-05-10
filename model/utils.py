import torch.nn as nn
import torch


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def get_loss_acc(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        total += len(Y_batch)
        num_batches += 1
        outputs = model(X_batch)
        y_pred = torch.argmax(outputs, dim=1)
        correct += torch.sum(y_pred == Y_batch).cpu().numpy()
        loss = criterion(outputs, Y_batch)
        total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc