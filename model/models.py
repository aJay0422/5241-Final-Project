import torch.nn as nn
import torch.nn.functional as F
import torch



class NN(nn.Module):
    def __init__(self, input_size=784, hidden_size=100):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        x = x.reshape(-1, 784)
        return self.fc(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 6 * 64)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))

        return x


class CNNplus(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNplus, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class NN7(nn.module):
    def __init__(self):
        super(NN7, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 2, 200)
        self.fc2 = nn.Linear(200, 19)

    def forward(self, x):
        x = x.view(-1, 28 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN7(nn.Module):
    def __init__(self):
        super(CNN7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(6 * 13 * 32, 256)
        self.fc2 = nn.Linear(256, 19)

    def forward(self, x):
        x = x.view(-1, 1, 28, 56)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 6 * 13 * 32)
        x = self.fc2(F.relu(self.fc1(x)))

        return x


class DoubleCNN(nn.Module):
    def __init__(self):
        super(DoubleCNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.out = nn.Sequential(
            nn.Linear(6 * 6 * 64 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 19),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        img = x.reshape(-1, 1, 28, 56)
        img1 = img[:, :, :, :28]
        img2 = img[:, :, :, 28:]
        output1 = self.CNN(img1).reshape(-1, 6 * 6 * 64)
        output2 = self.CNN(img2).reshape(-1, 6 * 6 * 64)
        output = torch.concat((output1, output2), dim=1)
        output = self.out(output)

        return output

class ConvAE(nn.Module):
  def __init__(self):
    super(ConvAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=2),
        nn.MaxPool2d(2,2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.MaxPool2d(2,2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2)
    )
    self.fc = nn.Sequential(
        nn.Linear(128 * 3 * 3, 400),
        nn.ReLU(inplace=True),
        nn.Linear(400, 128 * 3 * 3)
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = x.reshape(-1, 1, 28, 56)
    img1 = x[:,:,:,:28]
    img2 = x[:,:,:,28:]

    output1 = self.encoder(img1)
    output1 = output1.reshape(-1, 128 * 3 * 3)
    output1 = self.fc(output1)
    output1 = output1.reshape(-1, 128, 3, 3)
    output1 = self.decoder(output1)

    output2 = self.encoder(img2)
    output2 = output2.reshape(-1, 128 * 3 * 3)
    output2 = self.fc(output2)
    output2 = output2.reshape(-1, 128, 3, 3)
    output2 = self.decoder(output2)

    output = torch.concat((output1, output2), dim=3)
    output = output.squeeze()
    return output

class Prediction(nn.Module):
  def __init__(self, clf):
    super(Prediction, self).__init__()
    self.CNN = clf.cpu()

  def forward(self, x):
    img = x.reshape(-1, 1, 28, 56)
    img1 = img[:, :, :, :28]
    img2 = img[:, :, :, 28:]
    pred1 = torch.argmax(self.CNN(img1), dim=1)
    pred2 = torch.argmax(self.CNN(img2), dim=1)
    pred = pred1 + pred2
    return pred

class AEclf(nn.Module):
    def __init__(self, encoder, freeze=True):
        super(AEclf, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        if freeze:
            for params in self.encoder.parameters():
                params.require_grads = False