## Introduction
This is a project on hand-written digits classification. The dataset we use is MNIST.  
For question 3,4 and 5, we run several neural networks to predict hand-written digits.  
For question 6 and 7, we run several neural networks to predict the sum of 2 digits.  


## Dataset
We are using MNIST dataset for the project. The following codes create datasets and 
dataloader objects for training and testing:


```python
trainset = my_dataset(X_train_tensor, Y_train_tensor)
testset = my_dataset(X_test_tensor, Y_test_tensor)
trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = data.DataLoader(testset, batch_size=20)
```



## Models
All models are available in the `models.py` file.
### Single digit prediction
We used 3 different models for digits classification. They are NN(feedforward neural network 
with 1 hidden layer), CNN(convolutional neural network with 2 convolutional layers) and 
CNNplus(adding batchnorm layers in the previous structure). Here is the code for creating and
training the models(take NN for example).

```python
EPOCHS = 150
model = NN()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  
criterion = nn.CrossEntropy()  
model, training_record, best_state = train(model, EPOCHS, trainloader, testloader, optimizer, criterion)
```

The `train` function here is available in the `train.py` file. It not only trains the given model, but also record training and testing loss and 
accuracy during the training process. Meanwhile, it records the best model state according to the testloader.

The prediction accuracy we achieved is shown below:

|         | Test Acc |
|---------|----------|
| NN      | 97.95%   |
| CNN     | 99.25%   |
| CNNplus | 99.16%   |


### two digits sum prediction
For digits sum prediction, we used NN, CNN and DoubleCNN to directly predict the sum of 2 digits. We also 
trained an autoencoder to extract features from a single-digit image.  
Based on the image encoder and MNIST 
dataset, we trained a single-digit classifier. Then we use that classifier to predict digits from the double-
digits image and do the sum operation by hand. Here is an example of how we trained the autoencoder and use it as
a basic classifier to predict digits sum.

```python
EPOCHS = 40
convae = ConvAE()
optimizer = torch.optim.Adam(convae.parameters(), lr=0.003)
criterion = nn.BCELoss()
convae = train_ae(convae, EPOCHS, train_loader, val_loader, optimizer, criterion).cpu()

aeclf = AEclf(convae.encoder)
EPOCHS = 50
optimizer = torch.optim.SGD(aeclf.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
aeclf, training_record, best_state = train(aeclf, EPOCHS, trainloader, testloader, optimizer, criterion)
```

The prediction accuracy we achieved is shown below:

|           | Train Acc | Valid Acc | Test Acc |
|:---------:|:---------:|:---------:|:--------:|
|    NN     |  98.06%   |  68.88%   |  71.02%  |
|    CNN    |   100%    |  89.62%   |  91.02%  |
| DoubleCNN |   100%    |  91.42%   |  92.28%  |
|   AEclf   |   100%    |   100%    |  98.40%  |
