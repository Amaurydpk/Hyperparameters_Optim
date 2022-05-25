import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


def train(model, trainLoader, nbEpochs, fLoss, optimizer, learningRate, device):
    """
    Train the model on the training set 

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (dictionnary) HPs: a set of hyperparameters
    :param (int) nbEpochs: number of epochs 
    :param (object) fLoss: this criterion computes the loss between input and target.
    :param (str) optimizer: the name of the optimizer to use
    :param (float) learningRate: the value of the learning rate
    :param (torch.device) device: cuda or cpu

    :return: (None)
    """
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learningRate)
    #sched = scheduler.ExponentialLR(optimizer, gamma=0.9)
    model.train()
    for epoch in range(nbEpochs):  # loop over the dataset multiple times        
        for batch, (X, y) in enumerate(trainLoader):
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            optimizer.zero_grad()
            pred = model(X)
            loss = fLoss(pred, y)
            loss.backward()
            optimizer.step()
    

def test(model, testLoader, device):
    """
    Test the model by iterating over the test set and return the accuracy

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) testLoader: test set
    :param (torch.device) device: cuda or cpu

    :return: (float) the accuracy of the model on the test set in %
    """
    correct = 0 # Number of correct prediction
    model.eval()
    for batch, (X, y) in enumerate(testLoader):
        X, y = X.to(device), y.to(device)
        X = X.view(X.size(0), -1)
        pred = model(X)
        # the class with the highest value is what we choose as prediction
        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == y).sum().item()
    accuracy = 100 * correct / len(testLoader.sampler)
    return accuracy


def trainAndTest(model, trainLoader, testLoader, HPs):
    """
    Train and test the model and return the accuracy

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) testLoader: test set
    :param (dictionnary) HPs: a set of hyperparameters

    :return: (float) the accuracy of the model on the test set in %
    """
    # GPU or CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(use_cuda)
    # print(device)
    if use_cuda:
        model = model.cuda()
    # Training
    train(model, trainLoader, HPs['epochs'], nn.CrossEntropyLoss(), HPs['optimizer'], HPs['learningRate'], device)
    # Testing
    accuracy = test(model, testLoader, device)
    return accuracy

