import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


def trainOneEpoch(model, trainLoader, optimizer, device):
    """
    Train the model on the training set for a single epoch 

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (torch.optim) optimizer: the optimizer to use
    :param (torch.device) device: cuda or cpu

    :return: (nn.Sequential) the trained model
    """
    model.train()
    fLoss = torch.nn.CrossEntropyLoss()
    trainingLoss = 0
    for batch, (X, y) in enumerate(trainLoader):
        X, y = X.to(device), y.to(device)
        X = X.view(X.size(0), -1)
        # Forward pass
        pred = model(X)
        loss = fLoss(pred, y)
        trainingLoss += loss.item()
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def accuracy(model, loader, device):
    """
    Return the accuracy of the model on training, validation or test set

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) loader: the given dataset
    :param (torch.device) device: cuda or cpu

    :return: (float, float) the accuracy of the model on the set in % and the mean loss
    """
    # Disable graph and gradient computations for speed and memory efficiency
    with torch.no_grad():
        model.eval()
        loss, nbCorrect = 0.0, 0
        size = len(loader.dataset)
        fLoss = torch.nn.CrossEntropyLoss()
        for(X, y) in loader:
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            # Forward pass
            pred = model(X)
            # the class with the highest value is what we choose as prediction
            _, predicted = torch.max(pred.data, 1)
            # Losses and nb of correct predictions
            loss += fLoss(pred, y).item()
            nbCorrect += (predicted == y).sum().item()
    return float(nbCorrect / size) * 100, loss / size # accuracy and mean loss



def train(model, trainLoader, validLoader, nbEpochs, opt, learningRate, device):
    """
    Train on all epochs the model on the training set and select the best trained model 
    based on accuracy on the validation set

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) validLoader: validation set
    :param (int) nbEpochs: number of epochs 
    :param (str) opt: the name of the optimizer to use
    :param (float) learningRate: the value of the learning rate
    :param (torch.device) device: cuda or cpu

    :return: (nn.Sequential) the trained model
    """
    # if opt == "Adam":
    #     optimizer = optim.Adam(model.parameters(), betas=(b1, b2), lr=lr, weight_decay=0.05)
    # elif opt == "ASGD":
    #     optimizer = optim.ASGD(model.parameters(), lr=lr, lambd=b2, alpha=b1)
    # else:
    #     raise Exception("Optimizer must be a string between Adam or ASGD")

    optimizer = getattr(optim, opt)(model.parameters(), lr=learningRate)
    sched = scheduler.ReduceLROnPlateau(optimizer, 'min')

    bestPrecision = 0
    trainingLosses = []
    accuracies = []
    validationLosses = []

    for epoch in range(nbEpochs):  # loop over the dataset multiple times        
        # Train
        model = trainOneEpoch(model, trainLoader, optimizer, device)

        # Accuracy on training set
        trainPrecision, train_loss = accuracy(model, trainLoader, device)
        trainingLosses.append(train_loss)

        # Accuracy on validation set
        precision, validationLoss = accuracy(model, validLoader, device)
        validationLosses.append(validationLoss)
        accuracies.append(precision)

        if precision > bestPrecision:
            bestPrecision = precision
            bestModel = model

        # Scheduler
        sched.step(validationLoss)

    return bestModel
    








# def test(model, testLoader, device):
#     """
#     Test the model by iterating over the test set and return the accuracy

#     :param (nn.Sequential) model: a nn network
#     :param (DataLoader) testLoader: test set
#     :param (torch.device) device: cuda or cpu

#     :return: (float) the accuracy of the model on the test set in %
#     """
#     correct = 0 # Number of correct prediction
#     model.eval()
#     for batch, (X, y) in enumerate(testLoader):
#         X, y = X.to(device), y.to(device)
#         X = X.view(X.size(0), -1)
#         pred = model(X)
#         # the class with the highest value is what we choose as prediction
#         _, predicted = torch.max(pred.data, 1)
#         correct += (predicted == y).sum().item()
#     accuracy = 100 * correct / len(testLoader.sampler)
#     return accuracy


# def trainAndTest(model, trainLoader, testLoader, HPs):
#     """
#     Train and test the model and return the accuracy

#     :param (nn.Sequential) model: a nn network
#     :param (DataLoader) trainLoader: training set
#     :param (DataLoader) testLoader: test set
#     :param (dictionnary) HPs: a set of hyperparameters

#     :return: (float) the accuracy of the model on the test set in %
#     """
#     # GPU or CPU
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     # print(use_cuda)
#     # print(device)
#     if use_cuda:
#         model = model.cuda()
#     # Training
#     model = train(model, trainLoader, HPs['epochs'], nn.CrossEntropyLoss(), HPs['optimizer'], HPs['learningRate'], device)
#     # Testing
#     accuracy = test(model, testLoader, device)
#     return accuracy

