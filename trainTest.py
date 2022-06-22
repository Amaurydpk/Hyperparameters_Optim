import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import time
from constants import PRINT


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
            # Forward pass
            pred = model(X)
            # the class with the highest value is what we choose as prediction
            _, predicted = torch.max(pred.data, 1)
            # Losses and nb of correct predictions
            loss += fLoss(pred, y).item()
            nbCorrect += (predicted == y).sum().item()
    return float(nbCorrect / size) * 100, loss / size # accuracy and mean loss



def train(model, trainLoader, validLoader, nbEpochs, opt, learningRate, weightDecay, optimParam1, optimParam2, device):
    """
    Train on all epochs the model on the training set and select the best trained model 
    based on accuracy on the validation set

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) validLoader: validation set
    :param (int) nbEpochs: number of epochs 
    :param (str) opt: the name of the optimizer to use
    :param (float) learningRate: the value of the learning rate
    :param (float) weightDecay: the weight decay for the optimizer (L2 penalty)
    :param (float) optimParam1: a param for the optimizer
    :param (float) optimParam2: a param for the optimizer
    :param (torch.device) device: cuda or cpu

    :return: (nn.Sequential) the trained model
    """
    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), betas=(optimParam1, optimParam2), lr=learningRate, weight_decay=weightDecay)
    elif opt == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=learningRate, lambd=optimParam1, alpha=optimParam2, weight_decay=weightDecay)
    elif opt == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learningRate, lr_decay=optimParam1, initial_accumulator_value=optimParam2, weight_decay=weightDecay)
    elif opt == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learningRate, momentum=optimParam1, alpha=optimParam2, weight_decay=weightDecay)
    
    #optimizer = getattr(optim, opt)(model.parameters(), lr=learningRate)
    sched = scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    
    bestAcc = 0

    listTrainAcc = []
    listTrainLoss = []
    listValLoss = []
    listValAcc = []
    
    stop = False # For early stopping
    epoch = 1

    #if torch.cuda.is_available():
        #model = torch.nn.DataParallel(model)
    
    t0 = time.time()
    executionTime = 0
   
    while (not stop) and (epoch <= nbEpochs):  # loop over the dataset multiple times
        if PRINT:
            print("> Epoch {}".format(epoch))        
        
        # Train
        model = trainOneEpoch(model, trainLoader, optimizer, device)

        # Accuracy on training set
        trainAcc, trainLoss = accuracy(model, trainLoader, device)
        listTrainLoss.append(trainLoss)
        listTrainAcc.append(trainAcc)

        # Accuracy on validation set
        valAcc, valLoss = accuracy(model, validLoader, device)
        listValLoss.append(valLoss)
        listValAcc.append(valAcc)

        executionTime = time.time() - t0

        if PRINT:
            print("\tExecution time: {:.2f}, Train accuracy: {:.2f}, Val accuracy: {:.2f}".format(executionTime, trainAcc, valAcc))

        if valAcc > bestAcc:
            bestAcc = valAcc
            bestModel = model
        
        # Early stopping
        if (epoch >= nbEpochs/5) and (bestAcc < 20):
            stop = True
            if PRINT:
                print("\tEarly stopped")

        # Scheduler
        sched.step(valLoss)

        epoch += 1

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

