import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import optuna

# Train and evaluate the accuracy of neural network with the addition of pruning mechanism
def train_and_evaluate(trial, params, model, trainLoader, testLoader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(use_cuda)
    # print(device)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['learning_rate'])

    for epoch in range(params['epochs']):  # loop over the dataset multiple times        
        ## Training
        train_loss = 0.0
        train_correct = 0
        model.train()
        for batch, (X, y) in enumerate(trainLoader):
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*X.size(0)
            scores, predictions = torch.max(pred.data, 1)
            train_correct += (predictions == y).sum().item()
        train_loss = train_loss/len(trainLoader.sampler)
        train_acc = train_correct/len(trainLoader.sampler) * 100
    
        ## Validation
        valid_loss=0.0
        val_correct = 0
        model.eval()
        for batch, (X, y) in enumerate(testLoader):
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1)
            pred = model(X)
            loss = criterion(pred, y)
            valid_loss += loss.item()*X.size(0)
            scores, predictions = torch.max(pred.data, 1)
            val_correct += (predictions == y).sum().item()
        valid_loss = valid_loss/len(testLoader.sampler)
        val_acc = val_correct/len(testLoader.sampler) * 100
        
    
        #print("Epoch:{}/{} \t TrainLoss:{:.3f} \t ValLoss:{:.3f} \t TrainAcc:{:.2f}% \t ValAcc:{:.2f}%".format(epoch+1, nbEpochs, train_loss, valid_loss, train_acc, val_acc))
    
        accuracy = val_acc

        # Pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return accuracy