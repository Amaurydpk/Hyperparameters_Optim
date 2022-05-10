import matplotlib.pyplot as plt
import torch
import numpy as np

def train(model, trainLoader, validLoader, fLoss, optimizer, device, nbEpochs):
    """
    Train a model for one epoch, iterating over the dataloader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
        model       -- A torch.nn.Module object
        trainLoader -- A torch.utils.data.DataLoader for training
        validLoader -- A torch.utils.data.DataLoader for validation
        fLoss       -- The loss function, i.e. a loss Module
        optimizer   -- A torch.optim.Optimzer object
        device      -- a torch.device class specifying the device used for computation
        nbEpochs    -- int for the number of loop over the dataset
    Returns : None
    """

    train_loss_history=[]
    val_loss_history=[]
    train_acc_history=[]
    val_acc_history=[]

    for epoch in range(nbEpochs):  # loop over the dataset multiple times        
        ## Training
        train_loss = 0.0
        train_correct = 0
        model.train()
        for batch, (X, y) in enumerate(trainLoader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = fLoss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*X.size(0)
            scores, predictions = torch.max(pred.data, 1)
            train_correct += (predictions == y).sum().item()
        train_loss = train_loss/len(trainLoader.sampler)
        train_acc = train_correct/len(trainLoader.sampler) * 100
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        ## Validation
        valid_loss=0.0
        val_correct = 0
        model.eval()
        for batch, (X, y) in enumerate(validLoader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = fLoss(pred, y)
            valid_loss += loss.item()*X.size(0)
            scores, predictions = torch.max(pred.data, 1)
            val_correct += (predictions == y).sum().item()
        valid_loss = valid_loss/len(validLoader.sampler)
        val_acc = val_correct/len(validLoader.sampler) * 100
        val_loss_history.append(valid_loss)
        val_acc_history.append(val_acc)
    
        print("Epoch:{}/{} \t TrainLoss:{:.3f} \t ValLoss:{:.3f} \t TrainAcc:{:.2f}% \t ValAcc:{:.2f}%".format(epoch+1, nbEpochs, train_loss, valid_loss, train_acc, val_acc))
    
    # Display plots about loss and accuracy
    _, (ax1, ax2) = plt.subplots(2, 1)
    epochs = np.arange(1, nbEpochs+1)
    # Loss
    ax1.semilogy(epochs, train_loss_history, label='Train')
    ax1.semilogy(epochs, val_loss_history, label='Validation')
    ax1.set_xticks(epochs)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.legend()
    # Accuracy
    ax2.plot(epochs, train_acc_history, label='Train')
    ax2.plot(epochs, val_acc_history, label='Validation')
    ax2.set_xticks(epochs)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid()
    ax2.legend()
    
    plt.show()   


def test(model, dataloader, device):
    """
    Test a model by iterating over the loader
    Arguments :
        model      -- A torch.nn.Module object
        dataloader -- A torch.utils.data.DataLoader
        f_loss     -- The loss function, i.e. a loss Module
        device     -- The device to use for computation 
    Returns : None
    """
    correct, total = 0, 0  
    model.eval() # evaluation mode
    # We disable gradient computation which speeds up the computation and reduces the memory usage
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # We need to copy the data on the GPU if we use one
            outputs = model(X)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy of the network on the test set: {100 * correct // total} %')