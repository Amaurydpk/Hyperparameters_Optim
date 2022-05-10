import torch
import torch.optim.lr_scheduler as lr_scheduler # provides several methods to adjust the learning rate based on the number of epochs

def train(model, dataloader, fLoss, optimizer, device, nbEpochs):
    """
    Train a model for one epoch, iterating over the dataloader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
        model      -- A torch.nn.Module object
        dataloader -- A torch.utils.data.DataLoader
        fLoss      -- The loss function, i.e. a loss Module
        optimizer  -- A torch.optim.Optimzer object
        device     -- a torch.device class specifying the device used for computation
        nbEpochs   -- int for the number of loop over the dataset
    Returns :
    """
    size = len(dataloader.dataset)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    model.train() # We enter train mode
    for epoch in range(nbEpochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = fLoss(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Display progression
            if batch % 2000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:.3f}  [{current:>5d}/{size:>5d}]")

        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        print('\n')


def test(model, dataloader, device):
    """
    Test a model by iterating over the loader
    Arguments :
        model      -- A torch.nn.Module object
        dataloader -- A torch.utils.data.DataLoader
        f_loss     -- The loss function, i.e. a loss Module
        device     -- The device to use for computation 
    Returns :
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
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
            

