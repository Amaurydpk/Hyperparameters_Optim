import torch
import torch.nn as nn
import torch.optim as optim
from CnnCifar10 import CNN
from loadCifar10 import trainloader, testloader
from trainTestCifar10 import train, test

torch.manual_seed(19)

### MAIN ###
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    model.to(device)
    fLoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, trainloader, fLoss, optimizer, device, nbEpochs=2)
    test(model, testloader, device)