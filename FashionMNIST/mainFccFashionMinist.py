import torch
import torch.nn as nn
import torch.optim as optim
from FccFashionMnist import FullyConnectedReLU, FullyConnectedSigmoid
from loadFashionMnist import trainLoader, validLoader, testLoader
from trainTestFashionMnist import train, test

torch.manual_seed(19)

### PARAMS ###
INPUT_SIZE = 1*28*28
NUM_CLASSES = 10
EPOCHS = 5

### MAIN ###
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FullyConnectedReLU(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model.to(device)
    fLoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, trainLoader, validLoader, fLoss, optimizer, device, nbEpochs=EPOCHS)
    test(model, testLoader, device)