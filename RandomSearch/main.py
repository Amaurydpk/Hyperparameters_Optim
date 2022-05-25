## IMPORTS ##
import torch
from loadFashionMnist import loadFashionMNIST
from randomSearch import randomSearch, randomSearchWithStatOptim
from ..constants import hyperParamsRange

torch.manual_seed(19)

### MAIN ###
if __name__ == '__main__':
    trainLoader, testLoader = loadFashionMNIST()
    randomSearch(trainLoader, testLoader, hyperParamsRange, nbTrials=5)
    #randomSearchWithStatOptim(trainLoader, testLoader, hyperParamsRange, nbTrials=5)

