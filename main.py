## IMPORTS ##
import torch
from loadFashionMnist import loadDataSetFashionMNIST
from randomSearch import randomSearch, randomSearchWithStatOptim
from constants import HPrangeFashionMnist

torch.manual_seed(19)

### MAIN ###
if __name__ == '__main__':
    ## Random Search on Fashion Mnist ##
    trainLoader, testLoader = loadDataSetFashionMNIST()
    randomSearch(trainLoader, testLoader, HPrangeFashionMnist, nbTrials=5)
    #randomSearchWithStatOptim(trainLoader, testLoader, hyperParamsRange, nbTrials=5)

