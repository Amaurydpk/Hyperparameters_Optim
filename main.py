## IMPORTS ##
import torch
from blackBoxes import evaluateBlackboxFashion
from randomSearch import randomSearch, randomSearchWithStatOptim
from constants import HPrangeFashionMnist, DATASET_DIR, BATCH_SIZE_FASHION, VALID_RATIO

torch.manual_seed(19) # Set seed for reproducible results


### MAIN ###
if __name__ == '__main__':
    ## Random Search on Fashion MNIST ##
    randomSearch(evaluateBlackboxFashion, HPrangeFashionMnist, nbTrials=5)
    #randomSearchWithStatOptim(trainLoader, testLoader, hyperParamsRange, nbTrials=5)

