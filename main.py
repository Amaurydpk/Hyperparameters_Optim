## IMPORTS ##
import torch
from blackBoxes import evaluateBlackbox
from randomSearch import randomSearch
from constants import MAX_BB_EVAL
import time

torch.manual_seed(19) # Set seed for reproducible results


### MAIN ###
if __name__ == '__main__':
    
    t0 = time.time()

    # Random Search
    modelType = "cnn"
    #modelType = "cnn"
    dataSet = "fashion"
    #dataSet = "cifar-10"
    randomSearch(evaluateBlackbox, modelType, dataSet, nbTrials=MAX_BB_EVAL)
   
    print("\nExecution time : " + str(int(time.time() - t0)) + " seconds")