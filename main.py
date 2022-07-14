## IMPORTS ##
import torch
from constants import MAX_BB_EVAL
import time
from math import ceil
from blackBoxes import evaluateBlackbox
from randomSearch import randomSearch
from subproblemsStrategy import subproblemsNomad

torch.manual_seed(19) # Set seed for reproducible results

### MAIN ###
if __name__ == '__main__':
    
    t0 = time.time()

    modelType = "fcc"
    #modelType = "cnn"
    dataSet = "fashion"
    #dataSet = "cifar-10"
    
    ## Random search
    #randomSearch(evaluateBlackbox, modelType, dataSet, nbTrials=MAX_BB_EVAL)
    
    ## Subproblems strategy
    nTrials = 5
    budgetByTrial = int(MAX_BB_EVAL / nTrials)
    budgetLHbyTrial = ceil(budgetByTrial / 3) 
    subproblemsNomad(modelType, dataSet, nTrials, budgetByTrial, budgetLHbyTrial)
   
    print("\nExecution time : " + str(int(time.time() - t0)) + " seconds")
   
