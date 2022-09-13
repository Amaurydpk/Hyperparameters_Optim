## IMPORTS ##
import torch
from constants import MAX_BB_EVAL
import time
import datetime
from math import ceil
from blackBoxes import evaluateBlackbox
from randomSearch import randomSearch
from subproblemsStrategy import subproblemsNomad

torch.manual_seed(19) # Set seed for reproducible results

### MAIN ###
if __name__ == '__main__':
    
    t0 = time.time()

    #modelType = "fcc"
    modelType = "cnn"
    #dataSet = "fashion"
    dataSet = "cifar-10"
    
    # Random search
    #randomSearch(evaluateBlackbox, modelType, dataSet, nbTrials=MAX_BB_EVAL)
    
    # Subproblem strategy
    subproblemsNomad(modelType, dataSet, nTrials=4, budgetByTrial=25, budgetLHbyTrial=12)
    
    print("\nExecution time : " + str(datetime.timedelta(seconds=int(time.time() - t0))))
   