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
    
    ## Random search
    # for nTrial in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     print(f"Nb trials: {nTrial}")
    #     randomSearch(evaluateBlackbox, modelType, dataSet, nbTrials=nTrial)
    
    ## Subproblems strategy
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f"======== Nb trials: {n} =========")
        nTrials = 2 if n <= 50 else 3
        budgetByTrial = int(n / nTrials)
        budgetLHbyTrial = ceil(budgetByTrial / 3)
        subproblemsNomad(modelType, dataSet, nTrials, budgetByTrial, budgetLHbyTrial)
        #print(nTrials, budgetByTrial, budgetLHbyTrial)
   
    print("\nExecution time : " + str(datetime.timedelta(seconds=int(time.time() - t0))))
   
