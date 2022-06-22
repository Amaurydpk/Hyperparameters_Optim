## IMPORTS ##
import torch
from blackBoxes import evaluateBlackboxFashion, evaluateBlackboxCifar
from randomSearch import randomSearch
from constants import MAX_BB_EVAL
import time

torch.manual_seed(19) # Set seed for reproducible results


### MAIN ###
if __name__ == '__main__':
    
    t0 = time.time()

    # Random Search on Fashion MNIST with FCC #
    #randomSearch(evaluateBlackboxFashion, "fcc", nbTrials=MAX_BB_EVAL)

    # Random Search on CIFAR with CNN #
    randomSearch(evaluateBlackboxCifar, "cnn", nbTrials=MAX_BB_EVAL)

    # Random search improved #
    #randomSearchWithCategoricalHPfixed(evaluateBlackboxFashion, 
                            # HPrangeFCC, 
                            # giveRandomHPsFCC, 
                            # MAX_BB_EVAL, 
                            # 'optimizer', 
                            # 'optimizerList')
    
    print("\nExecution time : " + str(int(time.time() - t0)))