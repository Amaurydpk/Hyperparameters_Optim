from hyperparameters import setHyperparams
from constants import MAX_BB_EVAL, PRINT

#random.seed(19) # Set seed for reproducible results

def randomSearch(blackBox, modelType="fcc", dataSet="fashion", nbTrials=MAX_BB_EVAL):
    """
    Performs a random search for hyperparameters optimizitation and display 
    the best set of hyperparameters found in the finite number of trials
    
    :param (function) blackBox: the blackbox to use
    :param (str) modelType: the type of the NN model : "fcc" or "cnn"
    :param (str) dataSet: the chosen dataSet ("fashion" or "cifar-10")
    :param (int) nbTrials: number of trials to perform

    :return: (dictionnary) the best HPs' set found
    """
    trials, accuracies = [], []
    bestIndex = 0
    HPs = setHyperparams(modelType) # initialize a set of HPs
    for i in range (nbTrials):
        HPs.setRandom()
        trials.append(HPs)
        accuracy = blackBox(HPs, modelType, dataSet) # evaluate the model
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
            bestIndex = i
        if PRINT:
            print("============ Trial {} ============".format(i+1))
            print("--- Hyperparameters ---")
            HPs.display()
            print("\n> Training & Testing ...")
            print("\nAccuracy on test set = {}\n".format(round(accuracy, 3)))
    # Display the best set of HPs
    print("============ BEST ============")
    print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
    trials[bestIndex].display()
    return trials[bestIndex], accuracies[bestIndex]
