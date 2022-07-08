from hyperparameters import setHyperparams
from constants import MAX_BB_EVAL

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
        print("============ Trial {} ============".format(i+1))
        print("--- Hyperparameters ---")
        HPs.setRandom()
        trials.append(HPs)
        HPs.display()
        print("\n> Training & Testing ...")
        accuracy = blackBox(HPs, modelType, dataSet) # evaluate the model
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
            bestIndex = i
        print("\nAccuracy on test set = {}\n".format(round(accuracy, 3)))
    
    # Display the best set of HPs
    print("============ BEST ============")
    print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
    trials[bestIndex].display()
    return trials[bestIndex]



# def randomSearchWithCategoricalHPfixed(blackBox, HPrange, randomHPsfunction, nbBbEvaluation, fixedHPname, HPrangeName):
#     """
#     Performs random searches for hyperparameters optimizitation 
#     for each optimizer in HPrange we do a random search (with optim fixed) 
#     and compute the mean accuracy to know the best optimizer
   
#     :param (function) blackBox: the blackbox to use
#     :param (dictionnary) HPrange: dictionnary of range for each hyperparameter
#     :param (function) randomHPsfunction: the function that return a random set of HPs from given HPrange
#     :param (int) nbBbEvaluation: number blackbox evaluations allowed

#     :return: (dictionnary) the best HPs' set found
#     """
#     trials, accuracies = [], []
#     bestIndex = 0
#     bestMeanAccuracy = 0
#     bestHP = ""
#     nbHPs = len(HPrange[HPrangeName])
#     nbTrialsPerHp = nbBbEvaluation // nbHPs
#     nbTrialsLeft = nbBbEvaluation - (nbTrialsPerHp * nbHPs)

#     for hp in HPrange[HPrangeName]:
#         print("##### Random search with {}={} fixed #####".format(fixedHPname, hp))
#         meanAccuracy = 0
#         for i in range (nbTrialsPerHp):
#             print("============ Trial {} ============".format(i+1))
#             print("--- Hyperparameters ---")
#             HPs = randomHPsfunction(HPrange) # a random set of HPs
#             HPs[fixedHPname] = hp # fixed hp
#             trials.append(HPs)
#             printDictionnary(HPs)
#             print("\n> Training & Testing ...")
#             accuracy = blackBox(HPs) # evaluate the model
#             print("\nAccuracy on test set = {}\n".format(round(accuracy, 3)))
#             accuracies.append(accuracy)
#             if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
#                 bestIndex = i
#             meanAccuracy += accuracy
#         meanAccuracy /= nbTrialsPerHp
#         print("Mean accuracy with {} : {}\n".format(hp, meanAccuracy))
        
#         if meanAccuracy >= bestMeanAccuracy:
#             bestMeanAccuracy = meanAccuracy
#             bestHP = hp

#     # Display the best hp found
#     print("============ BEST ============")
#     print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
#     printDictionnary(trials[bestIndex])

#     print("\nBest {} : {}".format(fixedHPname, bestHP))

#     return bestHP, nbTrialsLeft
