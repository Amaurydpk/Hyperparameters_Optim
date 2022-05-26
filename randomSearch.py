import random
from functions import printDictionnary
from blackBoxes import evaluateBlackboxFashion

random.seed(19) # Set seed for reproducible results

def giveRandomHPs(HPrange):
    """
    Return a random set of hyperparameters from the HPrange dictionnary
   
    :param (dictionnary) HPrange: dictionnary of range for each hyperparameter

    :return: (dictionnary) a random set of HP
    """
    # Meta variables
    optim = random.choice(HPrange['optimizerList'])
    nLayers = random.randint(HPrange['nLayers'][0], HPrange['nLayers'][1])
    # Decreed variables
    nUnitsList = [random.randint(HPrange['nHiddenLayers'][0], HPrange['nHiddenLayers'][1]) for i in range(nLayers)]
    # Others variables
    dropout = round(random.uniform(HPrange['dropout'][0], HPrange['dropout'][1]), 3)
    learningRate = 10 ** random.randint(HPrange['learningRateExponent'][0], HPrange['learningRateExponent'][1])
    activationFunction = random.choice(HPrange['activationFunctionList'])
    # Build HPs set
    HPs = {
        'epochs': HPrange['epochs'],
        'optimizer': optim,
        'nLayers': nLayers,
        'nUnitsList': nUnitsList,
        'dropout': dropout,
        'learningRate': learningRate,
        'activation': activationFunction,
    }
    return HPs


def randomSearch(blackBox, HPrange, nbTrials):
    """
    Performs a random search for hyperparameters optimizitation and display 
    the best set of hyperparameters found in the finite number of trials
    
    :param (function) blackBox: the blackbox to use
    :param (dictionnary) HPrange: dictionnary of range for each hyperparameter
    :param (int) nbTrials: number of trials to perform

    :return: (dictionnary) the best HPs' set found
    """
    trials, accuracies = [], []
    bestIndex = 0
    for i in range (nbTrials):
        print("------------- Trial {} -------------".format(i+1))
        HPs = giveRandomHPs(HPrange) # a random set of HPs
        trials.append(HPs)
        printDictionnary(HPs)
        accuracy = blackBox(HPs) # evaluate the model
        print("\nACCURACY = {}\n".format(round(accuracy, 3)))
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
            bestIndex = i
    
    # Display the best set of HPs
    print("------------- BEST -------------")
    print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
    printDictionnary(trials[bestIndex])
    return trials[bestIndex]


def randomSearchWithStatOptim(blackBox, HPrange, nbTrials):
    """
    Performs random searches for hyperparameters optimizitation 
    for each optimizer in HPrange we do a random search (with optim fixed) 
    and compute the mean accuracy to know the best optimizer
   
    :param (function) blackBox: the blackbox to use
    :param (dictionnary) HPrange: dictionnary of range for each hyperparameter
    :param (int) nbTrials: number of trials to perform

    :return: (dictionnary) the best HPs' set found
    """
    trials, accuracies = [], []
    bestIndex = 0
    bestMeanAccuracy = 0
    bestOptim = ""

    for optim in HPrange['optimizerList']:
        print("---- Random search with optim={} fixed ----".format(optim))
        meanAccuracy = 0
        for i in range (nbTrials):
            HPs = giveRandomHPs(HPrange) # a random set of HPs
            HPs['optimizer'] = optim # fixed optimizer
            trials.append(HPs)
            #printDictionnary(HPs)
            accuracy = blackBox(HPs) # evaluate the model
            print("Trial {} : accuracy = {}".format(i+1, round(accuracy, 3)))
            accuracies.append(accuracy)
            if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
                bestIndex = i
            meanAccuracy += accuracy
        meanAccuracy /= nbTrials
        print("Mean accuracy with {} : {}\n".format(optim, meanAccuracy))
        if meanAccuracy >= bestMeanAccuracy:
            bestMeanAccuracy = meanAccuracy
            bestOptim = optim

    # Display the best optimizer
    print("Best Optimizer: {}".format(bestOptim))

    return trials[bestIndex]

