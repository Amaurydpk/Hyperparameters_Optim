import random
from FCCclassification import buildFCCModel
from trainTest import trainAndTest
from constants import INPUT_SIZE, NUM_CLASSES
from functions import printDictionnary


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


def evaluateBlackbox(trainLoader, testLoader, HPs):
    """
    Return the accuracy of the model trained with a givens set of HPs
   
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) testLoader: test set
    :param (dictionnary) HPs: a set of hyperparameters

    :return: (float) the objective function value (the accuracy of the model)
    """
    model = buildFCCModel(INPUT_SIZE, NUM_CLASSES, HPs['activation'], HPs['nLayers'], HPs['nUnitsList'], HPs['dropout'])
    accuracy = trainAndTest(model, trainLoader, testLoader, HPs)
    return accuracy


def randomSearch(trainLoader, testLoader, HPrange, nbTrials):
    """
    Performs a random search for hyperparameters optimizitation and display 
    the best set of hyperparameters found in the finite number of trials
   
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) testLoader: test set
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
        accuracy = evaluateBlackbox(trainLoader, testLoader, HPs) # evaluate the model
        print("\nACCURACY = {}\n".format(round(accuracy, 3)))
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]: # record the index if we improve accuracy
            bestIndex = i
    
    # Display the best set of HPs
    print("------------- BEST -------------")
    print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
    printDictionnary(trials[bestIndex])
    return trials[bestIndex]


def randomSearchWithStatOptim(trainLoader, testLoader, HPrange, nbTrials):
    """
    Performs random searches for hyperparameters optimizitation 
    for each optimizer in HPrange we do a random search (with optim fixed) 
    and compute the mean accuracy to know the best optimizer
   
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) testLoader: test set
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
            accuracy = evaluateBlackbox(trainLoader, testLoader, HPs) # evaluate the model
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

