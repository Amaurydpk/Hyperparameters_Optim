import random
from FCCclassification import buildFCCModel
from trainTest import trainAndEvaluate
from constants import INPUT_SIZE, NUM_CLASSES


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
    nHiddenLayersList = []
    for i in range(nLayers):
        nHiddenLayersList.append(random.randint(HPrange['nHiddenLayers'][0], HPrange['nHiddenLayers'][1]))
    # Others variables
    dropout = round(random.uniform(HPrange['dropout'][0], HPrange['dropout'][1]), 3)
    learningRateExponent = random.randint(HPrange['learningRateExponent'][0], HPrange['learningRateExponent'][1])
    learningRate = 10**learningRateExponent
    activationFunction = random.choice(HPrange['activationFunctionList'])
    # Build HPs set
    HPs = {
        'epochs': HPrange['epochs'],
        'optimizer': optim,
        'nLayers': nLayers,
        'nHiddenLayersList': nHiddenLayersList,
        'dropout': dropout,
        'learningRate': learningRate,
        'activation': activationFunction,
    }
    return HPs


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
        for key, value in HPs.items() :
            print(key + " : " + str(value))
        accuracy = evaluateBlackbox(trainLoader, testLoader, HPs) # evaluate the model
        print("\nACCURACY = {}\n".format(round(accuracy, 3)))
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]:
            bestIndex = i
    
    # Display the best set of HPs
    print("------------- BEST -------------")
    print("ACCURACY : {}\n".format(round(accuracies[bestIndex],3)))
    for key, value in trials[bestIndex].items() :
        print(key + " : " + str(value))
    return trials[bestIndex]


def evaluateBlackbox(trainLoader, testLoader, HPs):
    """
    Return the accuracy of the model trained with a givens set of HPs
   
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) testLoader: test set
    :param (dictionnary) HPs: a set of hyperparameters

    :return: (float) the objective function value (the accuracy of the model)
    """
    model = buildFCCModel(HPs, inputsize=INPUT_SIZE, numClasses=NUM_CLASSES)
    accuracy = trainAndEvaluate(model, trainLoader, testLoader, HPs)
    return accuracy