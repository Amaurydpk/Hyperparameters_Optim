import random
from FccFashionMnist import buildFCCModel
from trainTestFashionMnist import trainAndEvaluate
from constants import INPUT_SIZE, NUM_CLASSES

def giveRandomHPs(HPrange):
    """
    Return a random set of hyperparameters from the HPrange dictionnary
   
    :param (dictionnary) HPrange: dictionnary of range for each hyperparameter

    :return: (dictionnary) a random set of HP
    """
    # Meta variables
    optim = random.choice(HPrange['optimizerList'])
    nLayers = random.randint(HPrange['nLayersLB'], HPrange['nlayersUB'])
    # Decreed variables
    nHiddenLayersList = []
    for i in range(nLayers):
        nHiddenLayersList.append(random.randint(HPrange['nHiddenLayersLB'], HPrange['nHiddenLayersUB']))
    # Others variables
    dropout = random.uniform(HPrange['dropout'][0], HPrange['dropout'][1])
    learningRate = random.uniform(HPrange['learningRateLB'], HPrange['learningRateUB'])
    activationFunction = random.choice(HPrange['activationFunctionList'])
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
        print("----------- Trial {} -----------".format(i+1))
        HPs = giveRandomHPs(HPrange) # a random set of HPs
        trials.append(HPs)
        for key, value in HPs.items() :
            print(key + " : " + str(value))
        accuracy = evaluateBlackbox(trainLoader, testLoader, HPs) # evaluate the model
        print("\nACCURACY = {}\n".format(accuracy))
        accuracies.append(accuracy)
        if accuracy >= accuracies[bestIndex]:
            bestIndex = i
    
    # Display the best set of HPs
    print("----- BEST -----")
    print("Accuracy : {}\n".format(round(accuracies[bestIndex],3)))
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