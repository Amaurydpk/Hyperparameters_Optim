import PyNomad
import sys
import torch
from trainTest import train, accuracy
from loadFashionMnist import loadDataSetFashionMNIST
from models import FullyConnectedNeuralNet
from constants import EPOCHS, INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, HPrangeFCC, MAX_BB_EVAL
import time
import random

def bb(listUnit, dropout, lr):
    """
    
    """
    # Load training, validation and testing sets formatted with the batch size
    trainLoader, validLoader, testLoader = loadDataSetFashionMNIST()
    # Construct model
    model = FullyConnectedNeuralNet(INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, 'ReLU', 2, listUnit, dropout)

    # Decide whether CPU or GPU is used
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Trained model
    model = train(model, trainLoader, validLoader, EPOCHS, 'Adam', lr, device)
    
    # Final precision on trained model on test dataset 
    return -(accuracy(model, testLoader, device)[0]) 


def bbPynomad(x):
    """
    Black-box function formatted to software PyNomad
    """
    try:
        f = bb([int(x.get_coord(0)), int(x.get_coord(1))], x.get_coord(2), x.get_coord(3))
        x.setBBO(str(f).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1  # 1: success 0: failed evaluation

    




### MAIN ###
if __name__ == '__main__':

    budget = MAX_BB_EVAL
    nTrials = int(budget/10)
    budgetByTrials = int(budget / nTrials)

    for i in range(nTrials):
        
        # Random choice on meta and categorical variables
        optim = random.choice(HPrangeFCC['optimizerList'])
        nLayers = random.randint(HPrangeFCC['nLayers'][0], HPrangeFCC['nLayers'][1])
        activationFunction = random.choice(HPrangeFCC['activationFunctionList'])

        # Lower bound and upper bound of standard variables (units, )
        lb, ub = [], []
        for i in range(nLayers): # units
            lb.append(int(HPrangeFCC['nHiddenLayers'][0]))
            ub.append(int(HPrangeFCC['nHiddenLayers'][1]))
        # dropout
        lb.append(HPrangeFCC['dropout'][0])
        ub.append(HPrangeFCC['dropout'][1])
        # lr
        lb.append(HPrangeFCC['initialLearningRate'][0])
        ub.append(HPrangeFCC['initialLearningRate'][1])
        # weight decay
        lb.append(HPrangeFCC['optimWeightDecay'][0])
        ub.append(HPrangeFCC['optimWeightDecay'][1])
        # optim param 1
        lb.append(HPrangeFCC['optimParam1'][0])
        ub.append(HPrangeFCC['optimParam1'][1])
        # optim param 2
        lb.append(HPrangeFCC['optimParam2'][0])
        ub.append(HPrangeFCC['optimParam2'][1])

        # Formatting the parameters for PyNomad
        # R=real (float) and I=integer
        input_type = "BB_INPUT_TYPE (" + " I" * nLayers + "R R R R R)"
        dimension = "DIMENSION "
        max_nb_of_evaluations = "MAX_BB_EVAL 100"

        params = [max_nb_of_evaluations, dimension, input_type,
                "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL TRUE", "DISPLAY_STATS BBE OBJ (SOL)", "LH_SEARCH 50 0"]
        # "FIXED_VARIABLE 1"
        # "VNS_MADS_SEARCH TRUE"
        # LH_SEARCH 5 0


        # Important : PyNomad strictly minimizes the bb function
        t0 = time.time()
        PyNomad.optimize(bbPynomad, [], lb, ub, params)
        print(time.time() - t0)

  

        






