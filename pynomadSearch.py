import PyNomad
import sys
import torch
from trainTest import train, accuracy
from loadFashionMnist import loadDataSetFashionMNIST
from models import FullyConnectedNeuralNet
from constants import EPOCHS, INPUT_SIZE_FASHION, NUM_CLASSES_FASHION
import time


def bb(listUnit, dropout, lr, weightDecay, optimParam1, optimParam2):
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
    model = train(model, trainLoader, validLoader, EPOCHS, 'Adam', lr, weightDecay, optimParam1, optimParam2, device)
    
    # Final precision on trained model on test dataset 
    return -(accuracy(model, testLoader, device)[0]) 


def bbPynomad(x):
    """
    Black-box function formatted to software PyNomad
    """
    try:
        f = bb([int(x.get_coord(0)), int(x.get_coord(1))], x.get_coord(2), x.get_coord(3), x.get_coord(4), x.get_coord(5), x.get_coord(6))
        x.setBBO(str(f).encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1  # 1: success 0: failed evaluation

    
#unit1, unit2, dropout, lr
# Initial point x0, lower bound (lb) and upper bound(ub)

#x0 = [int(98), int(74), 0.15, 0.0021]
lb = [int(4), int(4), 0, 10**-5, 0, 0, 0]
ub = [int(256), int(256), 1, 10**-1, 1, 1, 1]

# sampler = LatinHypercube(d=4).random(n=1)
# x0 = scale(sampler, lb, ub)[0]
# x0 = [int(x0[0]), int(x0[1]), x0[2], x0[3]]

# Formatting the parameters for PyNomad
input_type = "BB_INPUT_TYPE (I I R R R R R)"  # R=real (float) and I=integer
dimension = "DIMENSION 7"
max_nb_of_evaluations = "MAX_BB_EVAL 100"

params = [max_nb_of_evaluations, dimension, input_type,
          "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL TRUE", "DISPLAY_STATS BBE OBJ (SOL)", "LH_SEARCH 15 0"]
# "FIXED_VARIABLE 1"
# "VNS_MADS_SEARCH TRUE"
# LH_SEARCH 5 0

# Important : PyNomad strictly minimizes the bb function
t0 = time.time()
PyNomad.optimize(bbPynomad, [], lb, ub, params)
print(int(time.time() - t0))