import torch
from FCCclassification import buildFCCModel
from loadFashionMnist import loadDataSetFashionMNIST
from trainTest import train, accuracy
from constants import INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, DATASET_DIR, BATCH_SIZE_FASHION, VALID_RATIO


def evaluateBlackboxFashion(HPs):
    """
    Return the accuracy of the model trained with a given set of HPs on Fashion MNIST
   
    :param (dictionnary) HPs: a set of hyperparameters

    :return: (float) the objective function value (the accuracy of the model)
    """
    # Load training, validation and testing sets formatted with the batch size
    trainLoader, validLoader, testLoader = loadDataSetFashionMNIST(dataDir=DATASET_DIR, batchSize=BATCH_SIZE_FASHION, validRatio=VALID_RATIO)

    # Construct model
    model = buildFCCModel(INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, HPs['activation'], HPs['nLayers'], HPs['nUnitsList'], HPs['dropout'])
    
    # Decide whether CPU or GPU is used
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Trained model
    model = train(model, trainLoader, validLoader, HPs['epochs'], HPs['optimizer'], HPs['learningRate'], device)
    
    # Final precision on trained model on test dataset 
    return accuracy(model, testLoader, device)[0] 




