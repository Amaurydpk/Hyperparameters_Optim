import torch
from loadFashionMnist import loadDataSetFashionMNIST
from loadCifar10 import loadCIFAR10
from trainTest import train, accuracy
from constants import INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR, EPOCHS
from models import ConvNeuralNet, FullyConnectedNeuralNet


def evaluateBlackboxFashion(HPs):
    """
    Return the accuracy of the model trained with a given set of HPs on Fashion MNIST
   
    :param (setHyperams) HPs: a set of hyperparameters

    :return: (float) the objective function value (the accuracy of the model)
    """
    # Load training, validation and testing sets formatted with the batch size
    trainLoader, validLoader, testLoader = loadDataSetFashionMNIST()

    # Construct model
    model = FullyConnectedNeuralNet(INPUT_SIZE_FASHION, NUM_CLASSES_FASHION, 
                            HPs.activationFunction.value, 
                            HPs.nFullLayers.value, 
                            HPs.listFullLayerSize.value, 
                            HPs.dropout.value)
    
    # Decide whether CPU or GPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)
    
    # Trained model
    try:
        model = train(model, trainLoader, validLoader, EPOCHS, 
                        HPs.optimizer.value, 
                        HPs.initialLearningRate.value, 
                        HPs.optimWeightDecay.value, 
                        HPs.optimParam1.value, 
                        HPs.optimParam2.value, 
                        device)
    except Exception as e:
        print("Stopped because error : "+ str(e))
        return 0
    
    # Final precision on trained model on test dataset 
    return accuracy(model, testLoader, device)[0] 


def evaluateBlackboxCifar(HPs):
    """
    Return the accuracy of the model trained with a given set of HPs on CIFAR-10
   
    :param (dictionnary) HPs: a set of hyperparameters

    :return: (float) the objective function value (the accuracy of the model)
    """
    # Load training, validation and testing sets formatted with the batch size
    trainLoader, validLoader, testLoader = loadCIFAR10()

    model =  ConvNeuralNet(HPs.nConvolutionalLayers.value, 
                            HPs.nFullLayers.value, 
                            HPs.listParamConvLayers.value, 
                            HPs.listFullLayerSize.value, 
                            HPs.dropout.value, 
                            HPs.activationFunction.value, 
                            INPUT_SIZE_CIFAR, 
                            NUM_CLASSES_CIFAR, 
                            INPUT_CHANNELS_CIFAR)

    # Decide whether CPU or GPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Training model
    try:
        model = train(model, trainLoader, validLoader, EPOCHS, 
                        HPs.optimizer.value, 
                        HPs.initialLearningRate.value, 
                        HPs.optimWeightDecay.value, 
                        HPs.optimParam1.value, 
                        HPs.optimParam2.value, 
                        device)
    except Exception as e:
        print("Stopped because error : "+ str(e))
        return 0

    # Final precision on trained model on test dataset 
    return accuracy(model, testLoader, device)[0] 