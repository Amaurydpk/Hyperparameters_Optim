import torch
from loadFashionMnist import loadFashionMNIST
from loadCifar10 import loadCIFAR10
from trainTest import train, accuracy
from constants import INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION, INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR, EPOCHS
from models import ConvNeuralNet, FullyConnectedNeuralNet


def evaluateBlackbox(HPs, modelType="fcc", dataSet="fashion"):
    """
    Return the accuracy of the given machine learning model trained with a given set of HPs on the given dataSet
   
    :param (setHyperparams) HPs: a set of hyperparameters
    :param (str) modelType: the chosen neural network model ("fcc" or "cnn")
    :param (str) dataSet: the chosen dataSet ("fashion" or "cifar-10")

    :return: (float) the objective function value (the accuracy of the model)
    """

    # Load training, validation and testing sets formatted with the batch size
    if dataSet == "fashion":
        trainLoader, validLoader, testLoader = loadFashionMNIST(batchSize=2**HPs.batchSizeExponent.value)
        inputSize, inputChannel, numClasses = INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION
    elif dataSet == "cifar-10":
        trainLoader, validLoader, testLoader = loadCIFAR10(batchSize=2**HPs.batchSizeExponent.value)
        inputSize, inputChannel, numClasses = INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR

    # Construct model
    if modelType == "fcc":
        model = FullyConnectedNeuralNet(inputSize * inputSize * inputChannel, 
                            numClasses, 
                            HPs.activationFunction.value, 
                            HPs.nFullLayers.value, 
                            HPs.nFullLayers.getChildrenValues(), 
                            HPs.dropout.value)
    elif modelType == "cnn":
        model =  ConvNeuralNet(HPs.nConvolutionalLayers.value, 
                            HPs.nFullLayers.value, 
                            HPs.nConvolutionalLayers.getChildrenValues(), 
                            HPs.nFullLayers.getChildrenValues(), 
                            HPs.dropout.value, 
                            HPs.activationFunction.value, 
                            inputSize, 
                            numClasses, 
                            inputChannel)

    # Decide whether CPU or GPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)
    
    # Model training
    try:
        model = train(model, trainLoader, validLoader, device, 
                        EPOCHS, 
                        HPs.optimizer.value, 
                        HPs.learningRateExponent.value, 
                        HPs.optimizer.getChildrenValues())
    except Exception as e:
        print("Stopped because error : "+ str(e))
        return 0
    
    # Final precision on trained model on test dataset 
    return accuracy(model, testLoader, device)[0] 