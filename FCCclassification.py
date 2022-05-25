import torch.nn as nn

def buildFCCModel(inputsize, numClasses, activation, nLayers, nUnitsList, dropout):
    """
    Build a fully connected NN model

    :param (int) inputsize: size of the input
    :param (int) numClasses: size of the outpout, number of classes
    :param (str) activation: the name of the activation function to use
    :param (int) nLayers: number of hidden layers
    :param (List) nUnitsList: list of units number for each layers
    :param (float) dropout: dropout rate

    :return: (nn.Sequential) a fully connected nn network
    """
    layers = []
    in_features = inputsize
    fActivation = nn.ReLU() if activation == 'ReLU' else nn.Sigmoid()

    for i in range(nLayers):
        out_features = nUnitsList[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(fActivation)
        layers.append(nn.Dropout(dropout))
        in_features = out_features

    layers.append(nn.Linear(in_features, numClasses))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)