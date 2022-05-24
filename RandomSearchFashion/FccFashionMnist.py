import torch.nn as nn

def buildFCCModel(HPs, inputsize, numClasses, dropout=False):
    """
    Build a fully connected NN model

    :param (dictionnary) HPs: a set of hyperparameters
    :param (int) inputsize: size of the input
    :param (int) numClasses: size of the outpout, number of classes
    :param (bool) dropout: True if we want to add dropout layer, False otherwise

    :return: (nn.Sequential) a fully connected nn network
    """
    layers = []
    in_features = inputsize
    activation = nn.ReLU() if HPs['activation'] == 'ReLU' else nn.Sigmoid()

    for i in range(HPs['nLayers']):
        out_features = HPs['nHiddenLayersList'][i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation)
        if dropout:
            layers.append(nn.Dropout(HPs['dropout']))
        in_features = out_features

    layers.append(nn.Linear(in_features, numClasses))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)