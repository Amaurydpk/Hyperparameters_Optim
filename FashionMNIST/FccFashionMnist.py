import torch.nn as nn

##### FULLY CONNECTED 2 HIDDEN LAYERS CLASSIFIER #####

## RELU ##
class FullyConnectedReLU(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnectedReLU, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

## SIGMOID ##
class FullyConnectedSigmoid(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnectedSigmoid, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y