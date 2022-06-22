import random

class Hyperparam():

    def __init__(self, name, typ, range, isMeta=False):
        """
        Initialize an hyperparameter

        :param (str) name: the name of the hyperparameter
        :param (str) type: 'int', 'real', 'cat', 'bool', 'list'
        :param (list) range: the range of the hp --> if 'int' or 'real' then [lb, ub], 
                                                     if 'cat' then [choices], 
                                                     if 'bool' then [False, True]
                                                     if 'list' then [(Hyperparam, ..., Hyperparam)] * size
        :param (bool) isMeta: a boolean if the variable is meta or not
        """
        self.name = name
        self.type = typ
        self.range = range
        self.isMeta = isMeta

        if self.type == 'real' or self.type == 'int':
            self.lb = self.range[0]
            self.ub = self.range[1]
        
        self.value = self.randomValue()

    def randomValue(self):
        if self.type == 'int':
            return random.randint(self.lb, self.ub)
        elif self.type == 'real':
            return random.uniform(self.lb, self.ub)
        elif self.type == 'list':
            val = []
            for elt in self.range:
                if type(elt) == Hyperparam:
                    val.append(elt.randomValue())
                else:
                    val.append([(hp.randomValue()) for hp in elt])
            return val
        else:
            return random.choice(self.range)
    
    def setValue(self, value):
        self.value = value
    
    def __repr__(self) -> str:
        if self.type == "list":
            s = self.name + ":"
            for elt in self.range:
                if type(elt) == Hyperparam:
                    s += "\n\t" + elt.name + ":" + str(elt.value)
                else:
                    s += "\n\t("
                    for hp in elt:
                        if hp != elt[-1]:
                            s += hp.name + ":" + str(hp.value) + ", "
                        else:
                            s += hp.name + ":" + str(hp.value) + ")"
            return s
        else:
            return self.name + ": " + str(self.value)


class setHyperparams:

    def __init__(self, model):
        """
        Initialize a set of hyperparameters for a given model

        :param (str) model: "cnn" or "fcc"
        """

        if model == "cnn":
            self.nConvolutionalLayers = Hyperparam('nConvolutionalLayers', 'int', [1, 3], isMeta=True)
            listParam = [
                (   Hyperparam('nOutputChannel', 'int', [1, 100]),
                    Hyperparam('kernelSize', 'int', [1, 5]),
                    Hyperparam('stride', 'int', [1, 3]),
                    Hyperparam('padding', 'int', [0, 2]),
                    Hyperparam('doPooling', 'bool', [False, True])
                ) for i in range (self.nConvolutionalLayers.value)
            ]
            self.listParamConvLayers = Hyperparam('listParamConvLayers', 'list', listParam)
        
        self.nFullLayers = Hyperparam('nFullLayers', 'int', [1, 3])
        
        self.listFullLayerSize = Hyperparam('listFullLayerSize', 'list', [(Hyperparam('fullLayerSize', 'int', [1, 500])) for i in range (self.nFullLayers.value)])

        self.optimizer = Hyperparam('optimizer', 'cat', ['ASGD', 'Adam', 'Adagrad', 'RMSprop'], isMeta=True)

        self.activationFunction = Hyperparam('activationFunction', 'cat', ['ReLU', 'Sigmoid', 'Tanh'])

        self.dropout = Hyperparam('dropout', 'real', [0.2, 0.5])
        self.initialLearningRate = Hyperparam('initialLearningRate', 'real', [10**-5, 10**-1])
        self.optimWeightDecay = Hyperparam('optimWeightDecay', 'real', [0, 1])
        self.optimParam1 = Hyperparam('optimParam1', 'real', [0, 1])
        self.optimParam2 = Hyperparam('optimParam2', 'real', [0, 1])

                
    def display(self):
        for key, hp in self.__dict__.items():
            print(hp)
                

### MAIN ###
if __name__ == '__main__':

    # S = setHyperparams("fcc")
    # S.display()
    # print()
    P = setHyperparams("cnn")
    P.display()
    

 

    
