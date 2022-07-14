import random
from typing import Tuple
from constants import DEFAULT_OPTIMIZER_SETTINGS


class Hyperparam():

    def __init__(self, name, typ, range, isMeta=False):
        """
        Initialize an hyperparameter

        :param (str) name: the name of the hyperparameter
        :param (str) typ: 'int', 'real', 'cat'
        :param (list) range: the range of the hp --> if 'int' or 'real' then [lb, ub], 
                                                     if 'cat' then [choices], 
                                                    note: bool variables are treated like int with [0,1]
        :param (bool) isMeta: a boolean if the variable is meta or not
        """
        self.name = name
        self.type = typ
        self.range = range
        self.isMeta = isMeta
        self.children = None
        self.value = self.range[0] # Initialisation

        if self.type == 'real':
            self.lb = self.range[0]
            self.ub = self.range[1]
        if self.type == 'int':
            self.lb = int(self.range[0])
            self.ub = int(self.range[1])
        
        if self.isMeta:
            self.setChildren()
        

    def setRandomValue(self):
        if self.type == 'int':
            self.value = random.randint(self.lb, self.ub)
        elif self.type == 'real':
            self.value = random.uniform(self.lb, self.ub)
        else:
            self.value = random.choice(self.range)
        
        if self.isMeta: # If is meta then the children may change with the value of the parent
            self.setChildren()
            for elt in self.children:
                if type(elt) == tuple:
                    for hp in elt:
                        hp.setRandomValue()
                else:
                    elt.setRandomValue()
    

    def setValue(self, value):
        self.value = value
        if self.isMeta: # If is meta then the children may change
            self.setChildren()


    def setChildren(self):
        # -- Nb of convolutional layers --  
        if self.name == 'nConvolutionalLayers':
            self.children = [
                (Hyperparam('nOutputChannel_' + str(i+1), 'int', [1, 100]),
                Hyperparam('kernelSize_' + str(i+1), 'int', [1, 5]),
                Hyperparam('stride_' + str(i+1), 'int', [1, 3]),
                Hyperparam('padding_' + str(i+1), 'int', [0, 2]),
                Hyperparam('doPooling_' + str(i+1), 'int', [0, 1])) 
                for i in range(self.value)] 
        
        # -- Nb of full layers --  
        elif self.name == 'nFullLayers':
            self.children = [(Hyperparam('fullLayerSize_'+ str(i+1), 'int', [1, 500])) for i in range(self.value)]
        
        # -- Optimizer --
        elif self.name == 'optimizer':
            if self.value == 'ASGD':
                self.children = [
                    Hyperparam('lambd', 'real', [0, 1]),        # default=1e-4  decay term
                    Hyperparam('alpha', 'real', [0, 1]),        # default=0.75  power for eta update
                    Hyperparam('t0', 'int', [1e3, 1e9])         # default=1e6   point at which to start averaging
                    #Hyperparam('weightDecay', 'real', [0, 1])  # default=0     weight decay (L2 penalty)
                ]
            elif self.value == 'Adam':
                self.children = [
                    Hyperparam('beta1', 'real', [0, 1]),        # default=0.9    coefficient used for computing running averages of gradient and its square
                    Hyperparam('beta2', 'real', [0, 1]),        # default=0.999  coefficient used for computing running averages of gradient and its square
                    Hyperparam('eps', 'real', [0, 1])           # default=1e-8   term added to the denominator to improve numerical stability
                    #Hyperparam('weightDecay', 'real', [0, 1])  # default=0      weight decay (L2 penalty)
                ]
            elif self.value == 'Adagrad':
                self.children = [
                    Hyperparam('lr_decay', 'real', [0, 1]),     # default=0     learning rate decay
                    Hyperparam('intial_acc', 'real', [0, 1]),   # default=0     initial accumulator ?? NOT DOCUMENTED
                    Hyperparam('eps', 'real', [0, 1])           # default=1e-10 term added to the denominator to improve numerical stability
                    #Hyperparam('weightDecay', 'real', [0, 1])  # default=0     weight decay (L2 penalty)
                ]
            elif self.value == 'RMSprop':
                self.children = [
                    Hyperparam('momentum', 'real', [0, 1]),     # default=0     momentum factor
                    Hyperparam('alpha', 'real', [0, 1]),        # default=0     smoothing constant
                    Hyperparam('eps', 'real', [0, 1])           # default=1e-8  term added to the denominator to improve numerical stability
                    #Hyperparam('weightDecay', 'real', [0, 1])  # default=0     weight decay (L2 penalty)
                ]
            else:
                self.children = None
       
        else:
            self.children = None
        

    def getChildrenValues(self):
        if self.isMeta:
            values = []
            for elt in self.children:
                if type(elt) == tuple:
                    values.append(tuple(hp.value for hp in elt))
                else:
                    values.append(elt.value)
            return values
        else:
            return None
    

    def printChildren(self):
        if self.isMeta:
            s = ""
            firstElt = True
            for elt in self.children:
                if not firstElt:
                    s += "\n"
                if type(elt) == Hyperparam:
                    s += "\t" + elt.name + ": " + str(elt.value)
                else:
                    s += "\t("
                    for hp in elt:
                        if hp != elt[-1]:
                            s += hp.name + ": " + str(hp.value) + ", "
                        else:
                            s += hp.name + ": " + str(hp.value) + ")"
                firstElt = False
            print(s)


    def __repr__(self) -> str:
        return self.name + ": " + str(self.value)


class setHyperparams:

    def __init__(self, model):
        """
        Initialize a set of hyperparameters for a given model

        :param (str) model: "cnn" or "fcc"
        """

        self.batchSizeExponent = Hyperparam('batchSizeExponent', 'int', [4, 9])

        if model == "cnn":
            self.nConvolutionalLayers = Hyperparam('nConvolutionalLayers', 'int', [1, 3], isMeta=True)
        
        self.nFullLayers = Hyperparam('nFullLayers', 'int', [1, 3], isMeta=True)

        self.activationFunction = Hyperparam('activationFunction', 'cat', ['ReLU', 'Sigmoid', 'Tanh'])

        self.dropout = Hyperparam('dropout', 'real', [0, 1])

        self.learningRateExponent = Hyperparam('learningRateExponent', 'int', [-5, -1])

        optimIsMeta = not DEFAULT_OPTIMIZER_SETTINGS # If we use default optimizer settings the optimizer is not considered as a meta variable
                                                     # If we do not use default optimizer settings the optimize is considered as a meta variable
        self.optimizer = Hyperparam('optimizer', 'cat', ['ASGD', 'Adam', 'Adagrad', 'RMSprop'], isMeta=optimIsMeta)

    
    def getHPsOfType(self, meta=False, hpType='all'):
        """
        Return a list of HPs corresponding to criteria : meta or not and the given type (hpType)

        :param (bool) meta: to say if we want meta HP or not (True or False)
        :param (str) hpType: type of HP wanted ('int', 'real', 'cat' or 'all')
        """
        HPs = []
        for key, hp in self.__dict__.items(): 
            if meta: # If we want only meta variables
                if hp.isMeta:
                    if hpType == 'all':
                        HPs.append(hp)
                    elif hp.type == hpType:
                        HPs.append(hp)
            else: # If we want no-meta variables
                if not hp.isMeta:
                    if hpType == 'all':
                        HPs.append(hp)
                    elif hp.type == hpType:
                        HPs.append(hp)
                else: # Check on the children of the meta variables
                    for child in hp.children:
                        if type(child) == tuple:
                            for elt in child:
                                if hpType == 'all':
                                    HPs.append(elt)
                                elif elt.type == hpType:
                                    HPs.append(elt)
                        elif type(child) == Hyperparam:
                            if hpType == 'all':
                                HPs.append(child)
                            elif child.type == hpType:
                                HPs.append(child)                
        return HPs
    

    def setRandom(self):
        """
        Initialize the HPs set with random value
        """
        for key, hp in self.__dict__.items():
            hp.setRandomValue()
   

    def display(self):
        for key, hp in self.__dict__.items():
            print(hp)
            if hp.isMeta:
                hp.printChildren()


# TO DO: enelver les tuples dans la liste des params convolutionnel et essayer de traiter ça avec les numéros de couches



### MAIN ###
if __name__ == '__main__':
    # S = setHyperparams("fcc")
    # S.display()
    # print()
   # P = setHyperparams("cnn")
    #P.display()
    #P.setRandom()
    #P.display()
    
    HPs = setHyperparams(model="fcc")
    HPs.display()
    print()
    MetaAndCatHPs = HPs.getHPsOfType(meta=True, hpType='all') + HPs.getHPsOfType(meta=False, hpType='cat')
    for hp in MetaAndCatHPs:
        hp.setRandomValue()
    HPs.display()
        
  

 

    
