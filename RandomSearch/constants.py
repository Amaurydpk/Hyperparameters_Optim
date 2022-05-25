
INPUT_SIZE = 1*28*28 
NUM_CLASSES = 10
DATASET_DIR = "./data"

BATCH_SIZE_FASHION  = 128   # Using minibatches of 128 samples

hyperParamsRange = {
    'epochs': 5,
    'optimizerList': ['Adam', 'SGD'],  
    'nLayers': (1, 3), 
    'nHiddenLayers': (4, 256),
    'activationFunctionList': ['ReLU', 'Sigmoid'], 
    'learningRateExponent': (-5, -1),
    'dropout': (0.2, 0.5)
}

BATCH_SIZE_CIFAR = 4