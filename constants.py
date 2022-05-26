
DATASET_DIR = "./data"
VALID_RATIO = 0.2  # Going to use 80%/20% split for train/valid

# Fashion MNIST
INPUT_SIZE_FASHION = 1*28*28 
NUM_CLASSES_FASHION = 10
BATCH_SIZE_FASHION = 128   # Using minibatches of 128 samples

HPrangeFashionMnist = {
    'epochs': 5,
    'optimizerList': ['Adam', 'ASGD'],  
    'nLayers': (1, 3), 
    'nHiddenLayers': (4, 256),
    'activationFunctionList': ['ReLU', 'Sigmoid'], 
    'learningRateExponent': (-5, -1),
    'dropout': (0.2, 0.5)
}

# CIFAR-10
BATCH_SIZE_CIFAR = 4

