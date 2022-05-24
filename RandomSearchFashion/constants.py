
INPUT_SIZE = 1*28*28 
NUM_CLASSES = 10
BATCH_SIZE  = 128   # Using minibatches of 128 samples
DATASET_DIR = "./data"

hyperParamsRange = {
    'epochs': 4,
    'optimizerList': ['Adam', 'SGD'], 
    'nLayersLB': 1, 
    'nlayersUB': 3, 
    'nHiddenLayersLB': 4, 
    'nHiddenLayersUB': 256, 
    'activationFunctionList': ['ReLU', 'Sigmoid'], 
    'learningRateLB': 1e-5, 
    'learningRateUB': 1e-1,
    'dropout': (0.2, 0.5)
}