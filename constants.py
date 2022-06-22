from hyperparameters import Hyperparam

DATASET_DIR = "./data"
VALID_RATIO = 0.2  # Going to use 80%/20% split for train/valid

MAX_BB_EVAL = 5 # Max number of blackbox evaluations

EPOCHS = 5

NUM_WORKERS = 4

PRINT = True

# Fashion MNIST --------------------------------------------
INPUT_SIZE_FASHION = 1*28*28 
NUM_CLASSES_FASHION = 10
BATCH_SIZE_FASHION = 128 # Using minibatches of 128 samples

# CIFAR-10 ------------------------------------------------
INPUT_SIZE_CIFAR = 32 # size image 32*32
INPUT_CHANNELS_CIFAR = 3 # 3 channels : RGB
NUM_CLASSES_CIFAR = 10
BATCH_SIZE_CIFAR = 128


