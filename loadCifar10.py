import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from constants import DATASET_DIR, VALID_RATIO, NUM_WORKERS


torch.manual_seed(19)


def loadCIFAR10(batchSize):
    """
    Load the CIFAR-10 dataset and return a training set and a test set

    :param (int) batchSize : the batch size value

    :return: (DataLoader, DataLoader) the training and the test set
    """    
    # The output of torchvision datasets are PIL images of range [0, 1]. 
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
    # Load the the training set
    trainValidDataset = torchvision.datasets.CIFAR10(root=DATASET_DIR, train=True, download=True, transform=transform)
    # Split into training and validation sets
    nb_train = int((1 - VALID_RATIO) * len(trainValidDataset))
    nb_valid = int(VALID_RATIO * len(trainValidDataset))
    trainDataset, validDataset = random_split(trainValidDataset, [nb_train, nb_valid])

    # Load the test set
    testDataset = torchvision.datasets.CIFAR10(root=DATASET_DIR, train=False, download=True, transform=transform)

    # Transform into DataLoader
    trainLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=NUM_WORKERS) # <-- this reshuffles the data at every epoch
    validLoader = DataLoader(dataset=validDataset, batch_size=batchSize, shuffle=False, num_workers=NUM_WORKERS)
    testLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False, num_workers=NUM_WORKERS)
    return trainLoader, validLoader, testLoader
    

def displayExamples(trainLoader):
    """
    Display some images with their label from the trainLoader

    :param (DataLoader) train_loader: training set

    :return: None
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # get some random training images
    dataiter = iter(trainLoader)
    images, labels = dataiter.next()
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(32)))
    # show images
    imshow(torchvision.utils.make_grid(images))
    

def imshow(img):
    """
    Plot the given image
    """
    img = img / 2 + 0.5 # unnormalize
    print(img.type())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


### MAIN ###
if __name__ == '__main__':
    trainLoader, validLoader, testLoader = loadCIFAR10(batchSize=128)
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)