import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import BATCH_SIZE, DATASET_DIR

torch.manual_seed(19)

class DatasetTransformer(torch.utils.data.Dataset):
    """
    Transform PIL Images into pytorch tensors
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


def loadFashionMNIST():
    """
    Load the Fashion MNIST dataset and return a training set and a test set

    :return: (DataLoader, DataLoader) the training and the test set
    """
    # Load the the training set
    train_dataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=True, download=True, transform=None)
    train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=False, download=True, transform=None)
    test_dataset  = DatasetTransformer(test_dataset , transforms.ToTensor())
    # Transform into DataLoader
    trainLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) # <-- this reshuffles the data at every epoch
    testLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return trainLoader, testLoader


def displayExamples(trainLoader):
    """
    Display 10 images with their label from the train_loader

    :param (DataLoader) train_loader: training set

    :return: None
    """
    nsamples=10
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    imgs, labels = next(iter(trainLoader))
    fig=plt.figure(figsize=(20,5),facecolor='w')
    for i in range(nsamples):
        ax = plt.subplot(1,nsamples, i+1)
        plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


### MAIN ###
if __name__ == '__main__':
    trainLoader, testLoader = loadFashionMNIST()
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)




