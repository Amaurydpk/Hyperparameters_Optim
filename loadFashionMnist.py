import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import BATCH_SIZE_FASHION, DATASET_DIR, VALID_RATIO, NUM_WORKERS


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


def loadDataSetFashionMNIST():
    """
    Load the Fashion MNIST dataset and return a training set and a test set

    :param (string) dataDir: the path for the data folder
    :param (int) batchSize: the size of the batch
    :param (float) validRatio: the ratio between 0 and 1 to split train/valid

    :return: (DataLoader, DataLoader, DataLoader) the training, validation and test set
    """
    # Load the the training set
    trainValidDataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=True, download=True, transform=None)
    trainValidDataset = DatasetTransformer(trainValidDataset, transforms.ToTensor())
    # Split into training and validation sets
    nb_train = int((1 - VALID_RATIO) * len(trainValidDataset))
    nb_valid = int(VALID_RATIO * len(trainValidDataset))
    trainDataset, validDataset = random_split(trainValidDataset, [nb_train, nb_valid])

    # Load the test set
    testDataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=False, download=True, transform=None)
    testDataset  = DatasetTransformer(testDataset , transforms.ToTensor())

    # Transform into DataLoader
    trainLoader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE_FASHION, shuffle=True, num_workers=NUM_WORKERS) # <-- this reshuffles the data at every epoch
    validLoader = DataLoader(dataset=validDataset, batch_size=BATCH_SIZE_FASHION, shuffle=False, num_workers=NUM_WORKERS)
    testLoader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE_FASHION, shuffle=False, num_workers=NUM_WORKERS)
    return trainLoader, validLoader, testLoader


def displayExamples(trainLoader):
    """
    Display 10 images with their label from the trainLoader

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
    trainLoader, validLoader, testLoader = loadDataSetFashionMNIST(dataDir=DATASET_DIR, batchSize=BATCH_SIZE_FASHION, validRatio=VALID_RATIO)
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The validation set contains {} images, in {} batches".format(len(validLoader.dataset), len(validLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)




