import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

torch.manual_seed(19)

##### PARAMS #####

batch_size  = 128   # Using minibatches of 128 samples
dataset_dir = "./data"

##### DATASETS #####

# Load the dataset for the trainingn sets
train_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir, train=True, download=True, transform=None)
# Load the test set
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir, train=False, download=True, transform=None)

##### TRANSFORM PIL IMAGES INTO PYTORCH TENSORS #####

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)

train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
test_dataset  = DatasetTransformer(test_dataset , transforms.ToTensor())


##### DATALOADERS #####

trainLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # <-- this reshuffles the data at every epoch
testLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

##### DISPLAY FASHION EXAMPLES #####
def displayExamples(train_loader):
    """
    Show 10 images with their label from the train_loader
    Arguments :
        train_loader -- training set
    Returns : None
    """
    nsamples=10
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    imgs, labels = next(iter(train_loader))
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
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)




