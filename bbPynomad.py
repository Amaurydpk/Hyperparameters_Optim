import sys
import torch
from trainTest import train, accuracy
from loadFashionMnist import loadFashionMNIST
from models import FullyConnectedNeuralNet, ConvNeuralNet
from constants import EPOCHS, INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION, INPUT_CHANNELS_CIFAR, INPUT_SIZE_CIFAR, NUM_CLASSES_CIFAR

def bb(listUnit, dropout, lrExponent):
	trainLoader, validLoader, testLoader = loadFashionMNIST()
	model = FullyConnectedNeuralNet(INPUT_SIZE_FASHION*INPUT_SIZE_FASHION*INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION, 'ReLU', 2, listUnit, dropout)
	gpuAvailable = torch.cuda.is_available()
	device = torch.device("cuda" if gpuAvailable else "cpu")
	model = model.to(device)
	model = train(model, trainLoader, validLoader, device, EPOCHS, 'Adam', lrExponent)
	return -(accuracy(model, testLoader, device)[0])

def bbPynomad(x):
	try:
		f = bb([int(x.get_coord(0)), int(x.get_coord(1))], x.get_coord(2), x.get_coord(3), x.get_coord(4))
		x.setBBO(str(f).encode("UTF-8"))
	except:
		print("Unexpected eval error", sys.exc_info()[0])
		return 0
	return 1  # 1: success 0: failed evaluation
