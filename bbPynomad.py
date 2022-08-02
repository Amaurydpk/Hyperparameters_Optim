import sys
import torch
from trainTest import train, accuracy
from loadCifar10 import loadCIFAR10
from constants import EPOCHS, INPUT_CHANNELS_CIFAR, INPUT_SIZE_CIFAR, NUM_CLASSES_CIFAR
from models import ConvNeuralNet

def bb(batchSizeExponent, listConvol, listUnit, lrExponent, dropout):
	trainLoader, validLoader, testLoader = loadCIFAR10(batchSize=2**batchSizeExponent)
	inputSize, inputChannel, numClasses = INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR
	model = ConvNeuralNet(1, 
                            7, 
                            listConvol, 
                            listUnit, 
                            dropout, 
                            'Tanh', 
                            inputSize, 
                            numClasses, 
                            inputChannel)
	gpuAvailable = torch.cuda.is_available()
	device = torch.device("cuda" if gpuAvailable else "cpu")
	model = model.to(device)
	try:
		model = train(model, trainLoader, validLoader, device, EPOCHS, 'ASGD', lrExponent)
	except Exception as e:
		print("Stopped because error : "+ str(e))
		return 0
	return -(accuracy(model, testLoader, device)[0])

def bbPynomad(x):
	try:
		f = bb(int(x.get_coord(0)), [(int(x.get_coord(1)), int(x.get_coord(2)), int(x.get_coord(3)), int(x.get_coord(4)), int(x.get_coord(5)))], [int(x.get_coord(6)), int(x.get_coord(7)), int(x.get_coord(8)), int(x.get_coord(9)), int(x.get_coord(10)), int(x.get_coord(11)), int(x.get_coord(12))], int(x.get_coord(13)), x.get_coord(14))
		x.setBBO(str(f).encode("UTF-8"))
	except:
		print("Unexpected eval error", sys.exc_info()[0])
		return 0
	return 1  # 1: success 0: failed evaluation
