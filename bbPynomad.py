import sys
import torch
from trainTest import train, accuracy
from loadFashionMnist import loadFashionMNIST
from constants import EPOCHS, INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION
from models import ConvNeuralNet

def bb(batchSizeExponent, listConvol, listUnit, lrExponent, dropout):
	trainLoader, validLoader, testLoader = loadFashionMNIST(batchSize=2**batchSizeExponent)
	inputSize, inputChannel, numClasses = INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION
	model = ConvNeuralNet(1, 
                            4, 
                            listConvol, 
                            listUnit, 
                            dropout, 
                            'ReLU', 
                            inputSize, 
                            numClasses, 
                            inputChannel)
	gpuAvailable = torch.cuda.is_available()
	device = torch.device("cuda" if gpuAvailable else "cpu")
	model = model.to(device)
	try:
		model = train(model, trainLoader, validLoader, device, EPOCHS, 'Adagrad', lrExponent)
	except Exception as e:
		print("Stopped because error : "+ str(e))
		return 0
	acc = accuracy(model, testLoader, device)[0]
	exe = open("SuBaccuracies_cnn_fashion.txt",'a')
	exe.write('{}, '.format(acc))
	return -acc

def bbPynomad(x):
	try:
		f = bb(int(x.get_coord(0)), [(int(x.get_coord(1)), int(x.get_coord(2)), int(x.get_coord(3)), int(x.get_coord(4)), int(x.get_coord(5)))], [int(x.get_coord(6)), int(x.get_coord(7)), int(x.get_coord(8)), int(x.get_coord(9))], int(x.get_coord(10)), x.get_coord(11))
		x.setBBO(str(f).encode("UTF-8"))
	except:
		print("Unexpected eval error", sys.exc_info()[0])
		return 0
	return 1  # 1: success 0: failed evaluation
