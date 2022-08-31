
def writeFileBbPynomad(modelType, dataSet, HPs, dim):
    
    exe = open("bbPynomad.py",'w')
    #exe.truncate(0) # To erease the content
    # Imports
    exe.write("""import sys\n"""+
              """import torch\n"""+
              """from trainTest import train, accuracy\n""")

    if dataSet == "fashion":
        exe.write("""from loadFashionMnist import loadFashionMNIST\n""" + 
                  """from constants import EPOCHS, INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION\n""")
    elif dataSet == "cifar-10":
        exe.write("""from loadCifar10 import loadCIFAR10\n"""+
                  """from constants import EPOCHS, INPUT_CHANNELS_CIFAR, INPUT_SIZE_CIFAR, NUM_CLASSES_CIFAR\n""")

    if modelType == "fcc":
        exe.write("""from models import FullyConnectedNeuralNet\n""")
    elif modelType == "cnn":
        exe.write("""from models import ConvNeuralNet\n""")
    exe.write("\n")
        
    # black box function ------
    if modelType == "fcc":
        exe.write("""def bb(batchSizeExponent, listUnit, lrExponent, dropout""")
    elif modelType == "cnn":
        exe.write("""def bb(batchSizeExponent, listConvol, listUnit, lrExponent, dropout""")
    
    if HPs.optimizer.isMeta:
        exe.write(""", optimParams):\n""")
    else:
        exe.write("""):\n""")
    
    if dataSet == "fashion":
        exe.write("""\ttrainLoader, validLoader, testLoader = loadFashionMNIST(batchSize=2**batchSizeExponent)\n""")
        exe.write("""\tinputSize, inputChannel, numClasses = INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION\n""")
    elif dataSet == "cifar-10":
        exe.write("""\ttrainLoader, validLoader, testLoader = loadCIFAR10(batchSize=2**batchSizeExponent)\n""")
        exe.write("""\tinputSize, inputChannel, numClasses = INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR\n""")

    if modelType == "fcc":
        exe.write("""\tmodel = FullyConnectedNeuralNet(inputSize * inputSize * inputChannel, 
                            numClasses, 
                            '{}', 
                            {}, 
                            listUnit, 
                            dropout)\n""".format(HPs.activationFunction.value, HPs.nFullLayers.value))
    elif modelType == "cnn":
        exe.write("""\tmodel = ConvNeuralNet({}, 
                            {}, 
                            listConvol, 
                            listUnit, 
                            dropout, 
                            '{}', 
                            inputSize, 
                            numClasses, 
                            inputChannel)\n""".format(HPs.nConvolutionalLayers.value, HPs.nFullLayers.value, HPs.activationFunction.value))

    exe.write("""\tgpuAvailable = torch.cuda.is_available()\n"""+
            """\tdevice = torch.device("cuda" if gpuAvailable else "cpu")\n"""+
            """\tmodel = model.to(device)\n""")

    if HPs.optimizer.isMeta:
        exe.write("""\ttry:\n"""+
        """\t\tmodel = train(model, trainLoader, validLoader, device, EPOCHS, '{}', lrExponent, optimParams)\n""".format(HPs.optimizer.value))
    else:
        exe.write("""\ttry:\n"""+
        """\t\tmodel = train(model, trainLoader, validLoader, device, EPOCHS, '{}', lrExponent)\n""".format(HPs.optimizer.value))
    
    exe.write("""\texcept Exception as e:\n"""+
        """\t\tprint("Stopped because error : "+ str(e))\n"""+
        """\t\treturn 0\n""")
    
    exe.write("""\tacc = accuracy(model, testLoader, device)[0]\n""")
    exe.write("""\texe = open("SuBaccuracies_{}_{}.txt",'a')\n""".format(modelType, dataSet))
    exe.write("""\texe.write('{}, '.format(acc))\n""")

    exe.write("""\treturn -acc\n\n""")

    # bb Pynomad CAN BE IMPROVE BUT WORKS    
    exe.write("""def bbPynomad(x):\n""" +
              """\ttry:\n""")
    paramX = "(int(x.get_coord(0)), " # batchSizeExponent
    index = 1
    if modelType == "cnn":
        listConvol = """["""
        for i in range(HPs.nConvolutionalLayers.value):
            listConvol += "("
            for j in range(len(HPs.nConvolutionalLayers.children[i])):
                if HPs.nConvolutionalLayers.children[i][j].type == 'int':
                    stringTypeStart = "int("
                    stringTypeEnd = ")"
                else:
                    stringTypeStart = ""
                    stringTypeEnd = ""
                listConvol+= stringTypeStart + "x.get_coord({})".format(index) + stringTypeEnd
                index += 1
                if j == len(HPs.nConvolutionalLayers.children[i]) - 1: #last elt
                    listConvol += ")"
                else:
                    listConvol += ", "

            if i == HPs.nConvolutionalLayers.value - 1: # if it is the last
                listConvol+= "]"
            else:
                listConvol += ", "
            
        paramX += listConvol + ", "

    listUnit = """["""
    for i in range(HPs.nFullLayers.value):
        if i != HPs.nFullLayers.value - 1:
            listUnit+="int(x.get_coord({})), ".format(index)
        else:
            listUnit+="int(x.get_coord({}))]".format(index)
        index += 1
    paramX += listUnit + ", "

    # lr Exponent
    paramX += "int(x.get_coord({})), ".format(index)
    index += 1
    # dropout
    paramX += "x.get_coord({})".format(index)
    index += 1


    # params optimizer
    if HPs.optimizer.isMeta:
        paramX += ", ["
        for hp in HPs.optimizer.children:
            if hp.type == 'int':
                stringTypeStart = "int("
                stringTypeEnd = ")"
            else:
                stringTypeStart = ""
                stringTypeEnd = ""
            paramX+= stringTypeStart + "x.get_coord({})".format(index) + stringTypeEnd
            if index == dim-1: # the end
                paramX += "]"
            else:
                paramX += ", "
            index += 1
    paramX += ")"

    exe.write("""\t\tf = bb""" + paramX + """\n""")

    exe.write("""\t\tx.setBBO(str(f).encode("UTF-8"))\n""" +
    #"""\texcept Exception as e:\n""" +
    """\texcept:\n""" +
    """\t\tprint("Unexpected eval error", sys.exc_info()[0])\n""" +
    #"""\t\tprint(str(e), sys.exc_info()[0])\n""" +
    """\t\treturn 0\n""" +
    """\treturn 1  # 1: success 0: failed evaluation\n"""
    )

    exe.close()