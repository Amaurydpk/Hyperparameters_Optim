from unicodedata import name
import PyNomad
import sys
from constants import MAX_BB_EVAL, DEFAULT_OPTIMIZER_SETTINGS
from hyperparameters import setHyperparams
import time
import random


def writeFileBbPynomad(modelType, dataSet, HPs, dim):
    
    exe = open('bbPynomad.py','w')
    
    # Blackbox function
    exe.write("""import sys\n"""+
    """import torch\n"""+
    """from trainTest import train, accuracy\n"""+
    """from loadFashionMnist import loadFashionMNIST\n"""+
    """from models import FullyConnectedNeuralNet, ConvNeuralNet\n"""+
    """from constants import EPOCHS, INPUT_SIZE_FASHION, INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION, INPUT_CHANNELS_CIFAR, INPUT_SIZE_CIFAR, NUM_CLASSES_CIFAR\n"""+
    "\n"
    )
    if DEFAULT_OPTIMIZER_SETTINGS:
        exe.write("""def bb(listUnit, dropout, lrExponent):\n""")
    else:    
        exe.write("""def bb(listUnit, dropout, lrExponent, optimWeightDecay, optimParam1, optimParam2):\n""")
    
    exe.write("""\ttrainLoader, validLoader, testLoader = loadFashionMNIST()\n""")
    exe.write("""\tmodel = FullyConnectedNeuralNet(INPUT_SIZE_FASHION*INPUT_SIZE_FASHION*INPUT_CHANNELS_FASHION, NUM_CLASSES_FASHION, '{}', {}, listUnit, dropout)\n"""
            .format(HPs.activationFunction.value, HPs.nFullLayers.value))

    exe.write("""\tgpuAvailable = torch.cuda.is_available()\n"""+
            """\tdevice = torch.device("cuda" if gpuAvailable else "cpu")\n"""+
            """\tmodel = model.to(device)\n""")
    if DEFAULT_OPTIMIZER_SETTINGS:
        exe.write("""\tmodel = train(model, trainLoader, validLoader, device, EPOCHS, '{}', lrExponent)\n""".format(HPs.optimizer.value))
    else:
        exe.write("""\tmodel = train(model, trainLoader, validLoader, device, EPOCHS, '{}', lrExponent,  optimWeightDecay, optimParam1, optimParam2)\n""".format(HPs.optimizer.value))
    exe.write("""\treturn -(accuracy(model, testLoader, device)[0])\n\n""")

    # bb Pynomad
    exe.write("""def bbPynomad(x):\n""" +
              """\ttry:\n""")
    paramX = """("""
    index = 0
    listUnit = """["""
    for i in range(HPs.nFullLayers.value):
        if i != HPs.nFullLayers.value - 1:
            listUnit+="int(x.get_coord({})), ".format(i)
        else:
            listUnit+="int(x.get_coord({}))], ".format(i)
        index += 1
    paramX += listUnit
    for j in range(dim-index):
        if j != dim-index-1:
            paramX += """x.get_coord({}), """.format(index+j)
        else:
            paramX += """x.get_coord({}))""".format(index+j)
    
    exe.write("""\t\tf = bb""" + paramX + """\n""")

    exe.write("""\t\tx.setBBO(str(f).encode("UTF-8"))\n""" +
    """\texcept:\n""" +
    """\t\tprint("Unexpected eval error", sys.exc_info()[0])\n""" +
    """\t\treturn 0\n""" +
    """\treturn 1  # 1: success 0: failed evaluation\n"""
    )

    exe.close()

    
    

### MAIN ###
if __name__ == '__main__':

    modelType = "cnn"
    dataSet = "fashion"

    budget = 30 #MAX_BB_EVAL
    nTrials = 2
    budgetByTrials = int(budget / nTrials)
    lh_budget = int(budgetByTrials/5) # budget for LHS

    memoryHPsCatMetaTested = [] # Memory to keep set of meta and categorical HPs already tested
    
    t0 = time.time()

    HPs = setHyperparams(model=modelType)

    for i in range(nTrials):
        
        ## Meta and categorical HPs
        MetaAndCatHPs = HPs.getHPsOfType(meta=True, hpType='all') + HPs.getHPsOfType(meta=False, hpType='cat')
        different = False
        # TO ADD : stop criterion if exceed number of combination possible
        while not different: # to be sure to not test twice the same combination
            # random draw of meta and categorical HPs
            randomDraw = []
            for hp in MetaAndCatHPs:
                hp.setRandomValue()
                randomDraw.append(hp.value)
            if randomDraw not in memoryHPsCatMetaTested: 
                different = True
        # Keep memory of meta and categorical fixed values
        memoryHPsCatMetaTested.append(randomDraw)

        # Print
        print("========= Trial {} =========".format(i+1))
        print("Fixed HPs:")
        for hp in MetaAndCatHPs:
            print("\t" + repr(hp))
        print()
    
        ## NOMAD ---------------------
        input_type = "BB_INPUT_TYPE ("  # R=real (float) and I=integer
        dim = 0
        ## Bounds ----
        lb, ub  = [], []
        ## Standard variables ---
        standardHPs = HPs.getHPsOfType(meta=False, hpType='int') + HPs.getHPsOfType(meta=False, hpType='real')
        nameStandardHPs = []
        first = True # Just to have a space or not
        for hp in standardHPs:
            nameStandardHPs.append(hp.name)
            lb += [hp.lb]
            ub += [hp.ub]
            if first:
                typeLetter = 'R' if hp.type == 'real' else 'I'
                first = False
            else:
                typeLetter = ' R' if hp.type == 'real' else ' I'
            input_type += typeLetter
            dim += 1
        
        print("Standard HPs to optimize: {}".format(nameStandardHPs))
        print("Lower bounds : {}".format(lb))
        print("Upper bounds : {}".format(ub))
        print()
        
        # Formatting the parameters for PyNomad
        input_type += ")"
        dimension = "DIMENSION " + str(dim)
        max_nb_of_evaluations = "MAX_BB_EVAL " + str(budgetByTrials)
        lh_search_number = "LH_SEARCH "+ str(lh_budget) + " 0"
        params = [max_nb_of_evaluations, dimension, input_type,
        "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL TRUE", "DISPLAY_STATS BBE OBJ (SOL)", lh_search_number]
        print("Params given to NOMAD:")
        print(params)
        print()

        writeFileBbPynomad(modelType, dataSet, HPs, dim)

        import bbPynomad
        
        # try:
        #     sol = PyNomad.optimize(bbPynomad.bbPynomad, [], lb, ub, params)
        # except Exception as e:
        #     print("Stopped because error : "+ str(e))
        #     sol = None

        # print()
        # print(sol)
        # print()

        print()
    
    print(memoryHPsCatMetaTested)
    
    print()
    print(str(int(time.time() - t0)) + " seconds")

  

        






