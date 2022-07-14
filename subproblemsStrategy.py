from hyperparameters import setHyperparams
from overwriteBBpynomad import writeFileBbPynomad
import subprocess


def subproblemsNomad(modelType, dataSet, nTrials, budgetByTrial, budgetLHbyTrial):
    """
    Performs subproblem strategy using NOMAD. Do *nTrials* by fixing categorical and meta HPs and runs
    NOMAD on *budgetByTrial* iterations with *budgetLHbyTrial* evaluations used for Latin Hypercube

    :param (str) modelType: "cnn" or "fcc"
    :param (str) dataSet: "fashion" or "cifar-10"
    :param (int) nTrials: number of trials with fixed categorical and meta HPs
    :param (int) budgetByTrial: number of blackbox evaluations by trial
    :param (int) budgetLHbyTrial: number of evaluations used by trial to perform Latin Hypercube (to find a starting point)
    """
    memoryHPsCatMetaTested = [] # Memory to keep set of meta and categorical HPs already tested
    bestScore = 0
    HPs = setHyperparams(model=modelType)
    # Meta and categorical HPs to be fixed 
    MetaAndCatHPs = HPs.getHPsOfType(meta=True, hpType='all') + HPs.getHPsOfType(meta=False, hpType='cat')
    
    for i in range(nTrials):
        ## Random draw of categorical and meta HPs ---------
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

        # Print fixed HPs
        print("============= Trial {} =============".format(i+1))
        print("Fixed HPs:")
        for hp in MetaAndCatHPs:
            print("\t" + repr(hp))
        print()
    
        ## NOMAD ---------------------
        input_type = "BB_INPUT_TYPE ("  # R=real (float) and I=integer
        dim = 0 # Dimension
        lb, ub  = [], [] # Bounds
        # Standard variables (change at each trial because can be decreed by meta HPs)
        standardHPs = HPs.getHPsOfType(meta=False, hpType='int') + HPs.getHPsOfType(meta=False, hpType='real')
        nameStandardHPs = []
        first = True # Just to not have a space in BB_INPUT_TYPE for the first parameter
        for hp in standardHPs:
            nameStandardHPs.append(hp.name)
            if hp.type == 'int':
                lb += [int(hp.lb)]
                ub += [int(hp.ub)]
                typeLetter = 'I' if first else ' I'
            else:
                lb += [float(hp.lb)]
                ub += [float(hp.ub)]
                typeLetter = 'R' if first else ' R'
            first = False
            input_type += typeLetter
            dim += 1
        
        # Print
        print("Standard HPs to optimize:\n{}".format(nameStandardHPs))
        print()
        print("Lower bounds : {}".format(lb))
        print("Upper bounds : {}".format(ub))
        print()
        
        # Formatting the parameters for PyNomad
        input_type += ")"
        dimension = "DIMENSION " + str(dim)
        max_nb_of_evaluations = "MAX_BB_EVAL " + str(budgetByTrial)
        lh_search_number = "LH_SEARCH "+ str(budgetLHbyTrial) + " 0"
        params = [max_nb_of_evaluations, dimension, input_type,
        "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL TRUE", "DISPLAY_STATS BBE OBJ (SOL)", lh_search_number]
        print("Params given to NOMAD:")
        print(params)
        print()
        
        # Write python functions used by PyNomad (in bbPynomad.py)
        writeFileBbPynomad(modelType, dataSet, HPs, dim)

        try:
            # Run PyNomad
            result = subprocess.run("""python runPynomad.py "{}" "{}" "{}" """.format(lb, ub, params), shell=True, text=True, capture_output=True)
            # Save and print result
            result = result.stdout
            print(result)
            dicResult = eval(result.split("\n")[-2]) # To have the dictionnary result of the pynomad run
            xBest, fBest = dicResult["x_best"], -dicResult["f_best"]
            # Save trial if it beats the current best
            if fBest > bestScore:
                bestTrialsInfo = {
                                "metaAndCatHPsValue": randomDraw, 
                                "nameStandardHPs": nameStandardHPs, 
                                "xBest": xBest
                                }
                bestScore = fBest
        except Exception as e:
            print("Stopped because error : "+ str(e))
    
    print("Meta and categorical HPs tested:")
    print(memoryHPsCatMetaTested)
    
    # Display best HPs and score found
    print()
    print("####### BEST ########\n")
    print("Best Accuracy : " + str(bestScore))
    print("\nFixed HPs:")
    for i in range (len(bestTrialsInfo["metaAndCatHPsValue"])):
        print("\t" + MetaAndCatHPs[i].name + ": " + str(bestTrialsInfo["metaAndCatHPsValue"][i]))
    print("Best HPs found:")
    for i in range (len(bestTrialsInfo["nameStandardHPs"])):
        print("\t" + str(bestTrialsInfo["nameStandardHPs"][i]) + ": " + str(bestTrialsInfo["xBest"][i]))
