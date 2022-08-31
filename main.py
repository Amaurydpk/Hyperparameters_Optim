## IMPORTS ##
import torch
from constants import MAX_BB_EVAL
import time
import datetime
from math import ceil
from blackBoxes import evaluateBlackbox
from randomSearch import randomSearch
from subproblemsStrategy import subproblemsNomad

torch.manual_seed(19) # Set seed for reproducible results

### MAIN ###
if __name__ == '__main__':
    
    t0 = time.time()

    modelType = "fcc"
    #modelType = "cnn"
    dataSet = "fashion"
    #dataSet = "cifar-10"

    open("SuBaccuracies_{}_{}.txt".format(modelType, dataSet),'w')
    
    # Random search
    #randomSearch(evaluateBlackbox, modelType, dataSet, nbTrials=MAX_BB_EVAL)
    
    # Subproblems strategy
    # for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     print(f"\n======== NB TRIALS: {n} =========\n")
    #     nTrials = 2 if n <= 50 else 3
    #     budgetByTrial = int(n / nTrials)
    #     budgetLHbyTrial = ceil(budgetByTrial / 3)
    #     subproblemsNomad(modelType, dataSet, nTrials, budgetByTrial, budgetLHbyTrial)
    #     print(nTrials, budgetByTrial, budgetLHbyTrial)
    

    subproblemsNomad(modelType, dataSet, 4, 25, 12)
 
    #close("Vaccuracies_{}_{}.txt".format(modelType, dataSet))

    print("\nExecution time : " + str(datetime.timedelta(seconds=int(time.time() - t0))))
   

# BO
    # bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    #     {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    #     {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    #     {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    #     {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]


    # # Optimization objective 
    # def score(parameters):
    #     parameters = parameters[0]
    #     evaluateBlackbox(HPs, modelType="fcc", dataSet="fashion")
    #     score = cross_val_score(
    #                 XGBRegressor(learning_rate=parameters[0],
    #                             gamma=int(parameters[1]),
    #                             max_depth=int(parameters[2]),
    #                             n_estimators=int(parameters[3]),
    #                             min_child_weight = parameters[4]), 
    #                 X, Y, scoring='neg_mean_squared_error').mean()
    #     score = np.array(score)
    #     return score

    # optimizer = BayesianOptimization(f=cv_score, 
    #                                 domain=bds,
    #                                 model_type='GP',
    #                                 acquisition_type ='EI',
    #                                 acquisition_jitter = 0.05,
    #                                 exact_feval=True, 
    #                                 maximize=True)

    # # Only 20 iterations because we have 5 initial random points
    # optimizer.run_optimization(max_iter=20)