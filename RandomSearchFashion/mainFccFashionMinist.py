## IMPORTS ##
import torch
from FccFashionMnist import build_model_custom
from trainTestFashionMnist import train_and_evaluate
from loadFashionMnist import trainLoader, testLoader
import optuna
import matplotlib.pyplot as plt

## PARAMS ##
torch.manual_seed(19)
INPUT_SIZE = 1*28*28
NUM_CLASSES = 10

## FUNCTIONS ##

# Define a set of meta hyperparameter values, build the model, train the model, and evaluate the accuracy
def objectiveMeta(trial):
    params = {
        'epochs': 3, # Fixed
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        'n_layers': trial.suggest_int("n_layers", 1, 3),
        'learning_rate': 1e-2, # Fixed
        'activation': "ReLU", # Fixed
    }
    model = build_model_custom(trial, params, inputsize=INPUT_SIZE, num_classes=NUM_CLASSES, fixedNbUnits=True)
    accuracy = train_and_evaluate(trial, params, model, trainLoader, testLoader)
    return accuracy


# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
# Take previous best values for optimizer and number of layers
def objective(trial):
    params = {
        'epochs': 3, # Fixed
        'optimizer': bestMetaHP['optimizer'], # Fixed with best found earlier
        'n_layers': bestMetaHP['n_layers'], # Fixed with best found earlier
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'activation': trial.suggest_categorical("activation", ["ReLU", "Sigmoid"])
    }
    model = build_model_custom(trial, params, inputsize=INPUT_SIZE, num_classes=NUM_CLASSES, fixedNbUnits=False)
    accuracy = train_and_evaluate(trial, params, model, trainLoader, testLoader)
    return accuracy


### MAIN ###
if __name__ == '__main__':
    # First, random search on meta hyperparameters
    studyMeta = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    studyMeta.optimize(objectiveMeta, n_trials=2)
    bestTrialMeta = studyMeta.best_trial
    bestMetaHP = bestTrialMeta.params
    
    # Then, random search on other hyperparameters
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=2)
    bestTrial = study.best_trial

    # Printing 
    for key, value in bestMetaHP.items():
        print("{}: {}".format(key, value))
    for key, value in bestTrial.params.items():
        print("{}: {}".format(key, value))
    
    # Display
    # optuna.visualization.plot_intermediate_values(study).show()
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_parallel_coordinate(study).show()
    # optuna.visualization.plot_param_importances(study).show()

