import torch.nn as nn

##### FULLY CONNECTED #####

# Build a model by implementing define-by-run design from Optuna
def build_model_custom(trial, params, inputsize, num_classes, fixedNbUnits=True, dropout=False):
    layers = []
    in_features = inputsize
    n_layers = params['n_layers']

    activation = nn.ReLU() if params['activation'] == 'ReLU' else nn.Sigmoid()

    for i in range(n_layers):
        if fixedNbUnits:
            out_features = 64
        else:
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation)
        if dropout:
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, num_classes))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)