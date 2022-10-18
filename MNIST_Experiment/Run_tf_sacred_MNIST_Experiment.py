#%% Sweep Hyperparameters Sequentially
from tf_sacred_MNIST_Experiment import ex
from sklearn.model_selection import ParameterGrid

parameters = {'num_epochs': [5],
              'num_neurons': [10],
              'activation_type':["softmax"],
}

parameter_combinations = list(ParameterGrid(parameters))

for parameter_combo in parameter_combinations:
    ex.run(config_updates=parameter_combo)