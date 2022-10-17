#%% Sweep Hyperparameters Sequentially
from WINE_Experiment import ex
from sklearn.model_selection import ParameterGrid
from SLURM_Script_Factory import save_script
import sys

# Note that the regularizers follow the format "Mode=param" to avoid duplicate experiments with no regularizer (None)
parameters = {'encoding_method': ["Amplitude_Phase"],
              'cutoff_dimension':[5],
              'num_layers':[1, 2, 4],
              'num_classical':[1,2,4],
              'classical_size':[10,20],
              'n_qumodes': [2, 4],
              'n_circuits': [1],
              'regularizer_string': ["L1=0.01", "L1=0.1", "L2=0.01", "L2=0.1", "None"],
              'max_initial_weight': [None],
              'activation':["Sigmoid", "TanH"],
              'norm_threshold': [0.99],
              'ff_activation': ['relu',  None]
              }

"""
parameters = {'encoding_method': ["Amplitude_Phase"],
              'cutoff_dimension':[4],
              'num_layers':[1],
              'num_classical':[2],
              'classical_size':[10],
              'ff_activation':['relu'],
              'n_qumodes': [4],
              'n_circuits': [1],
              'regularizer_string': ["L1=0.01"],
              'max_initial_weight': [0.1],
              'activation':["Sigmoid"],
              'norm_threshold': [0.99]
              }"""

fields = {
        'memory': 16,
        'job_name': 'WINE_Experiment_Sweep',
        'time_h': 48,
        'num_cpus': 4
        }

# Either run the grid directly, or build a slurm script to run it on the cluster
if(sys.argv[1]=='run'):
    for parameter_combo in list(ParameterGrid(parameters)):
        ex.run(config_updates=parameter_combo)
elif(sys.argv[1]=='slurm'):
    save_script('Run_WINE_Experiment.sh', 'WINE_Experiment.py', fields, parameters)
# A smaller testing version to verify code is working.
elif(sys.argv[1]=='run_test'):
    parameters = {'encoding_method': ["Amplitude_Phase"],
              'cutoff_dimension':[4],
              'num_layers':[1],
              'num_classical':[2],
              'classical_size':[10,],
              'ff_activation':['relu'],
              'n_qumodes': [4],
              'n_circuits': [1],
              'regularizer_string': ["L1=0.01"],
              'max_initial_weight': [0.1],
              'activation':["Sigmoid"],
              'norm_threshold': [0.99]
              }
    for parameter_combo in list(ParameterGrid(parameters)):
        ex.run(config_updates=parameter_combo)
    
else:
    print("please enter 'run' or 'build_slurm'")
