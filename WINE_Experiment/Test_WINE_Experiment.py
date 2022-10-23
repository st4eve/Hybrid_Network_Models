#%% Sweep Hyperparameters Sequentially
from WINE_NoisyTest_Experiment import ex
import numpy as np
from sklearn.model_selection import ParameterGrid
from common_packages.SLURM_Script_Factory import save_script
from findMaxAccuracy import findMaxAcc, getConfig
import sys

fields = {
        'memory': 16,
        'job_name': 'WINE_Experiment',
        'time_h': 24,
        'num_cpus': 4
        }

# Find highest testing accuracy experiment
exp, max_val, epoch = findMaxAcc('WINE') 
parameters = getConfig('WINE', exp)
# Updater parameters based on experiment
for key in parameters:
    parameters[key] = [parameters[key]]

# If running noisy data leave commented. If testing precision with DEAP, uncomment and comment sigma and second precision update.
#parameters['precision'] = [int(2**i) for i in range(10)]
parameters['shots'] = [int(2**i) for i in range(10)]
parameters['sigma'] = list(np.logspace(-10, 0, 20))
parameters['precision'] = [int(2**16)]
parameters['max_epoch'] = [epoch] 
parameters['exp_train'] = [exp]

# Either run the grid directly, or build a slurm script to run it on the cluster
if(sys.argv[1]=='run'):
    for parameter_combo in list(ParameterGrid(parameters)):
        ex.run(config_updates=parameter_combo)
elif(sys.argv[1]=='slurm'):
    save_script('Test_WINE_NOISY_Experiment.sh', 'WINE_NoisyTest_Experiment.py', fields, parameters)
    
else:
    print("please enter 'run' or 'build_slurm'")
