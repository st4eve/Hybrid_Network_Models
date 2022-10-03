#%% Sweep Hyperparameters Sequentially
from WINE_NoisyTest_Experiment import ex
import numpy as np
from sklearn.model_selection import ParameterGrid
from SLURM_Script_Factory import save_script
from findMaxAccuracy import findMaxAcc, getConfig
import sys

fields = {
        'memory': 16,
        'job_name': 'WINE_Experiment',
        'time_h': 24,
        'num_cpus': 4
        }

exp, max_val, epoch = findMaxAcc('WINE') 
parameters = getConfig('WINE', exp)
for key in parameters:
    parameters[key] = [parameters[key]]
#parameters['precision'] = [int(2**i) for i in range(10)]
parameters['shots'] = [int(2**i) for i in range(10)]
parameters['sigma'] = list(np.logspace(-10, -2, 11))
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
