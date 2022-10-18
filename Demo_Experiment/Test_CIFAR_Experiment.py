#%% Sweep Hyperparameters Sequentially
from CIFAR_PWBTest_Experiment import ex
from sklearn.model_selection import ParameterGrid
from SLURM_Script_Factory import save_script
from findMaxAccuracy import findMaxAcc, getConfig
import sys

fields = {
        'memory': 16,
        'job_name': 'CIFAR_Experiment_Test',
        'time_h': 24,
        'num_cpus': 4
        }

exp, max_val, epoch = findMaxAcc('CIFAR')[:, -1] 
parameters = getConfig('CIFAR', exp)
for key in parameters:
    parameters[key] = [parameters[key]]
parameters['precision'] = [int(2**i) for i in range(10)]
parameters['shots'] = [int(2**i) for i in range(10)]
parameters['max_epoch'] = [epoch]
parameters['exp_train'] = [exp]
parameters['max_initial_weight'] = [0.2]


# Either run the grid directly, or build a slurm script to run it on the cluster
if(sys.argv[1]=='run'):
    for parameter_combo in list(ParameterGrid(parameters)):
        ex.run(config_updates=parameter_combo)
elif(sys.argv[1]=='slurm'):
    save_script('Test_CIFAR_Experiment.sh', 'CIFAR_PWBTest_Experiment.py', fields, parameters)
    
else:
    print("please enter 'run' or 'build_slurm'")
