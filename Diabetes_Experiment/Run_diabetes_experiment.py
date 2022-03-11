#%% Sweep Hyperparameters Sequentially
from Diabetes_Experiment import ex
from sklearn.model_selection import ParameterGrid
from SLURM_Script_Factory import save_script
import sys

if __name__ == "__main__":
    args = sys.argv[1:]

    parameters = {'encoding_strategy': ["None", "Sigmoid", "Sigmoid_BatchNorm"],
                  'cutoff_dimension':[15]
    }

    fields = {
        'memory': 32,
        'job_name': 'test_sweep',
        'time_h': 12,
        'num_cpus': 23
    }

    parameter_combinations = list(ParameterGrid(parameters))

    if(args[0]=='run'):
        for parameter_combo in parameter_combinations:
             ex.run(config_updates=parameter_combo)

    elif(args[0]=='build_slurm'):
        save_script('Diabetes.sh', 'Iris_Experiment.py', fields, parameters)

    else:
        print("please enter 'run' or 'build_slurm'")





