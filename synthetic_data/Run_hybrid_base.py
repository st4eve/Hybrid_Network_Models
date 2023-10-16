#%% Sweep Hyperparameters Sequentially
#from hybrid_base import ex
from sklearn.model_selection import ParameterGrid
from common_packages.SLURM_Script_Factory import save_script
import sys
import json

with open('hybrid_base.json', 'r') as json_file:
    parameters = json.load(json_file)

print(parameters)
fields = {
            'memory': 32,
            'job_name': 'Synthetic_Cutoff_10',
            'time_h': 24,
            'num_cpus': 6
        }

save_script('Run_hybrid_base.sh', 'hybrid_base.py', fields, parameters['parameters'])
