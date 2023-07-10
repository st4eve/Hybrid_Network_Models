#%% Sweep Hyperparameters Sequentially
#from hybrid_base import ex
from common_packages.SLURM_Script_Factory import save_script
import sys
import json

with open('quantum_base.json', 'r') as json_file:
    parameters = json.load(json_file)

print(parameters)
fields = {
            'memory': 32,
            'job_name': 'Synthetic_Cutoff_Sweep',
            'time_h': 128,
            'num_cpus': 4
        }

save_script('Run_quantum_base.sh', 'quantum_base.py', fields, parameters['parameters'])
