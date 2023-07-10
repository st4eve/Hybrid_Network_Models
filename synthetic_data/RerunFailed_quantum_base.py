#%% Sweep Hyperparameters Sequentially
#from hybrid_base import ex
from common_packages.SLURM_Script_Factory import save_script
import sys
import json
import os
from Plotting.generate_database import get_directories, get_config, get_metrics

with open('quantum_base.json', 'r') as json_file:
    parameters = json.load(json_file)

for key, val in parameters.

def initialize_from_folder(self, experiment_folder, verify=True):
    """Create a blob of data for the results directory"""
    experiment_names = [
        x for x in get_directories(experiment_folder) if x.isdigit()
    ]
    for experiment in experiment_names:
        experiment_path = experiment_folder + "/" + experiment
        config = get_config(experiment_path)
        metrics = get_metrics(experiment_path)

        if metrics

fields = {
            'memory': 48,
            'job_name': 'Synthetic_Cutoff_Sweep_Rerun',
            'time_h': 128,
            'num_cpus': 6
        }

print(parameters)
#save_script('Run_quantum_base.sh', 'quantum_base.py', fields, parameters['parameters'])
