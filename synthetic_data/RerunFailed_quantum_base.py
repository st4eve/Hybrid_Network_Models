#%% Sweep Hyperparameters Sequentially
#from hybrid_base import ex
from common_packages.SLURM_Script_Factory import DEFAULT_SLURM_FIELDS, TEMPLATE
import sys
import json
import os
import copy
from Plotting.generate_database import get_directories, get_config, get_metrics
ex_path = "/home/st4eve/Mounts/graham/Hybrid_Network_Models/synthetic_data/Synthetic_Quantum_Base_Experiment_cutoff_sweep"
#%%
with open('quantum_base.json', 'r') as json_file:
    parameters = json.load(json_file)
for key, val in parameters['parameters'].items():
    parameters['parameters'][key] = []
print(parameters)

#%%
failed_exps = []
def initialize_from_folder(experiment_folder):
    """Create a blob of data for the results directory"""
    experiment_names = [
        x for x in get_directories(experiment_folder) if x.isdigit()
    ]
    for experiment in experiment_names:
        experiment_path = experiment_folder + "/" + experiment
        config = get_config(experiment_path)
        metrics = get_metrics(experiment_path)

        if metrics is None and config is not None:
            print("No metrics found for experiment: " + experiment)
            config.pop('__doc__', None)
            failed_exps.append(copy.deepcopy(config))

fields = {
            'memory': 48,
            'job_name': 'Synthetic_Cutoff_Sweep_Rerun',
            'time_h': 128,
            'num_cpus': 6
        }


TEMPLATE = '''
#!/bin/env bash
#SBATCH --array=0-{num_jobs}
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --mem={memory}{memory_unit}
#SBATCH --time={time_d}-{time_h}:{time_m}:{time_s}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=17tna@queensu.ca
#SBATCH --account=def-bshastri
input$SLURM_ARRAY_TASK_ID.dat

trial=${{SLURM_ARRAY_TASK_ID}}

{call_function}
'''.strip()

initialize_from_folder(ex_path)

call_function = ""
for exp in failed_exps:
    call_function += f"srun python quantum_base.py with "
    for key, val in exp.items():
        call_function += f"{key}={val} "
    call_function += '\n'

subs = {
    'num_jobs': len(failed_exps)-1,
    'call_function': call_function,
}
for key, val in DEFAULT_SLURM_FIELDS.items():
    subs[key] = val
for key, val in fields.items():
    subs[key] = val

script_str = TEMPLATE.format(**subs)
filename = 'Rerun_failed_quantum_base.sh'
with open(filename, 'w') as file:
    file.write(script_str)
# %%
