# Build SLURM Script to run
from common_packages.SLURM_Script_Factory import save_script

fields = {
    'memory': 16,
    'job_name': 'test_sweep',
    'time_h': 2,
    'num_cpus': 1
}

parameters = {'num_epochs': [5],
              'num_neurons': [10, 20],
              'activation_type':["softmax"],
}

save_script('../test2.sh', 'tf_sacred_MNIST_Experiment.py', fields, parameters)