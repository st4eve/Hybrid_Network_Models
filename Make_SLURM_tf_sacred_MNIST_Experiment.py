# Build SLURM Script to run

from SLURM_Script_Factory import save_script
fields = {
    'memory': 8,
    'job_name': 'test_sweep',
    'time_h': 2,
    'num_cpus': 1
}

params = {
    'num_neurons':[1, 2, 3],
    'activation':["relu", "softmax"]
}

save_script('test2.sh', 'tf_sacred_MNIST_Experiment.py', fields, params)