import json
from datetime import datetime, timedelta
from Plotting.generate_database import get_config, get_directories
import numpy as np
import os

def calc_time(metrics_path):
    try:
        with open(metrics_path, "r") as file:
            metrics = json.load(file)
    except FileNotFoundError:
        print("File %s not found"%metrics_path)
        return None 
    except json.JSONDecodeError:
        print("File %s is not a valid json file"%metrics_path)
        return None
    num_epochs = metrics['epoch']['values'][-1]
    if num_epochs < 199:
        return None
    time_arr = metrics['epoch']['timestamps']
    time1 = datetime.strptime(time_arr[0], '%Y-%m-%dT%H:%M:%S.%f')
    time2 = datetime.strptime(time_arr[-1], '%Y-%m-%dT%H:%M:%S.%f')
    return time2 - time1 

def calc_time_per_epoch(metrics_path):
    with open(metrics_path, 'r') as file:
        metrics = json.load(file)
    num_epochs = metrics['epoch']['values'][-1]
    
    time_arr = np.array(metrics['epoch']['timestamps'])
    time_arr = time_arr.astype('datetime64[ns]')
    time_diff = time_arr[1:] - time_arr[:-1]
    return time_diff 

def find_max_time(exp_path):
    exps = os.listdir(exp_path)
    if '_sources' in exps:
        exps.remove('_sources')
    max_time = timedelta(0)
    max_exp = 0
    for exp in exps:
        metrics_path = exp_path + '/' + exp + '/metrics.json'
        time = calc_time(metrics_path)
        if time > max_time:
            max_time = time
            max_exp = exp
    return max_exp, max_time

def generate_time_database(exp_path):
    experiment_names = [
            x for x in get_directories(exp_path) if x.isdigit()
        ]
    data = {}
    for experiment in experiment_names:
        experiment_path = exp_path + "/" + experiment
        config = get_config(experiment_path)
        time = calc_time(experiment_path+'/metrics.json')
        if config is not None and time is not None and time != 0:
            experiment_number = int(experiment)
            data[experiment_number] = {}
            data[experiment_number]["config"] = config
            data[experiment_number]["time"] = time
    
    return data

def print_max_time(exp_path):
    exp, time = find_max_time(exp_path)
    print('In %s the maximum experiment time was:'%exp_path)
    print("Experiment %s took %s"%(exp, time))
    with open(exp_path + '/' + exp + '/config.json', 'r') as file:
        config = json.load(file)
    config.pop('__doc__')
    print('Config:')
    for key, val in config.items():
        print("\t%s: %s"%(key.capitalize(), val)) 

def main():
    metrics_path = "../synthetic_data/Synthetic_Quantum_Base_Experiment3/6/metrics.json"
    calc_time(metrics_path)
    
    print_max_time("../synthetic_data/Synthetic_Quantum_Base_Experiment3") 
    
    calc_time_per_epoch(metrics_path)
if __name__ == "__main__":
    main()
    