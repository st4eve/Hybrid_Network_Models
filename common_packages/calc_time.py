import json
from datetime import datetime, timedelta
import os

def calc_time(metrics_path):
    with open(metrics_path, "r") as file:
        metrics = json.load(file)
        
    num_epochs = metrics['epoch']['values'][-1]
    
    time_arr = metrics['epoch']['timestamps']
    time1 = datetime.strptime(time_arr[0], '%Y-%m-%dT%H:%M:%S.%f')
    time2 = datetime.strptime(time_arr[-1], '%Y-%m-%dT%H:%M:%S.%f')
    return time2 - time1 

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
if __name__ == "__main__":
    main()
    