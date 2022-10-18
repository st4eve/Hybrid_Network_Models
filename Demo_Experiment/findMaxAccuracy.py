import json
import os
import sys
import numpy as np

def getConfig(filedir, exp):
    dir_name = 'Experiment_Data_%s/%d/'%(filedir, exp)
    filename = dir_name + 'config.json'
    with open(filename) as json_file:
        return json.load(json_file)

def findMaxAcc(filedir):
    def getAccuracy(filedir):
        filename = filedir + 'metrics.json'
        with open(filename) as json_file:
            data = json.load(json_file)
        acc = data['val_accuracy']['values']
        return acc
        
    def findMax(arr):
        return np.max(arr), np.argmax(arr)
        
    def getConfig(filedir):
        filename = filedir + 'config.json'
        with open(filename) as json_file:
            return json.load(json_file)
        
    exp = []
    max_val = [0.0]
    epoch = []
    
    dir_name = 'Experiment_Data_%s'%filedir
    
    dirs = os.listdir(dir_name)
    dirs.remove('_sources')
    for directory in dirs:
        filedir = dir_name + '/' + directory + '/'
        acc = getAccuracy(filedir)
        curr_max, curr_epoch = findMax(acc)
        if curr_max > max_val[-1]:
            max_val.append(curr_max)
            epoch.append(curr_epoch)
            exp.append(int(directory))
    return np.array([exp, max_val[1:], epoch])

def main():
    if sys.argv[1] != None:
        print(findMaxAcc(sys.argv[1]))
    else:
        print('Please include an experiment to search.')
        
if __name__ == "__main__":
    main()
