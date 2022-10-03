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
        
    exp = 1
    max_val = 0
    epoch = 0
    
    dir_name = 'Experiment_Data_%s'%filedir
    
    dirs = os.listdir(dir_name)
    if '_sources' in dirs:
        dirs.remove('_sources')
    for directory in dirs:
        filedir = dir_name + '/' + directory + '/'
        try:
            acc = getAccuracy(filedir)
        except:
            print('Experiment %s was invalid'%directory)
        else:
            curr_max, curr_epoch = findMax(acc)
            if curr_max > max_val:
                max_val = curr_max
                epoch = curr_epoch
                exp = int(directory)
    return exp, max_val, epoch

def main():
    if sys.argv[1] != None:
        print(findMaxAcc(sys.argv[1]))
    else:
        print('Please include an experiment to search.')
        
if __name__ == "__main__":
    main()
