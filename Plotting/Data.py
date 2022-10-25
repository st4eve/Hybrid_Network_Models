import json
from os import listdir
from os.path import isfile, isdir, join
import copy
import collections
from statistics import mean

def get_filenames(path):
    """Utility: Get the filenames in a path"""
    return [f for f in listdir(path) if isfile(join(path,f))]

def get_directories(path):
    """Utility: Get the directories in a path"""
    return [d for d in listdir(path) if isdir(join(path, d))]

def get_config(experiment_path):
    """Utility: Get the config for an experiment"""
    try: 
        config_path = experiment_path + "/config.json"
        with open(config_path) as json_file:
            config = json.load(json_file)
        
        config_copy = copy.deepcopy(config)
        for key, val in config_copy.items(): 
            if val is None: 
                config[key] = 'None'
            if key == 'seed': 
                del config[key]
        return config
    except: 
        # For now, when we have an issue reading from a file, return None
        error_str = "Error reading from config file " + experiment_path + ". Ignoring file..."
        print(error_str)
        return None

def get_metrics(experiment_path):
    """Utility: Get the values for all metrics tracked in an experiment"""
    try: 
        metrics_path = experiment_path + "/metrics.json"
        with open(metrics_path) as json_file:
            metrics = json.load(json_file)

        simplified_metrics = {}
        for key in metrics:
            # For now, omit the traces - figure out those plots later
            if type(metrics[key]["values"][0]) is dict and '_storage' in metrics[key]["values"][0].keys():
                pass
            else: 
                simplified_metrics[key] = metrics[key]["values"]
        return simplified_metrics
    except: 
        # For now, when we have an issue reading from a file, return None
        error_str = "Error reading from metrics file " + experiment_path + ". Ignoring file..."
        print(error_str)
        return None


class ResultsBlobber(): 
    def __init__(self):
        pass 

    def initialize_from_saved_json(self, name):
        """Read blob from JSON file"""
        full_name = name + ".json"
        with open(full_name) as json_file:
            data = json.load(json_file)
        self.data = data

    def initialize_from_folder(self, experiment_folder):
        """Create a blob of data for the results directory"""
        experiment_names = [x for x in get_directories(experiment_folder) if x.isdigit()]
        data = {}
        for experiment in experiment_names:
            experiment_path = experiment_folder + '/' + experiment
            data[int(experiment)] = {}
            data[int(experiment)]['config'] = get_config(experiment_path)
            data[int(experiment)]['metrics'] = get_metrics(experiment_path)

            # If we had an issue reading metrics, ignore the experiment
            # This could be changed later
            if data[int(experiment)]['metrics'] is None or data[int(experiment)]['config'] is None: 
                del data[int(experiment)]
        self.data = data

    def verify_parameter_consistency(self): 
        self.check_parameter_consistency(self.data, 'config')
        self.check_parameter_consistency(self.data, 'metrics')
    
    def verify_trial_consistency(self):
        self.check_trial_consistency(self.data)
        print("There are a total of " + str(len(self.data)) + " valid experiments to use for plotting.")

    def check_parameter_consistency(self, data, type):
        """Check that the parameter keys in each experiment match"""
        data_copy = copy.deepcopy(self.data)
        reference = min(list(data_copy.keys()))
        for exp in data_copy:
            if data_copy[exp][type].keys() != data_copy[reference][type].keys():
                # For now, if we have an inconsistent config, delete the experiment. This assumes a correct reference at file 1
                error_str = "Inconsistent config parameters found in experiment folder between reference and file" + str(exp) + ". Ignoring experiment..."
                print(error_str)
                # raise Exception(error_str)
                del data[exp]

    def get_num_epochs(self): 
        num_epochs = []
        for exp in self.data:
            for key in self.data[exp]['metrics'].keys():
                num_epochs.append(len(self.data[exp]['metrics'][key]))
                break 
        counter = collections.Counter(num_epochs)
        return counter 

    def limit_epochs(self, limit_epochs): 
        if limit_epochs is not None: 
            for exp in self.data: 
                for key in self.data[exp]['metrics'].keys(): 
                    self.data[exp]['metrics'][key] = self.data[exp]['metrics'][key][:limit_epochs]
        return 

    def check_trial_consistency(self):
        """Check that every experiment has the same number of epochs"""
        reference = min(list(self.data.keys()))
        data_copy = copy.deepcopy(self.data) # Create a copy, so we can delete entries without modifying during iteration
        for exp in data_copy:
            for key in data_copy[reference]['metrics'].keys():
                if len(data_copy[exp]['metrics'][key]) != len(data_copy[reference]['metrics'][key]):
                    # For now, if we have an inconsistent trial, delete the experiment. This assumes a correct reference at file 1
                    error_str = "Inconsistent epoch numbers found in experiment folder between reference with " + \
                                str(len(self.data[reference]['metrics'][key])) + " epochs and file " + str(exp) + " with " + \
                                str(len(self.data[exp]['metrics'][key])) + " epochs"  + ". Ignoring experiment..."
                    print(error_str)
                    # raise Exception(error_str)
                    del self.data[exp]
                    break

    def get_data(self): 
        """Get the blob"""
        return self.data 

    def save_blob(self, name):
        """Save blob to JSON file"""
        full_name = name + ".json"
        with open(full_name, "w") as write_file:
            json.dump(self.data, write_file, indent=4)

class ResultsQueryEngine():
    def __init__(self, data):
        self.data = data

    def get_min_experiment(self):
        return min(list(self.data.keys()))

    def get_max_experiment(self):
        return max(list(self.data.keys()))

    def get_num_experiments(self): 
        return len(list(self.data.keys()))

    def get_hyperparameters(self): 
        reference = min(list(self.data.keys()))
        hyperparam_dict = self.data[reference]['config']
        hyperparameters = list(hyperparam_dict.keys())
        return hyperparameters

    def get_metrics(self): 
        reference = min(list(self.data.keys()))
        metrics_dict = self.data[reference]['metrics']
        metrics = list(metrics_dict.keys())
        return metrics

    def get_metric(self, experiment_number, metric): 
        return self.data[experiment_number]['metrics'][metric]

    def get_hyperparameter(self, experiment_number, hyperparameter): 
        return self.data[experiment_number]['config'][hyperparameter]

    def get_all_configs(self): 
        configs = []
        for exp in self.data: 
            configs.append(self.data[exp]['config'])

    def get_unique_hyperparameter_dict(self): 
        reference = self.get_min_experiment()
        hyperparam_dict = self.data[reference]['config'].copy()

        for key, value in hyperparam_dict.items():
            hyperparam_dict[key] = set([value])

        for exp in self.data:
            if exp != reference:
                for key, value in self.data[exp]['config'].items():
                    hyperparam_dict[key].add(value)

        for key, value in hyperparam_dict.items():
            hyperparam_dict[key] = list(value)

        return hyperparam_dict

    def get_average(self, target_hyperparameter, target_hyperparameter_value, target_metric): 
        data_to_average = []
        for exp in self.data: 
            if self.data[exp]['config'][target_hyperparameter] == target_hyperparameter_value: 
                data_to_average.append(self.data[exp]['metrics'][target_metric])

        averaged_data = len(data_to_average[0])*[0]
        for i in range(len(data_to_average[0])): 
            averaged_data[i] = mean([row[i] for row in data_to_average])
        
        return averaged_data

    def get_hyperparameter_strings(self, mode = "str_list"): 
        hyperparam_dict = self.get_unique_hyperparameter_dict()
        str_list = []
        key_to_str = {}
        str_to_key = {}
        for key, value in hyperparam_dict.items():
            full_str = "{}: {}".format(key, sorted(hyperparam_dict[key]))
            str_list.append(full_str)
            key_to_str[key] = full_str
            str_to_key[full_str] = key
        if mode == "str_list": 
            return str_list
        if mode == "key_to_str_dict":
            return key_to_str
        if mode == "str_to_key_dict": 
            return str_to_key

    def get_data(self): 
        """Get the blob"""
        return self.data 

    def get_metric_from_config(self, config, metric): 
        for exp in self.data: 
            if self.data[exp]['config'] == config: 
                return self.get_metric(exp, metric)