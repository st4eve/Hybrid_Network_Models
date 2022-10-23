import json
from os import listdir
from os.path import isfile, isdir, join

def get_filenames(path):
    """Utility: Get the filenames in a path"""
    return [f for f in listdir(path) if isfile(join(path,f))]

def get_directories(path):
    """Utility: Get the directories in a path"""
    return [d for d in listdir(path) if isdir(join(path, d))]

def get_config(experiment_path):
    """Utility: Get the config for an experiment"""
    config_path = experiment_path + "/config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config

def get_metrics(experiment_path):
    """Utility: Get the values for all metrics tracked in an experiment"""
    metrics_path = experiment_path + "/metrics.json"
    with open(metrics_path) as json_file:
        metrics = json.load(json_file)

    simplified_metrics = {}
    for key in metrics:
        simplified_metrics[key] = metrics[key]["values"]
    return simplified_metrics

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
        metrics = {}
        for experiment in experiment_names:
            experiment_path = experiment_folder + '/' + experiment
            data[int(experiment)] = {}
            data[int(experiment)]['config'] = get_config(experiment_path)
            data[int(experiment)]['metrics'] = get_metrics(experiment_path)
        self.check_parameter_consistency(data, 'config')
        self.check_parameter_consistency(data, 'metrics')
        self.check_trial_consistency(data)
        self.data = data

    def check_parameter_consistency(self, data, type):
        """Check that the parameter keys in each experiment match"""
        reference = min(list(data.keys()))
        for exp in data:
            if data[exp][type].keys() != data[reference][type].keys():
                error_str = "Inconsistent config parameters found in experiment folder between reference and file" + str(exp)
                raise Exception(error_str)

    def check_trial_consistency(self, data):
        """Check that every experiment has the same number of epochs"""
        reference = min(list(data.keys()))
        for exp in data:
            for key in data[reference]['metrics'].keys():
                if len(data[exp]['metrics'][key]) != len(data[reference]['metrics'][key]):
                    error_str = "Inconsistent epoch numbers found in experiment folder between reference with " + \
                                str(len(data[reference]['metrics'][key])) + " epochs and file " + str(exp) + " with " + \
                                str(len(data[exp]['metrics'][key])) + " epochs"
                    raise Exception(error_str)

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