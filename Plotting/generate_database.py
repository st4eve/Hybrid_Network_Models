"""Hybrid Network Models 2022"""
import collections
import copy
import json
from os import listdir
from os.path import isdir, isfile, join

import numpy as np


def get_filenames(path):
    """Utility: Get the filenames in a path"""
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_directories(path):
    """Utility: Get the directories in a path"""
    return [d for d in listdir(path) if isdir(join(path, d))]


def get_config(experiment_path):
    """Utility: Get the config for an experiment"""
    try:
        config_path = experiment_path + "/config.json"
        with open(file=config_path, mode="r", encoding="utf-8") as json_file:
            config = json.load(json_file)

        config_copy = copy.deepcopy(config)
        for key, val in config_copy.items():
            if val is None:
                config[key] = "None"
            if key == "seed":
                del config[key]
        return config
    except Exception as exception:
        # For now, when we have an issue reading from a file, return None
        print(f"Exception {exception}")
        print(f"Error reading from config file {experiment_path} . Ignoring file...")
        return None


def get_metrics(experiment_path):
    """Utility: Get the values for all metrics tracked in an experiment"""
    try:
        metrics_path = experiment_path + "/metrics.json"
        with open(file=metrics_path, mode="r", encoding="utf-8") as json_file:
            metrics = json.load(json_file)

        simplified_metrics = {}
        for key in metrics:
            if key == "traces":
                avg_traces = []
                std_traces = []
                for trace_data in metrics[key]["values"]:
                    avg_traces.append(
                        np.mean([trace["value"] for trace in trace_data["_storage"]])
                    )
                    std_traces.append(
                        np.std([trace["value"] for trace in trace_data["_storage"]])
                    )
                simplified_metrics["traces_average"] = avg_traces
                simplified_metrics["traces_std"] = std_traces
            else:
                simplified_metrics[key] = metrics[key]["values"]
        return simplified_metrics
    except Exception as exception:
        # For now, when we have an issue reading from a file, return None
        print(f"Exception {exception}")
        print(f"Error reading from metrics file {experiment_path} . Ignoring file...")
        return None


class ResultsDatabaseGenerator:
    """Class to generate database

    Schema:
    data[experiment]["config"][config_param_name] -> value
    data[experiment]["metrics"][metric_name] -> array of values over epoch

    """

    def __init__(self):
        self.data = None

    def initialize_from_saved_json(self, name):
        """Read blob from JSON file"""
        full_name = name + ".json"
        with open(file=full_name, mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        self.data = data

    def initialize_from_folder(self, experiment_folder, verify=True):
        """Create a blob of data for the results directory"""
        experiment_names = [
            x for x in get_directories(experiment_folder) if x.isdigit()
        ]
        data = {}
        for experiment in experiment_names:
            experiment_path = experiment_folder + "/" + experiment
            config = get_config(experiment_path)
            metrics = get_metrics(experiment_path)
            if config is not None and metrics is not None and metrics != '{}' and metrics != {}:
                experiment_number = int(experiment)
                data[experiment_number] = {}
                data[experiment_number]["config"] = config
                data[experiment_number]["metrics"] = metrics
        
        self.data = data
        if verify:
            self.verify_parameter_consistency()

    def verify_parameter_consistency(self):
        """Check consistency of metrics and config"""
        self.check_parameter_consistency(self.data, "config")
        self.check_parameter_consistency(self.data, "metrics")

    def check_parameter_consistency(self, data, param_type):
        """Check that the parameter keys in each experiment match"""
        data_copy = copy.deepcopy(self.data)
        reference = min(list(data_copy.keys()))
        reference_data = data_copy[reference]
        for exp, data in data_copy.items():
            for key1, key2 in zip(data[param_type].keys(), reference_data[param_type].keys()):
                if key1 != key2:
                    print(
                        f"Inconsistent config parameter {key1}!={key2} found in experiment folder between file and reference {exp}. Ignoring experiment..."
                    )
                    del self.data[exp]
                    break
            
    def get_num_epochs(self):
        """Get the distribution of the number of epochs"""
        num_epochs = []
        for exp in self.data:
            for _, val in self.data[exp]["metrics"].items():
                num_epochs.append(len(val))
                break
        counter = collections.Counter(num_epochs)
        return counter

    def limit_epochs(self, limit_epochs):
        """Restrict reading to a specific number of epochs"""
        if limit_epochs is not None:
            for exp in self.data:
                for metric, data in self.data[exp]["metrics"].items():
                    self.data[exp]["metrics"][metric] = data[:limit_epochs]
        return

    def check_trial_consistency(self):
        """Check that every experiment has the same number of epochs."""
        reference = min(list(self.data.keys()))
        data_copy = copy.deepcopy(self.data)
        reference_data = data_copy[reference]["metrics"]
        for exp, exp_data in data_copy.items():
            for key, val in reference_data.items():
                num_reference_epochs = len(val)
                num_current_epochs = len(exp_data["metrics"][key])
                if num_reference_epochs != num_current_epochs:
                    print(
                        f"Inconsistent epoch numbers found in experiment folder between reference with {num_reference_epochs} epochs and file {exp} with {num_current_epochs} epochs. Ignoring experiment..."
                    )
                    del self.data[exp]
                    break

    def save_blob(self, name):
        """Save blob to JSON file"""
        full_name = name + ".json"
        with open(file=full_name, mode="w", encoding="utf-8") as write_file:
            json.dump(self.data, write_file, indent=4)
