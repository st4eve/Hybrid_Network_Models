import json
from abc import abstractmethod

from PyInquirer import prompt
from sklearn.model_selection import ParameterGrid

from Data import *
from Plot import *


class PlotterInterface:
    @abstractmethod
    def get_plotter(data_list, selected_metric, selected_hyperparameter):
        pass


class HyperparameterFinalMetric(PlotterInterface):
    def get_plotter(data_list, selected_metric, selected_hyperparameter):
        x = []
        y = []
        for data in data_list:
            config = data["config"]
            metric = data["metric"]
            x.append(config[selected_hyperparameter])
            y.append(metric[-1])
        x_label = selected_hyperparameter
        y_label = selected_metric
        return BasicPlot(x, y, x_label, y_label)


class MetricOverEpochByHyperparameter(PlotterInterface):
    def get_plotter(data_list, selected_metric, selected_hyperparameter):
        x = []
        y = []
        labels = []
        for data in data_list:
            config = data["config"]
            metric = data["metric"]
            y.append(metric)
            x.append([i for i in range(len(y[len(y) - 1]))])
            labels.append(
                str(selected_hyperparameter)
                + "="
                + str(config[selected_hyperparameter])
            )
        x_label = "Epochs"
        y_label = selected_metric
        return MultiPlot(x, y, x_label, y_label, labels)


class MetricOverEpochBySingleHyperparameter(PlotterInterface):
    def get_plotter(data_list, selected_metric, selected_hyperparameter=None):
        config = data_list[0]["config"]
        y = data_list[0]["metric"]
        x = [i for i in range(len(y))]
        x_label = "Epochs"
        y_label = selected_metric
        return BasicPlot(x, y, x_label, y_label)


def get_combos(dict):
    dict_list = []
    for parameter_combo in list(ParameterGrid(dict)):
        dict_list.append(parameter_combo)
    return dict_list


def list_dict_to_element_dict(dict):
    for key, value in dict.items():
        dict[key] = value[0]
    return dict


def command(cmd_type, message, choices=None, predefined_output=None):
    if predefined_output is None:
        name = "variable"
        question = [
            {"type": cmd_type, "name": name, "message": message, "choices": choices}
        ]
        if cmd_type == "input":
            question = [{"type": cmd_type, "name": name, "message": message}]
        elif cmd_type == "list":
            question = [
                {"type": cmd_type, "name": name, "message": message, "choices": choices}
            ]
        answer = prompt(question)
        return answer[name]
    else:
        return predefined_output


class CLI:
    def __init__(self):
        self.path = None
        self.query_engine = None
        self.plotter = None
        self.selected_metric = None
        self.selected_hyperparameter = None
        self.config_filter = None
        self.plotting_data = None

    def run(self, input_path=None):
        self.path = self.get_experiment_path(input_path)

        # Create the blob. Limit number of epochs in case of inconsistency:
        data_blob = ResultsBlobber()
        data_blob.initialize_from_folder(self.path)
        data_blob.verify_parameter_consistency()
        self.num_epochs = self.get_num_epochs(data_blob.get_num_epochs())
        data_blob.limit_epochs(self.num_epochs)
        data_blob.check_trial_consistency()

        self.query_engine = ResultsQueryEngine(data_blob.data)

        while True:
            next_functions, plotting_factory = self.select_plotting_mode()

            for function in next_functions:
                function()

            if len(self.plotting_data) > 0:
                self.plotter = plotting_factory.get_plotter(
                    self.plotting_data,
                    self.selected_metric,
                    self.selected_hyperparameter,
                )
                self.plot(self.plotter)
            else:
                print(
                    "No valid data found for target configurations due to dropped experiments."
                )

    def get_experiment_path(self, path=None):
        cmd_type = "input"
        message = "Please enter the path to the experiment"
        return command(cmd_type, message, predefined_output=path)

    def get_num_epochs(self, epoch_distribution):
        if len(epoch_distribution) > 1:
            cmd_type = "input"
            message = "In the case of multiple unfinished experiments, limit the plots to a specific number of epochs. Otherwise, leave blank. The distribution of finished epochs is: {}".format(
                epoch_distribution
            )
            num_epochs = command(cmd_type, message)
            if num_epochs == "":
                return None
            else:
                return int(num_epochs)
        else:
            return None

    def select_plotting_mode(self):
        cmd_type = "list"
        message = "Select plotting mode"
        choices = {
            "Final Metric vs Hyperparameter": {
                "functions": [
                    self.select_metric,
                    self.select_hyperparameter,
                    self.select_hyperparameter_constants,
                    self.prepare_plotting_data,
                ],
                "plotting_factory": HyperparameterFinalMetric,
            },
            "Metric vs Epochs for each Hyperparameter": {
                "functions": [
                    self.select_metric,
                    self.select_hyperparameter,
                    self.select_hyperparameter_constants,
                    self.prepare_plotting_data,
                ],
                "plotting_factory": MetricOverEpochByHyperparameter,
            },
            "Metric vs Epochs for single Hyperparameter": {
                "functions": [
                    self.select_metric,
                    self.select_hyperparameter_constants,
                    self.prepare_plotting_data,
                ],
                "plotting_factory": MetricOverEpochBySingleHyperparameter,
            },
            "Averaged Metric vs Epochs for each Hyperparameter": {
                "functions": [
                    self.select_metric,
                    self.select_hyperparameter,
                    self.prepare_average_plotting_data,
                ],
                "plotting_factory": MetricOverEpochByHyperparameter,
            },
        }
        plotting_mode = command(cmd_type, message, list(choices.keys()))
        return (
            choices[plotting_mode]["functions"],
            choices[plotting_mode]["plotting_factory"],
        )

    def prepare_plotting_data(self):
        target_configs = get_combos(self.config_filter)
        if self.selected_hyperparameter is not None:
            target_configs.sort(key=lambda dict: dict[self.selected_hyperparameter])
        target_configs = target_configs

        self.plotting_data = []
        for target_config in target_configs:
            data_dict = {}
            data_dict["config"] = target_config
            data_dict["metric"] = self.query_engine.get_metric_from_config(
                target_config, self.selected_metric
            )
            # In case of errors, ignore the data poitn
            if data_dict["metric"] is not None:
                self.plotting_data.append(data_dict)

    def prepare_average_plotting_data(self):
        hyperparameter_values = self.query_engine.get_unique_hyperparameter_dict()[
            self.selected_hyperparameter
        ]
        self.plotting_data = []
        for hyperparameter_value in hyperparameter_values:
            data_dict = {}
            data_dict["config"] = {self.selected_hyperparameter: hyperparameter_value}
            data_dict["metric"] = self.query_engine.get_average(
                self.selected_hyperparameter, hyperparameter_value, self.selected_metric
            )
            self.plotting_data.append(data_dict)

    def select_metric(self):
        type = "list"
        message = "Please select the metric to plot: "
        self.selected_metric = command(type, message, self.query_engine.get_metrics())

    def select_hyperparameter(self):
        type = "list"
        message = "Please select the hyperparameter of interest: "
        selected_str = command(
            type, message, self.query_engine.get_hyperparameter_strings()
        )
        str_to_key_dict = self.query_engine.get_hyperparameter_strings(
            mode="str_to_key_dict"
        )
        self.selected_hyperparameter = str_to_key_dict[selected_str]

    def select_hyperparameter_constants(self):
        # This could be refactored...
        self.config_filter = self.query_engine.get_unique_hyperparameter_dict().copy()
        for hyperparam in self.query_engine.get_hyperparameters():
            hyperparam_elements = self.query_engine.get_unique_hyperparameter_dict()[
                hyperparam
            ]
            if (
                hyperparam != self.selected_hyperparameter
                and len(hyperparam_elements) > 1
            ):
                cmd_type = "list"
                message = "Please select a single value for " + hyperparam + ":"
                if type(hyperparam_elements[0]) == float:
                    val = command(
                        cmd_type, message, [str(x) for x in sorted(hyperparam_elements)]
                    )
                    self.config_filter[hyperparam] = [float(val)]
                elif type(hyperparam_elements[0]) == int:
                    val = command(
                        cmd_type, message, [str(x) for x in sorted(hyperparam_elements)]
                    )
                    self.config_filter[hyperparam] = [int(val)]
                else:
                    val = command(cmd_type, message, hyperparam_elements)
                    self.config_filter[hyperparam] = [val]

    def plot(self, plotter):
        plotter.plot()

        while True:
            options = plotter.get_options()
            option_strings = []
            option_string_map = {}
            for key, val in options.items():
                option_string = key + " = " + str(val)
                option_strings.append(option_string)
                option_string_map[option_string] = key

            option_strings.append("save")
            option_strings.append("exit")
            option_strings.insert(0, "plot")
            option_strings.append("settings from file")
            option_strings.append("new plot")

            cmd_type = "list"
            message = "Please select plotting option to modify, or select plot, save, settings from file, or exit"
            choices = option_strings
            option = command(cmd_type, message, choices)

            if option == "save":
                cmd_type = "input"
                message = "Enter name of folder to save to:"
                save_name = command(cmd_type, message)
                plotter.save(save_name)
                plotter.close()
            elif option == "plot":
                plotter.close()
                plotter.plot()
            elif option == "exit":
                exit()
            elif option == "settings from file":
                cmd_type = "input"
                message = "Enter path to saved settings"
                path = command(cmd_type, message)
                plotter.apply_saved_settings(path)
            elif option == "new plot":
                return
            else:
                cmd_type = "input"
                message = "Please enter the new value for " + option + ":"
                value = command(cmd_type, message, choices)
                plotter.set_option(option_string_map[option], value)
