"""Hybrid Network Models 2022"""
import copy
import json
from os import listdir
from os.path import isdir, join

import tensorflow as tf
from WINE_Dataset import *
from hybrid_base import EXPERIMENT_NAME as _BASE_EXPERIMENT_NAME

from hybrid_base import LOSS_FUNCTION, NUM_EPOCHS, OPTIMIZER, RANDOM_SEED
from keras import Model, layers, models, regularizers
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from PWBLayer_TF import PWBLinearLayer
from common_packages.utilities import get_num_parameters_per_layer

EXPERIMENT_NAME = "WINE_Hybrid_PWB_Experiment"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


def get_config(experiment_path):
    """Utility: Get the config for an experiment"""
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


@ex.capture
def log_performance(_run, val_accuracy, val_loss, epoch):
    """Logs performance with sacred framework"""
    _run.log_scalar("val_loss", float(val_loss))
    _run.log_scalar("val_accuracy", float(val_accuracy))
    _run.log_scalar("epoch", int(epoch), epoch)


@ex.config
def confnet_config():
    """Default config"""
    precision = 2**10  # pylint: disable=W0612
    num_qumodes = 4  # pylint: disable=W0612
    network_type = "classical"  # pylint: disable=W0612
    #network_type = "quantum"


@ex.automain
def define_and_train(precision, num_qumodes, network_type):
    """Build and run the network"""

    tf.random.set_seed(RANDOM_SEED)

    class Net(Model):  # pylint: disable=W0223
        """Neural network model to train on"""

        def __init__(self):
            super().__init__()

            self.base_model = models.Sequential(
                [
                    PWBLinearLayer(
                        40,
                        input_dim=9,
                        activation="relu",
                        precision=precision),
                ]
            )
            
            self.classical1 = models.Sequential([
                    layers.Flatten(),
                    PWBLinearLayer(
                        2 * num_qumodes,
                        activation=None,
                        precision=precision
                    )
            ])
            
            classical_size = int(get_num_parameters_per_layer(num_qumodes) // (1 + 2*num_qumodes) + 1)
            if network_type == "classical":
                self.quantum_substitue = models.Sequential(
                    [
                        PWBLinearLayer(
                            classical_size,
                            activation="relu",
                            precision=precision
                        ),
                    ]
                )

            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=num_qumodes,
                n_circuits=1,
                n_layers=1,
                cutoff_dim=5,
                encoding_method="Amplitude_Phase",
                regularizer=regularizers.L1(l1=0.01),
                max_initial_weight=0.2,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("TanH", self.quantum_layer.encoding_object)

            self.classical2 = PWBLinearLayer(
                        2,
                        activation="softmax",
                        precision=precision
                    )

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.base_model(inputs)
            output = self.classical1(output)
            if network_type == "quantum":
                output = self.quantum_preparation_layer(output)
                output = self.quantum_layer(output)
            elif network_type == "classical":
                output = self.quantum_substitue(output)
            else:
                raise ValueError("Invalid network type specified.")
            output = self.classical2(output)
            return output

    x_val,y_val = load_validation()
    validate_data = [x_val.astype(np.float32),y_val.astype(np.float32)]
    model = Net()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    model(x_val[0:2].astype(np.float32))
    #print([l.weights for l in model.layers])
    if network_type == 'quantum':
        BASE_EXPERIMENT_NAME = "Experiment_Data_WINE"
        target_experiment_path = f"{BASE_EXPERIMENT_NAME}/65"
    else:
        BASE_EXPERIMENT_NAME = _BASE_EXPERIMENT_NAME
        sub_folders = [folder for folder in listdir(BASE_EXPERIMENT_NAME) if isdir(join(BASE_EXPERIMENT_NAME, folder))]
        experiment_numbers = [sub_folder for sub_folder in sub_folders if sub_folder.isdigit()]
        for experiment_num in experiment_numbers:
            config = get_config(f"{BASE_EXPERIMENT_NAME}/{experiment_num}")
            if config["num_qumodes"] == num_qumodes and config["network_type"] == network_type:
                target_experiment_path = f"{BASE_EXPERIMENT_NAME}/{experiment_num}"
                break
    for i in range(NUM_EPOCHS):
        model.load_weights(f"{target_experiment_path}/weights/weight{i}.ckpt", by_name=False)
        val_loss, val_acc = model.evaluate(*validate_data, verbose=3)
        log_performance(val_accuracy=val_acc, val_loss=val_loss, epoch=i)  # pylint: disable=E1120
