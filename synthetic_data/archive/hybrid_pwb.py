"""Hybrid Network Models 2022"""
import copy
import json
from os import listdir
from os.path import isdir, join

import tensorflow as tf
from data import generate_synthetic_dataset
from hybrid_base import EXPERIMENT_NAME as BASE_EXPERIMENT_NAME
from hybrid_base import LOSS_FUNCTION, NUM_EPOCHS, OPTIMIZER, RANDOM_SEED
from keras import Model, layers, models, regularizers
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from PWBLayer_TF import PWBLinearLayer
from common_packages.utilities import get_equivalent_classical_layer_size

EXPERIMENT_NAME = "Synthetic_Hybrid_PWB_Experiment_Test2"
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
    sigma = 1 # pylint: disable=W0612
    num_qumodes = 2  # pylint: disable=W0612
    network_type = "classical"  # pylint: disable=W0612

@ex.automain
def define_and_train(sigma, num_qumodes, network_type):
    """Build and run the network"""

    tf.random.set_seed(RANDOM_SEED)

    class Net(Model):  # pylint: disable=W0223
        """Neural network model to train on"""

        def __init__(self):
            super().__init__()
            self.gaussian = layers.GaussianNoise(sigma)
            precision = int(2**15 - 1)
            self.base_model = models.Sequential(
                [
                    PWBLinearLayer(20, activation="relu", precision=precision),
                    PWBLinearLayer(20, activation="relu", precision=precision),
                    PWBLinearLayer(2 * num_qumodes, activation=None, precision=precision),
                ]
            )

            if network_type == "classical":
                self.quantum_substitue = models.Sequential(
                    [
                        layers.Dense(
                            get_equivalent_classical_layer_size(num_qumodes, 2 * num_qumodes, 3),
                            activation="relu",
                            bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                            kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        ),
                    ]
                )

            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=num_qumodes,
                n_circuits=1,
                n_layers=1,
                cutoff_dim=3,
                encoding_method="Amplitude_Phase",
                regularizer=regularizers.L1(l1=0.1),
                max_initial_weight=0.15,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("Sigmoid", self.quantum_layer.encoding_object)

            self.final_layer = models.Sequential([PWBLinearLayer(3, activation="softmax", precision=precision)])

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.gaussian(inputs, training=True)
            output = self.base_model(output)
            if network_type == "quantum":
                output = self.quantum_preparation_layer(output)
                output = self.quantum_layer(output)
            elif network_type == "classical":
                output = self.quantum_substitue(output)
            else:
                raise ValueError("Invalid network type specified.")
            output = self.final_layer(output)
            return output

    _, validate_data = generate_synthetic_dataset()
    model = Net()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    sub_folders = [folder for folder in listdir(BASE_EXPERIMENT_NAME) if isdir(join(BASE_EXPERIMENT_NAME, folder))]
    experiment_numbers = [sub_folder for sub_folder in sub_folders if sub_folder.isdigit()]
    target_experiment_path_list = []
    for experiment_num in experiment_numbers:
        config = get_config(f"{BASE_EXPERIMENT_NAME}/{experiment_num}")
        if config["num_qumodes"] == num_qumodes and config["network_type"] == network_type:
            target_experiment_path_list.append(f"{BASE_EXPERIMENT_NAME}/{experiment_num}")
    
    for target_experiment_path in target_experiment_path_list:
        model.load_weights(f"{target_experiment_path}/weights/weight{NUM_EPOCHS-1}.ckpt", by_name=False)
        for i in range(10):
            val_loss, val_acc = model.evaluate(*validate_data, verbose=2)
            log_performance(val_accuracy=val_acc, val_loss=val_loss, epoch=i)  # pylint: disable=E1120
