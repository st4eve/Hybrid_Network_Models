"""Hybrid Network Models 2022"""
import copy
import json
from os import listdir
from os.path import isdir, join

import tensorflow as tf
from WINE_Dataset import *
from hybrid_base import EXPERIMENT_NAME as _BASE_EXPERIMENT_NAME

from hybrid_base import LOSS_FUNCTION, NUM_EPOCHS, OPTIMIZER, RANDOM_SEED, BATCH_SIZE
from keras import Model, layers, models, regularizers
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from PWBLayer_TF import PWBLinearLayer
from common_packages.utilities import get_num_parameters_per_layer

from WINE_Dataset import SIGMAS

EXPERIMENT_NAME = "WINE_Noisy_Train1"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds

NUM_EPOCHS = 100
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
def log_performance(_run, logs, epoch, model):
    """Logs performance with sacred framework"""
    _run.log_scalar("loss", float(logs.get("loss")), epoch)
    _run.log_scalar("accuracy", float(logs.get("accuracy")), epoch)
    _run.log_scalar("val_loss", float(logs.get("val_loss")), epoch)
    _run.log_scalar("val_accuracy", float(logs.get("val_accuracy")), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212


@ex.config
def confnet_config():
    """Default config"""
    sigma = SIGMAS[0]  # pylint: disable=W0612
    num_qumodes = 4  # pylint: disable=W0612
    network_type = "classical"  # pylint: disable=W0612
    #network_type = "quantum"
    PWB = 0

class RandomizeNoise(Callback):
    def on_epoch_begin(self,epoch):
        noise = tf.random.normal(x_train.shape, stddev=sigma)
        input_data = [noise + x_train.astype(np.float32),y_train.astype(np.float32)]  # pylint: disable=E1120
class LogPerformance(Callback):
    """Logs performance"""

    def on_epoch_end(self, epoch, logs=None):
        """Log key metrics on epoch end"""
        log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120 

@ex.automain
def define_and_train(sigma, num_qumodes, network_type, PWB):
    """Build and run the network"""

    tf.random.set_seed(RANDOM_SEED)
    precision = int(2**16-1)

    class Net(Model):  # pylint: disable=W0223
        """Neural network model to train on"""

        def __init__(self):
            super().__init__()
            self.gaussian = layers.GaussianNoise(sigma)
            if PWB:
                self.base_model = models.Sequential(
                        [
                            PWBLinearLayer(40,input_dim=9,activation='relu', precision=precision)
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
                self.classical2 = PWBLinearLayer(
                            2,
                            activation="softmax",
                            precision=precision
                        )


            else:
                self.base_model = models.Sequential(
                    [
                        layers.Dense(
                            40,
                            input_dim=9,
                            activation="relu",
                            )]
                )
                    
                self.classical1 = models.Sequential([
                        layers.Flatten(),
                        layers.Dense(
                            2 * num_qumodes,
                            activation=None,
                        )
                ])
                
                classical_size = int(get_num_parameters_per_layer(num_qumodes) // (1 + 2*num_qumodes) + 1)
                if network_type == "classical":
                    self.quantum_substitue = models.Sequential(
                        [
                            layers.Dense(
                                classical_size,
                                activation="relu"
                            ),
                        ]
                    )
                self.classical2 = layers.Dense(
                            2,
                            activation="softmax",
                        )

            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=num_qumodes,
                n_circuits=1,
                n_layers=1,
                cutoff_dim=5,
                encoding_method="Amplitude_Phase",
                max_initial_weight=None,
                regularizer=regularizers.L1(l1=0.01),
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("TanH", self.quantum_layer.encoding_object)

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.gaussian(inputs)
            output = self.base_model(output)
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

    x_train, x_test, y_train, y_test = prepare_dataset()
    model = Net()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    model.fit(
        x_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=[x_test, y_test],
        callbacks=[LogPerformance()],
    )
