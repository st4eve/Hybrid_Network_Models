"""Hybrid Network Models 2022"""
import tensorflow as tf
from data import generate_synthetic_dataset_easy
from keras import Model, layers, models, regularizers, activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import os
import os

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from common_packages.utilities import get_equivalent_classical_layer_size

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RANDOM_SEED = 30
BATCH_SIZE = 32
NUM_EPOCHS = 200


OPTIMIZER = Adam(learning_rate=0.001)
LOSS_FUNCTION = "categorical_crossentropy"
EXPERIMENT_NAME = "Classical_Small_kerr_no_hidden"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, logs, epoch, model):
    """Logs performance with sacred framework"""
    _run.log_scalar("loss", float(logs.get("loss")), epoch)
    _run.log_scalar("accuracy", float(logs.get("accuracy")), epoch)
    _run.log_scalar("val_loss", float(logs.get("val_loss")), epoch)
    _run.log_scalar("val_accuracy", float(logs.get("val_accuracy")), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    if (epoch == 5) | (epoch == 195):
        model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212

@ex.capture
def save_num_params(_run, logs, model, epoch):
    _run.log_scalar('num_params', int(model.count_params()), NUM_EPOCHS)
#    model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212

class LogPerformance(Callback):
    """Logs performance"""

    def on_epoch_end(self, epoch, logs=None):
        """Log key metrics on epoch end"""
        log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120
    
    def on_train_end(self, epoch, logs=None):
        save_num_params(logs=logs, model=self.model, epoch=epoch)

@ex.config
def confnet_config():
    """Default config"""
    network_type = "classical"  # pylint: disable=W0612
    num_qumodes = 1  # pylint: disable=W0612
    cutoff=-1
    n_layers=0
    input_nl = None
    iteration=-1

class Net(Model):  # pylint: disable=W0223
    """Neural network model to train on"""
    def __init__(
            self,
            network_type,
            num_qumodes,
            cutoff,
            n_layers,
            input_nl,
            max_initial_weight=None,
            shots=None,):
        super().__init__(
        )

        self.network_type = network_type
        self.num_qumodes = num_qumodes
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.max_initial_weight = max_initial_weight
        if network_type=='classical':
            input_activation = input_nl
        else:
            input_activation = None
        self.input_layer = models.Sequential(
            [
                layers.Dense(
                    5*num_qumodes,
                    activation=input_activation,
                    trainable=True,
                    bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                ),
            ]
        )

        if (network_type == "classical")  and (n_layers > 0):
            initial_layer_size = get_equivalent_classical_layer_size(num_qumodes, 5*num_qumodes, num_qumodes)
            self.quantum_substitue = [
                    layers.Dense(
                        initial_layer_size,
                        activation=input_activation,
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )]
            layer_size = get_equivalent_classical_layer_size(num_qumodes, initial_layer_size, num_qumodes)
            if n_layers == 2:
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size,
                        activation=input_activation,
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
            elif n_layers > 2:
                layer_size1 = get_equivalent_classical_layer_size(num_qumodes, initial_layer_size, num_qumodes)
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size1,
                        activation=input_activation,
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
                layer_size2 = get_equivalent_classical_layer_size(num_qumodes, layer_size1, layer_size1)
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size2,
                        activation=input_activation,
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    ) for i in range(n_layers-3)
                ]
                self.quantum_substitue += [
                    layers.Dense(
                        get_equivalent_classical_layer_size(num_qumodes, layer_size2, num_qumodes),
                        activation=input_activation,
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
        
            self.quantum_substitue = models.Sequential(self.quantum_substitue)
        if network_type=='quantum':
            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=num_qumodes,
                n_circuits=1,
                n_layers=n_layers,
                cutoff_dim=cutoff,
                encoding_method="Kerr",
                regularizer=regularizers.L2(l2=0.1),
                max_initial_weight=max_initial_weight,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=shots,
            )

            self.quantum_preparation_layer = Activation_Layer("Sigmoid", self.quantum_layer.encoding_object)
        
        self.final_layer = models.Sequential(
            [
                layers.Dense(
                    4,
                    activation="softmax",
                    trainable=True,
                    bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                ),
            ]
        )
    def call(self, inputs):  # pylint: disable=W0221
        """Call the network"""
        output = self.input_layer(inputs)
        if self.network_type == "quantum":
            output = self.quantum_preparation_layer(output)
            output = self.quantum_layer(output)
        elif (self.network_type == "classical") and (self.n_layers > 0):
            output = self.quantum_substitue(output)
        elif self.network_type != "classical" and self.network_type != "quantum":
            raise ValueError("Invalid network type specified.")
        output = self.final_layer(output)
        return output

train_data, test_data = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=8, n_classes=4)

@ex.automain
def define_and_train(network_type, num_qumodes, cutoff, n_layers, input_nl):
    """Build and run the network"""
    model = Net(
        network_type=network_type,
        num_qumodes=num_qumodes,
        cutoff=cutoff,
        n_layers=n_layers,
        input_nl=input_nl
    )
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    model.fit(
        *train_data,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=test_data,
        callbacks=[LogPerformance()],
    )
        

