"""Hybrid Network Models 2022"""
import tensorflow as tf
from data import generate_synthetic_dataset
from hybrid_base import EXPERIMENT_NAME as BASE_EXPERIMENT_NAME
from hybrid_base import LOSS_FUNCTION, NUM_EPOCHS, OPTIMIZER, RANDOM_SEED
from keras import Model, layers, models, regularizers
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from common_packages.PWBLayer_TF import PWBLinearLayer

EXPERIMENT_NAME = "Synthetic_Hybrid_PWB_Experiment"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, val_accuracy, val_loss, epoch):
    """Logs performance with sacred framework"""
    _run.log_scalar("val_loss", float(val_loss))
    _run.log_scalar("val_accuracy", float(val_accuracy))
    _run.log_scalar("epoch", int(epoch), epoch)


@ex.config
def confnet_config():
    """Default config"""
    precision = 8  # pylint: disable=W0612
    exp_num = 1  # pylint: disable=W0612
    network_type = "classical"  # pylint: disable=W0612


@ex.automain
def define_and_train(precision, exp_num, network_type):
    """Build and run the network"""

    tf.random.set_seed(RANDOM_SEED)

    class Net(Model):  # pylint: disable=W0223
        """Neural network model to train on"""

        def __init__(self):
            super().__init__()

            self.base_model = models.Sequential(
                [
                    PWBLinearLayer(10, activation="relu", precision=precision),
                    PWBLinearLayer(6, activation=None, precision=precision),
                ]
            )

            self.quantum_substitue = models.Sequential(
                [
                    layers.Dense(
                        33,
                        activation="softmax",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    ),
                ]
            )

            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=3,
                n_circuits=1,
                n_layers=1,
                cutoff_dim=5,
                encoding_method="Amplitude_Phase",
                regularizer=regularizers.L1(l1=0.1),
                max_initial_weight=None,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("Sigmoid", self.quantum_layer.encoding_object)

            self.final_layer = models.Sequential([PWBLinearLayer(3, activation="softmax", precision=precision)])

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.base_model(inputs)
            if network_type == "quantum":
                output = self.quantum_preparation_layer(output)
                output = self.quantum_layer(output)
            elif network_type == "classical":
                output = self.quantum_substitue(output)
            else:
                raise ValueError("Invalid network type specified.")
            output = self.final_layer(output)
            return output

    _, _, validate_data = generate_synthetic_dataset()
    model = Net()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    exp_path = f"{BASE_EXPERIMENT_NAME}/{exp_num}"
    for i in range(NUM_EPOCHS):
        model.load_weights(f"{exp_path}/weights/weight{i}.ckpt")
        val_loss, val_acc = model.evaluate(*validate_data, verbose=3)
        log_performance(val_accuracy=val_acc, val_loss=val_loss, epoch=i)  # pylint: disable=E1120
