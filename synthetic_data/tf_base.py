"""Hybrid Network Models 2022"""
import tensorflow as tf
from data import generate_synthetic_dataset
from keras import Model, layers, models, regularizers
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer_MultiQunode
from common_packages.utilities import get_equivalent_classical_layer_size

RANDOM_SEED = 30
BATCH_SIZE = 36
NUM_EPOCHS = 100
OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"
EXPERIMENT_NAME = "Synthetic_Hybrid_Base_Experiment_100Epochs"
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
    model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212


class LogPerformance(Callback):
    """Logs performance"""

    def on_epoch_end(self, epoch, logs=None):
        """Log key metrics on epoch end"""
        log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120


@ex.config
def confnet_config():
    """Default config"""
    network_type = "classical_tf"  # pylint: disable=W0612
    num_qumodes = 3  # pylint: disable=W0612


@ex.automain
def define_and_train(network_type, num_qumodes):
    """Build and run the network"""

    tf.random.set_seed(RANDOM_SEED)

    class Net(Model):  # pylint: disable=W0223
        """Neural network model to train on"""

        def __init__(self):
            super().__init__()

            self.base_model = models.Sequential(
                [
                    layers.Dense(
                        20,
                        input_dim=15,
                        activation="relu"
                    ),
                    layers.Dense(
                        20,
                        input_dim=20,
                        activation="relu"
                    ),
                    layers.Dense(
                        2 * num_qumodes,
                        activation=None
                    ),
                ]
            )

            if network_type == "classical_tf":
                self.quantum_substitue = models.Sequential(
                    [
                        layers.Dense(
                            get_equivalent_classical_layer_size(num_qumodes, 2 * num_qumodes, 3),
                            activation="relu"
                        ),
                    ]
                )
            self.quantum_layer = QuantumLayer_MultiQunode(
                n_qumodes=num_qumodes,
                n_circuits=1,
                n_layers=1,
                cutoff_dim=5,
                encoding_method="Amplitude_Phase",
                regularizer=regularizers.L1(l1=0.1),
                max_initial_weight=0.2,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("Sigmoid", self.quantum_layer.encoding_object)

            self.final_layer = models.Sequential(
                [
                    layers.Dense(
                        3,
                        activation="softmax"
                    ),
                ]
            )

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.base_model(inputs)
            if network_type == "quantum_tf":
                output = self.quantum_preparation_layer(output)
                output = self.quantum_layer(output)
            elif network_type == "classical_tf":
                output = self.quantum_substitue(output)
            else:
                raise ValueError("Invalid network type specified.")
            output = self.final_layer(output)
            return output

    train_data, test_data = generate_synthetic_dataset()
    model = Net()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    model.fit(
        *train_data,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=test_data,
        callbacks=[LogPerformance()],
    )
