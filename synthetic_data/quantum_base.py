"""Hybrid Network Models 2022"""
import tensorflow as tf
from data import generate_synthetic_dataset_easy
from keras import Model, layers, models, regularizers, activations
from keras.callbacks import Callback
from keras.utils.layer_utils import count_params
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from common_packages.CV_quantum_layers import Activation_Layer, CV_Measurement, QuantumLayer
from common_packages.utilities import get_equivalent_classical_layer_size

RANDOM_SEED = 30
BATCH_SIZE = 64
NUM_EPOCHS = 200
OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"
EXPERIMENT_NAME = "Synthetic_Quantum_Base_Experiment_loss_test"
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


@ex.capture
def save_num_params_weights(_run, logs, model, epoch):
    _run.log_scalar('num_params', int(count_params(model.trainable_weights)), NUM_EPOCHS)
    model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212

class LogPerformance(Callback):
    """Logs performance"""

    def on_epoch_end(self, epoch, logs=None):
        """Log key metrics on epoch end"""
        log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120
    
    def on_train_end(self, epoch, logs=None):
        save_num_params_weights(logs=logs, model=self.model, epoch=epoch)

@ex.config
def confnet_config():
    """Default config"""
    network_type = "classical"  # pylint: disable=W0612
    num_qumodes = 4  # pylint: disable=W0612
    cutoff=5
    n_layers=5
    iteration=-1

class Net(Model):  # pylint: disable=W0223
    """Neural network model to train on"""

    def __init__(self,
                    network_type,
                    num_qumodes,
                    cutoff,
                    n_layers,
                    max_initial_weight=None,
                    ):
        super().__init__()
        self.network_type = network_type
        self.num_qumodes = num_qumodes
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.max_initial_weight = max_initial_weight
        self.input_layer = models.Sequential(
            [
                layers.Dense(
                    2*num_qumodes,
                    activation=None,
                    trainable=True,
                    bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                ),
            ]
        )

        if network_type == "classical":
            initial_layer_size = get_equivalent_classical_layer_size(num_qumodes, 2*num_qumodes, num_qumodes)
            self.quantum_substitue = [
                    layers.Dense(
                        initial_layer_size,
                        activation="relu",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )]
            if n_layers == 2:
                layer_size = get_equivalent_classical_layer_size(num_qumodes, initial_layer_size, num_qumodes)
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size,
                        activation="relu",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
            if n_layers > 2:
                layer_size1 = get_equivalent_classical_layer_size(num_qumodes, initial_layer_size, num_qumodes)
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size1,
                        activation="relu",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
                layer_size2 = get_equivalent_classical_layer_size(num_qumodes, layer_size1, layer_size1)
                self.quantum_substitue += [
                    layers.Dense(
                        layer_size2,
                        activation="relu",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    ) for i in range(n_layers-3)
                ]
                self.quantum_substitue += [
                    layers.Dense(
                        get_equivalent_classical_layer_size(num_qumodes, layer_size2, num_qumodes),
                        activation="relu",
                        bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                        kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                    )
                ]
            
            self.quantum_substitue = models.Sequential(self.quantum_substitue)
        if network_type=='quantum':
            self.quantum_layer = QuantumLayer(
                n_qumodes=num_qumodes,
                n_layers=n_layers,
                cutoff_dim=cutoff,
                encoding_method="Amplitude_Phase",
                regularizer=regularizers.L1(l1=0.1),
                max_initial_weight=max_initial_weight,
                measurement_object=CV_Measurement("X_quadrature"),
                shots=None,
            )

            self.quantum_preparation_layer = Activation_Layer("Sigmoid", self.quantum_layer.encoding_object)
        
        self.final_layer = models.Sequential(
            [
                layers.Dense(
                    4,
                    activation="softmax",
                    trainable=False,
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
        elif self.network_type == "classical":
            output = self.quantum_substitue(output)
        else:
            raise ValueError("Invalid network type specified.")
        output = self.final_layer(output)
        return output


@ex.automain
def define_and_train(network_type, num_qumodes, cutoff, n_layers):
    """Build and run the network"""

    train_data, test_data = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=8, n_classes=4)
    model = Net(network_type=network_type,
                num_qumodes=num_qumodes,
                cutoff=cutoff,
                n_layers=n_layers)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])
    model.fit(
        *train_data,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=test_data,
        callbacks=[LogPerformance()],
    )
        

