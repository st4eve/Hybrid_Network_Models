"""Hybrid Network Models 2022"""
import tensorflow as tf
from classical_base import EXPERIMENT_NAME as BASE_EXPERIMENT_NAME
from classical_base import LOSS_FUNCTION, NUM_EPOCHS, OPTIMIZER, RANDOM_SEED
from data import generate_synthetic_dataset
from keras import Model, models
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

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
    precision = 128  # pylint: disable=W0612
    exp_num = 1  # pylint: disable=W0612


@ex.automain
def define_and_train(precision, exp_num):
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

            self.final_layer = models.Sequential([PWBLinearLayer(3, activation="softmax", precision=precision)])

        def call(self, inputs):  # pylint: disable=W0221
            """Call the network"""
            output = self.base_model(inputs)
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
