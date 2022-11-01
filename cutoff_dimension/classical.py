"""Hybrid Network Models 2022"""
from keras import Model, layers, models
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from data import generate_synthetic_dataset

EXPERIMENT_NAME = "cutoff_classical"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(f"Experiment_{EXPERIMENT_NAME}"))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, logs, epoch):
    """Logs performance with sacred framework"""
    _run.log_scalar("loss", float(logs.get("loss")), epoch)
    _run.log_scalar("accuracy", float(logs.get("accuracy")), epoch)
    _run.log_scalar("val_loss", float(logs.get("val_loss")), epoch)
    _run.log_scalar("val_accuracy", float(logs.get("val_accuracy")), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)


class LogPerformance(Callback):
    """Logs performance"""

    def on_epoch_end(self, epoch, logs=None):
        """Log key metrics on epoch end"""
        log_performance(
            logs=logs,
            epoch=epoch,
        )


@ex.automain
def define_and_train():
    """Build and run the network"""

    class Net(Model):
        """Neural network model to train on"""

        def __init__(self):
            super(Net, self).__init__()

            self.base_model = models.Sequential(
                [
                    layers.Dense(
                        10,
                        input_dim=10,
                        activation="relu",
                    ),
                    layers.Dense(
                        6,
                        activation="relu",
                    ),
                    layers.Dense(
                        33,
                        activation="relu",
                    ),
                    layers.Dense(
                        3,
                        activation="softmax",
                    ),
                ]
            )

        def call(self, inputs, training=None, mask=None):
            """Call the network"""
            output = self.base_model(inputs)
            return output

    train_data, validate_data, _ = generate_synthetic_dataset()
    model = Net()
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        *train_data,
        epochs=50,
        batch_size=36,
        validation_data=validate_data,
        callbacks=[LogPerformance()],
    )
