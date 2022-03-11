#%% Imports

# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# NumPy
import numpy as np

# Custom Quantum Layers
from CV_quantum_layers import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

# Other
import time

#%% Setup Experiment
ex = Experiment('Iris_test')
ex.observers.append(FileStorageObserver('Experiment_Data2'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

#%% Experiment Parameters
@ex.config
def confnet_config():
    initial_weight_amplitudes = 1.5
    initial_input_amplitude = 1.5
    loss_coefficient = 2
    cutoff_management = "Loss"
    cutoff_dimension = 30

#%% Logs
@ex.capture
def log_performance(_run, logs, epoch, time, weight_values, weight_names):
    _run.log_scalar("loss", float(logs.get('loss')), epoch)
    _run.log_scalar("accuracy", float(logs.get('accuracy')), epoch)
    _run.log_scalar("val_loss", float(logs.get('val_loss')), epoch)
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')), epoch)
    _run.log_scalar("normalization", float(logs.get('net_norm')), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    _run.log_scalar("time", float(time), epoch)

#%% Main
@ex.automain
def define_and_train(initial_weight_amplitudes, initial_input_amplitude, loss_coefficient, cutoff_management, cutoff_dimension):

    # Fixed parameters
    batch_size = 10
    learning_rate = 0.01
    num_epochs = 30

    # Metric Logging Callback
    class LogPerformance(Callback):
        def __init__(self):
            super(LogPerformance, self).__init__()

        def on_epoch_begin(self, epoch, logs={}):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs={}):
            stop_time = time.time()
            duration = stop_time - self.start_time
            weight_names = weights = [weight.name for weight in model.trainable_weights]
            weight_values = [weight.numpy()[0] for weight in model.trainable_weights]
            log_performance(logs=logs, epoch=epoch, time=duration, weight_values=weight_values,
                            weight_names=weight_names)

    # Load dataset
    iris = load_iris()
    x = iris['data']
    y = iris['target']

    # Eliminate species at index 1 to reduce output classes to 2.
    filter = np.where((y == 0) | (y == 2))
    x, y = x[filter], y[filter]
    y[np.where(y == 0)] = 0
    y[np.where(y == 2)] = 1

    # Angle Normalization
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    x[:, 1:4:2] = scaler.fit_transform(x[:, 1:4:2])

    # Amplitude Normalization
    scaler = MinMaxScaler(feature_range=(0, initial_input_amplitude))
    x[:, 0:4:2] = scaler.fit_transform(x[:, 0:4:2])

    # Split training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)

    # Convert to tensors
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    # Build Model
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()
            self.main = QuantumLayer(n_qumodes=2,
                                     n_outputs=2,
                                     n_layers=1,
                                     cutoff_dim=cutoff_dimension,
                                     encoding_method=AmplitudePhaseDisplacementEncoding,
                                     cutoff_management=cutoff_management,
                                     cutoff_management_coefficient=loss_coefficient,
                                     input_amplitude=initial_weight_amplitudes,
                                     )

        def call(self, inputs):
            x = self.main(inputs)
            return x

    # Compile Model
    model = Net()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy']
                  )

    # Fit Model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              initial_epoch=0,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[LogPerformance()]
              )


