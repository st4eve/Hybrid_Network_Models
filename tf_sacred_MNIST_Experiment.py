#%% Imports

# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras

# NumPy
import numpy as np

# Custom Quantum Layers
from tf_quantum_layers import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%% Setup Experiment
ex = Experiment('MNIST_TEST')
ex.observers.append(FileStorageObserver('MNIST_Experiment'))

#%% Experiment Parameters
@ex.config
def confnet_config():
    num_neurons = 20
    activation_type = 'softmax'

#%% Logs
@ex.capture
def log_performance(_run, logs):
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("accuracy", float(logs.get('accuracy')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
    _run.result = float(logs.get('val_accuracy'))

#%% Main Experiment
@ex.automain
def define_and_train(num_neurons, activation_type):

    # Metric Logging Callback
    class LogPerformance(Callback):
        def on_epoch_end(self, _, logs={}):
            log_performance(logs=logs)

    # Load Data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # Build Model
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()

            self.main = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28,28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_neurons, activation=activation_type),
                #QuantumLayer(10, "cv_neural_net", n_qubits=2, n_blocks=1),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

        def call(self, inputs):
            x = self.main(inputs)
            return x

    # Compile Model
    model = Net()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(0.01),
                  metrics=['accuracy']
                  )

    # Train Model
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=3,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[LogPerformance()]
              )

