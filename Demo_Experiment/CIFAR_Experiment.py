#%% Imports
# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

# NumPy
import numpy as np

# Custom
from CV_quantum_layers import *
from CIFAR_Dataset import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

#%% Setup Experiment
ex_name = 'CIFAR'
ex = Experiment(ex_name)
ex.observers.append(FileStorageObserver('Experiment_Data_%s'%ex_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds

#%% Experiment Parameters
@ex.config
def confnet_config():
    encoding_method = "Amplitude_Phase"
    cutoff_dimension = 5
    num_layers = 1
    activation="Sigmoid"
    n_qumodes = 4
    n_circuits = 1
    regularizer_string = "L1=0.01"
    max_initial_weight = 0.1
    norm_threshold = 0.99

#%% Logs
@ex.capture
def log_performance(_run, logs, epoch, traces, model):
    _run.log_scalar("loss", float(logs.get('loss')), epoch)
    _run.log_scalar("accuracy", float(logs.get('accuracy')), epoch)
    _run.log_scalar("val_loss", float(logs.get('val_loss')), epoch)
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    _run.log_scalar("traces", traces, epoch)
    
    model.save_weights('Experiment_Data_%s/%s/weights/weight%d.ckpt' % (_run.experiment_info["name"], _run._id, epoch))

#%% Metric Logging Callback Class
class LogPerformance(Callback):
    def __init__(self):
        super(LogPerformance, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        log_performance(logs=logs, epoch=epoch, traces=self.model.quantum_layer.traces, model=self.model)

        # Reset the traces for the next epoch
        self.model.quantum_layer.traces = []

#%% Get regularizer
def get_regularizer(regularizer_string):
    type = regularizer_string.split('=')[0]
    value = float(regularizer_string.split('=')[1])
    if(type=="L1"):
        return tf.keras.regularizers.L1(l1=value)
    if(type=="L2"):
        return tf.keras.regularizers.L2(l2=value)
    else:
        return None

#%% Main
@ex.automain
def define_and_train(encoding_method, cutoff_dimension, num_layers, activation, n_qumodes, n_circuits, regularizer_string, max_initial_weight, norm_threshold):

    # Create neural network class using the parameters
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()

            # Base model for transfer learning
            self.base_model = VGG16(weights="imagenet", include_top=False, input_shape=x_train[0].shape)
            self.base_model.trainable = False

            # Quantum Layer
            regularizer = get_regularizer(regularizer_string)
            self.quantum_layer = QuantumLayer_MultiQunode( n_qumodes=n_qumodes,
                                                      n_circuits=n_circuits,
                                                      n_layers=num_layers,
                                                      cutoff_dim=cutoff_dimension,
                                                      encoding_method=encoding_method,
                                                      regularizer=regularizer,
                                                      max_initial_weight=max_initial_weight,
                                                      measurement_object=CV_Measurement("X_quadrature"),
                                                      trace_tracking=True)

            # Quantum preparation layer with custom activation (classical)
            # Use the encoding conversion factor to get the number of inputs right
            # Example: 4 qumodes
            # Phase or amplitude encoding: conversion=1 -> 4 classical outputs to feed into quantum circuit
            # Phase+amplitude encoding: conversion=2 -> 8 classical outputs to feed into quantum circuit
            self.classical1 = models.Sequential([
                layers.Flatten(),
                layers.Dense(n_qumodes*self.quantum_layer.encoding_object.conversion, activation=None,
                                bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                                kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0))])
            self.activation = Activation_Layer(activation, self.quantum_layer.encoding_object)

            # Post quantum layer (classical)
            self.classical2 = layers.Dense(n_qumodes, activation='softmax')

        def call(self, inputs):
            x = self.base_model(inputs)
            x = self.classical1(x)
            x = self.activation(x)
            x = self.quantum_layer(x)
            x = self.classical2(x)
            return x

    # Get dataset
    x_train, x_test, y_train, y_test = prepare_dataset()

    # Build and train model
    model = Net()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test),callbacks=[LogPerformance()])
