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

# Custom PWB TensorFlow layer
from PWBLayer_TF import PWBLinearLayer

# json and os python package
import json
import os

# NumPy
import numpy as np

# Custom
from CV_quantum_layers import *
from WINE_Dataset import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

#%% Setup Experiment
ex_name = 'WINE'
ex = Experiment(ex_name)
ex.observers.append(FileStorageObserver('Experiment_Test_%s'%ex_name))
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
    num_classical = 1
    classical_size = 24
    regularizer_string = "L1=0.01"
    max_initial_weight = 0.1
    norm_threshold = 0.99
    n_classes = 11
    precision = 127
    shots = None
    max_epoch = 1
    exp_train = 1
    ff_activation = None
    noisy = False

#%% Logs
@ex.capture
def log_performance(_run, val_accuracy, val_loss):
    _run.log_scalar('val_accuracy', float(val_accuracy))
    _run.log_scalar('val_loss', float(val_loss))
    
    
def findMaxAcc():
    def getAccuracy(filedir):
        filename = filedir + 'metrics.json'
        with open(filename) as json_file:
            data = json.load(json_file)
    
        acc = data['val_accuracy']['values']
        return acc
    
    def getConfig(filedir):
        filename = filedir + 'config.json'
        with open(filename) as json_file:
            return json.load(json_file)
        
    def findMax(arr):
        return np.max(arr), np.argmax(arr)
        
    exp = 1
    max_val = 0
    epoch = 0
    
    dir_name = 'Experiment_Data_%s'%ex_name
    
    dirs = os.listdir(dir_name)[1::]
    for directory in dirs:
        filedir = dir_name + '/' + directory + '/'
        acc = getAccuracy(filedir)
        curr_max, curr_epoch = findMax(acc)
        if curr_max > max_val:
            max_val = curr_max
            epoch = curr_epoch
            exp = int(directory)
    print(dirs)
    return exp, max_val, epoch

    
    
    
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
def define_and_test(encoding_method, 
                    cutoff_dimension, 
                    num_layers, activation, 
                    n_qumodes, n_circuits, 
                    regularizer_string, 
                    max_initial_weight, 
                    norm_threshold,
                    n_classes,
                    num_classical,
                    classical_size,
                    precision, 
                    shots, 
                    max_epoch,
                    exp_train,
                    ff_activation,
                    noisy):

    # Create neural network class using the parameters
    class Net(tf.keras.Model):
        def __init__(self, shots=shots, precision=precision):
            super(Net, self).__init__()

            # Base model for transfer learning
            self.base_model = keras.models.Sequential([PWBLinearLayer(classical_size, activation=ff_activation)
                                                        for i in range(num_classical)])

            # Quantum Layer
            regularizer = get_regularizer(regularizer_string)
            self.quantum_layer = QuantumLayer_MultiQunode(n_qumodes=n_qumodes,
                                                      n_circuits=n_circuits,
                                                      n_layers=num_layers,
                                                      cutoff_dim=cutoff_dimension,
                                                      encoding_method=encoding_method,
                                                      regularizer=regularizer,
                                                      max_initial_weight=max_initial_weight,
                                                      measurement_object=CV_Measurement("X_quadrature"),
                                                      trace_tracking=True,
                                                      shots=shots)

            # Quantum preparation layer with custom activation (classical)
            # Use the encoding conversion factor to get the number of inputs right
            # Example: 4 qumodes
            # Phase or amplitude encoding: conversion=1 -> 4 classical outputs to feed into quantum circuit
            # Phase+amplitude encoding: conversion=2 -> 8 classical outputs to feed into quantum circuit
            self.classical1 = models.Sequential([
                layers.Flatten(),
                PWBLinearLayer(n_qumodes*self.quantum_layer.encoding_object.conversion, activation=None, precision=precision)])
            self.activation = Activation_Layer(activation, self.quantum_layer.encoding_object)

            # Post quantum layer (classical)
            self.classical2 = PWBLinearLayer(n_classes, activation='softmax', precision=precision)

        def call(self, inputs):
            x = self.base_model(inputs)
            x = self.classical1(x)
            x = self.activation(x)
            x = self.quantum_layer(x)
            x = self.classical2(x)
            return x

    # Get dataset
    if noisy:
        x_train, x_test, y_train, y_test = load_dataset_noisy()
    else:
        x_train, x_test, y_train, y_test = prepare_dataset()

    # Build and train model
    model = Net()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('./Experiment_Data_%s/%d/weights/weight%d.ckpt'%(ex_name, exp_train, max_epoch))
    if shots != None:
        n_tests = 100
        for i in range(n_tests):
            val_loss, val_acc = model.evaluate(x_train, y_train, verbose=3)
            log_performance(val_accuracy=val_acc,
                        val_loss=val_loss)

    else:
        val_loss, val_acc = model.evaluate(x_train, y_train, verbose=3)
        log_performance(val_accuracy=val_acc,
                        val_loss=val_loss)
    
    
    
    
    
    
    
    
    
    
