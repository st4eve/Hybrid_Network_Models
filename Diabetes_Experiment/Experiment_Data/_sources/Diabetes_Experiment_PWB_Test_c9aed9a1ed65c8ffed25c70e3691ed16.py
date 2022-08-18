# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

# NumPy
import numpy as np

# Custom Quantum Layers
from CV_quantum_layers import *

#PWB Layer
from PWBLayer_TF import PWBLinearLayer

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


# Other
import time
import itertools
import os
import sys


# Load pre-processed dataset
from Diabetes_dataset import data_ingredient, load_data

#%% Setup Experiment
ex = Experiment('Diabetes_Test', ingredients=[data_ingredient])
ex.observers.append(FileStorageObserver('Experiment_Data'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

#%% Experiment Parameters
@ex.config
def confnet_config():
    encoding_strategy = "None"
    cutoff_dimension = 5
    num_layers = 1
    num_pre_classical = 2

#%% Logs
@ex.capture
def log_performance(_run, logs, epoch, time):
    _run.log_scalar("loss", float(logs.get('loss')), epoch)
    _run.log_scalar("accuracy", float(logs.get('accuracy')), epoch)
    _run.log_scalar("val_loss", float(logs.get('val_loss')), epoch)
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    _run.log_scalar("time", float(time), epoch)

    _run.log_scalar("avg_norm", float(logs.get('avg_norm')), epoch)
    for i in range(100):
        name = "norm_" + str(i)
        _run.log_scalar(name, float(logs.get(name)), epoch)


#%% Main
@ex.automain
def define_and_test(encoding_strategy, cutoff_dimension, num_layers, num_pre_classical):

    # Fixed parameters
    batch_size = 20
    learning_rate = 0.01
    loss_coefficient = 0.01
    cutoff_management = "L2"
    num_epochs = 50
    seed = 12

    # Metric Logging Callback
    class LogPerformance(Callback):
        def __init__(self):
            super(LogPerformance, self).__init__()

        def on_epoch_begin(self, epoch, logs={}):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs={}):
            stop_time = time.time()
            duration = stop_time - self.start_time
            log_performance(logs=logs, epoch=epoch, time=duration)

    # Load dataset
    x_train, y_train, x_test, y_test = load_data()

    # Finding input encoding value:
    def find_max_displacement(cutoff_dim, min_normalization):
        cutoff_dim = int(cutoff_dim)
        dev = qml.device("strawberryfields.tf", wires=1, cutoff_dim=cutoff_dim)

        @qml.qnode(dev, interface="tf")
        def qc(a):
            qml.Displacement(a, 0, wires=0)
            return qml.probs(wires=0)

        a = 0
        norm = 1
        while (norm > min_normalization):
            fock_dist = qc(a)
            norm = np.sum(fock_dist)
            a += 0.02

        return a

    max_input_val = find_max_displacement(cutoff_dimension, 0.999)
    
    PWB_ON = False

    # Define model
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()


            pre_quantum_network = []
            for i in range(num_pre_classical):
                if PWB_ON:  pre_quantum_network.append(PWBLinearLayer(8, activation='relu'))
                else: pre_quantum_network.append(layers.Dense(8,activation="relu",
                                bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                                kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0)))

            if PWB_ON: pre_quantum_network.append(PWBLinearLayer(8))
            else: pre_quantum_network.append(layers.Dense(8,
                                bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                                kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0)))

            self.sequential_1 = tf.keras.Sequential(pre_quantum_network)

            self.quantum_layer = QuantumLayer_MultiQunode(8, 4, 2, 2, num_layers, cutoff_dimension, AmplitudePhaseDisplacementEncoding,
                                         cutoff_management, loss_coefficient)

            if PWB_ON:
                self.sequential_2 = tf.keras.Sequential([
                    self.quantum_layer,
                    PWBLinearLayer(4, activation="relu"),
                    PWBLinearLayer(2, activation="sigmoid")])

            else:
                self.sequential_2 = tf.keras.Sequential([
                    self.quantum_layer,
                    layers.Dense(4, activation="relu",
                                bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                                kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0)),
                    layers.Dense(2, activation="sigmoid",
                                bias_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0),
                                kernel_constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0))])

            self.batch_norm = layers.BatchNormalization()
            self.layer_norm = layers.LayerNormalization()

        def call(self, inputs):
            x = self.sequential_1(inputs)
            if (encoding_strategy == "Sigmoid_BatchNorm"):
                x = self.batch_norm(x)

            if(encoding_strategy =="Sigmoid_LayerNorm"):
                x = self.layer_norm(x)

            if(encoding_strategy=="Sigmoid" or
               encoding_strategy == "Sigmoid_BatchNorm" or
               encoding_strategy=="Sigmoid_LayerNorm"):

                x = activations.sigmoid(x)

                # Scale sigmoid outputs based on number of circuits
                # This code sucks. Need some desperate cleanup
                # Currently hardcoded to 2 circuits
                x_split = list(tf.split(x, 4, axis=1))
                x_split[0] = max_input_val * x_split[0]
                x_split[1] = 2 * np.pi * x_split[1]
                x_split[2] = max_input_val * x_split[2]
                x_split[3] = 2 * np.pi * x_split[3]
                x = tf.concat([x_split[i] for i in range(4)], axis=1)

            if(encoding_strategy=="RELU"):
                x_split = list(tf.split(x, 4, axis=1))
                x_split[0] = activations.relu(x_split[0], max_value=max_input_val)
                x_split[1] = activations.relu(x_split[1], max_value=2*np.pi)
                x_split[2] = activations.relu(x_split[2], max_value=max_input_val)
                x_split[3] = activations.relu(x_split[3], max_value=2*np.pi)
                x = tf.concat([x_split[i] for i in range(4)], axis=1)

            x = self.sequential_2(x)
            return x

    # Compile model
    model = Net()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy']
                  )

    # Test model
    if PWB_ON: print('Photonic Weight Banks Engaged')
    else: print('Tensorflow Dense Engaged')
    loss, acc, norm = model.evaluate(x_test, y_test, verbose=3)[0:3]
    print('Untrained Network')
    print('Loss: %.3f \t Accuracy: %.3f \t Norm:%.3f' % (loss, acc, norm))
    model.load_weights('../Experiment_Data7/54/weights/weight48')
    print('Pretained Network')
    loss, acc, norm = model.evaluate(x_test, y_test, verbose=3)[0:3]
    print('Loss: %.3f \t Accuracy: %.3f \t Norm:%.3f' % (loss, acc, norm))
    
    

