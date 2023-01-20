# %% Imports
import tensorflow as tf
from tensorflow import keras
import sys

# We will need the DEAP library
sys.path.append('../DEAP')
from deap_tf.mappers import PWBMapper
import pennylane as qml
import numpy as np
# Gives some numpy functionality for the DEAP library
# to function properly.
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# %% PWB LinearLayer
"""
This class implements a photonic weight bank simulated linear dense layer for Tensorflow.
Weights and biases should be limited to [-1,1] to maintain physical accuracy.
"""
class PWBLinearLayer(keras.layers.Layer):
    def __init__(self, num_outputs, precision=127, name='LinearLayer', input_dim=None, activation=None, constraint=lambda t: tf.clip_by_value(t, -1.0, 1.0)):
        super(PWBLinearLayer, self).__init__()
        self.num_outputs = num_outputs
        # Local PWB mapper class
        self.PWBMapper = PWBMapper        
        # Precision of the PWB. Typically 127 or 7 bits of precision
        self.precision = precision
        self.activation = keras.activations.get(activation)
        self.constraint = constraint
        self.num_inputs = None
        
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        if (self.num_inputs == None):
            self.num_inputs = input_shape[-1]
        self.weight = self.add_weight(
            'kernel',
            shape=[self.num_inputs,self.num_outputs],
            initializer=w_init,
            dtype=tf.float32,
            trainable=False,
            constraint=self.constraint
            )
        self.bias = self.add_weight(
            'bias',
            shape=[self.num_outputs],
            initializer=w_init,
            dtype=tf.float32,
            trainable=False,
            constraint=self.constraint
            )
        self.setPrecision(self.precision)
        self.neurons = [self.PWBMapper.build(i) for i in tf.transpose(self.weight)]
        self.bias_neuron = self.PWBMapper.build(self.bias)
    
    # Simulate PWB
    def PWB(self, inputs):
        res = tf.convert_to_tensor([n.step(inputs) for n in self.neurons])
        b_inputs = tf.eye(res.shape[-1], dtype=tf.float32)
        bias = tf.convert_to_tensor([self.bias_neuron.step(b) for b in b_inputs])
        return res + bias
    # Update precision
    def setPrecision(self, precision):
        self.precision = precision
        self.PWBMapper.setPrecision(precision)
       
    # Layer call function
    def call(self, inputs):
        # First we have to update weights in our photonic neurons
        weights = tf.transpose(self.weight)
        for i,n in enumerate(self.neurons):
            self.PWBMapper.updateWeights(n, weights[i])
        
        self.PWBMapper.updateWeights(self.bias_neuron, self.bias)
        # Simulate the PWB over the batch inputs
        if len(inputs.get_shape()) > 1:
            res = []
            for i,data in enumerate(inputs):
                r = self.PWB(data)
                res.append(r)
        else:
            res = self.PWB(inputs)
        res = tf.stack(res)

        return self.activation(res)
