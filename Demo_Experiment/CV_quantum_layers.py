# %% Imports
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
import numpy as np
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

# %% CV Data Encoding
"""This class wraps around the encoding methods to make them simpler to access"""
class CV_Encoding():
    def __init__(self, mode, phase_amplitude=0):
        if(mode=="Amplitude"):
            self.conversion = 1
        elif(mode=="Phase"):
            self.conversion = 1
        elif(mode=="Amplitude_Phase"):
            self.conversion = 2
        else:
            raise ValueError('Please specify a valid encoding type. Valid type are Amplitude, Phase, and Amplitude_Phase.')
        self.mode = mode
        self.phase_amplitude = phase_amplitude

    def get_encoding(self, features, wires):
        if(self.mode=="Amplitude"):
            qml.DisplacementEmbedding(features, wires, method='amplitude', c=0)
        if(self.mode=="Phase"):
            qml.DisplacementEmbedding(features, wires, method='phase', c=self.phase_amplitude)
        if(self.mode=="Amplitude_Phase"):
            for i in range(int(len(features) / 2)):
                qml.Displacement(features[i], features[i + int(len(features) / 2)], wires=wires[i])


# %% CV Measurement
"""This class wraps around the measurement methods to make them simpler to access"""
class CV_Measurement():
    def __init__(self, mode):
        if(not(mode=="X_quadrature" or mode=="Fock")):
            raise ValueError('Please specify a valid measurement type. Valid types are X_quadrature and Fock.')
        self.mode = mode

    def get_measurement(self, n_qumodes):
        if(self.mode=="X_quadrature"):
            return [qml.expval(qml.X(wires=i)) for i in range(n_qumodes)]
        if(self.mode=="Fock"):
            return [qml.probs(wires=i) for i in range(n_qumodes)]

# %% Quantum Activation Layer
class Activation_Layer():
    """This class builds the activation layers that must be placed before the quantum layer"""
    def __init__(self, activation_type, encoding_object):
        self.activation_type = activation_type
        self.encoding_object = encoding_object
        self.encoding_object.phase_amplitude

    def __call__(self, x):
        # Normalize inputs to [-0.5, 0.5] range, regarldess of the activation type
        if (self.activation_type == "ReLU"):
            x = activations.relu
            x -= 0.5
        if (self.activation_type == "Sigmoid"):
            x = activations.sigmoid(x)
            x -= 0.5
        if (self.activation_type == "TanH"):
            x = activations.tanh(x)
            x /= 2

        if self.encoding_object.mode == "Amplitude":
            x *= 2*self.encoding_object.phase_amplitude
        if self.encoding_object.mode == "Phase":
            x *= 2*np.pi
        if self.encoding_object.mode == "Amplitude_Phase":
            x_split = list(tf.split(x, 2, axis=1))
            x_split[0] += 0.5 #Move range to [0, 1]
            x_split[0] *= self.encoding_object.phase_amplitude
            x_split[1] *= 2*np.pi
            x = tf.concat([x_split[i] for i in range(2)], axis=1)

        return x

# %% CV Quantum Nodes
def build_cv_quantum_node(n_qumodes, cutoff_dim, encoding_object, measurement_object):
    """
    Create CV quantum node
    :param n_qumodes: Number of qumodes in the circuit
    :param cutoff_dim: Cutoff dimension to simulate
    :param encoding_object: Encoding object
    :param measurement_object: Measurement object
    :return: CV qnode
    """
    dev = qml.device("strawberryfields.tf", wires=n_qumodes, cutoff_dim=cutoff_dim)

    @qml.qnode(dev, interface="tf")
    def cv_nn(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
        encoding_object.get_encoding(inputs, wires=range(n_qumodes))
        qml.templates.CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=range(n_qumodes))
        return measurement_object.get_measurement(n_qumodes)

    return cv_nn

# %% Full CV Keras Layers
class QuantumLayer_MultiQunode(keras.Model):

    def __init__(self, n_qumodes, n_circuits, n_layers, cutoff_dim, encoding_object = CV_Encoding("Amplitude_Phase"), regularizer = None, max_initial_weight = 0.1, measurement_object=CV_Measurement("X_quadrature")):
        """
        Initialize Keras NN layer. Example:
        8 inputs coming in from previous layer
        4 qumodes -> n_qumodes = 4
        1 circuit -> n_circuits = 1
        Amplitude & Phase encoding -> encoding_object = CV_Encoding("Amplitude_Phase")
        This results in a single circuit with 4 qumodes because the amplitude+phase encoding
        divides the inputs from the previous layer by 2

        :param n_qumodes: Total number of qumodes
        :param n_circuits: Number of circuits in the layer
        :param n_layers: Number of layers within the CV circuits
        :param cutoff_dim: Cutoff dimension for the layer simulation
        :param encoding_object: Encoding method
        :param regularizer: Regularizer object
        :param max_initial_weight: Maximum value allowed for non-phase parameters
        :param measurement_method: Measurement method
        """
        super().__init__()
        self.n_circuits = n_circuits

        # Calculate number of qumodes based on the down-scaling from encoding and number of circuits
        n_qumodes_per_circuit = n_qumodes/n_circuits
        if(n_qumodes_per_circuit.is_integer()):
            n_qumodes_per_circuit = int(n_qumodes_per_circuit)
        else:
            raise ValueError('Please ensure the number of inputs divides evenly into the encoding method & number of circuits')

        # Create the specified number of circuits
        self.circuit_layer = []
        for i in range(n_circuits):

            # Make quantum node
            cv_nn = build_cv_quantum_node(n_qumodes_per_circuit, cutoff_dim, encoding_object, measurement_object)

            # Define weight shapes
            weight_shapes = self.define_weight_shapes(L = n_layers, M = n_qumodes_per_circuit)

            # Define weight specifications
            weight_specs = self.define_weight_specs(max_initial_weight = max_initial_weight, regularizer = regularizer)

            # Build circuit
            circuit = qml.qnn.KerasLayer(cv_nn, weight_shapes, output_dim=n_qumodes_per_circuit, weight_specs=weight_specs)

            self.circuit_layer.append(circuit)

    def define_weight_specs(self, max_initial_weight, regularizer):
        """
        Define the initial weights and regularizers on each parameter
        :param max_initial_weight: maximum allowable value for random initializaiton for non-phase parameters
        :param regularizer: L1, L2 or None regularizer
        :return: dictionary of parameters with their initializers and regularizers
        """
        weight_specs = {
            "theta_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "phi_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "varphi_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},

            "r": {"initializer": tf.random_uniform_initializer(minval=0, maxval=max_initial_weight),
                  "regularizer": regularizer},

            "phi_r": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "theta_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "phi_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "varphi_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},

            "a": {"initializer": tf.random_uniform_initializer(minval=0, maxval=max_initial_weight),
                  "regularizer": regularizer},

            "phi_a": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
            "k": {"initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)}
        }
        return weight_specs

    def define_weight_shapes(self, L, M):
        """
        Define the weight shapes for the circuit
        :param L: Number of layers
        :param M: Number of qumodes
        :return: dictionary of parameters with sets of dimensions
        """
        K = int(M * (M - 1) / 2)
        weight_shapes = {"theta_1": (L, K),
                         "phi_1": (L, K),
                         "varphi_1": (L, M),
                         "r": (L, M),
                         "phi_r": (L, M),
                         "theta_2": (L, K),
                         "phi_2": (L, K),
                         "varphi_2": (L, M),
                         "a": (L, M),
                         "phi_a": (L, M),
                         "k": (L, M)
                         }
        return weight_shapes


    def call(self, x):
        """
        Call the neural network
        :param x: Input tensor
        :return: Output tensor
        """
        x_split = list(tf.split(x, self.n_circuits, axis=1))
        output = tf.concat([self.circuit_layer[i](x_split[i]) for i in range(self.n_circuits)], axis=1)
        return output

#%% Accessory Algorithms
# TO-DO: refactor this section and add clarification
def find_max_displacement(cutoff_dim, norm_threshold):
    """Increase the displacement until de-normlized"""
    cutoff_dim = int(cutoff_dim)
    dev = qml.device("strawberryfields.tf", wires=1, cutoff_dim=cutoff_dim)

    @qml.qnode(dev, interface="tf")
    def qc(a):
        qml.Displacement(a, 0, wires=0)
        qml.Kerr(np.pi / 4, wires=0)
        return qml.probs(wires=0)

    a = 0
    norm = 1
    while (norm > norm_threshold):
        a += 0.01
        fock_dist = qc(a)
        norm = np.sum(fock_dist)

    return a

def get_max_input(initial_weight, cutoff_dim, n_layers, n_qumodes, norm_threshold):

    # Use this function to give us an initial guess
    guess = find_max_displacement(cutoff_dim, norm_threshold)
    t1 = guess*tf.ones(n_qumodes)

    # We create a network initialized with random values over and over again
    # and test to see how many counts we get above our normalized threshold
    # Once the counts are above some number, the max input can be considered
    # to be a "safe" value
    count = 0
    while(count<20):
        t2 = tf.random.uniform([n_qumodes], minval=0, maxval=2 * np.pi)
        inputs = tf.concat([t1, t2], 0)

        keras_network = QuantumLayer_MultiQunode(n_qumodes=n_qumodes,
                                                 n_circuits=1,
                                                 n_layers=n_layers,
                                                 cutoff_dim=cutoff_dim,
                                                 encoding_object=CV_Encoding("Amplitude_Phase"),
                                                 regularizer=None,
                                                 max_initial_weight = initial_weight,
                                                 measurement_object=CV_Measurement("Fock"))
        network = keras_network.circuit_layer[0]

        fock_dist = network(inputs)
        norms = [np.sum(dist) for dist in fock_dist]
        norm = sum(norms)/n_qumodes
        if(norm<norm_threshold):
            print("Norm: ", norm, "Inputs: ", inputs[0], "Count: ", count)
            t1*=0.99
            count=0
        else:
            count+=1

        if(inputs[0]<0):
            print("Need a larger cutoff dimension")
            return

    print("max input: ", t1[0])
    return t1[0]