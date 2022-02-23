# %% Imports
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
import numpy as np

#%% CV Data Encoding
"""These functions described different options for encoding
PennyLane provides Displacement for amplitude OR phase, and Squeezing."""

def AmplitudePhaseDisplacementEncoding(features, wires):
    """Encode N data points in N/2 qumodes where pairs of features are
    encoded as amplitudes and phases of the displacement gate.
    Later, this should be modified to alter each pair to be complex valued,
    and then converted to angle and phase to put the data points on equal footing."""
    for idx, f in enumerate(features):
        if(idx%2==0):
            qml.Displacement(f, features[idx+1], wires=wires[int(idx/2)])

#%% CV Quantum Nodes
"""Builds a quantum node based on the specifications"""

def build_cv_quantum_node(n_qumodes, n_outputs, cutoff_dim, encoding_method, meas_method="X_quadrature"):
    dev = qml.device("strawberryfields.tf", wires=n_qumodes, cutoff_dim=cutoff_dim)
    @qml.qnode(dev, interface="tf")
    def cv_nn(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
        encoding_method(inputs, wires=range(n_qumodes))
        qml.templates.CVNeuralNetLayers(theta_1,
                                        phi_1,
                                        varphi_1,
                                        r,
                                        phi_r,
                                        theta_2,
                                        phi_2,
                                        varphi_2,
                                        a,
                                        phi_a,
                                        k,
                                        wires=range(n_qumodes))
        if(meas_method=="X_quadrature"):
            return [qml.expval(qml.X(wires=i)) for i in range(n_outputs)]
        elif(meas_method=="Fock"):
            return [qml.probs(wires=i) for i in range(n_outputs)]
        else:
            print("Please enter valid measurement type")

    return cv_nn

#%% Single CV Keras Layer
"""Builds a Keras Layer using a quantum node"""

def build_cv_neural_network(n_qumodes, n_outputs, n_layers, cutoff_dim, encoding_method, meas_method="X_quadrature"):

    # Make quantum node
    cv_nn = build_cv_quantum_node(n_qumodes, n_outputs, cutoff_dim, encoding_method, meas_method)

    # Define weight shapes
    L = n_layers
    M = n_qumodes
    K = int(M * (M - 1) / 2)
    weight_shapes = {"theta_1":     (L, K),
                     "phi_1":       (L, K),
                     "varphi_1":    (L, M),
                     "r":           (L, M),
                     "phi_r":       (L, M),
                     "theta_2":     (L, K),
                     "phi_2":       (L, K),
                     "varphi_2":    (L, M),
                     "a":           (L, M),
                     "phi_a":       (L, M),
                     "k":           (L, M)
                     }

    # Build Keras layer and initialize weights
    circuit = qml.qnn.KerasLayer(cv_nn, weight_shapes, output_dim=n_outputs, weight_specs={
        "a": {"initializer": tf.random_uniform_initializer(minval=-0.05, maxval=0.05)},
        "r": {"initializer": tf.random_uniform_initializer(minval=-0.05, maxval=0.05)}
    })

    return circuit

#%% Full CV Keras Layers
class QuantumLayer(keras.Model):
    """ This class builds a quantum layer that can be directly used in model building.
    Note that while it is not strictly necessary for this case, it will be expanding upon
    to build more complex layers later on. """
    def __init__(self, n_qumodes, n_outputs, n_layers, cutoff_dim, encoding_method):
        super().__init__()
        self.n_outputs = n_outputs
        self.circuit_layer = build_cv_neural_network(n_qumodes, n_outputs, n_layers, cutoff_dim, encoding_method, meas_method = "X_quadrature")
        self.normalization_qnode= build_cv_quantum_node(n_qumodes, n_outputs, cutoff_dim, encoding_method, meas_method = "Fock")

    def call(self, x):
        x = self.circuit_layer(x)
        return x

    def check_normalization(self, x):
        """Currently only accepts single inputs"""
        weights = tuple(self.trainable_weights)
        fock_dist = self.normalization_qnode(x, *weights)
        integral = np.sum(fock_dist)/self.n_outputs
        return integral

