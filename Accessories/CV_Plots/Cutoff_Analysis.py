# %% Imports
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
import numpy as np
from numpy import random

#%%
def quantum_layer(initial_weight, cutoff_dim, n_layers, n_qumodes):
    dev = qml.device("strawberryfields.tf", wires=n_qumodes, cutoff_dim=cutoff_dim)

    @qml.qnode(dev, interface="tf")
    def cv_nn(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
        qml.DisplacementEmbedding(inputs, wires=range(n_qumodes))
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
        return [qml.probs(wires=i) for i in range(n_qumodes)]

    L = n_layers
    M = n_qumodes
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

    circuit = qml.qnn.KerasLayer(cv_nn, weight_shapes, output_dim=n_qumodes, weight_specs={
        "theta_1": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "phi_1": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "varphi_1": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},

        "r": {"initializer": tf.keras.initializers.Constant(initial_weight)},

        "phi_r": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "theta_2": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "phi_2": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "varphi_2": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},

        "a": {"initializer": tf.keras.initializers.Constant(initial_weight)},

        "phi_a": {
            "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)},
        "k": {"initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)}
    })

    return circuit

#%%
def get_max_input(initial_weight, cutoff_dim, n_layers, n_qumodes, norm_threshold):
    def find_max_displacement(cutoff_dim, norm_threshold):
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

    guess = find_max_displacement(cutoff_dim, norm_threshold)
    inputs = guess*tf.ones(n_qumodes)
    print(inputs)

    count = 0
    while(count<1000):
        network = quantum_layer(initial_weight, cutoff_dim, n_layers, n_qumodes)
        fock_dist = network(inputs)
        norms = [np.sum(dist) for dist in fock_dist]
        norm = sum(norms)/n_qumodes
        if(norm<norm_threshold):
            print("Norm: ", norm, "Inputs: ", inputs, "Count: ", count)
            inputs*=0.99
            count=0
        else:
            count+=1

        if(inputs[0]<0):
            print("Need a larger cutoff dimension")
            return

    print("Norm: ", norm, "Inputs: ", inputs, "Count: ", count)


#%%
get_max_input(initial_weight=0.1, cutoff_dim=5, n_layers=1, n_qumodes=4, norm_threshold=0.99)