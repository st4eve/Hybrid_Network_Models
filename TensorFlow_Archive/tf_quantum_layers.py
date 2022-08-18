# %% Imports
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
from itertools import combinations
import numpy as np

#%% Real Amplitudes Circuit
"""Returns a real amplitudes circuit with some number of qubits and some number of blocks (i.e. depth)"""


def real_amplitudes_circuit(n_qubits, n_blocks):
    dev = qml.device("default.qubit.tf", wires=n_qubits)

    @qml.qnode(dev, interface="tf")
    def real_amplitudes(inputs, weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)  # Angle encoding
        for i in range(n_blocks):
            for combo in combinations(list(range(0, n_qubits)), 2):
                qml.CNOT(wires=list(combo))
            for j in range(n_qubits):
                qml.RY(weights[i * n_qubits + j], wires=j)

        measured_qubits = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return measured_qubits

    weight_shapes = {"weights": n_blocks * n_qubits}
    circuit = qml.qnn.KerasLayer(real_amplitudes, weight_shapes, output_dim=n_qubits)
    return circuit

#%% CV Neural Net
def CustomDisplacementEmbedding(features, wires):
    for idx, f in enumerate(features):
        if(idx%2==0):
            qml.Displacement(f, features[idx+1], wires=wires[int(idx/2)])
            #print("amp: ", f, "phase: ", features[idx+1], "idx: ", int(idx/2))

def cv_neural_net(n_qubits, n_blocks, output_size, cutoff_dim):
    dev = qml.device("strawberryfields.tf", wires=n_qubits, cutoff_dim=cutoff_dim)

    @qml.qnode(dev, interface="tf")
    def cv_nn(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
        CustomDisplacementEmbedding(inputs, wires=range(n_qubits))
        qml.templates.CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=range(n_qubits))

        return [qml.expval(qml.X(wires=i)) for i in range(output_size)]

    L = n_blocks
    M = n_qubits
    K = int(M*(M-1)/2)
    weight_shapes = {"theta_1":(L,K), "phi_1":(L,K), "varphi_1":(L,M), "r":(L,M), "phi_r":(L,M), "theta_2":(L,K), "phi_2":(L,K), "varphi_2":(L,M), "a":(L,M), "phi_a":(L,M), "k":(L,M)}
    circuit = qml.qnn.KerasLayer(cv_nn, weight_shapes, output_dim=2, weight_specs={
        "a": {"initializer": tf.random_uniform_initializer(minval=-0.05, maxval=0.05)},
        "r": {"initializer": tf.random_uniform_initializer(minval=-0.05, maxval=0.05)}
    })

    return circuit

def normalization_check(cutoff_dim, inputs, weights):
    @qml.qnode(qml.device("strawberryfields.tf", wires=2, cutoff_dim=cutoff_dim), interface="tf")
    def cv_nn_normalization_test(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
        CustomDisplacementEmbedding(inputs, wires=range(2))
        qml.templates.CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=range(2))
        return [qml.probs(wires=i) for i in range(2)]
    return cv_nn_normalization_test(inputs, *weights)

#%% Main Quantum Layer
class QuantumLayer(keras.Model):
    def __init__(self, input_size, output_size, circuit_layer, **kwargs):
        super().__init__()

        # Get circuit params
        if (circuit_layer == "real_amplitudes"):
            n_qubits = kwargs.get("n_qubits")
            n_blocks = kwargs.get("n_blocks")

            self.input_size = input_size
            self.output_size = self.input_size

            self.num_wires = n_qubits
            if (input_size % self.num_wires != 0):
                raise ValueError("Please ensure the input size is a multiple of the number of wires")
            self.num_circuits = int(input_size / self.num_wires)

            self.circuit_layer = [real_amplitudes_circuit(n_qubits, n_blocks) for i in range(self.num_circuits)]
        if(circuit_layer == "cv_neural_net"):
            n_qubits = kwargs.get("n_qubits")
            n_blocks = kwargs.get("n_blocks")

            self.input_size = input_size
            self.output_size = self.input_size

            self.num_wires = n_qubits
            if (input_size % self.num_wires != 0):
                raise ValueError("Please ensure the input size is a multiple of the number of wires")
            self.num_circuits = int(input_size / self.num_wires)

            self.circuit_layer = [cv_neural_net(n_qubits, n_blocks, output_size, kwargs['cutoff_dim']) for i in range(self.num_circuits)]

    def call(self, x):
        x_split = list(tf.split(x, self.num_circuits, axis=1))
        x = tf.concat([self.circuit_layer[i](x_split[i]) for i in range(self.num_circuits)], axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
