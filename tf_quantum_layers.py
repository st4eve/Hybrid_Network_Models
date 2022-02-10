# %% Imports
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import pennylane as qml
from itertools import combinations


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

def cv_neural_net(n_qubits, n_blocks):
    dev = qml.device("strawberryfields.tf", wires=n_qubits, cutoff_dim=15)

    @qml.qnode(dev, interface="tf")
    def cv_nn(inputs, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10):
        qml.templates.DisplacementEmbedding(inputs, wires=range(n_qubits))
        qml.templates.CVNeuralNetLayers(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, wires=range(n_qubits))
        return [qml.expval(qml.X(wires=i)) for i in range(n_qubits)]

    L = n_blocks
    M = n_qubits
    K = int(M*(M-1)/2)
    weight_shapes = {"w0":(L,K), "w1":(L,K), "w2":(L,M), "w3":(L,M), "w4":(L,M), "w5":(L,K), "w6":(L,K), "w7":(L,M), "w8":(L,M), "w9":(L,M), "w10":(L,M)}
    circuit = qml.qnn.KerasLayer(cv_nn, weight_shapes, output_dim=2)
    return circuit

class QuantumLayer(keras.Model):
    def __init__(self, input_size, circuit_layer, **kwargs):
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

            self.circuit_layer = [cv_neural_net(n_qubits, n_blocks) for i in range(self.num_circuits)]

    def call(self, x):
        x_split = list(tf.split(x, self.num_circuits, axis=1))
        x = tf.concat([self.circuit_layer[i](x_split[i]) for i in range(self.num_circuits)], axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape