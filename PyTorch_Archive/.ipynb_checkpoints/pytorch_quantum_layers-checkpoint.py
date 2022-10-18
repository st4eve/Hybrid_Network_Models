import pennylane as qml
from itertools import combinations
import torch
import torch.nn as nn

# %% Real Amplitudes Circuit
"""Returns a real amplitudes circuit with some number of qubits and some number of blocks (i.e. depth)"""


def real_amplitudes_circuit(n_qubits, n_blocks):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
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
    circuit = qml.qnn.TorchLayer(real_amplitudes, weight_shapes)
    return circuit


# %% Quantum Layer
"""This generates a quantum layer using any specified circuit and takes care of executing multiple side by side
circuits. This was created with the help of https://pennylane.ai/qml/demos/tutorial_qnn_module_torch.html

Example of how to implement this layer: 
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=10, out_features=6),
            nn.ReLU(),
            QuantumLayer(6, "real_amplitudes", n_qubits=2, n_blocks=1),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features=2),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.main(x)
        return x
"""


class QuantumLayer(nn.Module):
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

            self.circuit_layer = nn.ModuleList(
                real_amplitudes_circuit(n_qubits, n_blocks) for i in range(self.num_circuits))

    def forward(self, x):
        x_split = list(torch.split(x, self.num_wires, dim=1))
        x = torch.cat([self.circuit_layer[i](x_split[i]) for i in range(self.num_circuits)], dim=1)
        return x

# %% Tests
x = torch.Tensor([0.1, 0.3, 0.4, 0.5])
test_circuit = real_amplitudes_circuit(4, 1)
print("Circuit Output: ", test_circuit(x))
print("Circuit Weights: ", test_circuit.weights, "\n\n")

x = torch.Tensor([[0.1, 0.3, 0.4, 0.5],[0.1, 0.3, 0.4, 0.5]])
test_layer = QuantumLayer(4, "real_amplitudes", n_qubits=2, n_blocks=1)
print("Layer Output: ", test_layer(x))
print("Weights: ", test_layer.state_dict())
