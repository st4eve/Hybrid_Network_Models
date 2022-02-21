#%% Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os.path
import torch.utils.data as data_utils
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from pytorch_train import *
from pytorch_quantum_layers import *

#%% Parameters
batch_size_train = 500
batch_size_test = 100
num_epochs = 2
learning_rate = 0.01

#%% Load Data
training_data = torchvision.datasets.MNIST("./MNIST_Experiment/root", train=True, download=True, transform=ToTensor())
testing_data = torchvision.datasets.MNIST("./MNIST_Experiment/root", train=False, download=True, transform=ToTensor())

indices = torch.arange(1000)
training_data_subset = data_utils.Subset(training_data, indices)

train_dataloader = DataLoader(training_data_subset, batch_size=batch_size_train, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size_test, shuffle=True)

print(training_data_subset[0])
#%% Main Model
for i in range(2, 20, 2):
    # Network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.main = nn.Sequential(
                nn.Linear(in_features=784, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=i),
                nn.ReLU(),
                QuantumLayer(i, "real_amplitudes", n_qubits=2, n_blocks=1),
                nn.Linear(in_features=i, out_features=10),
                nn.LogSoftmax(dim=1)
            )

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.main(x)
            return x

    # File to save network to
    fileName = "./MNIST_Experiment/qc_layer_size_" + str(i) + ".pt"

    # Train Network
    network = Net()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train(num_epochs, network, optimizer, loss_function, train_dataloader, fileName, continue_training=False)

    # Test Network
    test(network, loss_function, test_dataloader, fileName)
