#%% Imports
from keras.utils.layer_utils import count_params
from TensorFlow_Archive.tf_quantum_layers import *
import time
import pandas as pd
import os
from math import comb
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%% Non-CV Timing Function for depth and qubits
def timing(circuit_name, file_name, max_qubits, max_blocks, min_qubits = 2, min_blocks = 1, samples=1):
    qubit_range = range(min_qubits, max_qubits+1)
    block_range = range(min_blocks, max_blocks+1)

    save_qubits = []
    save_blocks = []
    save_times = []
    save_weights = []
    save_gates = []

    for n_qubits in qubit_range:
        for n_blocks in block_range:
            print(n_qubits, " ", n_blocks)
            test_circuit = circuit_name(n_qubits, n_blocks)
            input = tf.random.uniform([n_qubits])
            start = time.time()
            for i in range(samples):
                output = test_circuit(input)
            end = time.time()
            dt = (end - start)/samples
            n_weights = count_params(test_circuit.trainable_weights)

            if(circuit_name == real_amplitudes_circuit):
                save_gates.append(3*n_qubits+n_blocks*(n_qubits+comb(n_qubits,2)))

            save_qubits.append(n_qubits)
            save_blocks.append(n_blocks)
            save_times.append(dt)
            save_weights.append(n_weights)

    data = {'n_qubits': save_qubits, 'n_blocks': save_blocks, 'time': save_times, 'n_weights': save_weights, 'n_gates': save_gates}
    dataframe = pd.DataFrame(data=data)
    file_name = file_name + ".pkl"
    dataframe.to_pickle(file_name)

#%% CV Timing Function for depth and qubits and cutoff_dim
def cv_timing(circuit_name, file_name, max_qubits, max_blocks, min_qubits = 2, min_blocks = 1, min_cutoff=2, max_cutoff=15, samples=1):
    qubit_range = range(min_qubits, max_qubits+1)
    block_range = range(min_blocks, max_blocks+1)
    cutoff_range = range(min_cutoff, max_cutoff+1, 2)

    save_qubits = []
    save_blocks = []
    save_cutoff = []
    save_times = []
    save_weights = []
    save_gates = []

    for n_qubits in qubit_range:
        for n_blocks in block_range:
            for cutoff_dim in cutoff_range:
                print(n_qubits, " ", n_blocks, " ", cutoff_dim)
                test_circuit = circuit_name(n_qubits, n_blocks, cutoff_dim)
                input = tf.random.uniform([n_qubits])
                start = time.time()
                for i in range(samples):
                    output = test_circuit(input)
                end = time.time()
                dt = (end - start)/samples
                n_weights = count_params(test_circuit.trainable_weights)

                if(circuit_name == cv_neural_net):
                    save_gates.append(2*n_qubits+n_blocks*(7*n_qubits+4*int(n_qubits*(n_qubits-1)/2)))

                save_qubits.append(n_qubits)
                save_blocks.append(n_blocks)
                save_cutoff.append(cutoff_dim)
                save_times.append(dt)
                save_weights.append(n_weights)

    data = {'n_qubits': save_qubits,
            'n_blocks': save_blocks,
            'cutoff_dim': save_cutoff,
            'time': save_times,
            'n_weights': save_weights,
            'n_gates': save_gates}

    dataframe = pd.DataFrame(data=data)
    file_name = file_name + ".pkl"
    dataframe.to_pickle(file_name)

#%% Neural Network Comparison
def nn_timing(circuit_name, file_name, max_qubits, max_blocks, min_qubits = 2, min_blocks = 1, batch_size = 20):
    qubit_range = range(min_qubits, max_qubits + 1)
    block_range = range(min_blocks, max_blocks + 1)

    save_qubits = []
    save_blocks = []
    save_times = []
    save_gates = []

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')
    x_train = x_train[:batch_size]
    y_train = y_train[:batch_size]

    for n_qubits in qubit_range:
        for n_blocks in block_range:
            print(n_qubits, " ", n_blocks)
            class Net(tf.keras.Model):
                def __init__(self):
                    super(Net, self).__init__()

                    self.main = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(28, 28)),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(n_qubits, activation='relu'),
                        QuantumLayer(n_qubits, "real_amplitudes", n_qubits=n_qubits, n_blocks=n_blocks),
                        tf.keras.layers.Dense(10, activation='softmax')
                    ])

                def call(self, inputs):
                    x = self.main(inputs)
                    return x

            model = Net()

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy']
                          )

            start = time.time()

            model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

            end = time.time()
            dt = (end - start)

            if(circuit_name == real_amplitudes_circuit):
                save_gates.append(3*n_qubits+n_blocks*(n_qubits+comb(n_qubits,2)))

            save_qubits.append(n_qubits)
            save_blocks.append(n_blocks)
            save_times.append(dt)

    data = {'n_qubits': save_qubits, 'n_blocks': save_blocks, 'time': save_times, 'n_gates': save_gates}
    dataframe = pd.DataFrame(data=data)
    file_name = file_name + ".pkl"
    dataframe.to_pickle(file_name)

#%% CV Neural Network Comparison
def nn_cv_timing(circuit_name, file_name, max_qubits, max_blocks, min_qubits = 2, min_blocks = 1, min_cutoff=2, max_cutoff=15, batch_size = 20):
    qubit_range = range(min_qubits, max_qubits + 1)
    block_range = range(min_blocks, max_blocks + 1)
    cutoff_range = range(min_cutoff, max_cutoff + 1, 2)

    save_qubits = []
    save_blocks = []
    save_times = []
    save_gates = []
    save_cutoff = []

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')
    x_train = x_train[:50]
    y_train = y_train[:50]

    for n_qubits in qubit_range:
        for n_blocks in block_range:
            for cutoff_dim in cutoff_range:
                print("new network")
                class Net(tf.keras.Model):
                    def __init__(self):
                        super(Net, self).__init__()

                        self.main = tf.keras.models.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(n_qubits, activation='relu'),
                            QuantumLayer(n_qubits, "cv_neural_net", n_qubits=n_qubits, n_blocks=n_blocks, cutoff_dim=cutoff_dim),
                            tf.keras.layers.Dense(10, activation='softmax')
                        ])

                    def call(self, inputs):
                        x = self.main(inputs)
                        return x

                model = Net()

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy']
                              )

                start = time.time()

                model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

                end = time.time()
                dt = (end - start)

                if(circuit_name == cv_neural_net):
                    save_gates.append(2*n_qubits+n_blocks*(7*n_qubits+4*int(n_qubits*(n_qubits-1)/2)))

                save_qubits.append(n_qubits)
                save_cutoff.append(cutoff_dim)
                save_blocks.append(n_blocks)
                save_times.append(dt)

    data = {'n_qubits': save_qubits, 'n_blocks': save_blocks, 'cutoff_dim': save_cutoff,'time': save_times, 'n_gates': save_gates}
    dataframe = pd.DataFrame(data=data)
    file_name = file_name + ".pkl"
    dataframe.to_pickle(file_name)

#%% Non-CV Timing Experiments
timing(real_amplitudes_circuit, "real_amplitudes2", 15, 15, 2, 1, 2)
#%%
nn_timing(real_amplitudes_circuit, "real_amplitudes-batch-5", 10, 10, 2, 1, 5)
#%%
nn_timing(real_amplitudes_circuit, "real_amplitudes-batch-10", 10, 10, 2, 1, 10)
#%%
nn_timing(real_amplitudes_circuit, "real_amplitudes-batch-25", 10, 10, 2, 1, 25)

#%% CV Timing Experiments
cv_timing(cv_neural_net, "cv_neural_net", 5, 10, 2, 1, 4, 4, 2)

#%%
nn_cv_timing(cv_neural_net, "cv_neural_net-batch-5", 3, 4, 2, 1, 2, 20, 5)
nn_cv_timing(cv_neural_net, "cv_neural_net-batch-10", 3, 4, 2, 1, 2, 20, 10)
nn_cv_timing(cv_neural_net, "cv_neural_net-batch-25", 3, 4, 2, 1, 2, 20, 25)

#%% Reading
# output = pd.read_pickle("a_file.pkl")
