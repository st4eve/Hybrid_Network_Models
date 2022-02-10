#%% Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_quantum_layers import *
from tf_train import *

#%% Load Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.convert_to_tensor(x_train, dtype='float32')
x_test = tf.convert_to_tensor(x_test, dtype='float32')
x_train = x_train[:1000]
y_train = y_train[:1000]

#%% Main Model
class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.main = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            #QuantumLayer(10, "real_amplitudes", n_qubits=2, n_blocks=1),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        x = self.main(inputs)
        return x

#%% Parameters
batch_size_train = 50
batch_size_test = 100
num_epochs = 6
learning_rate = 0.01
optimizer=tf.keras.optimizers.Adam(learning_rate)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
model_path = "./quantumtest/model"
metrics_path = "./quantumtest/metrics.npz"

#%% Train
model = Net()
train(num_epochs, batch_size_train, model, optimizer, loss_function, x_train, y_train, model_path, metrics_path,
          continue_training=False)

#%% Test Network
test(model, x_test, y_test, batch_size_test, metrics_path)


