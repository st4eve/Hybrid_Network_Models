#%% Imports

# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# NumPy
import numpy as np

# Custom Quantum Layers
from CV_quantum_layers import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

# Other
import time

# Experiment to be integrated into sacred workflow
cutoff_dim = 10
num_epochs = 50

# Normalization callback check
class CheckNormalization(Callback):
    def __init__(self):
        super(CheckNormalization, self).__init__()

    def on_batch_end(self, batch, logs={}):
        """Check normalization by instantiating an identical circuit
        with the model weights and testing two random inputs. This
        needs to be done in a more intelligent way."""
        random_ints = np.random.randint(0, 80, 2, dtype=int)
        for i in range(len(random_ints)):
            norm = model.main.check_normalization(x_train[i])
            if(norm<0.97):
                print("Normalization starting to get out of bounds: ", norm)

# Load Dataset
def load_iris_data():
    iris = load_iris()
    x = iris['data']
    y = iris['target']

    # Eliminate Setosa Species at index 0 to reduce output classes to 2.
    filter = np.where((y == 1) | (y == 2) )
    x, y = x[filter], y[filter]
    y[np.where(y == 1)] = 0
    y[np.where(y == 2)] = 1

    def normalize_data(x, x_max):
        min_val = np.min(x)
        max_val = np.max(x)
        range = max_val - min_val
        x = (x - min_val)/range
        x = x*x_max
        return x

    # Angle Normalization
    for i in range(1,4,2):
        x[:,i] = normalize_data(x[:,i],2*np.pi)

    # Amplitude Normalization
    for i in range(0,4,2):
        x[:,i] = normalize_data(x[:,i],5)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_iris_data()

# Build Model
class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.main = QuantumLayer(n_qumodes=2, n_outputs=2, n_layers=1, cutoff_dim=cutoff_dim, encoding_method=AmplitudePhaseDisplacementEncoding)

        #self.final = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.main(inputs)
        #x = self.final(x)
        return x

# Compile Model
model = Net()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(0.01),
              metrics=['accuracy']
              )

# Fit Model
model.fit(x_train, y_train,
          batch_size=10,
          epochs=num_epochs,
          initial_epoch=0,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[CheckNormalization()]
          )


