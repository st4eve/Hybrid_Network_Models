#%% Imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

#%% Preprocess dataset
def prepare_dataset():
    """This function should return the data ready to feed into the network"""
    # Import data subset and normalize from 0-1
    # https://huggingface.co/datasets/cifar100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

    n1 = 300
    n2 = 60
    # indices = [13, 8, 48, 90] # bus, bicycle, motorcycle, train
    indices = [22, 39, 86, 87]  # clock, keyboard, telephone, television

    filter_train = np.array([np.where(y_train.flatten() == idx)[0][:n1] for idx in indices]).flatten()
    filter_test = np.array([np.where(y_test.flatten() == idx)[0][:n2] for idx in indices]).flatten()

    x_train = x_train[filter_train] / 255.0
    y_train = y_train[filter_train]
    x_test = x_test[filter_test] / 255.0
    y_test = y_test[filter_test]
    for i in range(len(indices)):
        y_train[y_train == indices[i]] = i
        y_test[y_test == indices[i]] = i

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test