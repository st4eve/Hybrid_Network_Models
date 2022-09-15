#%% Imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import json

#%% Preprocess dataset
def save_dataset():
    """This function should return the data ready to feed into the network"""
    # Import data subset and normalize from 0-1
    # https://huggingface.co/datasets/cifar100i
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

    # Photos to use
    # indices = [13, 8, 48, 90] # bus, bicycle, motorcycle, train
    indices = [22, 39, 86, 87]  # clock, keyboard, telephone, television

    # Filter out based on desired dataset size
    num_training_samples = 300
    num_testing_samples = 60
    filter_train = np.array([np.where(y_train.flatten() == idx)[0][:num_training_samples] for idx in indices]).flatten()
    filter_test = np.array([np.where(y_test.flatten() == idx)[0][:num_testing_samples] for idx in indices]).flatten()

    # Normalize inputs to [0,1]
    x_train = x_train[filter_train] / 255.0
    y_train = y_train[filter_train]
    x_test = x_test[filter_test] / 255.0
    y_test = y_test[filter_test]

    # Change categories from indices selected above to [0:N]
    for i in range(len(indices)):
        y_train[y_train == indices[i]] = i
        y_test[y_test == indices[i]] = i

    # Make y values categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    
    data = np.array([x_train, x_test, y_train, y_test], dtype='object')
    np.save('./CIFAR_Dataset/CIFAR.npy', data, allow_pickle=True)
    return 0

def prepare_dataset():
    return np.load('./CIFAR_Dataset/CIFAR.npy', allow_pickle=True)


if __name__ == '__main__':
    save_dataset()
    x_train, x_test, y_train, y_test = prepare_dataset()
    print(x_train)



