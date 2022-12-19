#%% Imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import json
from wine import prepare_wine_dataset

SIGMAS = np.linspace(0.5, 0.001, 30)


#%% Preprocess dataset
def save_dataset():
    """This function should return the data ready to feed into the network"""
    train_data, test_data, validate_data = prepare_wine_dataset({5,7},40)
    x_train, y_train = train_data
    x_test, y_test = test_data
    x_val, y_val = validate_data

  # Save data
    data = np.array([x_train, x_test, y_train, y_test], dtype=object)
    np.save('./WINE_Dataset/WINE.npy', data, allow_pickle=True)
    data = np.array([np.array([x_val], dtype=object), np.array(y_val, dtype=object)], dtype=object)
    np.save('./WINE_Dataset/WINE_val.npy', data, allow_pickle=True)
    
    # Add gaussian noise for ENOB 
#    for sigma in SIGMAS:  
#        x_train_noisy = np.random.normal(0, sigma, x_train.shape) + x_train
#        x_test_noisy = np.random.normal(0, sigma, x_test.shape) + x_test
#        x_val_noisy = np.random.normal(0,sigma, x_val.shape) + x_val
#        data = np.array([x_train_noisy, x_test_noisy, y_train, y_test], dtype=object)
#        np.save('./WINE_Dataset/WINE_Noisy%f.npy'%sigma, data, allow_pickle=True)
#        data = np.array([np.array([x_val_noisy], dtype=object), np.array(y_val, dtype=object)], dtype=object)
#        np.save('./WINE_Dataset/WINE_val_Noisy%f.npy'%sigma, data, allow_pickle=True) 
    return 0

def prepare_dataset(n_samples=None):
    if (n_samples == None):
        return np.load('./WINE_Dataset/WINE.npy', allow_pickle=True)
    elif(n_samples > 0):
        x_train, x_test, y_train, y_test = np.load('./WINE_Dataset/WINE.npy', allow_pickle=True)
        return x_train[0:n_samples], x_test[0:n_samples//2], y_train[0:n_samples], y_test[0:n_samples//2]
    else:
        return None

def load_validation(n_samples=None):
    if (n_samples == None):
        x_test, y_test = np.load('./WINE_Dataset/WINE_val.npy', allow_pickle=True)
        return x_test[0], y_test
    elif(n_samples > 0):
        x_test, y_test = np.load('./WINE_Dataset/WINE_val.npy', allow_pickle=True)
        return x_test[0][0:n_samples], y_test[0:n_samples]
    else:
        return None

def load_noisy_data(sigma, n_samples=None):
    if (n_samples == None):
        x_train, x_test, y_train, y_test = np.load('./WINE_Dataset/WINE_Noisy%f.npy'%sigma, allow_pickle=True)
        return x_train, x_test, y_train, y_test
    elif(n_samples > 0):
        x_train, x_test, y_train, y_test = np.load('./WINE_Dataset/WINE%f.npy'%sigma, allow_pickle=True)
        return x_train[0:n_samples], x_test[0:n_samples//2], y_train[0:n_samples], y_test[0:n_samples//2]
    else:
        return None


def load_noisy_validation(sigma, n_samples=None):
    if (n_samples == None):
        x_test, y_test = np.load('./WINE_Dataset/WINE_val_Noisy%f.npy'%sigma, allow_pickle=True)
        return x_test[0], y_test
    elif(n_samples > 0):
        x_test, y_test = np.load('./WINE_Dataset/WINE_val.npy', allow_pickle=True)
        return x_test[0][0:n_samples], y_test[0:n_samples]
    else:
        return None


# Run script once to save datasets and verify the function works properly.
if __name__ == '__main__':
    save_dataset()
    x_train, x_test, y_train, y_test = prepare_dataset()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_test, y_test = load_validation()
    print(x_test.shape, y_test.shape)

    print(np.mean(x_train), np.std(x_train))
    x_train_noisy, x_test_noisy, y_train, y_test = load_noisy_data(SIGMAS[-1])

    print(x_train, x_train_noisy, np.mean(x_train-x_train_noisy), np.std(x_train-x_train_noisy))
    print('Maxes:', np.max(x_train), np.max(x_train_noisy), np.max(x_test), np.max(x_test_noisy))
