#%% Imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from sklearn import preprocessing
import numpy as np
import json

#%% Preprocess dataset
def save_dataset():
    """This function should return the data ready to feed into the network"""
    # Import data subset and normalize from 0-1
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = tfds.as_numpy(tfds.load(
        'wine_quality', 
        split=['train[:40%]', 'train[40%:50%]', 'train[50%:60%]'], 
        as_supervised=True, 
        shuffle_files=True, 
        batch_size=-1))

    # Preprocessing stuff
    x_train = np.array(list(x_train.values())).T
    x_test = np.array(list(x_test.values())).T
    x_val = np.array(list(x_val.values())).T

    # Mean of 0 and STD of 1
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    
    # Scale between 0 and 1
    x_train = (x_train - np.min(x_train, axis=1, keepdims=True))/np.ptp(x_train, axis=1, keepdims=True)
    x_test = (x_test - np.min(x_test, axis=1, keepdims=True))/np.ptp(x_test, axis=1, keepdims=True)
    x_val = (x_val - np.min(x_val, axis=1, keepdims=True))/np.ptp(x_val, axis=1, keepdims=True)


    #Make y values categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
     
    data = np.array([x_train, x_test, y_train, y_test], dtype=object)
    np.save('./WINE_Dataset/WINE.npy', data, allow_pickle=True)
    data = np.array([np.array([x_val], dtype=object), np.array(y_val, dtype=object)], dtype=object)
    np.save('./WINE_Dataset/WINE_val.npy', data, allow_pickle=True)
    
    # Add gaussian noise for ENOB
    sigmas = np.logspace(-10, 0, 20)
    for sigma in sigmas:  
        x_train_noisy = np.random.normal(0, sigma, x_train.shape) + x_train
        x_test_noisy = np.random.normal(0, sigma, x_test.shape) + x_test
        data = np.array([x_train_noisy, x_test_noisy, y_train, y_test], dtype=object)
        np.save('./WINE_Dataset/WINE_Noisy%f.npy'%sigma, data, allow_pickle=True)
    
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
        x_train, x_test, y_train, y_test = np.load('./WINE_Dataset/WINE02.npy', allow_pickle=True)
        return x_train[0:n_samples], x_test[0:n_samples//2], y_train[0:n_samples], y_test[0:n_samples//2]
    else:
        return None


# Run script once to save datasets and verify the function works properly.
if __name__ == '__main__':
    save_dataset()
    x_train, x_test, y_train, y_test = prepare_dataset()
    print(x_train.shape, y_train.shape)
    x_test, y_test = load_validation()
    print(x_test.shape, y_test.shape)

    print(np.mean(x_train), np.std(x_train))
    sigmas = np.logspace(-10, -2, 11)
    x_train_noisy, x_test_noisy, y_train, y_test = load_noisy_data(sigmas[-1])
    print(x_train, x_train_noisy, np.mean(x_train-x_train_noisy), np.std(x_train-x_train_noisy))
    print('Maxes:', np.max(x_train), np.max(x_train_noisy), np.max(x_test), np.max(x_test_noisy))
    