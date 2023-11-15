import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

def split_data(x_data, y_data, train_ratio, seed=10):
    """Split up X, Y data into train, test and validate groups

    Args:
        x_data (numpy.array): X dataset
        y_data (numpy.array): Y dataset
        train_ratio (float): fraction to use for training
        seed (int): seed to use for shuffling

    Returns:
        tuple: tuple of train, test, and validate data tuples
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_ratio, random_state=seed, stratify=y_data
    )

    return (x_train, y_train), (x_test, y_test)


def generate_synthetic_dataset(num_datapoints=1000, n_features=15, n_classes=3):
    """Generates synthetic dataset for cutoff dimension analysis

    Returns:
        tuple: Tuple of numpy arrays of x,y data
    """
    x_data, y_data = make_classification(
        n_samples=num_datapoints,
        n_features=n_features,
        n_informative=n_features//3 * 2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=3,
        class_sep=2,
        flip_y=0.05,
        random_state=17,
    )
    scaler = MinMaxScaler().fit(x_data)
    x_data = scaler.transform(x_data)
    y_data = to_categorical(y_data, num_classes=len(np.unique(y_data)))
    train_data, test_data = split_data(x_data, y_data, 0.7)
    return train_data, test_data

def generate_synthetic_dataset_easy(num_datapoints=5000, n_features=15, n_classes=3):
    """Generates synthetic dataset for cutoff dimension analysis

    Returns:
        tuple: Tuple of numpy arrays of x,y data
    """
    x_data, y_data = make_classification(
        n_samples=num_datapoints,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=3,
        class_sep=3.0,
        flip_y=0.02,
        random_state=17,
    )
    scaler = MinMaxScaler().fit(x_data)
    x_data = scaler.transform(x_data)
    y_data = to_categorical(y_data, num_classes=len(np.unique(y_data)))
    train_data, test_data = split_data(x_data, y_data, 0.7, seed=17)
    return train_data, test_data

def generate_synthetic_dataset_easy_raw(num_datapoints=5000, n_features=15, n_classes=3):
    """Generates synthetic dataset for cutoff dimension analysis

    Returns:
        tuple: Tuple of numpy arrays of x,y data
    """
    x_data, y_data = make_classification(
        n_samples=num_datapoints,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=3,
        class_sep=3.0,
        flip_y=0.02,
        random_state=17,
    )
    train_data, test_data = split_data(x_data, y_data, 0.7, seed=17)
    return train_data, test_data




def save_training_data(x_train, x_val, y_train, y_val):
    """Save training data to numpy file

    Args:
        x_train (Numpy array): x data used for training
        x_val (Numpy array): x data used for validation
        y_train (Numpy array): y data used for training
        y_val (Numpy array): y data used for validation
    """
    data = np.array([x_train, x_val, y_train, y_val], dtype=object)
    np.save("./synthetic_data/synthetic_training.npy", data, allow_pickle=True)


def save_testing_data(x_test, y_test):
    """Save testing data to numpy file

    Args:
        x_test (Numpy array): x data used for testing
        y_test (Numpy array): y data used for testing
    """
    data = np.array([np.array([x_test], dtype=object), np.array(y_test, dtype=object)], dtype=object)
    np.save("./synthetic_data/synthetic_testing.npy", data, allow_pickle=True)
