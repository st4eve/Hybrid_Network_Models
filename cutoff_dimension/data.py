import numpy as np
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def split_data(x_data, y_data, train_ratio, test_ratio, validate_ratio, seed=10):
    """Split up X, Y data into train, test and validate groups

    Args:
        x_data (numpy.array): X dataset
        y_data (numpy.array): Y dataset
        train_ratio (float): fraction to use for training
        test_ratio (float): fraction to use for testing
        validate_ratio (float): fraction to use for validation
        seed (int): seed to use for shuffling

    Raises:
        ValueError: If ratios do not sum to 1

    Returns:
        tuple: tuple of train, test, and validate data tuples
    """
    if train_ratio + test_ratio + validate_ratio != 1.00:
        raise ValueError("Ratios must add up to one")

    x_train, x_test_validate, y_train, y_test_validate = train_test_split(
        x_data, y_data, train_size=train_ratio, random_state=seed, stratify=y_data
    )
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_test_validate,
        y_test_validate,
        train_size=test_ratio / (test_ratio + validate_ratio),
        random_state=seed,
        stratify=y_test_validate,
    )
    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def generate_synthetic_dataset():
    """Generates synthetic dataset for cutoff dimension analysis

    Returns:
        tuple: Tuple of numpy arrays of x,y data
    """
    x_data, y_data = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=2.0,
        flip_y=0.05,
        random_state=17,
    )
    y_data = to_categorical(y_data, num_classes=len(np.unique(y_data)))
    train_data, test_data, validate_data = split_data(x_data, y_data, 0.7, 0.15, 0.15)
    return train_data, test_data, validate_data


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
    data = np.array(
        [np.array([x_test], dtype=object), np.array(y_test, dtype=object)], dtype=object
    )
    np.save("./synthetic_data/synthetic_testing.npy", data, allow_pickle=True)
