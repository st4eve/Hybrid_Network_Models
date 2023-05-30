"""Hybrid Network Models 2022"""
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy import stats


def drop_outliers(dataframe, field_name):
    """Drops outliers for a specified column

    Drop the outliers using 1.5 iqr method

    Args:
        dataframe (pandas.DataFrame): Dataframe of dataset
        field_name (str): The column name to drop outliers for
    """
    iqr = 1.5 * (
        np.percentile(dataframe[field_name], 75)
        - np.percentile(dataframe[field_name], 25)
    )
    dataframe.drop(
        dataframe[
            dataframe[field_name] > (iqr + np.percentile(dataframe[field_name], 75))
        ].index,
        inplace=True,
    )
    dataframe.drop(
        dataframe[
            dataframe[field_name] < (np.percentile(dataframe[field_name], 25) - iqr)
        ].index,
        inplace=True,
    )


def load_wine_dataset(type="white"):
    """Loads the wine dataset

    Args:
        type (string): Type of wine to load

    Returns:
        pandas.DataFrame: Pandas dataframe containing all of the wine data
    """
    if type not in ["white", "red", "all"]:
        raise ValueError("only 'white', 'red', or 'all' types of wine are accepted")

    white_wine_dataframe = tfds.as_dataframe(
        *tfds.load(
            "wine_quality/white",
            split="train[:100%]",
            shuffle_files=True,
            with_info=True,
        )
    )
    red_wine_dataframe = tfds.as_dataframe(
        *tfds.load(
            "wine_quality/red", split="train[:100%]", shuffle_files=True, with_info=True
        )
    )
    if type == "white":
        final_dataframe = white_wine_dataframe
    if type == "red":
        final_dataframe = red_wine_dataframe
    if type == "all":
        final_dataframe = pd.concat([red_wine_dataframe, white_wine_dataframe])

    final_dataframe.columns = final_dataframe.columns.str.lstrip("features")
    final_dataframe.columns = final_dataframe.columns.str.lstrip("/")
    return final_dataframe


def drop_wine_correlated_features(wine_dataset):
    """Drops correlated features in the wine dataset

    Free sulfur dioxide is dropped due to correlation with total sulfur dioxide
    Residual sugar is dropped due to correlation with density and both sulfurs

    Args:
        wine_dataset (pandas.DataFrame): Pandas dataframe of the wine dataset
    """
    wine_dataset.drop(columns=["free sulfur dioxide", "residual sugar"], inplace=True)


def drop_wine_outliers(wine_dataset):
    """Drops outliers in the wine dataset for all features

    Args:
        wine_dataset (pandas.DataFrame): Pandas dataframe of the wine dataset
    """
    features = [feature for feature in wine_dataset.columns if feature != "quality"]
    for feature in features:
        drop_outliers(wine_dataset, feature)


def eliminate_skews(wine_dataset):
    """Eliminate skews

    We could consider updating this to attempt skew fixes based on the skew value.
    Currently, alcohol, sulphates and volatile acidity have been identified as
    having large skews and eliminated with log and boxcox method.

    Args:
        wine_dataset (pandas.DataFrame): Pandas dataframe of the wine dataset
    """
    wine_dataset["alcohol"] = stats.boxcox(wine_dataset["alcohol"])[0]
    wine_dataset["volatile acidity"] = stats.boxcox(wine_dataset["volatile acidity"])[0]
    wine_dataset["sulphates"] = np.log(wine_dataset["sulphates"])


def balance_wine_dataset(wine_dataset, output_qualities):
    """Filter down to a few output features and balance them

    Remove all qualities not specified in output_qualities.
    Balance remaining qualities to the lowest count.
    Only allow 5, 6, 7 because all other qualities have insignificant counts.
    We could consider updating this to use oversampling with something like SMOTE.


    Args:
        wine_dataset (pandas.DataFrame): Pandas dataframe of the wine dataset
        output_qualities (dict, optional): Set of output qualities to use for classifying.
        Defaults to {5,6,7}.

    Raises:
        ValueError: 2 or 3 target features must be supplied
        ValueError: Target features must be in the set {5, 6, 7}
    """
    output_qualities = list(output_qualities)
    if len(output_qualities) not in [2, 3]:
        raise ValueError("Must supply 2 or 3 target features")
    for output_quality in output_qualities:
        if output_quality not in [5, 6, 7]:
            raise ValueError("Must supply output qualities in the set {5, 6, 7}")

    unique_qualities = wine_dataset["quality"].unique()
    for quality in unique_qualities:
        if quality not in output_qualities:
            index_names = wine_dataset[wine_dataset["quality"] == quality].index
            wine_dataset.drop(index_names, inplace=True)

    grouped_dataset = wine_dataset.groupby("quality")
    balanced_wine_dataset = pd.DataFrame(
        grouped_dataset.apply(
            lambda x: x.sample(grouped_dataset.size().min()).reset_index(drop=True)
        )
    )
    return balanced_wine_dataset


def prepare_xy_data(wine_dataset):
    """Convert dataframes to numpy arrays and apply processing

    Args:
        wine_dataset (pandas.DataFrame): wine dataset dataframe

    Returns:
        tuple: X, Y numpy.array data tuple where Y is one-hot encoded
    """
    x_data = wine_dataset.drop("quality", axis=1)
    x_data = x_data.to_numpy()
    scaler = StandardScaler().fit(x_data)
    x_data = scaler.transform(x_data)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_data)
    x_data = scaler.transform(x_data)

    y_data = wine_dataset["quality"]
    y_data = y_data.to_numpy()

    for i, quality in enumerate(np.unique(y_data)):
        y_data[y_data == quality] = i

    y_data = to_categorical(y_data, num_classes=len(np.unique(y_data)))
    return x_data, y_data


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


def prepare_wine_dataset(output_qualities, seed):
    """Prepares the wine dataset for use in neural networks

    Returns:
        tuple: Tuple of train, test and validate data tuples
    """
    wine_dataset = load_wine_dataset()
    drop_wine_correlated_features(wine_dataset=wine_dataset)
    drop_wine_outliers(wine_dataset=wine_dataset)
    balanced_wine_dataset = balance_wine_dataset(
        wine_dataset=wine_dataset, output_qualities=output_qualities
    )
    eliminate_skews(balanced_wine_dataset)
    x_data, y_data = prepare_xy_data(wine_dataset=balanced_wine_dataset)
    train_data, test_data, validate_data = split_data(
        x_data=x_data,
        y_data=y_data,
        train_ratio=0.7,
        test_ratio=0.15,
        validate_ratio=0.15,
        seed=seed,
    )
    return train_data, test_data, validate_data

def prepare_dataset_for_visualization(output_qualities, seed):
    wine_dataset = load_wine_dataset()
    drop_wine_correlated_features(wine_dataset=wine_dataset)
    drop_wine_outliers(wine_dataset=wine_dataset)
    balanced_wine_dataset = balance_wine_dataset(
        wine_dataset=wine_dataset, output_qualities=output_qualities
    )
    eliminate_skews(balanced_wine_dataset)
    return balanced_wine_dataset
    
def save_training_data(x_train, x_val, y_train, y_val): 
    data = np.array([x_train, x_val, y_train, y_val], dtype=object)
    np.save('./WINE_Dataset/WINE.npy', data, allow_pickle=True)
    
def save_testing_data(x_test, y_test):
    data = np.array([np.array([x_test], dtype=object), np.array(y_test, dtype=object)], dtype=object)
    np.save('./WINE_Dataset/WINE_val.npy', data, allow_pickle=True)

if __name__ == "__main__":
    train_data, test_data, validate_data = prepare_wine_dataset({5, 6, 7}, 10)
