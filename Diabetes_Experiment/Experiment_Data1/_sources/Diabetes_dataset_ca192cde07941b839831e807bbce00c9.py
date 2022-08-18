import numpy as np
from sacred import Ingredient
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    filename = 'diabetes.csv'

@data_ingredient.capture
def load_data():
    data = pd.read_csv("diabetes.csv")

    data_X = data.loc[:, data.columns != "Outcome"]
    data_Y = data[["Outcome"]]

    train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y,
                                                        test_size=0.2,
                                                        stratify=data_Y,
                                                        random_state=0)

    # Pregancies pre-processing
    Q1 = train_X["Pregnancies"].quantile(0.25)
    Q3 = train_X["Pregnancies"].quantile(0.75)
    q95th = train_X["Pregnancies"].quantile(0.95)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR
    train_X["Pregnancies"] = np.where(train_X["Pregnancies"] > UW, q95th, train_X["Pregnancies"])

    # Blood pressure pre-processing
    med = train_X["BloodPressure"].median()
    q5th = train_X["BloodPressure"].quantile(0.05)
    q95th = train_X["BloodPressure"].quantile(0.95)
    Q1 = train_X["BloodPressure"].quantile(0.25)
    Q3 = train_X["BloodPressure"].quantile(0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5 * IQR
    UW = Q3 + 1.5 * IQR

    train_X["BloodPressure"] = np.where(train_X["BloodPressure"] == 0, med, train_X["BloodPressure"])
    train_X["BloodPressure"] = np.where(train_X["BloodPressure"] < LW, q5th, train_X["BloodPressure"])
    train_X["BloodPressure"] = np.where(train_X["BloodPressure"] > UW, q95th, train_X["BloodPressure"])

    # Skin Thickness pre-processing
    med = train_X["SkinThickness"].median()
    q95th = train_X["SkinThickness"].quantile(0.95)
    Q1 = train_X["SkinThickness"].quantile(0.25)
    Q3 = train_X["SkinThickness"].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR

    train_X["SkinThickness"] = np.where(train_X["SkinThickness"] == 0, med, train_X["SkinThickness"])
    train_X["SkinThickness"] = np.where(train_X["SkinThickness"] > UW, q95th, train_X["SkinThickness"])

    # Insulin pre-processing
    q60th = train_X["Insulin"].quantile(0.60)
    q95th = train_X["Insulin"].quantile(0.95)
    Q1 = train_X["Insulin"].quantile(0.25)
    Q3 = train_X["Insulin"].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR

    train_X["Insulin"] = np.where(train_X["Insulin"] == 0, q60th, train_X["Insulin"])
    train_X["Insulin"] = np.where(train_X["Insulin"] > UW, q95th, train_X["Insulin"])

    # BMI pre-processing
    med = train_X["BMI"].median()
    q95th = train_X["BMI"].quantile(0.95)
    Q1 = train_X["BMI"].quantile(0.25)
    Q3 = train_X["BMI"].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR

    train_X["BMI"] = np.where(train_X["BMI"] == 0, med, train_X["BMI"])
    train_X["BMI"] = np.where(train_X["BMI"] > UW, q95th, train_X["BMI"])

    # Diabetes Pedigree
    q95th = train_X["DiabetesPedigreeFunction"].quantile(0.95)
    Q1 = train_X["DiabetesPedigreeFunction"].quantile(0.25)
    Q3 = train_X["DiabetesPedigreeFunction"].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR

    train_X["DiabetesPedigreeFunction"] = np.where(train_X["DiabetesPedigreeFunction"] > UW, q95th,
                                                   train_X["DiabetesPedigreeFunction"])

    # Age pre-processing
    q95th = train_X["Age"].quantile(0.95)
    Q1 = train_X["Age"].quantile(0.25)
    Q3 = train_X["Age"].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + 1.5 * IQR

    train_X["Age"] = np.where(train_X["Age"] > UW, q95th, train_X["Age"])

    # Standardization
    feature_names = train_X.columns
    scaler = StandardScaler()

    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    train_X = pd.DataFrame(train_X, columns=feature_names)

    test_X = scaler.transform(test_X)
    test_X = pd.DataFrame(test_X, columns=feature_names)

    x_train = tf.convert_to_tensor(train_X.values)
    y_train = tf.convert_to_tensor(train_Y.values)
    x_test = tf.convert_to_tensor(test_X.values)
    y_test = tf.convert_to_tensor(test_Y.values)

    return x_train, y_train, x_test, y_test