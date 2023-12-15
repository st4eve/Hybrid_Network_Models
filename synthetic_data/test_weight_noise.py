import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

df = pd.read_pickle('./dataframes/df_kerr8.pkl') 

print(df.head())