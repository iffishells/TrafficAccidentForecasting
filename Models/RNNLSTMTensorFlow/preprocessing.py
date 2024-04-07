
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def create_dataset(data, time_steps,future_step=10):
    X, y = [], []
    for i in range(len(data) - time_steps-future_step):
        # print( f"input shape {(i,i + time_steps)} output shape : {(i + time_steps,i + time_steps+future_step)}")
        # print()
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps:i + time_steps+future_step ])
    return np.array(X), np.array(y)

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return scaled_data, scaler
