import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def build_lstm_model(input_shape, units,output_steps):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(None, 1), return_sequences=True),
        tf.keras.layers.LSTM(32,return_sequences=True),
        tf.keras.layers.LSTM(units, return_sequences=False),
        tf.keras.layers.Dense(output_steps,activation='relu')

    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))
    return history
