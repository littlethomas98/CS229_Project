import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from sklearn.utils import shuffle
import os

os.chdir('../CS229_Project/ThomCode')

def importData():
    data = pd.read_csv('CleanMergedData.csv')
    data = data.to_numpy()
    return data

def splitData(data, splitFrac = 0.7):
    data = shuffle(data, random_state = 4)
    data = data[~np.isnan(data).any(axis=1)]

    train_len  = int(data.shape[0] * splitFrac)
    valid_len = train_len + int((data.shape[0] - train_len) / 2)

    data_train = data[0:train_len, :]
    data_valid = data[train_len:valid_len, :]
    data_test = data[valid_len:, :]

    mag_train = data_train[:,3:]
    so2_train = data_train[:,2]
    mag_valid = data_valid[:,3:]
    so2_valid = data_valid[:,2]
    mag_test = data_test[:,3:]
    so2_test = data_test[:,2]


    return mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test

def createModel():
    model = keras.Sequential()
    model.add(tf.keras.Input(shape = (2,)))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 1))
    print(model.summary())

    return model

def trainModel(x_train, y_train, x_valid, y_valid):
    #Create an instance of the model
    model = createModel()
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanSquaredError(),
        # metrics = [keras.metrics.mean_squared_error]
    )

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_valid, y_valid),
    )

    return model

def testModel(model, x_test, y_test):
    #Evaluate the model on the test data
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss, test acc:", results)

    print("Generate predictions")
    predictions = model.predict(x_test)
    plt.plot(x_test, y_test, 'bo', alpha = 0.2)
    plt.plot(x_test, predictions, 'ro', alpha = 0.2)
    plt.show()

    return predictions

def main():
    data = importData()
    mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test = splitData(data)
    model = trainModel(mag_train, so2_train, mag_valid, so2_valid)
    testModel(model, mag_test, so2_test)
    return

main()