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
    """
    Function: importData
        This function loads all realizations from files

    Parameters: none

    Returns:    data - a numpy array of all data
    """
    
    data = pd.read_csv('CleanMergedData.csv')
    data = data.to_numpy()
    return data


def splitData(data, train_Frac = 0.7):
    """
    Function: splitData
        This function splits the data into training, validation, and testing sets.

    Parameters: data - a numpy array of all data
                train_Frac - the fraction of data that you would like to be training data (the test and validation 
                             sets will be evenly split between the remaining data)


    Returns:    x_train - training data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_train - training data outputs (SO2 concentration)
                x_valid - validation data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_valid - validation data outputs (SO2 concentration)
                x_test - testing data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_test - testing data outputs (SO2 concentration)
    """
    data = shuffle(data, random_state = 4)
    data = data[~np.isnan(data).any(axis=1)]

    train_len  = int(data.shape[0] * train_Frac)
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
    """
    Function: createModel
        This function creates a Sequential TensorFlow model

    Parameters: none

    Returns:    model - an untrained sequential TensorFlow model
    """    
    model = keras.Sequential()
    model.add(tf.keras.Input(shape = (2,)))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 1))
    print(model.summary())

    return model


def trainModel(x_train, y_train, x_valid, y_valid):
    """
    Function: trainModel
        This function trains the Sequential TensorFlow model on the provided training set

    Parameters: x_train - training data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_train - training data outputs (SO2 concentration)
                x_valid - validation data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_valid - validation data outputs (SO2 concentration)

    Returns:    model - a trained sequential TensorFlow model
    """ 

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
    """
    Function: testModel
        This function trains the Sequential TensorFlow model on the provided training set

    Parameters: x_test - testing data inputs (EQ magnitude and distrance between EQ epicenter and SO2 recording)
                y_test - testing data outputs (SO2 concentration)

    Returns:    none
    """

    #Evaluate the model on the test data
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss, test acc:", results)

    print("Generate predictions")
    predictions = model.predict(x_test)

    #Plot read data
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x_test[:,0], x_test[:,1], y_test, label = 'Real Data')

    #Plot read data
    ax.scatter(x_test[:,0], x_test[:,1], predictions)
    ax.set_xlabel('EQ Magnitude')
    ax.set_ylabel('Distance (km)')
    ax.set_zlabel('SO2 Concentration (ppm)')

    #Determined that 3rd dimension is not relavent for current analysis, therefore reduce to 2d plot
    plt.figure()
    line1, line2 = plt.plot(x_test, y_test, 'bo', alpha = 0.2)
    line3, line4 = plt.plot(x_test, predictions, 'ro', alpha = 0.2, label = 'Predicted Data')
    plt.legend([line1, line3], ['Real Data', 'Predicted Data'], loc='best')
    plt.xlabel('Distance (km)')
    plt.ylabel('SO2 Concentration (ppm)')
    plt.savefig('Predictions.png')
    plt.show()

    return 

def main():
    """
    Function: createModel
        This function creates a Sequential TensorFlow model

    Parameters: none

    Returns:    model - an untrained sequential TensorFlow model
    """   
    data = importData()
    mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test = splitData(data)
    model = trainModel(mag_train, so2_train, mag_valid, so2_valid)
    testModel(model, mag_test, so2_test)
    return

main()