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

    ##TODO TEMPORARY############################################
    data.sort_values(by=['Date'], inplace=True)
    plt.plot(data['Date'], data['Daily Max 1-hour SO2 Concentration'])
    plt.plot(data['Date'], 10**data['mag']/100, 'ro', alpha=0.5)
    plt.show()
    ##############################################################################

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
    data = shuffle(data, random_state = 2)
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


def createModel(InputShape):
    """
    Function: createModel
        This function creates a Sequential TensorFlow model

    Parameters: none

    Returns:    model - an untrained sequential TensorFlow model
    """    
    model = keras.Sequential()
    model.add(tf.keras.Input(shape = (InputShape,)))
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

    model = createModel(x_train.shape[1])
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

    finalLoss = [history.history['loss'][-1], history.history['val_loss'][-1]]

    #Plot train vs validation error
    # trainLoss = history.history['loss']
    # validLoss = history.history['val_loss']
    # plt.plot(trainLoss, label='Train')
    # plt.plot(validLoss, label='Validation')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('Model Loss')
    # plt.legend(loc = 'best')
    # plt.show()

    return model, finalLoss


def testModel(model, x_test, y_test, BV=False):
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

    if not BV:
        #Plot 3D data
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(x_test[:,0], x_test[:,1], y_test, label = 'Real Data')
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

    #Plugin to visualize model
    # keras.utils.plot_model(
    #     model,
    #     to_file='ModelPlot.png',
    #     show_shapes=True,
    #     show_dtype=False,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96,
    #     layer_range=None,
    #     show_layer_activations=False
    # )

    return results
    
def biasVar(mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test):
    trainHist = np.zeros([10,2])
    testHist = np.zeros(10)
    for i in range(1,11):
        mag_train_sec = mag_train[0:int(i/10 * mag_train.shape[0]),:]
        so2_train_sec = so2_train[0:int(i/10 * so2_train.shape[0])]
        mag_valid_sec = mag_valid[0:int(i/10 * mag_valid.shape[0]),:]
        so2_valid_sec = so2_valid[0:int(i/10 * so2_valid.shape[0])]
        mag_test_sec = mag_test[0:int(i/10 * mag_test.shape[0]),:]
        so2_test_sec = so2_test[0:int(i/10 * so2_test.shape[0])]
        model, trainLoss = trainModel(mag_train_sec, so2_train_sec, mag_valid_sec, so2_valid_sec)
        results = testModel(model, mag_test_sec, so2_test_sec, BV=True)
        trainHist[i-1] = trainLoss
        testHist[i-1] = results

    plt.plot(np.linspace(0.1,1,10),trainHist)
    plt.plot(np.linspace(0.1,1,10),testHist)
    plt.legend(['Train', 'Validation', 'Test'], loc='best')
    plt.title('Train and Test Loss vs. Dataset Size')
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Loss')
    plt.savefig('BiasVarStudy.png')

    return

def main(BiasVarStudy = False):
    """
    Function: createModel
        This function creates a Sequential TensorFlow model

    Parameters: BiasVarStudy - boolean value for whether to run a bias-variance study or
                               whether to run typical model training

    Returns:    model - an untrained sequential TensorFlow model
    """   
    data = importData()
    mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test = splitData(data)

    #Conduct Bias Variance Study?
    if BiasVarStudy:
        biasVar(mag_train, so2_train, mag_valid, so2_valid, mag_test, so2_test)
    else:
        model, _ = trainModel(mag_train, so2_train, mag_valid, so2_valid)
        testModel(model, mag_test, so2_test)
    return

main(BiasVarStudy=False)