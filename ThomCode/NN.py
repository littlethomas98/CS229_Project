import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##########################




def importData():
    """
    Function: importData
        This function loads all realizations from files

    Parameters: none

    Returns:    x_train - training data set - inputs
                so2_train - training data set - outputs (SO2 hazard levels)
                x_valid - validation data set - inputs
                so2_valid - validation data set - outputs (SO2 hazard levels)
                x_test - testing data set - inputs
                so2_test - testing data set - outputs (SO2 hazard levels)
    """
    
    x_train = pd.read_csv('ThomCode/Trainning_Data.csv')
    x_train = x_train.dropna()
    x_train.drop(['Date','Month'],inplace=True)
    x_train.to_numpy()
    y_train = tf.one_hot(x_train.pop('Risk_Label'),6)

    x_valid = pd.read_csv('ThomCode/Validation_Data.csv')
    x_valid = x_valid.dropna()
    x_valid.drop(['Date','Month'],inplace=True)
    x_valid.to_numpy()
    y_valid = tf.one_hot(x_valid.pop('Risk_Label'),6)

    x_test = pd.read_csv('ThomCode/Testing_Data.csv')
    x_test = x_test.dropna()
    x_test.drop(['Date','Month'],inplace=True)
    x_test.to_numpy()
    y_test = tf.one_hot(x_test.pop('Risk_Label'),6)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test


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
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 6, activation = 'softmax'))
    print(model.summary())

    return model


def trainModel(x_train, y_train, x_valid, y_valid):
    """
    Function: trainModel
        This function trains the Sequential TensorFlow model on the provided training set

    Parameters: x_train - training data inputs 
                y_train - training data outputs (SO2 concentration)
                x_valid - validation data inputs 
                y_valid - validation data outputs (SO2 concentration)

    Returns:    model - a trained sequential TensorFlow model
    """ 

    model = createModel(x_train.shape[1])
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = [keras.metrics.Accuracy(), keras.metrics.Recall()]
    )

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=100,
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

    Parameters: model - trained sequential TensorFlow model
                x_test - testing data inputs 
                y_test - testing data outputs (SO2 concentration)
                BV - boolean value for whether to run a bias-variance study or
                     whether to run typical model testing

    Returns:    none
    """

    #Evaluate the model on the test data
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss, test acc:", results)
    print("Generate predictions")
    predictions = model.predict(x_test)

    if not BV:
        plt.figure()
        plt.plot(x_test['Date'], np.argmax(y_test, axis=1), 'bx', markersize=8, alpha = 0.2)
        plt.plot(x_test['Date'], np.argmax(predictions, axis=1), 'ro', alpha = 0.2, label = 'Predicted Data')
        plt.legend(['Real Data', 'Predicted Data'], loc='best')
        plt.xlabel('Date (Noramlized to 2018)')
        plt.ylabel('SO2 Hazard Level')
        plt.savefig('Predictions.png')

    return results


def biasVar(x_train, so2_train, x_valid, so2_valid, x_test, so2_test):
    """
    Function: biasVar
        This function conducts a study to check if the model has high bias or variance.
        It trains a model on training samples of increasing size and plots the train and
        validation losses as the sample size increases.

    Parameters: x_train - training data set - inputs
                so2_train - training data set - outputs (SO2 hazard levels)
                x_valid - validation data set - inputs
                so2_valid - validation data set - outputs (SO2 hazard levels)
                x_test - testing data set - inputs
                so2_test - testing data set - outputs (SO2 hazard levels)
                
    Returns:    none
    """
    trainHist = np.zeros([10,2])
    testHist = np.zeros(10)

    ##Plot training and validation loss with varying 
    for i in range(1,11):
        x_train_sec = x_train[0:int(i/10 * x_train.shape[0]),:]
        so2_train_sec = so2_train[0:int(i/10 * so2_train.shape[0])]
        x_valid_sec = x_valid[0:int(i/10 * x_valid.shape[0]),:]
        so2_valid_sec = so2_valid[0:int(i/10 * so2_valid.shape[0])]
        x_test_sec = x_test[0:int(i/10 * x_test.shape[0]),:]
        so2_test_sec = so2_test[0:int(i/10 * so2_test.shape[0])]
        model, trainLoss = trainModel(x_train_sec, so2_train_sec, x_valid_sec, so2_valid_sec)
        results = testModel(model, x_test_sec, so2_test_sec, BV=True)
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
    Function: main
        This function calls all previously defined methods.

    Parameters: BiasVarStudy - boolean value for whether to run a bias-variance study or
                               whether to run typical model training

    Returns:    none
    """   
    x_train, y_train, x_valid, y_valid, x_test, y_test = importData()

    #Conduct Bias Variance Study?
    if BiasVarStudy:
        biasVar(x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        model, _ = trainModel(x_train, y_train, x_valid, y_valid)
        testModel(model, x_test, y_test)
    return

main(BiasVarStudy=False)