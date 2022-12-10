import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from sklearn.utils import shuffle
import seaborn as sns
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##########################


from numpy.random import seed
seed(1)
tf.random.set_seed(8)


def loadData():

    data = pd.read_csv('ThomCode/CleanMergedData.csv')
    # data = data.replace(np.nan,0)
    data = data.dropna(axis=0)

    data = shuffle(data, random_state=4)

    train_len = int(data.shape[0]*0.7)
    valid_len = (data.shape[0] - train_len) // 2 + train_len
    x_train = data.iloc[:train_len,3:]
    y_train = data.iloc[:train_len,2]
    y_train = tf.one_hot(y_train,6)

    x_valid = data.iloc[train_len:valid_len,3:]
    y_valid = data.iloc[train_len:valid_len,2]
    y_valid = tf.one_hot(y_valid,6)

    x_test = data.iloc[valid_len:,3:]
    y_test = data.iloc[valid_len:,2]
    y_test = tf.one_hot(y_test,6)

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
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dense(units = 128, activation = 'relu'))
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
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = 'categorical_crossentropy',
        metrics = [keras.metrics.Accuracy(), keras.metrics.Recall()]
    )

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=500,
        validation_data=(x_valid, y_valid),
    )

    finalLoss = [history.history['loss'][-1], history.history['val_loss'][-1]]

    return model, finalLoss


def plot_confusion_matrix(actual, predicted, labels, ds_type):
    cm = tf.math.confusion_matrix(actual, predicted)

    #Normalize to get percentage:
    cm /= np.sum(cm)

    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('Confusion Matrix')
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)


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

    #Plot confusion matrix
    plot_confusion_matrix(np.argmax(y_test,axis=1), np.argmax(predictions,axis=1), [0,1,2,3,4,5], 'testing')

    #Plot data
    if not BV:
        plt.figure()
        x=list(range(0,x_test.shape[0]))
        plt.plot(x,np.argmin(predictions,axis=1),'o', alpha = 0.2)
        plt.plot(x,np.argmin(y_test,axis=1),'o', alpha = 0.2)


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
        x_train_sec = x_train.iloc[0:int(i/10 * x_train.shape[0]),:]
        so2_train_sec = so2_train.iloc[0:int(i/10 * so2_train.shape[0])]
        x_valid_sec = x_valid.iloc[0:int(i/10 * x_valid.shape[0]),:]
        so2_valid_sec = so2_valid.iloc[0:int(i/10 * so2_valid.shape[0])]
        x_test_sec = x_test.iloc[0:int(i/10 * x_test.shape[0]),:]
        so2_test_sec = so2_test.iloc[0:int(i/10 * so2_test.shape[0])]
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


def replaceLatLongwithDistance(MergedData):
    #MLO data collectino site:
    v_lat = 19.5362
    v_long = -155.5763
    R = 6371 #Radius of Earth in km

    #Get latitude and longitude data for earthquake epicenter and SO2 data collection site
    lat1 = v_lat * np.pi / 180
    long1 = v_long * np.pi / 180
    lat2 = MergedData['SO2_lat'] * np.pi / 180
    long2 = MergedData['SO2_long'] * np.pi / 180
    lat3 = MergedData['EQ_lat'] * np.pi / 180
    long3 = MergedData['EQ_long'] * np.pi / 180

    #Calc distance between SO2 collection site and volcano
    a_SO2 = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((long2-long1)/2)**2
    c_SO2 = 2 * np.arctan2(np.sqrt(a_SO2),np.sqrt(1-a_SO2))
    MergedData['SO2_to_volc_Distance'] = R * c_SO2

    #Calc distance between EQ epicenter and volcano
    a_EQ = np.sin((lat3-lat1)/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin((long3-long1)/2)**2
    c_EQ = 2 * np.arctan2(np.sqrt(a_EQ),np.sqrt(1-a_EQ))
    MergedData['EQ_to_volc_Distance'] = R * c_EQ
    
    #Calculate bearing angle from volcano to SO2 data collection site
    X = np.cos(lat2) * np.sin(long1-long2)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin((lat1) * np.cos(lat2) * np.cos(long1-long2))
    MergedData['SO2_station_bearing_angle'] = -np.arctan2(X,Y) * 180 / np.pi
    MergedData['SO2_station_bearing_angle'][MergedData['SO2_station_bearing_angle'] < 0] = MergedData['SO2_station_bearing_angle'][MergedData['SO2_station_bearing_angle'] < 0] + 360
    MergedData.drop(['SO2_lat', 'SO2_long', 'EQ_lat', 'EQ_long'], axis = 1, inplace = True)

    #Calculate alignment between bearing angle to SO2 data collection site and wind direction
    v_SO2_Site = np.array([np.cos(MergedData['SO2_station_bearing_angle'] * np.pi / 180), np.sin(MergedData['SO2_station_bearing_angle'] * np.pi / 180)])
    v_volcano = np.array([np.cos(MergedData['WIND DIRECTION'] * np.pi / 180), np.sin(MergedData['WIND DIRECTION'] * np.pi / 180)])
    MergedData['Wind Alignment'] = [np.dot(v_SO2_Site[:,i].T,v_volcano[:,i]) for i in range(MergedData.shape[0])]

    return MergedData


def plotOutput(model, data):
    #Get Mauna Loa Latitude and Longitude
    Lat = np.linspace(18.9,20.3,30)
    Long = np.linspace(-156,-154.8,20)

    start = 7
    num_events = 10
    All_pred = np.zeros([len(Lat)*len(Long),num_events-start])
    for n in range(start,num_events):
        x_row = np.tile(data.iloc[n,:],(len(Lat)*len(Long),1))
        x_test = pd.DataFrame(x_row)
        x_test.columns = data.columns
        pred = np.zeros(len(Lat))

        #Loop over discritized map of Hawaii and plot output
        for i,lat in enumerate(Lat):
            for j,long in enumerate(Long):

                #MLO data collection site:
                v_lat = 19.5362
                v_long = -155.5763
                R = 6371 #Radius of Earth in km

                #Get latitude and longitude for location being considered
                lat1 = v_lat * np.pi / 180
                long1 = v_long * np.pi / 180
                lat2 = lat * np.pi / 180
                long2 = long * np.pi / 180

                #Calc distance between SO2 collection site and volcano
                a_SO2 = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((long2-long1)/2)**2
                c_SO2 = 2 * np.arctan2(np.sqrt(a_SO2),np.sqrt(1-a_SO2))
                x_test['SO2_to_volc_Distance'][i*j+j] = R * c_SO2

                #Calculate bearing angle from volcano to SO2 data collection site
                X = np.cos(lat2) * np.sin(long1-long2)
                Y = np.cos(lat1) * np.sin(lat2) - np.sin((lat1) * np.cos(lat2) * np.cos(long1-long2))
                x_test['SO2_station_bearing_angle'][i*j+j] = -np.arctan2(X,Y) * 180 / np.pi
                if x_test['SO2_station_bearing_angle'][i*j+j] < 0: x_test['SO2_station_bearing_angle'][i*j+j] += 360

                #Calculate alignment between bearing angle to SO2 data collection site and wind direction
                v_SO2_Site = np.array([np.cos(x_test['SO2_station_bearing_angle'][i*j+j] * np.pi / 180), np.sin(x_test['SO2_station_bearing_angle'][i*j+j] * np.pi / 180)])
                v_volcano = np.array([np.cos(x_test['WIND DIRECTION'][i*j+j] * np.pi / 180), np.sin(x_test['WIND DIRECTION'][i*j+j] * np.pi / 180)])
                x_test['Wind Alignment'][i*j+j] = np.dot(v_SO2_Site.T,v_volcano)
                
        pred = model.predict(x_test)
        All_pred[:,n-start] = np.argmax(pred,axis=1)

    pred = np.argmax(All_pred, axis=1)
    x_axis = x_axis=np.tile(Long,(len(Lat),1))
    y_axis=np.tile(Lat,(len(Long),1))


    west, south, east, north = (
        -156,
        18.979,
        -154.9,
        20.3
                )

    df = pd.DataFrame(
        {'Latitude': [19.4069, 19.4721],
        'Longitude': [-155.2834, -155.5922]}
    )

    hawaii_img, hawaii_ext = cx.bounds2img(west,
                                     south,
                                     east,
                                     north,
                                     ll=True,
                                     source=cx.providers.Stamen.TerrainBackground
                                    )

    LatScaleF = 1.7372*10**7/156.061628                   
    LongScaleF = 2.144*10**6 / 18.9100782  

    f, ax = plt.subplots(1)
    ax.imshow(hawaii_img, extent=hawaii_ext)  

    plt.scatter(x_axis*LatScaleF,y_axis.T*LongScaleF,c=pred,cmap='Reds', alpha = 0.65)
    plt.show()

    return


def main(BiasVarStudy = False):
    """
    Function: main
        This function calls all previously defined methods.

    Parameters: BiasVarStudy - boolean value for whether to run a bias-variance study or
                               whether to run typical model training

    Returns:    none
    """   

    x_train, y_train, x_valid, y_valid, x_test, y_test = loadData()

    #Conduct Bias Variance Study?
    if BiasVarStudy:
        biasVar(x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        model, _ = trainModel(x_train, y_train, x_valid, y_valid)
        testModel(model, x_test, y_test)
        plotOutput(model, x_test)
    return

main(BiasVarStudy=False)