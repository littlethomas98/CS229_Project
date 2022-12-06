import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def loadData():
    """
    Function: loadEQData
        This function loads all earthquake realizations from the NGA West2 database files and
        combines them into a single csv file for ease of access in later methods.

    Parameters: TODO

    Returns:    TODO
    """
    RainData = pd.read_csv('Weather Data/2018 Tempperature-Humidity.csv')
    EQData = pd.read_csv('EQ Data/2018 EQData.csv')

    return EQ_Data, GMM_RS

def trainModel(x_train, y_train, x_valid, y_valid):
    dtrain = xgb.DMatrix(x_train, label = y_train)
    dvalid = xgb.DMatrix(x_valid, label = y_valid)

    param = {'objective':'reg:squarederror', 
             'eval_metric': 'mape', 
             'seed':2,
             'min_child_weight': 0.1, 
             'max_depth': 5, 
             'learning_rate': 0.1, 
             'gamma': 0.1}

    evallist =[(dtrain, 'train'), (dvalid, 'eval')]

    num_round = 2
    model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds = 100)

    return model
    
def testModel(model, x_test, y_test):
    dtest = xgb.DMatrix(x_test)
    ypred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    #Plot Feature Importance
    xgb.plot_importance(model)
    plt.rcParams.update({'font.sans-serif':'Times New Roman'})
    plt.savefig('Feature Importance')
    return ypred