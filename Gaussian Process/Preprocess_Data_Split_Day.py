# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:51:31 2022

@author: raula
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import random

# Load SO2 Data
def loadSO2Data():
    SO2_datafiles = os.listdir('./SO2 Data')
    SO2data = dict()
    for filename in SO2_datafiles:
        SO2data[f'df_{filename}'] = pd.read_csv(f'./SO2 Data/{filename}')
    SO2_Data = pd.concat(SO2data)
    return SO2_Data

# Load EQ Data
def loadEQData():
    EQ_datafiles = os.listdir('./EQ Data')
    EQdata = dict()
    for filename in EQ_datafiles:
        EQdata[f'df_{filename}'] = pd.read_csv(f'./EQ Data/{filename}')
    EQ_Data = pd.concat(EQdata)
    return EQ_Data

# Convert Dates to Same Format and Remove Unnecessary Data
# Modified to be a code in case leap years exists
def cleanData(EQ_Data, SO2_Data):
    SO2_Dates = np.zeros(len(SO2_Data['Date']))
    for i, data in enumerate(SO2_Data['Date']):
        month = int(data[0:2])
        day = int(data[3:5])
        year = int(data[6:10])
        
        # Store dates as numbers
        # YYYY-MM-DD
        d = datetime.date(year, month, day)
        SO2_Dates[i] = d.toordinal()

    EQ_Dates = np.zeros(len(EQ_Data['time']))
    # EQ_Months = np.zeros(len(EQ_Data['time']))
    for i, data in enumerate(EQ_Data['time']):
        month = int(data[5:7])
        day = int(data[8:10])
        year = int(data[0:4])
        
        # Store dates as numbers
        # YYYY-MM-DD
        d = datetime.date(year, month, day)
        EQ_Dates[i] = d.toordinal()

    SO2_Data['Date'] = SO2_Dates
    EQ_Data['time'] = EQ_Dates

    EQ_relaventData = EQ_Data[['time','latitude','longitude','mag']]
    EQ_relaventData.rename(columns={'time' : 'Date'}, inplace=True)
    SO2_relaventData = SO2_Data[['Date','SITE_LATITUDE','SITE_LONGITUDE','Daily Max 1-hour SO2 Concentration']]
    
    # Remove Sensors out of Island
    SO2_relaventData = SO2_relaventData[SO2_relaventData['SITE_LONGITUDE'] > -157]  
    
    # Remove Negative SO2 levels
    SO2_relaventData = SO2_relaventData.drop(SO2_relaventData.index[SO2_relaventData['Daily Max 1-hour SO2 Concentration'] < 0])
    
    return EQ_relaventData, SO2_relaventData

# Merge SO2, Wind and EQ data into single dataframe and delete
# the first days without earthquakes since they dont matter to us
def mergeData(EQ_Data, SO2_Data, Wind_Data):
    # Merge S02 and EQ
    MergedData = SO2_Data.merge(EQ_Data, how = 'left', on = 'Date')
    MergedData.rename(columns={'SITE_LATITUDE' : 'SO2_lat', 'SITE_LONGITUDE' : 'SO2_long', 'latitude' : 'EQ_lat', 'longitude' : 'EQ_long'}, inplace=True) 
    
    # Find Wind Dataframe with shared dates to EQ and SO2
    Wind_temp = MergedData.merge(Wind_Data, how = 'left', on = 'Date')
    Wind_temp = Wind_temp[['Date', 'WIND DIRECTION', 'WIND SPEED', 'TEMPERATURE AT 2 METERS','RELATIVE HUMIDITY']]
    Wind_temp = Wind_temp.groupby(['Date'], as_index = False).mean()
    
    # Fill Missing Values
    Wind_temp.interpolate(method ='linear', limit_direction ='forward', inplace = True)
    
    MergedData = MergedData.merge(Wind_temp, how = 'left', on = 'Date')
    MergedData['mag'].fillna(0, inplace = True)
    
    # Delete the first rows where there was no previous earthquake
    MergedData = MergedData.sort_values(by = ['Date'])
    magnitudes = MergedData['mag'].to_numpy()
    first_non_zero = np.nonzero(magnitudes)[0][0]
    MergedData = MergedData.iloc[first_non_zero: , :]
    
    return MergedData

def loadFormatWindData():
    # Load Wind Data
    wind_df = pd.read_excel('Wind_Data.xlsx')
    
    return wind_df

def NormalizeData(LabeledData):
    
    # Latitude
    min_SO2_lat = LabeledData['SO2_lat'].min()
    min_EQ_lat = LabeledData['EQ_lat'].min()
    min_lat = min(min_SO2_lat, min_EQ_lat)
    
    max_SO2_lat = LabeledData['SO2_lat'].max()
    max_EQ_lat = LabeledData['EQ_lat'].max()
    max_lat = max(max_SO2_lat, max_EQ_lat)
    
    LabeledData['SO2_lat'] = (LabeledData['SO2_lat'] - min_lat) / (max_lat - min_lat)
    LabeledData['EQ_lat'] = (LabeledData['EQ_lat'] - min_lat) / (max_lat - min_lat)
    
    # Longitude
    min_SO2_long = LabeledData['SO2_long'].min()
    min_EQ_long = LabeledData['EQ_long'].min()
    min_long = min(min_SO2_long, min_EQ_long)
    
    max_SO2_long = LabeledData['SO2_long'].max()
    max_EQ_long = LabeledData['EQ_long'].max()
    max_long = max(max_SO2_long, max_EQ_long)
    
    LabeledData['SO2_long'] = (LabeledData['SO2_long'] - min_long) / (max_long - min_long)
    LabeledData['EQ_long'] = (LabeledData['EQ_long'] - min_long) / (max_long - min_long)
    
    # Wind Speed
    min_Wind_Speed = LabeledData['WIND SPEED'].min()
    max_Wind_Speed = LabeledData['WIND SPEED'].max()
    LabeledData['WIND SPEED'] = (LabeledData['WIND SPEED'] - min_Wind_Speed) / (max_Wind_Speed - min_Wind_Speed)

    # Wind Direction
    min_Wind_Dir = LabeledData['WIND DIRECTION'].min()
    max_Wind_Dir = LabeledData['WIND DIRECTION'].max()
    LabeledData['WIND DIRECTION'] = (LabeledData['WIND DIRECTION'] - min_Wind_Dir) / (max_Wind_Dir - min_Wind_Dir)

    # Relative Humidity
    min_Wind_RH = LabeledData['RELATIVE HUMIDITY'].min()
    max_Wind_RH = LabeledData['RELATIVE HUMIDITY'].max()
    LabeledData['RELATIVE HUMIDITY'] = (LabeledData['RELATIVE HUMIDITY'] - min_Wind_RH) / (max_Wind_RH - min_Wind_RH)

    # Wind Temperature
    min_Wind_Temp = LabeledData['TEMPERATURE AT 2 METERS'].min()
    max_Wind_Temp = LabeledData['TEMPERATURE AT 2 METERS'].max()
    LabeledData['TEMPERATURE AT 2 METERS'] = (LabeledData['TEMPERATURE AT 2 METERS'] - min_Wind_Temp) / (max_Wind_Temp - min_Wind_Temp)

    # SO2 Concentrations
    max_SO2 = LabeledData['Daily Max 1-hour SO2 Concentration'].max()
    LabeledData['Daily Max 1-hour SO2 Concentration'] = LabeledData['Daily Max 1-hour SO2 Concentration'] / max_SO2

    # Days since Last EQ
    min_Days_EQ = LabeledData['Days_Since_EQ_Activity'].min()
    max_Days_EQ = LabeledData['Days_Since_EQ_Activity'].max()
    LabeledData['Days_Since_EQ_Activity'] = (LabeledData['Days_Since_EQ_Activity'] - min_Days_EQ) / (max_Days_EQ - min_Days_EQ)
    
    # EQ Magnitude
    min_EQ_Mag = LabeledData['mag'].min()
    max_EQ_Mag = LabeledData['mag'].max()
    LabeledData['mag'] = (LabeledData['mag'] - min_EQ_Mag) / (max_EQ_Mag - min_EQ_Mag)
    
    # Replace NaN of EQ Coordinates to -1
    LabeledData['EQ_lat'] = LabeledData['EQ_lat'].fillna(-1)
    LabeledData['EQ_long'] = LabeledData['EQ_long'].fillna(-1)
    
    return LabeledData

def ClusterData(MergedData):
    
    # Calculate Days since Last EQ Event
    Compact_Data = MergedData.groupby(['Date'], as_index=False).mean()
    Date_idx = Compact_Data['Date'].to_numpy()
    Magnitudes = Compact_Data['mag'].to_numpy()
    Days_since_last_EQ = np.zeros(len(Date_idx))
    
    for i in range(len(Date_idx)):
        if Magnitudes[i] == 0:
            Days_since_last_EQ[i] = Days_since_last_EQ[i - 1] + 1

    Compact_Clustered_Data = pd.DataFrame({'Date': Date_idx, 'Days_Since_EQ_Activity': Days_since_last_EQ})
    Clustered_Data = MergedData.merge(Compact_Clustered_Data, how = 'left', on = 'Date')
    return Clustered_Data

def split_data(ClusteredData, train_ratio, test_ratio):
    
    # Randomely select trainning and testing cluster ids
    days_id = ClusteredData["Date"].unique()
    n_days = days_id.shape[0]
    n_train_days = int(n_days * train_ratio)
    n_test_days = int(n_days * test_ratio)
    trainning_days = np.random.choice(days_id, n_train_days, replace = False)
    test_valid_days = np.setdiff1d(days_id, trainning_days)
    testing_days = np.random.choice(test_valid_days, n_test_days, replace = False)
    validation_days = np.setdiff1d(test_valid_days, testing_days)
    
    # Obtain Trainning and Testing Data
    Trainning_Data = ClusteredData[ClusteredData["Date"].isin(test_valid_days) == False]
    
    trainning_valid_days = np.concatenate((trainning_days, validation_days), axis=0)
    Testing_Data = ClusteredData[ClusteredData["Date"].isin(trainning_valid_days) == False]
    
    trainning_testing_days = np.concatenate((trainning_days, testing_days), axis=0)
    Validation_Data = ClusteredData[ClusteredData["Date"].isin(trainning_testing_days) == False]
    
    return Trainning_Data, Validation_Data, Testing_Data


def AddRiskLabel(MergedData):
    
    # Green     0 - 100 ppb / Label = 0
    # Yellow  100 - 200 ppb / Label = 1
    # Orange  200 - 1000 ppb / Label = 2
    # Red    1000 - 3000 ppb / Label = 3
    # Purple 3000 - 5000 ppb / Label = 4
    # Maroon > 5000 ppb / Label = 5
    
    SO2_Labels = np.zeros(len(MergedData['Date']))
    SO2_Values = MergedData['Daily Max 1-hour SO2 Concentration'].to_numpy()
    
    for i in range(len(SO2_Labels)):
        if SO2_Values[i] > 5000:
            SO2_Labels[i] = 5
        elif SO2_Values[i] > 3000:
            SO2_Labels[i] = 4
        elif SO2_Values[i] > 1000:
            SO2_Labels[i] = 3
        elif SO2_Values[i] > 200:
            SO2_Labels[i] = 2
        elif SO2_Values[i] > 100:
            SO2_Labels[i] = 1
        else:
            SO2_Labels[i] = 0
        
    MergedData['Risk_Label'] = SO2_Labels
    
    return MergedData

def main():
    #Load Data
    SO2_Data = loadSO2Data()
    EQ_Data = loadEQData()
    Wind_Data = loadFormatWindData()

    #CleanData
    EQ_Data, SO2_Data = cleanData(EQ_Data, SO2_Data)
    EQ_Data.to_csv('EQ_Data_Clean.csv', index = False)
    SO2_Data.to_csv('SO2_Data_Clean.csv', index = False)
    
    # Fill Wind Missing Values and Merge Data
    MergedData = mergeData(EQ_Data, SO2_Data, Wind_Data)
    
    # Add Days since last EQ Cluster Activity
    ClusteredData = ClusterData(MergedData)
    
    # Add Risk Label
    LabeledData = AddRiskLabel(ClusteredData)
    LabeledData.to_csv('All_Data.csv', index = False)
    
    # Normalize and change NAN to -1
    NormalizedData = NormalizeData(LabeledData)
    NormalizedData.to_csv('All_Data_Normalized.csv', index = False)
    
    # Split Data on Training, Testing and Validation Based on Day
    # Training, Testing and Validation - 0.5, 0.25, 0.25
    Trainning_Data, Validation_Data, Testing_Data = split_data(LabeledData, 0.7, 0.15)
    Trainning_Data.to_csv('Trainning_Data.csv', index = False)
    Testing_Data.to_csv('Testing_Data.csv', index = False)
    Validation_Data.to_csv('Validation_Data.csv', index = False)
    
    return

main()