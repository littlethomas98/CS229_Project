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

def NormalizeLatLongData(MergedData):
    
    # Latitude
    min_SO2_lat = MergedData['SO2_lat'].min()
    min_EQ_lat = MergedData['EQ_lat'].min()
    min_lat = min(min_SO2_lat, min_EQ_lat)
    
    max_SO2_lat = MergedData['SO2_lat'].max()
    max_EQ_lat = MergedData['EQ_lat'].max()
    max_lat = max(max_SO2_lat, max_EQ_lat)
    
    MergedData['SO2_lat'] = (MergedData['SO2_lat'] - min_lat) / (max_lat - min_lat)
    MergedData['EQ_lat'] = (MergedData['EQ_lat'] - min_lat) / (max_lat - min_lat)
    
    # Longitude
    min_SO2_long = MergedData['SO2_long'].min()
    min_EQ_long = MergedData['EQ_long'].min()
    min_long = min(min_SO2_long, min_EQ_long)
    
    max_SO2_long = MergedData['SO2_long'].max()
    max_EQ_long = MergedData['EQ_long'].max()
    max_long = max(max_SO2_long, max_EQ_long)
    
    MergedData['SO2_long'] = (MergedData['SO2_long'] - min_long) / (max_long - min_long)
    MergedData['EQ_long'] = (MergedData['EQ_long'] - min_long) / (max_long - min_long)
    
    return MergedData

def ClusterData(NormalizedData):
    
    # Calculate Days since Last EQ Event
    Compact_Data = NormalizedData.groupby(['Date'], as_index=False).mean()
    Date_idx = Compact_Data['Date'].to_numpy()
    Magnitudes = Compact_Data['mag'].to_numpy()
    Days_since_last_EQ = np.zeros(len(Date_idx))
    
    for i in range(len(Date_idx)):
        if Magnitudes[i] == 0:
            Days_since_last_EQ[i] = Days_since_last_EQ[i - 1] + 1
            
    # Cluster data based on EQ-Activity vs No Activity repeating patterns
    Clusters = np.ones(len(Date_idx))
    cluster_id = 1
    for i in range(1,len(Date_idx)):
        if Days_since_last_EQ[i] == 0 and Magnitudes[i - 1] == 0:
            cluster_id += 1 
            
        Clusters[i] = cluster_id
    
    Compact_Clustered_Data = pd.DataFrame({'Date': Date_idx, 'Days_Since_EQ_Activity': Days_since_last_EQ, 'Cluster_ID': Clusters})
    Clustered_Data = NormalizedData.merge(Compact_Clustered_Data, how = 'left', on = 'Date')
    return Clustered_Data

def split_data(ClusteredData, train_ratio, test_ratio):
    
    # Randomely select trainning and testing cluster ids
    clusters_id = ClusteredData["Cluster_ID"].unique()
    n_clusters = clusters_id.shape[0]
    n_train_clusters = int(n_clusters*train_ratio)
    n_test_clusters = int(n_clusters*test_ratio)
    trainning_clusters = np.random.choice(clusters_id, n_train_clusters, replace = False)
    test_valid_clusters = np.setdiff1d(clusters_id, trainning_clusters)
    testing_clusters = np.random.choice(test_valid_clusters, n_test_clusters, replace = False)
    validation_clusters = np.setdiff1d(test_valid_clusters, testing_clusters)
    
    # Obtain Trainning and Testing Data
    Trainning_Data = ClusteredData[ClusteredData["Cluster_ID"].isin(test_valid_clusters) == False]
    
    trainning_valid_clusters = np.concatenate((trainning_clusters, validation_clusters), axis=0)
    Testing_Data = ClusteredData[ClusteredData["Cluster_ID"].isin(trainning_valid_clusters) == False]
    
    trainning_testing_clusters = np.concatenate((trainning_clusters, testing_clusters), axis=0)
    Validation_Data = ClusteredData[ClusteredData["Cluster_ID"].isin(trainning_testing_clusters) == False]
    
    return Trainning_Data, Testing_Data, Validation_Data

def AddRiskLabel(ClusteredData):
    
    # Green     0 - 100 ppb / Label = 0
    # Yellow  100 - 200 ppb / Label = 1
    # Orange  200 - 1000 ppb / Label = 2
    # Red    1000 - 3000 ppb / Label = 3
    # Purple 3000 - 5000 ppb / Label = 4
    # Maroon > 5000 ppb / Label = 5
    
    SO2_Labels = np.zeros(len(ClusteredData['Date']))
    SO2_Values = ClusteredData['Daily Max 1-hour SO2 Concentration'].to_numpy()
    
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
        
    ClusteredData['Risk_Label'] = SO2_Labels
    
    return ClusteredData

def AddMonth(NormalizedData):
    
    NormalizedData.sort_values(by = ['Date'])
    NormalizedData.reset_index(drop = True, inplace = True)
    
    Months = np.zeros(len(NormalizedData['Date']))
    for i, data in enumerate(NormalizedData['Date']):
        date = datetime.date.fromordinal(int(data))
        month = date.month
        Months[i] = month
    
    # Add month columns
    NormalizedData['Month'] = Months
    
    return NormalizedData

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
    # Normalize
    NormalizedData = NormalizeLatLongData(MergedData)
    # Add Month
    AddMonthData = AddMonth(NormalizedData)
    
    # Cluster data based on Earthquake Activity - No Earthquake Activity
    ClusteredData = ClusterData(AddMonthData)
    
    # Add Risk Label
    LabeledData = AddRiskLabel(ClusteredData)
    LabeledData.to_csv('All_Data.csv', index = False)
    
    # Split Data on Training, Testing and Validation Based on Clusters
    # Training, Testing and Validation - 0.5, 0.25, 0.25
    Trainning_Data, Testing_Data, Validation_Data = split_data(LabeledData, 0.5, 0.25)
    Trainning_Data.to_csv('Trainning_Data.csv', index = False)
    Testing_Data.to_csv('Testing_Data.csv', index = False)
    Validation_Data.to_csv('Validation_Data.csv', index = False)
    
    return

main()