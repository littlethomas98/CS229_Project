import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

#Convert Dates to Same Format and Remove Unnecessary Data
def cleanData(EQ_Data, SO2_Data):
    SO2_Dates = np.zeros(len(SO2_Data['Date']))
    for i, data in enumerate(SO2_Data['Date']):
        SO2_Dates[i] = int(data[0:2])*30 + int(data[3:5]) + (int(data[6:10])-2000)*365
        # SO2_Dates[i] = data[0:2] + data[3:5] + data[6:10]

    EQ_Dates = np.zeros(len(EQ_Data['time']))
    for i, data in enumerate(EQ_Data['time']):
        EQ_Dates[i] = int(data[5:7])*30 + int(data[8:10]) + (int(data[0:4])-2000)*365
        # EQ_Dates[i] = data[5:7] + data[8:10] + data[0:4]

    SO2_Data['Date'] = SO2_Dates
    EQ_Data['time'] = EQ_Dates

    EQ_relaventData = EQ_Data[['time','latitude','longitude','mag']]
    EQ_relaventData.rename(columns={'time' : 'Date'}, inplace=True)
    SO2_relaventData = SO2_Data[['Date','SITE_LATITUDE','SITE_LONGITUDE','Daily Max 1-hour SO2 Concentration']]

    return EQ_relaventData, SO2_relaventData

#Merge SO2 and EQ data into single dataframe 
def mergeData(EQ_Data, SO2_Data):

    MergedData = SO2_Data.merge(EQ_Data, how = 'left', on = 'Date')
    MergedData.rename(columns={'SITE_LATITUDE' : 'SO2_lat', 'SITE_LONGITUDE' : 'SO2_long', 'latitude' : 'EQ_lat', 'longitude' : 'EQ_long'}, inplace=True) 
    
    return MergedData

#Save relavent output data
def saveData(EQ_Data, SO2_Data, MergedData):
    EQ_Data.to_csv('EQ_Data.csv')
    SO2_Data.to_csv('SO2_Data.csv')
    MergedData.to_csv('MergedData.csv')
    return 

#Plot SO2 concentrations and EQ magnitudes
def plotData(MergedData):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('Days (Normalized to 2018)')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_title('Relationship Between SO2 Concentration and EQ Event')

    #Scale relative magnitude of SO2 concentration and EQ magnitude 
    #   Note that EQ scale is exponentially scaled, this is because the moment magnitude 
    #   scale used to evaluate eartquakes is logrithmic in nature.
    SO2Scale = 0.0015
    EQScale = -2
    DayAdjustment = 80

    plt.plot(MergedData['Date']-(np.min(MergedData['Date'])+DayAdjustment), MergedData['Daily Max 1-hour SO2 Concentration']*SO2Scale, 'o')
    plt.plot(MergedData['Date']-(np.min(MergedData['Date'])+DayAdjustment), MergedData['mag']**EQScale, 'o')

    plt.xlim([-10, 1600])
    plt.legend(['SO2 Conc.', 'EQ Mag.'])
    plt.savefig('SO2_vs_EQ.png')
    return

def main():
    #Load Data
    SO2_Data = loadSO2Data()
    EQ_Data = loadEQData()

    #Clean and Merge Data
    EQ_Data, SO2_Data = cleanData(EQ_Data, SO2_Data)
    MergedData = mergeData(EQ_Data, SO2_Data)

    #Save and Plot Data
    # saveData(EQ_Data, SO2_Data)
    # plotData(MergedData)

    return

main()