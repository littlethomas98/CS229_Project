import numpy as np
import pandas as pd
import os
# import tensorflow as tf

# Load SO2 Data
def loadSO2Data():
    SO2_datafiles = os.listdir('../SO2 Data')
    SO2data = dict()
    for filename in SO2_datafiles:
        SO2data[f'df_{filename}'] = pd.read_csv(f'../SO2 Data/{filename}')
    SO2_Data = pd.concat(SO2data)
    return SO2_Data

# Load EQ Data
def loadEQData():
    EQ_datafiles = os.listdir('../EQ Data')
    EQdata = dict()
    for filename in EQ_datafiles:
        EQdata[f'df_{filename}'] = pd.read_csv(f'../EQ Data/{filename}')
    EQ_Data = pd.concat(EQdata)
    return EQ_Data

#Convert Dates to Same Format
def convertDateFormat(EQ_Data, SO2_Data):
    SO2_Dates = np.zeros(len(SO2_Data['Date']))
    for i, data in enumerate(SO2_Data['Date']):
        SO2_Dates[i] = data[0:2] + data[3:5] + data[6:10]

    EQ_Dates = np.zeros(len(EQ_Data['time']))
    for i, data in enumerate(EQ_Data['time']):
        EQ_Dates[i] = data[5:7] + data[8:10] + data[0:4]

    SO2_Data['Date'] = SO2_Dates
    EQ_Data['time'] = EQ_Dates

    return EQ_Data, SO2_Data

#Pull only data corresponding to EQ events
def pullOnlyEventDates(EQ_Data, SO2_Data):
    minLengthDataSet = min(len(EQ_Data['time']), len(SO2_Data['Date']))
    if len(EQ_Data['time']) < len(SO2_Data['Date']):
        DataShort = EQ_Data['time']
        DataLong = SO2_Data['Date']
    else: 
        DataShort = SO2_Data['time']
        DataLong = EQ_Data['time']

    for i in range(len(DataShort)):
        

        print("egg1")
    return 

def plotData(EQ_Data, SO2_Data):
    
    return

def main():
    SO2_Data = loadSO2Data()
    EQ_Data = loadEQData()
    convertDateFormat(EQ_Data, SO2_Data)
    # pullOnlyEventDates(EQ_Data, SO2_Data)
    print("egg")
    return
