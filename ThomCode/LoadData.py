import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        # SO2_Dates[i] = int(data[0:2])*30 + int(data[3:5]) + (int(data[6:10])-2000)*365
        SO2_Dates[i] = data[0:2] + data[3:5] + data[6:10]

    EQ_Dates = np.zeros(len(EQ_Data['time']))
    for i, data in enumerate(EQ_Data['time']):
        # EQ_Dates[i] = int(data[5:7])*30 + int(data[8:10]) + (int(data[0:4])-2000)*365
        EQ_Dates[i] = data[5:7] + data[8:10] + data[0:4]

    SO2_Data['Date'] = SO2_Dates
    EQ_Data['time'] = EQ_Dates

    return EQ_Data, SO2_Data

#Add zeros to EQ data
def addZeros(EQ_Data, SO2_Data):
    EQ_Full = np.zeros([len(SO2_Data),2])
    EQ_Full = SO2_Data['Date']
    EQ = EQ_Data['time']
    
    
    SO2_Data.drop_duplicates(subset = "Date")
    SO2 = SO2_Data['Date']
    # EQ_Full[0:len(EQ_Data),0] = EQ_Data['time']
    # EQ_Full[0:len(EQ_Data),1] = EQ_Data['mag']
    
    # SO2_Full = np.zeros([len(SO2_Data),2])
    # SO2_Full[0:len(SO2_Data),0] = SO2_Data['Date']

    for el in EQ:
        ind = np.where(SO2 == el) 
        EQ_Full[ind] = el
    
    
    
    
    return 

#Save output data
def saveData(EQ_Data, SO2_Data):
    EQ_Data.to_csv('EQ_Data.csv')
    SO2_Data.to_csv('SO2_Data.csv')
    return 

#Plot SO2 concentrations and EQ magnitudes
def plotData(EQ_Data, SO2_Data):
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

    plt.plot(SO2_Data['Date']-(np.min(EQ_Data['time'])+DayAdjustment), SO2_Data['Daily Max 1-hour SO2 Concentration']*SO2Scale, 'o')
    plt.plot(EQ_Data['time']-(np.min(EQ_Data['time'])+DayAdjustment), EQ_Data['mag']**EQScale, 'o')
    plt.xlim([-10, 1600])
    plt.legend(['SO2 Conc.', 'EQ Mag.'])
    return

def main():
    SO2_Data = loadSO2Data()
    EQ_Data = loadEQData()
    convertDateFormat(EQ_Data, SO2_Data)
    addZeros(EQ_Data, SO2_Data)
    saveData(EQ_Data, SO2_Data)
    plotData(EQ_Data, SO2_Data)
    print("egg")
    return

main()