import numpy as np
import pandas as pd
import os
# import tensorflow as tf

# Load SO2 Data
SO2_datafiles = os.listdir('SO2 Data')
SO2data = dict()
for filename in SO2_datafiles:
    SO2data[f'df_{filename}'] = pd.read_csv(f'SO2 Data/{filename}')
SO2_Data = pd.concat(SO2data)

# Load EQ Data
EQ_datafiles = os.listdir('EQ Data')
EQdata = dict()
for filename in EQ_datafiles:
    EQdata[f'df_{filename}'] = pd.read_csv(f'EQ Data/{filename}')
EQ_Data = pd.concat(EQdata)

#Convert Dates to Same Format
SO2_Dates = np.zeros(len(SO2_Data['Date']))
for i, data in enumerate(SO2_Data['Date']):
    SO2_Dates[i] = data[0:2] + data[3:5] + data[6:10]

EQ_Dates = np.zeros(len(EQ_Data['time']))
for i, data in enumerate(EQ_Data['time']):
    EQ_Dates[i] = data[5:7] + data[8:10] + data[0:4]

SO2_Data['Date'] = SO2_Dates
EQ_Data['time'] = EQ_Dates

minLengthDataSet = min(len(EQ_Data['time']), len(SO2_Data['time']))
if len(EQ_Data['time']) < len(SO2_Data['time']):
    DataShort = EQ_Data['time']
    DataLong = SO2_Data['time']
else: 
    DataShort = SO2_Data['time']
    DataLong = EQ_Data['time']

for i in range(len(DataShort)):
    

print("egg1")


