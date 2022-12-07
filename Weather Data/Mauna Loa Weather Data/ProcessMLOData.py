import pandas as pd
import numpy as np
import datetime
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##########################

##Dataset from: https://gml.noaa.gov/aftp/met/mlo/

##IMPORT DATA
datafiles = os.listdir('Weather Data/Mauna Loa Weather Data/By Year')
data = dict()
for filename in datafiles:
    if filename != 'weather_text_to_csv.py' and filename != 'SS':
        data[f'df_{filename}'] = pd.read_csv(f'Weather Data/Mauna Loa Weather Data/By Year/{filename}')
data = pd.concat(data)
data.rename(columns={'Column1': 'SITE CODE',
                     'Column2': 'YEAR',
                     'Column3': 'MONTH',
                     'Column4': 'DAY',
                     'Column5': 'HOUR', 
                     'Column6': 'MINUTE',
                     'Column7': 'WIND DIRECTION', #North is 0, east is 90, south is 180 and west is 270 (missing vals are -999)
                     'Column8': 'WIND SPEED', #m/s (missing vals are -999.9)
                     'Column9': 'WIND STEADINESS FACTOR', #100 times the ratio of the vector wind speed to the average wind speed for the hour (missing vals are -9)
                     'Column10': 'BAROMETRIC PRESSURE', #hPa (missing vals are -999.90)
                     'Column11': 'TEMPERATURE AT 2 METERS', #Celsius (missing vals are -999.9)
                     'Column12': 'TEMPERATURE AT 10 METERS', #Celsius (missing vals are -999.9)
                     'Column13': 'TEMPERATURE AT TOWER TOP', #Celsius (missing vals are -999.9)
                     'Column14': 'RELATIVE HUMIDITY', #Percent (missing vals are -99)
                     'Column15': 'PRECIPITATION INTENSITY', #Amount of rainfall per hour (mm / hour) (missing vals are -99)
                     }, inplace=True)


##Convert Day/Month/Year to standard format for comparison with other datasets
Combined_Dates = np.zeros(data.shape[0])
for i in range (data.shape[0]):
    month = data['MONTH'][i]
    day = data['DAY'][i]
    year = data['YEAR'][i]
    
    # Store dates as numbers
    # YYYY-MM-DD
    d = datetime.date(year, month, day)
    Combined_Dates[i] = d.toordinal()
data['Date'] = Combined_Dates

##Remove all rows with missing values
data['WIND DIRECTION'] = data['WIND DIRECTION'].replace(-999,np.nan)
data['WIND SPEED'] = data['WIND SPEED'].replace(-999.9,np.nan)
data['BAROMETRIC PRESSURE'] = data['BAROMETRIC PRESSURE'].replace(-999.90,np.nan)
data['TEMPERATURE AT 2 METERS'] = data['TEMPERATURE AT 2 METERS'].replace(-999.9,np.nan)
data['TEMPERATURE AT 10 METERS'] = data['TEMPERATURE AT 10 METERS'].replace(-999.9,np.nan)
data['TEMPERATURE AT TOWER TOP'] = data['TEMPERATURE AT TOWER TOP'].replace(-999.9,np.nan)
data['RELATIVE HUMIDITY'] = data['RELATIVE HUMIDITY'].replace(-99,np.nan)
data['PRECIPITATION INTENSITY'] = data['PRECIPITATION INTENSITY'].replace(-99,np.nan)
data.dropna(axis = 'rows', inplace=True)

##Take daily mean of all values
data.drop(columns = ['YEAR','MONTH','DAY','HOUR','MINUTE','WIND STEADINESS FACTOR'], inplace=True)
data = data.groupby('Date').mean().reset_index()
data.to_csv('Weather Data/MLO Data.csv')