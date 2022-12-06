import pandas as pd
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##########################

##Dataset from: https://gml.noaa.gov/aftp/met/mlo/

##IMPORT DATA
datafiles = os.listdir('Weather Data/Mauna Loa Weather Data')
data = dict()
for filename in datafiles:
    if filename != 'SS' and filename != 'ProcessMLOData.py':
        data[f'df_{filename}'] = pd.read_csv(f'Weather Data/Mauna Loa Weather Data/{filename}')
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


##TAKE DAILY AVERAGES FOR COMPARISON TO OTHER DATA
data['Date'] = data['MONTH'].astype(str) + data['YEAR'].astype(str) + data['DAY'].astype(str) #Weird order to remove duplicates (118 could be jan 18 or nov 8)
data = data.groupby('Date').mean().reset_index()
data = data.drop(columns = ['WIND STEADINESS FACTOR'])
data.to_csv('Weather Data/MLO Data.csv')