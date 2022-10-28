# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:05:18 2022

@author: THom
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

EQ_Data = pd.read_csv('EQ_Data.csv')
SO2_Data = pd.read_csv('SO2_Data.csv')

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel('Days (Normalized to 2018)')
ax1.set_ylabel('Relative Magnitude')
ax1.set_title('Relationship Between SO2 Concentration and EQ Event')

#Scale relative magnitude of SO2 concentration and EQ magnitude 
#Note that EQ scale is exponentially scaled, this is due to the natrual 
#phenomenon of earthquake
SO2Scale = 0.0015
EQScale = -2
DayAdjustment = 80

plt.plot(SO2_Data['Date']-(np.min(EQ_Data['time'])+DayAdjustment), SO2_Data['Daily Max 1-hour SO2 Concentration']*SO2Scale, 'o')
plt.plot(EQ_Data['time']-(np.min(EQ_Data['time'])+DayAdjustment), EQ_Data['mag']**EQScale, 'o')
plt.xlim([-10, 1600])
plt.legend(['SO2 Conc.', 'EQ Mag.'])
