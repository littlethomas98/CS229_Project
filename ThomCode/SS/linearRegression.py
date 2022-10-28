import torch
import pandas as pd

def inportData():
    EQ_Data = pd.read_csv('EQ_Data.csv')
    SO2_Data = pd.read_csv('SO2_Data.csv')
    
    return EQ_Data, SO2_Data

def mse(y, y_hat):
    delta = y - y_hat
    return torch.sum(delta * delta) / delta.numel()

def linearRegression(EQ_Data, SO2_Data):
    
    return 

