import torch
import pandas as pd
from __future__ import print_function
from itertools import count

import torch.nn.functional as F





def importData():
    data = pd.read_csv('MergedData.csv')
    return data


