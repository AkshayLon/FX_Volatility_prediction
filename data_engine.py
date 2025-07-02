import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class DataSet:

    def __init__(self, datafile):
        self.data = pd.read_csv(datafile, encoding='utf-8', float_precision='high')
        self.data['LogPrice'] = np.log(self.data['Close'])
        self.data['LogReturns'] = self.data['LogPrice'].diff().fillna(0)
        self.data['DemeanedLogReturns'] = self.data['LogReturns'] - self.data['LogReturns'].mean()
        self.data['ReturnsSquared'] = np.square(self.data['DemeanedLogReturns'])
        self.data['RealisedVolatility'] = self.data['ReturnsSquared'].rolling(window=10).mean().fillna(0)
        self.data['NormalisedVolatility'] = (1/self.data['RealisedVolatility'].std())*(self.data['RealisedVolatility'] - self.data['RealisedVolatility'].mean())
        
        self.training_data = sliding_window_view(self.data['NormalisedVolatility'].values, 50)[:-1:10, :]
        self.test_data = sliding_window_view(self.data['NormalisedVolatility'].values, 10)[50::10, :]

    def __len__(self):
        return self.training_data.shape[0]
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        return self.training_data[index], self.test_data[index]

    