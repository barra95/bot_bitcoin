# Code to transform the data

#Data viz
import plotly.graph_objs as go

import numpy as np
import pandas as pd

class Data:
    
    def __init__(self,trans=True):
        self.trans = trans

    def target_train(self,dataset):
        ''' Clean and Transform Data'''
        dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()
        dataset=dataset.reset_index(drop=True)

        # Create short simple moving average over the short window
        dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()

        # Create long simple moving average over the long window
        dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()

        # Create signals or target
        dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
        return dataset

    #calculation of exponential moving average
    def EMA(self, df, n):
        EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
        return EMA

    #calculation of rate of change
    def ROC(self,df, n):  
        M = df.diff(n - 1)  
        N = df.shift(n - 1)  
        ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))   
        return ROC

    #Calculation of price momentum
    def MOM(self,df, n):   
        MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))   
        return MOM

    #calculation of relative strength index
    def RSI(self,series, period):
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])
        rs = u.ewm(com=period-1, adjust=False).mean() / \
        d.ewm(com=period-1, adjust=False).mean()
        return 100 - 100 / (1 + rs)

    #calculation of stochastic osillator.

    def STOK(self,close, low, high, n): 
        STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
        return STOK

    def STOD(self,close, low, high, n):
        STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
        STOD = STOK.rolling(3).mean()
        return STOD

    #Calculation of moving average
    def MA(self,df, n):
        MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        return MA

    

    

    


        


        

