#Code to get the data

#Data Source
import yfinance as yf
    
def get_data_train():
    ''' Function to get the data for train'''
    data = yf.download(tickers='BTC-USD',period = '60d', interval = '5m')
    return data

def get_data_process():
    ''' Function to get the data for train'''
    data = yf.download(tickers='BTC-USD',period = '2d', interval = '5m')
    return data
    