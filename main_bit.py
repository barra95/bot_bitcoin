# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from dateutil.relativedelta import relativedelta
import datetime
from time import time, sleep


from get_transform_data import Data
from get_data import get_data_process, get_data_train
from train_model_bit import Train_Model

def main():

    
    semanas = []
    acciones = []
    while True:
        sleep(60 - time() % 60)


        today_date = datetime.datetime.today().isocalendar()
        week = today_date.week
        semanas.append(week)
        if week == semanas[0]:
            semanas[0] = week
            #Get Data
            data = get_data_process()
            #Get Y
            data = Data().target_train(data)

            #Create variables
            data['EMA10'] = Data().EMA(data, 10)
            data['EMA30'] = Data().EMA(data, 30)
            data['EMA200'] = Data().EMA(data, 200)
            data['ROC10'] = Data().ROC(data['Close'], 10)
            data['ROC30'] = Data().ROC(data['Close'], 30)
            data['MOM10'] = Data().MOM(data['Close'], 10)
            data['MOM30'] = Data().MOM(data['Close'], 30)
            data['RSI10'] = Data().RSI(data['Close'], 10)
            data['RSI30'] = Data().RSI(data['Close'], 30)
            data['RSI200'] = Data().RSI(data['Close'], 200)
            data['%K10'] = Data().STOK(data['Close'], data['Low'], data['High'], 10)
            data['%D10'] = Data().STOD(data['Close'], data['Low'], data['High'], 10)
            data['%K30'] = Data().STOK(data['Close'], data['Low'], data['High'], 30)
            data['%D30'] = Data().STOD(data['Close'], data['Low'], data['High'], 30)
            data['%K200'] = Data().STOK(data['Close'], data['Low'], data['High'], 200)
            data['%D200'] = Data().STOD(data['Close'], data['Low'], data['High'], 200)
            data['MA21'] = Data().MA(data, 10)
            data['MA63'] = Data().MA(data, 30)
            data['MA252'] = Data().MA(data, 200)
            
            #Droping innecesary features
            data=data.drop(['High','Low','Open','signal'], axis=1)

            momentum = data.iloc[-1:]

            #Load Model
            model = Train_Model().load_model('bitcoin_rf_model.pkl')
            
            #Predict best action
            action = model.predict(momentum)
            acciones.append(action[0])

            if len(acciones) < 2:
                if action[0] == 1:
                    print('First action: Buy')
                    print('Price: ', momentum['Close'].iloc[0])
                else:
                    print('First action: Don´t Buy')
                    print('Price: ', momentum['Close'].iloc[0])
            else:
                if acciones[0] !=  action[0]:
                    if action[0] == 1:
                        print('Buy')
                        print('Price: ', momentum['Close'].iloc[0])
                    else:
                        print('Sell')
                        print('Price: ', momentum['Close'].iloc[0])
                else:
                    if action[0] == 1:
                        print(action[0])
                        print('Hold')
                        print('Price: ', momentum['Close'].iloc[0])
                    else:
                        print(action[0])
                        print('Don´t Buy')
                        print('Price: ', momentum['Close'].iloc[0])

            acciones[0] = action[0]
            print('Wait 1 min')

        else:
            #Get Data
            data = get_data_train()
            
            #Get Y
            data = Data().target_train(data)

            #Create variables
            data['EMA10'] = Data().EMA(data, 10)
            data['EMA30'] = Data().EMA(data, 30)
            data['EMA200'] = Data().EMA(data, 200)
            data['ROC10'] = Data().ROC(data['Close'], 10)
            data['ROC30'] = Data().ROC(data['Close'], 30)
            data['MOM10'] = Data().MOM(data['Close'], 10)
            data['MOM30'] = Data().MOM(data['Close'], 30)
            data['RSI10'] = Data().RSI(data['Close'], 10)
            data['RSI30'] = Data().RSI(data['Close'], 30)
            data['RSI200'] = Data().RSI(data['Close'], 200)
            data['%K10'] = Data().STOK(data['Close'], data['Low'], data['High'], 10)
            data['%D10'] = Data().STOD(data['Close'], data['Low'], data['High'], 10)
            data['%K30'] = Data().STOK(data['Close'], data['Low'], data['High'], 30)
            data['%D30'] = Data().STOD(data['Close'], data['Low'], data['High'], 30)
            data['%K200'] = Data().STOK(data['Close'], data['Low'], data['High'], 200)
            data['%D200'] = Data().STOD(data['Close'], data['Low'], data['High'], 200)
            data['MA21'] = Data().MA(data, 10)
            data['MA63'] = Data().MA(data, 30)
            data['MA252'] = Data().MA(data, 200)
            
            #Droping innecesary features
            data=data.drop(['High','Low','Open'], axis=1)
            data = data.dropna(axis=0)

            #Train Test Split
            Y_train, X_train, Y_validation, X_validation = Train_Model().get_train_val(data)

            #Training Model
            model = Train_Model().model_training(Y_train, X_train, Y_validation, X_validation)

            #Print results
            Train_Model().print_results(model, Y_validation, X_validation)

            #Saving Model
            Train_Model().saving_model(model)

if __name__ == '__main__':
    main()




