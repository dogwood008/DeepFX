
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


class HistData:
    def __init__(self, csv_path=None, begin_date=None, end_date=None, range_limited=True, sep=';'):
        self.csv_path = csv_path
        with open(self.csv_path) as f:
            try:
                int(f.readline()[0])
                print('no header')
                header = None
                names = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')
            except:
                print('header is included')
                header = 0
                names = None
        self.csv_data = pd.read_csv(self.csv_path,
                                    names=names, header=header,
                                    index_col='Date', parse_dates=['Date'],
                                    dtype={'Date': 'str', 'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'},
                                    sep=sep)
        self.begin_date = begin_date
        self.end_date = end_date
        if range_limited:
            self.csv_data = self.data()
            if(len(self.csv_data) is 0):
                raise ValueError('Given csv in the range was empty.')
        
    def data(self):
        if self.begin_date is None and self.end_date is None:
            return self.csv_data
        else:
            is_in_date_array = (self.csv_data.index >= self.begin_date) &                                 (self.csv_data.index <= self.end_date)
            return self.csv_data.ix[is_in_date_array]
    
    def steps(self):
        return len(self.csv_data) - 1;

    def max_value(self):
        return self.csv_data.max()['Close']

    def min_value(self):
        return self.csv_data.min()['Close']

    def date_at(self, index):
        return self.csv_data.iloc[index].name
    
    def close_at(self, index):
        return self.csv_data.ix[[index], ['Close']].Close[0]
    
    def values_at(self, index):
        return self.csv_data.ix[[index], :]


# In[ ]:


class BitcoinHistData(HistData):        
    # https://www.kaggle.com/mczielinski/bitcoin-historical-data/data
    # coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv
    import datetime
    def __init__(self, csv_path=None, begin_date=None, end_date=None, range_limited=True, sep=','):
        self.csv_path = csv_path
        with open(self.csv_path) as f:
            try:
                int(f.readline()[0])
                print('no header')
                header = None
                names = ('Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)')
            except:
                print('header is included')
                header = 0
                names = None
        self.csv_data = pd.read_csv(self.csv_path,
                                    names=names, header=header,
                                    parse_dates=['Timestamp'],
                                    dtype={'Timestamp': 'int', 'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'},
                                    sep=sep)
        self.csv_data['Date'] =  pd.to_datetime(self.csv_data['Timestamp'], unit='s')
        self.csv_data.index = self.csv_data['Date']
        self.begin_date = begin_date
        self.end_date = end_date
        if range_limited:
            self.csv_data = self.data()
            if(len(self.csv_data) is 0):
                raise ValueError('Given csv in the range was empty.')


# In[ ]:


if __name__ == '__main__':
    import numpy as np
    begin = '2017-10-01T00:00:00'
    end = '2017-10-07T23:59:59'
    
    hd = BitcoinHistData(csv_path = 'historical_data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv',
                     begin_date=begin,
                     end_date=end)
    #print(hd.data())
    print(hd.max_value())
    print(hd.min_value())
    print(hd.date_at(1))
    print(hd.close_at(1))
    print(hd.values_at(1))

