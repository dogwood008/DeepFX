
# coding: utf-8

# In[ ]:


# histdata.comでDLした1分足のデータを任意の足に変換する

# http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/usdjpy/2017/10


# In[ ]:


import pandas as pd
import numpy as np
from hist_data import HistData, BitcoinHistData


# In[ ]:


def get_new_index(old_dataframe, freq='5min'):
    start = hd.data()[0:1].index[0]
    end = hd.data()[-1:].index[0]
    new_index = pd.date_range(start, end, None, freq)
    return new_index

def create_dataframe(dataarray):
    new_df = pd.DataFrame.from_records(dataarray,                          index=['Date'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return new_df

def create_new_dataarray(hist_data, new_index, i):
    old_dataframe = hist_data.data()
    start = new_index[i].to_pydatetime()
    end = new_index[i+1].to_pydatetime()
    slice = old_dataframe.loc[start:end][:-1]
    if len(slice) is 0:
        return
    open = slice.ix[0:1]['Open'][0]
    high = max(slice['High'])
    low = min(slice['Low'])
    close = slice.ix[-1:]['Close'][0]
    if type(hist_data) == HistData:
        volume = slice.sum()['Volume']
    elif type(hist_data) == BitcoinHistData:
        volume = slice.sum()['Volume_(BTC)']
    return np.array([start, open, high, low, close, volume])

def create_new_dataframe(hist_data, freq='5min'):
    old_dataframe = hist_data.data()
    new_index = get_new_index(old_dataframe, freq)
    datalist = [create_new_dataarray(hist_data, new_index, i) for i in range(len(new_index) - 1)]
    none_removed_array = np.array([x for x in datalist if x is not None])
    new_df = create_dataframe(none_removed_array)
    return new_df


# In[ ]:


if False:
    read_filepath = 'historical_data/DAT_ASCII_USDJPY_M1_201710.csv'
    write_filepath = 'historical_data/DAT_ASCII_USDJPY_M1_201710_h1.csv'
    hd = HistData(read_filepath)
    new_df = create_new_dataframe(hd, freq='1h')
    new_df.to_csv(write_filepath, sep=';', header=['Open', 'High', 'Low', 'Close', 'Volume'])


# In[ ]:


if True:
    read_filepath = 'historical_data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv'
    write_filepath = 'historical_data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20_h1.csv'
    hd = BitcoinHistData(read_filepath)
    new_df = create_new_dataframe(hd, freq='1h')
    new_df.to_csv(write_filepath, sep=';', header=['Open', 'High', 'Low', 'Close', 'Volume'])

