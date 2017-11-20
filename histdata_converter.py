
# coding: utf-8

# In[ ]:


# histdata.comでDLした1分足のデータを任意の足に変換する

# http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/usdjpy/2017/10


# In[ ]:


import pandas as pd
import numpy as np
from hist_data import HistData


# In[ ]:


def get_new_index(old_dataframe, freq='5min'):
    start = hd.data()[0:1].index[0]
    end = hd.data()[-1:].index[0]
    new_index = pd.date_range(start, end, None, freq)
    return new_index

def create_dataframe(dataarray):
    new_df = pd.DataFrame.from_records(dataarray,                          index=['Date'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return new_df

def create_new_dataarray(old_dataframe, new_index, i):
    start = new_index[i].to_pydatetime()
    end = new_index[i+1].to_pydatetime()
    slice = old_dataframe.loc[start:end][:-1]
    if len(slice) is 0:
        return
    open = slice.ix[0:1]['Open'][0]
    high = max(slice['High'])
    low = min(slice['Low'])
    close = slice.ix[-1:]['Close'][0]
    volume = slice.sum()['Volume']
    return np.array([start, open, high, low, close, volume])

def create_new_dataframe(old_dataframe, freq='5min'):
    new_index = get_new_index(old_dataframe, freq)
    datalist = [create_new_dataarray(old_dataframe, new_index, i) for i in range(len(new_index) - 1)]
    none_removed_array = np.array([x for x in datalist if x is not None])
    new_df = create_dataframe(none_removed_array)
    return new_df


# In[ ]:


read_filepath = 'historical_data/DAT_ASCII_USDJPY_M1_201710.csv'
write_filepath = 'historical_data/DAT_ASCII_USDJPY_M1_201710_h1.csv'
hd = HistData(read_filepath)
new_df = create_new_dataframe(hd.data(), freq='1h')
new_df.to_csv(write_filepath, sep=';', header=['Open', 'High', 'Low', 'Close', 'Volume'])

