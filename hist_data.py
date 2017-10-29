
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


class HistData:
    def __init__(self, csv_path=None, begin_date=None, end_date=None, range_limited=True):
        self.csv_path = csv_path
        self.csv_data = pd.read_csv(self.csv_path, index_col=0, parse_dates=True, header=0)
        self.begin_date = begin_date
        self.end_date = end_date
        if range_limited:
            self.csv_data = self.data()
        
    def data(self):
        if self.begin_date is None and self.end_date is None:
            return self.csv_data
        else:
            is_in_date_array = (self.csv_data.index >= self.begin_date) &                                 (self.csv_data.index <= self.end_date)
            return self.csv_data.ix[is_in_date_array]

    def max_value(self):
        return self.data()[['High']].max()['High']

    def min_value(self):
        return self.data()[['Low']].min()['Low']

    def dates(self):
        return self.data().index.values

    ''' 引数の日時がデータフレームに含まれるか '''
    def has_datetime(self, datetime64_value):
        try:
            h.data().loc[datetime64_value]
            return True
        except KeyError:
            return False

    def _get_nearist_index(self, before_or_after, datetime):
        if before_or_after == 'before':
            offset = -1
        else:
            offset = 0
        index = max(h.data().index.searchsorted(datetime) + offset, 0)
        return self.data().ix[self.data().index[index]]

    ''' 引数の日時を含まない直前に存在する値を取得する '''        
    def get_last_exist_datetime(self, datetime64_value):
        return self._get_nearist_index('before', datetime64_value)
        
    ''' 引数の日時を含む直後に存在する値を取得する '''
    def get_next_exist_datetime(self, datetime64_value):
        return self._get_nearist_index('after', datetime64_value)
    
    ''' fromとtoの日時の差が閾値内にあるか否か '''
    def is_datetime_diff_in_threshould(self, from_datetime, to_datetime, threshold_timedelta):
        last_datetime = h.get_last_exist_datetime(from_datetime)
        next_exist_datetime = h.get_next_exist_datetime(to_datetime)
        delta = next_exist_datetime.name - last_datetime.name
        return delta <= threshold_timedelta


# In[ ]:


if __name__ == '__main__':
    begin = '2010-09-01T00:00:00'
    end = '2010-09-07T23:59:59'
    hd = HistData(begin_date=begin, end_date=end)
    print(hd.data())

