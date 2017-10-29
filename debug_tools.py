
# coding: utf-8

# In[ ]:


import datetime as dt


# In[ ]:


class DebugTools:
    def now():
        return dt.datetime.now() + dt.timedelta(hours=9)
    def now_str():    
        return DebugTools.now().strftime('%y/%m/%d %H:%M:%S')

