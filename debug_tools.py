
# coding: utf-8

# In[ ]:


import datetime as dt


# In[ ]:


class DebugTools:
    def now():
        return dt.datetime.now()
    def now_str():    
        return DebugTools.now().strftime('%y/%m/%d %H:%M:%S')
    def now_12():
        return DebugTools.now().strftime('%y%m%d_%H%M%S')


# In[ ]:


if __name__ == '__main__':
    print(DebugTools.now_12())

