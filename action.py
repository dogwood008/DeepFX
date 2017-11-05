
# coding: utf-8

# In[ ]:


import enum


# In[ ]:


class Action(enum.Enum):
    SELL = -1; STAY = 0; BUY = +1


# In[ ]:


if __name__ == '__main__':
    print(Action.BUY)
    print(repr(Action.BUY))
    print(len(Action))


# In[ ]:




