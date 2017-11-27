
# coding: utf-8

# In[ ]:


import enum


# In[ ]:


class Action(enum.Enum):
    SELL = 0; STAY = 1; BUY = 2


# In[ ]:


if __name__ == '__main__':
    print(Action.BUY)
    print(repr(Action.BUY))
    print(len(Action))

