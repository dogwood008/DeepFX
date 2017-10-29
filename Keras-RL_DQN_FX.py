
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[ ]:


import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, DEBUG, INFO

from hist_data import HistData
from fx_trade import FXTrade
from deep_fx import DeepFX


# In[ ]:


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


# In[ ]:


h = HistData('2010/9')


# In[ ]:


env = FXTrade(1000000, 0.08, h, logger=logger)
#env = FXTrade(1000000, 0.08, h, logger=logger)
prepared_model_filename = None #'Keras-RL_DQN_FX_model_meanq1.440944e+06_episode00003.h5'
dfx = DeepFX(env, 'test', prepared_model_filename=prepared_model_filename)


# In[ ]:


is_to_train = True
if is_to_train:
    dfx.train(is_for_time_measurement=True)
else:
    dfx.test(1, [EpisodeLogger()])


# In[ ]:


get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
data = h.data()['2010-09']['Close']
x = data.index
y  = data.values
plt.plot(x, y)


# In[ ]:


data.values


# In[ ]:




