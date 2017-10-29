
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[ ]:


import numpy as np
import pandas as pd
import talib
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


#import imp
#import sys
#del(hist_data)
#from hist_data import HistData
#del(hist_data)
#imp.reload(hist_data)
#imp.reload(sys.modules[hist_data.__module__])
hd = HistData(csv_path = 'historical_data/USDJPY.hst_.csv',
                     begin_date='2010-09-01T00:00:00',
                     end_date='2010-09-07T23:59:59')


# In[ ]:


hd.data()
#len(hist_data.data())


# In[ ]:


env = FXTrade(1000000, 0.08, hd, logger=logger)
#env = FXTrade(1000000, 0.08, h, logger=logger)
prepared_model_filename = None #'Keras-RL_DQN_FX_model_meanq1.440944e+06_episode00003.h5'
dfx = DeepFX(env, 'test', prepared_model_filename=prepared_model_filename)


# In[ ]:


#import imp
#import sys
#import deep_fx
#from deep_fx import DeepFX
##del(hist_data)
##from deep
##del(hist_data)
#imp.reload(deep_fx)
##imp.reload(sys.modules[deep_fx.__module__])
##reload(deep_fx)


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
data = hist_data.data()['Close']
x = data.index
y = data.values
sd = 1
upper, middle, lower = talib.BBANDS(data.values, timeperiod=20, matype=talib.MA_Type.SMA, nbdevup=sd, nbdevdn=sd)
[plt.plot(x, val) for val in [y, upper, middle, lower]]


# In[ ]:


data.values


# ## References
# 
# - [http://recruit.gmo.jp/engineer/jisedai/blog/deep-q-learning/](Deep Q-LearningでFXしてみた)

# In[ ]:




