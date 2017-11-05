
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[ ]:


import matplotlib as mpl
mpl.use('tkagg')
import numpy as np
import pandas as pd
import talib
from logging import getLogger, DEBUG, INFO, WARN
import os

from hist_data import HistData
from fx_trade import FXTrade
from deep_fx import DeepFX
from debug_tools import DebugTools


# In[ ]:


IS_SEND_LOG_TO_STACKDRIVER = True
if IS_SEND_LOG_TO_STACKDRIVER:
    import logging
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    jupyter_logger = logging.getLogger()
    jupyter_logger.setLevel(logging.WARN)
    client = google.cloud.logging.Client         .from_service_account_json(os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON_PATH'))
    client.setup_logging(log_level=logging.INFO)
    handler = CloudLoggingHandler(client, name='deepfx')
    logger = logging.getLogger()
else:
    from logging import StreamHandler
    logger = getLogger(__name__)
    handler = StreamHandler()
    logger.setLevel(INFO)
logger.addHandler(handler)
logger.info('DeepFX Started: %s' % DebugTools.now_str())


# In[ ]:


#import imp
#import sys
#del(hist_data)
#from hist_data import HistData
#del(hist_data)
#imp.reload(hist_data)
#imp.reload(sys.modules[hist_data.__module__])
hd = HistData(csv_path = 'historical_data/DAT_ASCII_USDJPY_M1_201710_m5.csv',
                     begin_date='2017-10-02T00:00:00',
                     end_date='2017-10-02T23:59:59')
                     #end_date='2017-10-09T23:59:59')


# In[ ]:


hd.data()
#len(hist_data.data())


# In[ ]:


env = FXTrade(1000000, 0.08, hd, logger=logger)
#env = FXTrade(1000000, 0.08, h, logger=logger)
prepared_model_filename = None #'Keras-RL_DQN_FX_model_meanq1.440944e+06_episode00003.h5'
dfx = DeepFX(env, prepared_model_filename=prepared_model_filename, steps = 100000, logger=logger)


# In[ ]:


is_to_train = True
if is_to_train:
    dfx.train(is_for_time_measurement=True)
else:
    dfx.test(1, [EpisodeLogger()])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
data = hd.data()['Close']
x = data.index
y = data.values
sd = 1
upper, middle, lower = talib.BBANDS(data.values, timeperiod=20, matype=talib.MA_Type.SMA, nbdevup=sd, nbdevdn=sd)
[plt.plot(x, val) for val in [y, upper, middle, lower]]


# In[ ]:


data.values


# In[ ]:


logger.info('DeepFX Finished: %s' % DebugTools.now_str())


# ## References
# 
# - [Deep Q-LearningでFXしてみた](http://recruit.gmo.jp/engineer/jisedai/blog/deep-q-learning/)
# - [slide](https://www.slideshare.net/JunichiroKatsuta/deep-qlearningfx)
