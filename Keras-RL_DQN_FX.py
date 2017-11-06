
# coding: utf-8

# [[Python] Keras-RLで簡単に強化学習(DQN)を試す](http://qiita.com/inoory/items/e63ade6f21766c7c2393)を参考に、エージェントを作成する。FXの自動取引を行い、利益を出すのが目標。

# In[ ]:


import matplotlib as mpl
mpl.use('tkagg')
import numpy as np
import pandas as pd
import talib
from logging import getLogger, DEBUG, INFO, WARN, ERROR, CRITICAL
import os
import logging
from logging import StreamHandler, LogRecord
import base64

from hist_data import HistData
from fx_trade import FXTrade
from deep_fx import DeepFX
from debug_tools import DebugTools


# In[ ]:


import crcmod
class LogRecordWithCRC16ThereadID(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crc16threadid = self._calc_crc16(self.process)

    def _calc_crc16(self, int_value):
        byte_array = str(int_value).encode('utf-8')
        crc16 = crcmod.Crc(0x18005)
        crc16.update(byte_array)
        return crc16.hexdigest()

def init_logger(sd_loglevel=logging.WARN, stream_loglevel=logging.CRITICAL):
    logging.setLogRecordFactory(LogRecordWithCRC16ThereadID)
    logger = logging.getLogger('deepfx')
    logger.setLevel(sd_loglevel)
    formatter = logging.Formatter('[%(crc16threadid)s] %(message)s')

    if sd_loglevel:
        import google
        from google.cloud.logging import Client
        from google.cloud.logging.handlers import CloudLoggingHandler
        client = google.cloud.logging.Client             .from_service_account_json(os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON_PATH'))
        handler = CloudLoggingHandler(client, name='deepfx')
        handler.setLevel(sd_loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = None

    if stream_loglevel:
        handler = StreamHandler()
        handler.setLevel(stream_loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = None

    return logger


# In[ ]:


deepfx_logger = init_logger()
deepfx_logger.critical('DeepFX Started: %s' % DebugTools.now_str())
deepfx_logger.debug   ('loglevel debug    test')
deepfx_logger.info    ('loglevel info     test')
deepfx_logger.warning ('loglevel warn     test')
deepfx_logger.error   ('loglevel error    test')
deepfx_logger.critical('loglevel critical test')


# In[ ]:


hd = HistData(csv_path = 'historical_data/DAT_ASCII_USDJPY_M1_201710_m5.csv',
                     begin_date='2017-10-02T00:00:00',
                     end_date='2017-10-09T23:59:59')


# In[ ]:


hd.data()
#len(hist_data.data())


# In[ ]:


env = FXTrade(1000000, 0.08, hd, logger=logger)
#env = FXTrade(1000000, 0.08, h, logger=logger)
prepared_model_filename = None #'Keras-RL_DQN_FX_model_meanq1.440944e+06_episode00003.h5'
dfx = DeepFX(env, prepared_model_filename=prepared_model_filename, steps = 100000, logger=deepfx_logger)


# In[ ]:


is_to_train = True
if is_to_train:
    dfx.train(is_for_time_measurement=True)
else:
    dfx.test(1, [EpisodeLogger()])


# In[ ]:


deepfx_logger.critical('DeepFX Finished: %s' % DebugTools.now_str())


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


# ## References
# 
# - [Deep Q-LearningでFXしてみた](http://recruit.gmo.jp/engineer/jisedai/blog/deep-q-learning/)
# - [slide](https://www.slideshare.net/JunichiroKatsuta/deep-qlearningfx)
