
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
from episode_logger import EpisodeLogger
from test_output_logger import TestOutputLogger

from hist_data import HistData, BitcoinHistData
from fx_trade import FXTrade
from bitcoin_trade import BitcoinTrade
from deep_fx import DeepFX
from debug_tools import DebugTools


# In[ ]:


import crcmod
class LogRecordWithHexThereadID(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hex_threadid = self._calc_hex(self.process)

    def _calc_hex(self, digit_value):
        return hex(digit_value)

def init_logger(sd_loglevel=logging.WARN, stream_loglevel=logging.CRITICAL):
    logging.setLogRecordFactory(LogRecordWithHexThereadID)
    logger = logging.getLogger('deepfx')
    logger.setLevel(sd_loglevel)
    formatter = logging.Formatter('[%(hex_threadid)s] %(message)s')

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


deepfx_logger = init_logger(stream_loglevel=None)
deepfx_logger.critical('DeepFX Started: %s' % DebugTools.now_str())
deepfx_logger.debug   ('loglevel debug    test')
deepfx_logger.info    ('loglevel info     test')
deepfx_logger.warning ('loglevel warn     test')
deepfx_logger.error   ('loglevel error    test')
deepfx_logger.critical('loglevel critical test')


# In[ ]:


is_for_fx = False
is_for_bitcoin = True


# In[ ]:


if is_for_fx:
    hd = HistData(csv_path = 'historical_data/DAT_ASCII_USDJPY_M1_201710_h1.csv',
                     begin_date='2017-10-02T00:00:00',
                     end_date='2017-10-02T01:59:59')
elif is_for_bitcoin:
    is_to_train = False
    csv_path = 'historical_data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20_h1.csv'
    if is_to_train:
        begin_date='2017-09-01T00:00:00'
        end_date='2017-09-30T23:59:59'
    else:
        begin_date='2017-10-01T00:00:00'
        end_date='2017-10-10T23:59:59'
        
    hd = HistData(csv_path, begin_date, end_date)


# In[ ]:


hd.data()
len(hd.data())


# In[ ]:


if is_for_fx:
    env = FXTrade(1000000, 0.08, hd, logger=deepfx_logger)
    #env = FXTrade(1000000, 0.08, h, logger=logger)
    prepared_model_filename = None #'Keras-RL_DQN_FX_model_meanq1.440944e+06_episode00003.h5'
    dfx = DeepFX(env, prepared_model_filename=prepared_model_filename, steps = 100000, logger=deepfx_logger)
elif is_for_bitcoin:
    env = BitcoinTrade(10000000, None, hd, logger=deepfx_logger, amount_unit=0.001)
    #env = FXTrade(1000000, 0.08, h, logger=logger)
    prepared_model_filename = 'Keras-RL_DQN_FX_model_meanq2.149058e+07_episode05184.h5'
    dfx = DeepFX(env, prepared_model_filename=prepared_model_filename, steps = 10000000, logger=deepfx_logger)
    #dfx = DeepFX(env, prepared_model_filename=prepared_model_filename, steps = 1000, logger=deepfx_logger)


# In[ ]:


if is_to_train:
    dfx.train(is_for_time_measurement=True)
else:
    dfx.test([TestOutputLogger(hd)])


# In[ ]:


deepfx_logger.critical('DeepFX Finished: %s' % DebugTools.now_str())


# In[ ]:


import os
import subprocess
if os.environ.get('SLACK_WEBHOOK_URL') and os.environ.get('GOOGLE_STACKDRIVER_URL'):
    google_stackdriver_url = os.environ.get('GOOGLE_STACKDRIVER_URL')
    payload = '{"username":"deepfx","icon_emoji":":+1:","channel":"deepfx","attachments":[{"color":"#36a64f","title":"DeepFX Finished","title_link":"%s","text":"<@%s> DeepFX Finished"}]}' % (google_stackdriver_url, os.environ.get('SLACK_NOTIFY_RECIEVE_USER'))
    command = ['curl']
    command.append('-XPOST')
    command.append('-HContent-Type: application/json')
    command.append("-d%s" % payload)
    command.append(os.environ.get('SLACK_WEBHOOK_URL'))
    subprocess.run(command)
else:
    print('Skipped Slack Notification.')


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

# ## TODO
# 
# 足の配列について、indexの外を読み出そうとしている節があるので直す。
# 
# ```json
# {
#  insertId:  "1l630l2g1k8tnms"  
#  jsonPayload: {…}  
#  logName:  "projects/deep-fx/logs/deepfx"  
#  receiveTimestamp:  "2017-11-18T17:12:18.459939016Z"  
#  resource: {…}  
#  severity:  "WARNING"  
#  timestamp:  "2017-11-18T17:12:18.459939016Z"  
# }
# ```
